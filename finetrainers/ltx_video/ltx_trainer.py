import os
import sys

base_repo_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.join(base_repo_path, "finetrainers"))

from trainer import Trainer
from constants import FINETRAINERS_LOG_LEVEL
from dataset import BucketSampler, VideoDatasetWithResizing
from dataclasses import dataclass
from utils.memory_utils import make_contiguous
from accelerate.logging import get_logger
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
import random
import torch
from peft import LoraConfig

logger = get_logger("finetrainers")
logger.setLevel(FINETRAINERS_LOG_LEVEL)


@dataclass
class LTXTrainingOutput:
    preds: torch.Tensor 
    targets: torch.Tensor
    sigmas: torch.Tensor


class LTXTrainer(Trainer):
    def prepare_models(self):
        logger.info("Initializing models")

        load_components_kwargs = {
            "text_encoder_dtype": torch.bfloat16,
            "transformer_dtype": torch.bfloat16,
            "vae_dtype": torch.bfloat16,
            "revision": self.args.revision,
            "cache_dir": self.args.cache_dir,
        }
        if self.args.pretrained_model_name_or_path is not None:
            load_components_kwargs["model_id"] = self.args.pretrained_model_name_or_path
        components = self._model_config_call(self.model_config["load_components"], load_components_kwargs)

        self.tokenizer = components.get("tokenizer")
        self.text_encoder = components.get("text_encoder")
        self.transformer = components.get("transformer")
        self.vae = components.get("vae")
        self.scheduler = components.get("scheduler")

        if self.vae is not None:
            if self.args.enable_slicing:
                self.vae.enable_slicing()
            if self.args.enable_tiling:
                self.vae.enable_tiling()

        self.transformer_config = self.transformer.config if self.transformer is not None else None

    def prepare_dataset(self):
        logger.info("Initializing dataset and dataloader")

        self.dataset = VideoDatasetWithResizing(
            data_root=self.args.data_root,
            caption_column=self.args.caption_column,
            video_column=self.args.video_column,
            resolution_buckets=self.args.video_resolution_buckets,
            dataset_file=self.args.dataset_file,
            id_token=self.args.id_token,
        )
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=1,
            sampler=BucketSampler(self.dataset, batch_size=self.args.batch_size, shuffle=True),
            collate_fn=self.model_config.get("collate_fn"),
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.pin_memory,
        )

    def prepare_trainable_parameters(self) -> None:
        logger.info("Initializing trainable parameters")

        self.text_encoder.requires_grad_(False)
        self.transformer.requires_grad_(False)
        self.vae.requires_grad_(False)

        # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        weight_dtype = torch.float32
        if self.state.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.state.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
            # due to pytorch#99272, MPS does not yet support bfloat16.
            raise ValueError(
                "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
            )

        # TODO: handle torch dtype from accelerator vs model dtype; refactor
        self.state.weight_dtype = weight_dtype
        self.text_encoder.to(self.state.accelerator.device, dtype=weight_dtype)
        self.transformer.to(self.state.accelerator.device, dtype=weight_dtype)
        self.vae.to(self.state.accelerator.device, dtype=weight_dtype)

        if self.args.gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()

        transformer_lora_config = LoraConfig(
            r=self.args.rank,
            lora_alpha=self.args.lora_alpha,
            init_lora_weights=True,
            target_modules=self.args.target_modules,
        )
        self.transformer.add_adapter(transformer_lora_config)

    def run_forward_pass_and_calculate_preds(
            self, batch, accelerator, weight_dtype, generator, scheduler_sigmas, return_dict=True
        ):
        videos = batch["videos"]
        prompts = batch["prompts"]
        batch_size = len(prompts)

        if self.args.caption_dropout_technique == "empty":
            if random.random() < self.args.caption_dropout_p:
                prompts = [""] * batch_size

        latent_conditions = self.model_config["prepare_latents"](
            vae=self.vae,
            image_or_video=videos,
            patch_size=self.transformer_config.patch_size,
            patch_size_t=self.transformer_config.patch_size_t,
            device=accelerator.device,
            dtype=weight_dtype,
            generator=generator,
        )
        latent_conditions = make_contiguous(latent_conditions)

        other_conditions = self.model_config["prepare_conditions"](
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            prompt=prompts,
            device=accelerator.device,
            dtype=weight_dtype,
        )
        other_conditions = make_contiguous(other_conditions)

        if self.args.caption_dropout_technique == "zero":
            if random.random() < self.args.caption_dropout_p:
                other_conditions["prompt_embeds"].fill_(0)
                other_conditions["prompt_attention_mask"].fill_(False)

                # TODO: refactor later
                if "pooled_prompt_embeds" in other_conditions:
                    other_conditions["pooled_prompt_embeds"].fill_(0)

        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.args.flow_weighting_scheme,
            batch_size=batch_size,
            logit_mean=self.args.flow_logit_mean,
            logit_std=self.args.flow_logit_std,
            mode_scale=self.args.flow_mode_scale,
        )
        indices = (u * self.scheduler.config.num_train_timesteps).long()
        sigmas = scheduler_sigmas[indices]
        timesteps = (sigmas * 1000.0).long()

        noise = torch.randn(
            latent_conditions["latents"].shape,
            generator=generator,
            device=accelerator.device,
            dtype=weight_dtype,
        )
        noisy_latents = (1.0 - sigmas) * latent_conditions["latents"] + sigmas * noise

        latent_conditions.update({"noisy_latents": noisy_latents})
        other_conditions.update({"timesteps": timesteps})

        # These weighting schemes use a uniform timestep sampling and instead post-weigh the loss
        pred = self.model_config["forward_pass"](
            transformer=self.transformer, **latent_conditions, **other_conditions
        )
        target = noise - latent_conditions["latents"]
        
        if not return_dict:
            return pred, target

        return LTXTrainingOutput(preds=pred, targets=target, sigmas=sigmas)
    
    def calculate_loss_weights(self, sigmas):
        weights = compute_loss_weighting_for_sd3(
            weighting_scheme=self.args.flow_weighting_scheme, sigmas=sigmas
        )
        return weights

    def calculate_loss(self, weights, preds, targets):
        loss = weights.float() * (preds["latents"].float() - targets.float()).pow(2)
        # Average loss across channel dimension
        loss = loss.mean(list(range(1, loss.ndim)))
        # Average loss across batch dimension
        loss = loss.mean()
        return loss

        
