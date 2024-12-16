import inspect
import json
import logging
import math
import os
import random
import shutil
from datetime import timedelta
from typing import Any, Dict
from pathlib import Path

import diffusers
import torch
import torch.backends
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs, ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from huggingface_hub import create_repo, upload_folder
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from tqdm import tqdm

from .args import Args
from .constants import FINETRAINERS_LOG_LEVEL
from .dataset import BucketSampler, VideoDatasetWithResizing
from .models import get_config_from_model_name
from .state import State
from .utils.optimizer_utils import get_optimizer, gradient_norm
from .utils.memory_utils import bytes_to_gigabytes, make_contiguous
from .utils.torch_utils import unwrap_model


logger = get_logger("finetrainers")
logger.setLevel(FINETRAINERS_LOG_LEVEL)

class Trainer:
    def __init__(self, args: Args) -> None:
        self.args = args
        _validate_args(args)
        
        self.state = State()
        
        # Tokenizers
        self.tokenizer = None
        self.tokenizer_2 = None
        self.tokenizer_3 = None
        
        # Text encoders
        self.text_encoder = None
        self.text_encoder_2 = None
        self.text_encoder_3 = None
        
        # Denoisers
        self.transformer = None
        self.unet = None

        # Autoencoders
        self.vae = None

        self._init_distributed()
        self._init_logging()
        self._init_directories_and_repositories()

        self.state.model_name = self.args.model_name
        self.model_config = get_config_from_model_name(self.args.model_name)
    
    def get_memory_statistics(self, precision: int = 3) -> Dict[str, Any]:
        memory_allocated = None
        memory_reserved = None
        max_memory_allocated = None
        max_memory_reserved = None
        
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            memory_allocated = torch.cuda.memory_allocated(device)
            memory_reserved = torch.cuda.memory_reserved(device)
            max_memory_allocated = torch.cuda.max_memory_allocated(device)
            max_memory_reserved = torch.cuda.max_memory_reserved(device)
        
        elif torch.mps.is_available():
            memory_allocated = torch.mps.current_allocated_memory()
        
        else:
            logger.warning("No CUDA, MPS, or ROCm device found. Memory statistics are not available.")
        
        return {
            "memory_allocated": round(bytes_to_gigabytes(memory_allocated), ndigits=precision),
            "memory_reserved": round(bytes_to_gigabytes(memory_reserved), ndigits=precision),
            "max_memory_allocated": round(bytes_to_gigabytes(max_memory_allocated), ndigits=precision),
            "max_memory_reserved": round(bytes_to_gigabytes(max_memory_reserved), ndigits=precision),
        }

    def prepare_models(self) -> None:
        logger.info("Initializing models")

        # TODO(aryan): refactor in future
        load_components_kwargs = {
            "text_encoder_dtype": torch.bfloat16,
            "transformer_dtype": torch.bfloat16,
            "vae_dtype": torch.bfloat16,
            "cache_dir": self.args.cache_dir,
        }
        if self.args.pretrained_model_name_or_path is not None:
            load_components_kwargs["model_id"] = self.args.pretrained_model_name_or_path
        components = self._model_config_call(self.model_config["load_components"], load_components_kwargs)

        self.tokenizer = components.get("tokenizer", None)
        self.text_encoder = components.get("text_encoder", None)
        self.transformer = components.get("transformer", None)
        self.vae = components.get("vae", None)
        self.scheduler = components.get("scheduler", None)

    def prepare_dataset(self) -> None:
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

        # TODO(aryan): refactor later. for now only lora is supported
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
        
        # TODO(aryan): handle torch dtype from accelerator vs model dtype
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

        # TODO: refactor
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if self.state.accelerator.is_main_process:
                transformer_lora_layers_to_save = None

                for model in models:
                    if isinstance(unwrap_model(self.state.accelerator, model), type(unwrap_model(self.state.accelerator, self.transformer))):
                        model = unwrap_model(self.state.accelerator, model)
                        transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                    else:
                        raise ValueError(f"Unexpected save model: {model.__class__}")

                    # make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()

                self.model_config["pipeline_cls"].save_lora_weights(
                    output_dir,
                    transformer_lora_layers=transformer_lora_layers_to_save,
                )

        def load_model_hook(models, input_dir):
            transformer_ = self.model_config["pipeline_cls"].from_pretrained(
                self.args.pretrained_model_name_or_path, subfolder="transformer"
            )
            transformer_.add_adapter(transformer_lora_config)

            lora_state_dict = self.model_config["pipeline_cls"].lora_state_dict(input_dir)

            transformer_state_dict = {
                f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.")
            }
            incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
            if incompatible_keys is not None:
                # check only for unexpected keys
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    logger.warning(
                        f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                        f" {unexpected_keys}. "
                    )

            # Make sure the trainable params are in float32. This is again needed since the base models
            # are in `weight_dtype`. More details:
            # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
            if self.args.mixed_precision == "fp16":
                # only upcast trainable parameters (LoRA) into fp32
                cast_training_params([transformer_])

        self.state.accelerator.register_save_state_pre_hook(save_model_hook)
        self.state.accelerator.register_load_state_pre_hook(load_model_hook)

        # Enable TF32 for faster training on Ampere GPUs: https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if self.args.allow_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
    
    def prepare_optimizer(self) -> None:
        logger.info("Initializing optimizer and lr scheduler")

        self.state.train_epochs = self.args.train_epochs
        self.state.train_steps = self.args.train_steps

        # Make sure the trainable params are in float32
        if self.args.mixed_precision == "fp16":
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params([self.transformer], dtype=torch.float32)
        
        self.state.learning_rate = self.args.lr
        if self.args.scale_lr:
            self.state.learning_rate = self.state.learning_rate * self.args.gradient_accumulation_steps * self.args.batch_size * self.state.accelerator.num_processes
        
        transformer_lora_parameters = list(filter(lambda p: p.requires_grad, self.transformer.parameters()))
        transformer_parameters_with_lr = {
            "params": transformer_lora_parameters,
            "lr": self.state.learning_rate,
        }
        params_to_optimize = [transformer_parameters_with_lr]
        self.state.num_trainable_parameters = sum(p.numel() for p in transformer_lora_parameters)

        # TODO(aryan): add deepspeed support
        optimizer = get_optimizer(
            params_to_optimize=params_to_optimize,
            optimizer_name=self.args.optimizer,
            learning_rate=self.args.lr,
            beta1=self.args.beta1,
            beta2=self.args.beta2,
            beta3=self.args.beta3,
            epsilon=self.args.epsilon,
            weight_decay=self.args.weight_decay,
        )

        num_update_steps_per_epoch = math.ceil(len(self.dataloader) / self.args.gradient_accumulation_steps)
        if self.state.train_steps is None:
            self.state.train_steps = self.state.train_epochs * num_update_steps_per_epoch
            self.state.overwrote_max_train_steps = True

        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.args.lr_warmup_steps * self.state.accelerator.num_processes,
            num_training_steps=self.state.train_steps * self.state.accelerator.num_processes,
            num_cycles=self.args.lr_num_cycles,
            power=self.args.lr_power,
        )

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
    
    def prepare_for_training(self) -> None:
        self.transformer, self.optimizer, self.dataloader, self.lr_scheduler = self.state.accelerator.prepare(
            self.transformer, self.optimizer, self.dataloader, self.lr_scheduler
        )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(self.dataloader) / self.args.gradient_accumulation_steps)
        if self.state.overwrote_max_train_steps:
            self.state.train_steps = self.state.train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.state.train_epochs = math.ceil(self.state.train_steps / num_update_steps_per_epoch)
    
    def prepare_trackers(self) -> None:
        logger.info("Initializing trackers")

        tracker_name = self.args.tracker_name or "finetrainers-experiment"
        self.state.accelerator.init_trackers(tracker_name, config=self.args.to_dict())
    
    def train(self) -> None:
        logger.info("Starting training")

        memory_statistics = self.get_memory_statistics()
        logger.info(f"Memory before training start: {json.dumps(memory_statistics, indent=4)}")

        self.state.train_batch_size = self.args.batch_size * self.state.accelerator.num_processes * self.args.gradient_accumulation_steps
        info = {
            "trainable parameters": self.state.num_trainable_parameters,
            "total samples": len(self.dataset),
            "train epochs": self.state.train_epochs,
            "train steps": self.state.train_steps,
            "batches per device": self.args.batch_size,
            "total batches observed per epoch": len(self.dataloader),
            "train batch size": self.state.train_batch_size,
            "gradient accumulation steps": self.args.gradient_accumulation_steps,
        }
        logger.info(f"Training configuration: {json.dumps(info, indent=4)}")

        # TODO(aryan): handle resume from checkpoint

        global_step = 0
        first_epoch = 0
        initial_global_step = 0
        progress_bar = tqdm(range(0, self.state.train_steps), initial=initial_global_step, desc="Training steps", disable=not self.state.accelerator.is_local_main_process)

        accelerator = self.state.accelerator
        weight_dtype = self.state.weight_dtype
        transformer_config = self.transformer.config
        scheduler_sigmas = self.scheduler.sigmas.clone().to(device=accelerator.device, dtype=weight_dtype)
        generator = torch.Generator(device=accelerator.device)
        if self.args.seed is not None:
            generator = generator.manual_seed(self.args.seed)
        
        for epoch in range(first_epoch, self.state.train_epochs):
            logger.debug(f"Starting epoch ({epoch + 1}/{self.state.train_epochs})")
            
            self.transformer.train()
            models_to_accumulate = [self.transformer]

            for step, batch in enumerate(self.dataloader):
                logger.debug(f"Starting step ({step + 1})")
                logs = {}

                with accelerator.accumulate(models_to_accumulate):
                    videos = batch["videos"]
                    prompts = batch["prompts"]
                    batch_size = len(prompts)

                    if self.args.caption_dropout_technique == "empty":
                        if random.random() < self.args.caption_dropout_p:
                            prompts = [""] * batch_size

                    latent_conditions = self.model_config["prepare_latents"](
                        vae=self.vae,
                        image_or_video=videos,
                        patch_size=transformer_config.patch_size,
                        patch_size_t=transformer_config.patch_size_t,
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
                    
                    # These weighting schemes use a uniform timestep sampling and instead post-weight the loss
                    weights = compute_density_for_timestep_sampling(
                        weighting_scheme=self.args.flow_weighting_scheme,
                        batch_size=batch_size,
                        logit_mean=self.args.logit_mean,
                        logit_std=self.args.logit_std,
                        mode_scale=self.args.mode_scale,
                    )
                    indices = (weights * self.scheduler.config_num_train_timesteps).long()
                    sigmas = scheduler_sigmas[indices]
                    timesteps = (sigmas * 1000.0).long()

                    noise = torch.randn(latent_conditions["latents"].shape, generator=generator, device=accelerator.device, dtype=weight_dtype)
                    noisy_latents = (1.0 - sigmas) * latent_conditions["latents"] + sigmas * noise
                    
                    latent_conditions.update({"noisy_latents": noisy_latents})
                    other_conditions.update({"timesteps": timesteps})

                    # These weighting schemes use a uniform timestep sampling and instead post-weight the loss
                    weights = compute_loss_weighting_for_sd3(weighting_scheme=self.args.flow_weighting_scheme, sigmas=sigmas)
                    pred = self.model_config["forward_pass"](transformer=self.transformer, **latent_conditions, **other_conditions)
                    target = noise - latent_conditions["latents"]
                    
                    loss = weights.float() * (pred["latents"].float() - target.float()).pow(2)
                    # Average loss across channel dimension
                    loss = loss.mean(list(range(1, loss.ndim)))
                    # Average loss across batch dimension
                    loss = loss.mean()

                    accelerator.backward(loss)
                    
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                
                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    # Checkpointing
                    if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
                        logger.info(f"Checkpointing at step {global_step}")
                        if global_step % self.args.checkpointing_steps == 0:
                            # _before_ saving state, check if this save would set us over the `checkpointing_limit`
                            if self.args.checkpointing_limit is not None:
                                checkpoints = os.listdir(self.args.output_dir)
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                                if len(checkpoints) >= self.args.checkpointing_limit:
                                    num_to_remove = len(checkpoints) - self.args.checkpointing_limit + 1
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    logger.info(
                                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                    )
                                    logger.info(f"Removing checkpoints: {', '.join(removing_checkpoints)}")

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(self.args.output_dir, removing_checkpoint)
                                        shutil.rmtree(removing_checkpoint)

                            save_path = os.path.join(self.args.output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")
                    
                logs = {"loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(logs)
                accelerator.log(logs, step=global_step)

                if global_step >= self.state.train_steps:
                    break
    
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            self.transformer = unwrap_model(accelerator, self.transformer)
            dtype = (
                torch.float16
                if self.args.mixed_precision == "fp16"
                else torch.bfloat16
                if self.args.mixed_precision == "bf16"
                else torch.float32
            )
            self.transformer = self.transformer.to(dtype)
            transformer_lora_layers = get_peft_model_state_dict(self.transformer)

            self.model_config["pipeline_cls"].save_lora_weights(
                save_directory=self.args.output_dir,
                transformer_lora_layers=transformer_lora_layers,
            )
        
        del self.tokenizer, self.text_encoder, self.transformer, self.vae, self.scheduler

    
    def evaluate(self) -> None:
        logger.info("Starting evaluation")
        # TODO: implement metrics for evaluation
    
    def _init_distributed(self) -> None:
        logging_dir = Path(self.args.output_dir, self.args.logging_dir)
        project_config = ProjectConfiguration(project_dir=self.args.output_dir, logging_dir=logging_dir)
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_process_group_kwargs = InitProcessGroupKwargs(backend="nccl", timeout=timedelta(seconds=self.args.nccl_timeout))
        mixed_precision = "no" if torch.backends.mps.is_available() else self.args.mixed_precision
        report_to = None if self.args.report_to.lower() == "none" else self.args.report_to

        accelerator = Accelerator(
            project_config=project_config,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with=report_to,
            kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
        )

        # Disable AMP for MPS.
        if torch.backends.mps.is_available():
            accelerator.native_amp = False
        
        self.state.accelerator = accelerator

        if self.args.seed is not None:
            self.state.seed = self.args.seed
            set_seed(self.args.seed)
    
    def _init_logging(self) -> None:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=FINETRAINERS_LOG_LEVEL,
        )
        if self.state.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()
        
        logger.info("Initialized FineTrainers")
        logger.info(self.state.accelerator.state, main_process_only=False)

    def _init_directories_and_repositories(self) -> None:
        if self.state.accelerator.is_main_process:
            self.args.output_dir = Path(self.args.output_dir)
            self.args.output_dir.mkdir(parents=True, exist_ok=True)
            self.state.output_dir = self.args.output_dir

            if self.args.push_to_hub:
                repo_id = self.args.hub_model_id or Path(self.args.output_dir).name
                self.state.repo_id = create_repo(token=self.args.hub_token, name=repo_id).repo_id
    
    def _model_config_call(self, fn, kwargs):
        accepted_kwargs = inspect.signature(fn).parameters.keys()
        kwargs = {k: v for k, v in kwargs.items() if k in accepted_kwargs}
        return fn(**kwargs)


def _validate_args(args: Args):
    return True
