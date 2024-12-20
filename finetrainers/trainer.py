import os
import sys

base_repo_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.join(base_repo_path, "finetrainers"))

import inspect
import json
import logging
import math
import os
from datetime import timedelta
from pathlib import Path

import diffusers
import torch
import torch.backends
import transformers
import wandb
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
    gather_object,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import export_to_video, load_image, load_video
from huggingface_hub import create_repo
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from tqdm import tqdm

from finetrainers.args import Args, validate_args
from finetrainers.constants import FINETRAINERS_LOG_LEVEL
from finetrainers.models import get_config_from_model_name
from finetrainers.state import State
from finetrainers.utils.file_utils import find_files, delete_files, string_to_filename
from finetrainers.utils.optimizer_utils import get_optimizer
from finetrainers.utils.memory_utils import get_memory_statistics, free_memory
from finetrainers.utils.torch_utils import unwrap_model


logger = get_logger("finetrainers")
logger.setLevel(FINETRAINERS_LOG_LEVEL)


class Trainer:
    def __init__(self, args: Args) -> None:
        validate_args(args)

        self.args = args
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
        self.model_config = get_config_from_model_name(self.args.model_name, self.args.training_type)

    def prepare_models(self) -> None:
        raise NotImplementedError

    def prepare_dataset(self) -> None:
        raise NotImplementedError

    def prepare_trainable_parameters(self) -> None:
        raise NotImplementedError

    # TODO: DeepSpeed support
    def save_model_hook(self, models, weights, output_dir):
        if self.state.accelerator.is_main_process:
            transformer_lora_layers_to_save = None

            for model in models:
                if isinstance(
                    unwrap_model(self.state.accelerator, model),
                    type(unwrap_model(self.state.accelerator, self.transformer)),
                ):
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

    # TODO: Only `transformer` for now. DeepSpeed support.
    def load_model_hook(self, models, input_dir):
        if not hasattr(self, "transformer_lora_config"):
            raise ValueError("Need `transformer_lora_config`.")
        
        transformer_lora_config = getattr(self, "transformer_lora_config")
        transformer_ = self.model_config["pipeline_cls"].from_pretrained(
            self.args.pretrained_model_name_or_path, subfolder="transformer"
        )
        transformer_.add_adapter(transformer_lora_config)

        lora_state_dict = self.model_config["pipeline_cls"].lora_state_dict(input_dir)

        transformer_state_dict = {
            f'{k.replace("transformer.", "")}': v
            for k, v in lora_state_dict.items()
            if k.startswith("transformer.")
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

    def register_save_load_hooks(self):    
        self.state.accelerator.register_save_state_pre_hook(self.save_model_hook)
        self.state.accelerator.register_load_state_pre_hook(self.load_model_hook)

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
            self.state.learning_rate = (
                self.state.learning_rate
                * self.args.gradient_accumulation_steps
                * self.args.batch_size
                * self.state.accelerator.num_processes
            )

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

    def calculate_loss(self, **kwargs):
        raise NotImplementedError
    
    def run_forward_pass_and_calculate_preds(self, **kwargs):
        raise NotImplementedError
    
    def calculate_loss_weights(self, **kwargs):
        raise NotImplementedError
    
    def sort_out_checkpoint_to_resume_from(self, accelerator):
        if self.args.resume_from_checkpoint != "latest":
            path = os.path.basename(self.args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(self.args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{self.args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            self.args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            num_update_steps_per_epoch = math.ceil(len(self.dataloader) / self.args.gradient_accumulation_steps)
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(self.args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

        return initial_global_step, global_step, first_epoch
    
    def save_intermediate_checkpoint(self, step, accelerator):
        if step % self.args.checkpointing_steps == 0:
            # before saving state, check if this save would set us over the `checkpointing_limit`
            if self.args.checkpointing_limit is not None:
                checkpoints = find_files(self.args.output_dir, prefix="checkpoint")

                # before we save the new checkpoint, we need to have at_most `checkpoints_total_limit - 1` checkpoints
                if len(checkpoints) >= self.args.checkpointing_limit:
                    num_to_remove = len(checkpoints) - self.args.checkpointing_limit + 1
                    checkpoints_to_remove = checkpoints[0:num_to_remove]
                    delete_files(checkpoints_to_remove)

            logger.info(f"Checkpointing at step {step}")
            save_path = os.path.join(self.args.output_dir, f"checkpoint-{step}")
            accelerator.save_state(save_path)
            logger.info(f"Saved state to {save_path}")

    def save_final_checkpoint(self, accelerator):
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
        logger.info(f"Checkpoint saved to {self.args.output_dir}.")

    def train(self) -> None:
        logger.info("Starting training")

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before training start: {json.dumps(memory_statistics, indent=4)}")

        self.state.train_batch_size = (
            self.args.batch_size * self.state.accelerator.num_processes * self.args.gradient_accumulation_steps
        )
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

        if not self.args.resume_from_checkpoint:
            initial_global_step = 0
            global_step = 0
            first_epoch = 0
        else:
            initial_global_step, global_step, first_epoch = self.sort_out_checkpoint_to_resume_from(accelerator=accelerator)

        progress_bar = tqdm(
            range(0, self.state.train_steps),
            initial=initial_global_step,
            desc="Training steps",
            disable=not self.state.accelerator.is_local_main_process,
        )

        accelerator = self.state.accelerator
        generator = torch.Generator(device=accelerator.device)
        if self.args.seed is not None:
            generator = generator.manual_seed(self.args.seed)
        self.state.generator = generator
        scheduler_sigmas = self.scheduler.sigmas.clone().to(device=accelerator.device, dtype=self.state.weight_dtype)

        for epoch in range(first_epoch, self.state.train_epochs):
            logger.debug(f"Starting epoch ({epoch + 1}/{self.state.train_epochs})")

            self.transformer.train()
            models_to_accumulate = [self.transformer]

            for step, batch in enumerate(self.dataloader):
                logger.debug(f"Starting step {step + 1}")
                logs = {}

                with accelerator.accumulate(models_to_accumulate):
                    forward_out = self.run_forward_pass_and_calculate_preds(
                        batch, 
                        accelerator, 
                        weight_dtype=self.state.weight_dtype, 
                        generator=generator, 
                        scheduler_sigmas=scheduler_sigmas
                    )
                    weights = self.calculate_loss_weights(sigmas=forward_out.sigmas)
                    loss = self.calculate_loss(weights, forward_out.preds, forward_out.targets)
                    accelerator.backward(loss)

                    if accelerator.sync_gradients and accelerator.distributed_type != DistributedType.DEEPSPEED:
                        accelerator.clip_grad_norm_(self.transformer.parameters(), self.args.max_grad_norm)

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                # Checkpointing
                if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
                    self.save_intermediate_checkpoint(step=global_step, accelerator=accelerator)

                # Maybe run validation
                should_run_validation = (
                    self.args.validation_every_n_steps is not None
                    and global_step % self.args.validation_every_n_steps == 0
                )
                if should_run_validation:
                    self.validate(global_step)

                # Log stuff to tracker.
                logs = {"loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(logs)
                accelerator.log(logs, step=global_step)

                if global_step >= self.state.train_steps:
                    break

            memory_statistics = get_memory_statistics()
            logger.info(f"Memory after epoch {epoch + 1}: {json.dumps(memory_statistics, indent=4)}")

            # Maybe run validation
            should_run_validation = (
                self.args.validation_every_n_epochs is not None
                and (epoch + 1) % self.args.validation_every_n_epochs == 0
            )
            if should_run_validation:
                self.validate(global_step)

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            self.save_final_checkpoint(accelerator=accelerator)

        del self.tokenizer, self.text_encoder, self.transformer, self.vae, self.scheduler
        free_memory()
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after training end: {json.dumps(memory_statistics, indent=4)}")
        accelerator.end_training()

    def validate(self, step: int) -> None:
        logger.info("Starting validation")

        accelerator = self.state.accelerator
        num_validation_samples = len(self.args.validation_prompts)

        if num_validation_samples == 0:
            logger.warning("No validation samples found. Skipping validation.")
            return

        self.transformer.eval()

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before validation start: {json.dumps(memory_statistics, indent=4)}")

        pipeline = self.model_config["initialize_pipeline"](
            model_id=self.args.pretrained_model_name_or_path,
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            tokenizer_2=self.tokenizer_2,
            text_encoder_2=self.text_encoder_2,
            transformer=unwrap_model(accelerator, self.transformer),
            vae=self.vae,
            device=accelerator.device,
            revision=self.args.revision,
            cache_dir=self.args.cache_dir,
            enable_slicing=self.args.enable_slicing,
            enable_tiling=self.args.enable_tiling,
            enable_model_cpu_offload=self.args.enable_model_cpu_offload,
        )

        all_processes_artifacts = []
        for i in range(num_validation_samples):
            # Skip current validation on all processes but one
            if i % accelerator.num_processes != accelerator.process_index:
                continue

            prompt = self.args.validation_prompts[i]
            image = self.args.validation_images[i]
            video = self.args.validation_videos[i]
            height = self.args.validation_heights[i]
            width = self.args.validation_widths[i]
            num_frames = self.args.validation_num_frames[i]

            if image is not None:
                image = load_image(image)
            if video is not None:
                video = load_video(video)

            logger.debug(
                f"Validating sample {i + 1}/{num_validation_samples} on process {accelerator.process_index}. Prompt: {prompt}",
                main_process_only=False,
            )
            validation_artifacts = self.model_config["validation"](
                pipeline=pipeline,
                prompt=prompt,
                image=image,
                video=video,
                height=height,
                width=width,
                num_frames=num_frames,
                num_videos_per_prompt=self.args.num_validation_videos_per_prompt,
                generator=self.state.generator,
            )

            prompt_filename = string_to_filename(prompt)[:25]
            artifacts = {
                "image": {"type": "image", "value": image},
                "video": {"type": "video", "value": video},
            }
            for i, (artifact_type, artifact_value) in enumerate(validation_artifacts):
                artifacts.update({f"artifact_{i}": {"type": artifact_type, "value": artifact_value}})
            logger.debug(
                f"Validation artifacts on process {accelerator.process_index}: {list(artifacts.keys())}",
                main_process_only=False,
            )

            for key, value in list(artifacts.items()):
                artifact_type = value["type"]
                artifact_value = value["value"]
                if artifact_type not in ["image", "video"] or artifact_value is None:
                    continue

                extension = "png" if artifact_type == "image" else "mp4"
                filename = f"validation-{step}-{accelerator.process_index}-{prompt_filename}.{extension}"
                filename = os.path.join(self.args.output_dir, filename)

                if artifact_type == "image":
                    logger.debug(f"Saving image to {filename}")
                    artifact_value.save(filename)
                    artifact_value = wandb.Image(filename)
                elif artifact_type == "video":
                    logger.debug(f"Saving video to {filename}")
                    export_to_video(artifact_value, filename, fps=15)
                    artifact_value = wandb.Video(filename, caption=prompt)

                all_processes_artifacts.append(artifact_value)

        all_artifacts = gather_object(all_processes_artifacts)

        if accelerator.is_main_process:
            for tracker in accelerator.trackers:
                if tracker.name == "wandb":
                    tracker.log({"validation": all_artifacts}, step=step)

        accelerator.wait_for_everyone()
        free_memory()
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after validation end: {json.dumps(memory_statistics, indent=4)}")
        self.transformer.train()

    def evaluate(self) -> None:
        raise NotImplementedError

    def _init_distributed(self) -> None:
        logging_dir = Path(self.args.output_dir, self.args.logging_dir)
        project_config = ProjectConfiguration(project_dir=self.args.output_dir, logging_dir=logging_dir)
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_process_group_kwargs = InitProcessGroupKwargs(
            backend="nccl", timeout=timedelta(seconds=self.args.nccl_timeout)
        )
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

        # Enable TF32 for faster training on Ampere GPUs: https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if self.args.allow_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True

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
