from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from diffusers import AutoencoderKLLTXVideo, FlowMatchEulerDiscreteScheduler, LTXPipeline, LTXVideoTransformer3DModel
from diffusers.utils import logging
from transformers import T5EncoderModel, T5Tokenizer

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def load_components(
    model_id: str = "Lightricks/LTX-Video",
    text_encoder_dtype: torch.dtype = torch.bfloat16,
    transformer_dtype: torch.dtype = torch.bfloat16,
    vae_dtype: torch.dtype = torch.bfloat16,
    cache_dir: Optional[str] = None,
) -> Dict[str, nn.Module]:
    tokenizer = T5Tokenizer.from_pretrained(model_id, subfolder="tokenizer", cache_dir=cache_dir)
    text_encoder = T5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=text_encoder_dtype, cache_dir=cache_dir)
    transformer = LTXVideoTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=transformer_dtype, cache_dir=cache_dir)
    vae = AutoencoderKLLTXVideo.from_pretrained(model_id, subfolder="vae", torch_dtype=vae_dtype, cache_dir=cache_dir)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler", cache_dir=cache_dir)
    return {"tokenizer": tokenizer, "text_encoder": text_encoder, "transformer": transformer, "vae": vae, "scheduler": scheduler}


def prepare_conditions(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    max_sequence_length: int = 128,
) -> torch.Tensor:
    device = device or text_encoder.device
    dtype = dtype or text_encoder.dtype
    
    if isinstance(prompt, str):
        prompt = [prompt]
    
    return _encode_prompt_t5(tokenizer, text_encoder, prompt, device, dtype, max_sequence_length)


def prepare_latents(
    vae: AutoencoderKLLTXVideo,
    image_or_video: torch.Tensor,
    patch_size: int = 1,
    patch_size_t: int = 1,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    device = device or vae.device
    dtype = dtype or vae.dtype

    if image_or_video.ndim == 4:
        image_or_video = image_or_video.unsqueeze(2)
    assert image_or_video.ndim == 5, f"Expected 5D tensor, got {image_or_video.ndim}D tensor"

    image_or_video = image_or_video.to(device=device, dtype=dtype)
    image_or_video = image_or_video.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, F, H, W] -> [B, F, C, H, W]
    latents = vae.encode(image_or_video).latent_dist.sample(generator=generator)
    _, _, num_frames, height, width = image_or_video.shape
    latents = _normalize_latents(latents, vae.latents_mean, vae.latents_std)
    latents = _pack_latents(latents, patch_size, patch_size_t)
    return {"latents": latents, "num_frames": num_frames, "height": height, "width": width}


def collate_fn_t2v(batch: List[List[Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
    return {
        "prompts": [x["prompt"] for x in batch[0]],
        "videos": torch.stack([x["video"] for x in batch[0]]),
    }


def forward_pass(
    transformer: LTXVideoTransformer3DModel,
    prompt_embeds: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    latents: torch.Tensor,
    noisy_latents: torch.Tensor,
    timesteps: torch.LongTensor,
    num_frames: int,
    height: int,
    width: int,
) -> torch.Tensor:
    # TODO(aryan): make configurable
    rope_interpolation_scale = [1 / 25, 32, 32]
    batch_size, num_channels, num_frames, height, width = latents.shape

    denoised_latents = transformer(
        hidden_states=noisy_latents,
        encoder_hidden_states=prompt_embeds,
        timestep=timesteps,
        encoder_attention_mask=prompt_attention_mask,
        num_frames=num_frames,
        height=height,
        width=width,
        rope_interpolation_scale=rope_interpolation_scale,
        return_dict=False,
    )[0]

    return {"latents": denoised_latents}


def _encode_prompt_t5(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: List[str],
    device: torch.device,
    dtype: torch.dtype,
    max_sequence_length,
) -> torch.Tensor:
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_attention_mask = text_inputs.attention_mask
    prompt_attention_mask = prompt_attention_mask.bool().to(device)

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    prompt_attention_mask = prompt_attention_mask.view(batch_size, -1)

    return {"prompt_embeds": prompt_embeds, "prompt_attention_mask": prompt_attention_mask}


def _normalize_latents(
    latents: torch.Tensor,
    latents_mean: torch.Tensor,
    latents_std: torch.Tensor,
    scaling_factor: float = 1.0
) -> torch.Tensor:
    # Normalize latents across the channel dimension [B, C, F, H, W]
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents = (latents - latents_mean) * scaling_factor / latents_std
    return latents


def _pack_latents(latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1) -> torch.Tensor:
    # Unpacked latents of shape are [B, C, F, H, W] are patched into tokens of shape [B, C, F // p_t, p_t, H // p, p, W // p, p].
    # The patch dimensions are then permuted and collapsed into the channel dimension of shape:
    # [B, F // p_t * H // p * W // p, C * p_t * p * p] (an ndim=3 tensor).
    # dim=0 is the batch size, dim=1 is the effective video sequence length, dim=2 is the effective number of input features
    batch_size, num_channels, num_frames, height, width = latents.shape
    post_patch_num_frames = num_frames // patch_size_t
    post_patch_height = height // patch_size
    post_patch_width = width // patch_size
    latents = latents.reshape(
        batch_size,
        -1,
        post_patch_num_frames,
        patch_size_t,
        post_patch_height,
        patch_size,
        post_patch_width,
        patch_size,
    )
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
    return latents


LTX_VIDEO_T2V_CONFIG = {
    "pipeline_cls": LTXPipeline,
    "load_components": load_components,
    "prepare_conditions": prepare_conditions,
    "prepare_latents": prepare_latents,
    "collate_fn": collate_fn_t2v,
}
