import torch


# Default values copied from https://github.com/huggingface/diffusers/blob/8957324363d8b239d82db4909fbf8c0875683e3d/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py#L47
def resolution_dependant_timestep_flow_shift(
    latents: torch.Tensor,
    sigmas: torch.Tensor,
    base_image_seq_len: int = 256,
    max_image_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> torch.Tensor:
    image_or_video_sequence_length = 0
    if latents.ndim == 4:
        image_or_video_sequence_length = latents.shape[2] * latents.shape[3]
    elif latents.ndim == 5:
        image_or_video_sequence_length = latents.shape[2] * latents.shape[3] * latents.shape[4]
    else:
        raise ValueError(f"Expected 4D or 5D tensor, got {latents.ndim}D tensor")

    m = (max_shift - base_shift) / (max_image_seq_len - base_image_seq_len)
    b = base_shift - m * base_image_seq_len
    mu = m * image_or_video_sequence_length + b
    sigmas = default_flow_shift(latents, sigmas, shift=mu)
    return sigmas


def default_flow_shift(sigmas: torch.Tensor, shift: float = 1.0) -> torch.Tensor:
    sigmas = (sigmas * shift) / (1 + (shift - 1) * sigmas)
    return sigmas
