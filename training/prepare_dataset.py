#!/usr/bin/env python3

# For folder structure dataset: python3 prepare_dataset.py --model_id THUDM/CogVideoX-2b --data_root /raid/aryan/video-dataset-disney/ --caption_column prompts.txt --video_column videos.txt --output_dir dump --height 480 --width 720 --max_num_frames 49 --max_sequence_length 226 --target_fps 8 --batch_size 1 --dtype fp32
# For latent/embed structure dataset: python3 prepare_dataset.py --model_id THUDM/CogVideoX-2b --data_root /raid/aryan/video-dataset-disney/ --caption_column prompts.txt --video_column videos.txt --output_dir dump --height 480 --width 720 --max_num_frames 49 --max_sequence_length 226 --target_fps 8 --batch_size 1 --dtype fp32 --save_tensors

import argparse
import gc
import pathlib
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from diffusers import AutoencoderKLCogVideoX
from diffusers.utils import export_to_video, get_logger
from torchvision import transforms
from transformers import T5EncoderModel, T5Tokenizer


# Must import after importing torch, otherwise there's a nasty segfault when loading text_encoder/vae
import decord  # isort:skip

decord.bridge.set_bridge("torch")

logger = get_logger(__name__)

DTYPE_MAPPING = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def get_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="THUDM/CogVideoX-2b",
        help="Hugging Face model ID to use for tokenizer, text encoder and VAE.",
    )
    parser.add_argument("--data_root", type=str, required=True, help="Path to where training data is located.")
    parser.add_argument(
        "--dataset_file", type=str, default=None, help="Path to CSV file containing metadata about training data."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="caption",
        help="If using a CSV file via the `--dataset_file` argument, this should be the name of the column containing the captions. If using the folder structure format for data loading, this should be the name of the file containing line-separated captions (the file should be located in `--data_root`).",
    )
    parser.add_argument(
        "--video_column",
        type=str,
        default="video",
        help="If using a CSV file via the `--dataset_file` argument, this should be the name of the column containing the video paths. If using the folder structure format for data loading, this should be the name of the file containing line-separated video paths (the file should be located in `--data_root`).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to output directory where preprocessed videos/latents/embeddings will be saved.",
    )
    parser.add_argument("--height", type=int, default=480, help="Height of the resized output video.")
    parser.add_argument("--width", type=int, default=720, help="Width of the resized output video.")
    parser.add_argument("--max_num_frames", type=int, default=49, help="Maximum number of frames in output video.")
    parser.add_argument(
        "--max_sequence_length", type=int, default=226, help="Max sequence length of prompt embeddings."
    )
    parser.add_argument(
        "--target_fps", type=int, default=8, help="Frame rate of output videos if `--save_tensors` is unspecified."
    )
    parser.add_argument(
        "--save_tensors",
        action="store_true",
        help="Whether to encode videos/captions to latents/embeddings and save them in pytorch serializable format.",
    )
    parser.add_argument(
        "--use_slicing",
        action="store_true",
        help="Whether to enable sliced encoding/decoding in the VAE. Only used if `--save_tensors` is also used.",
    )
    parser.add_argument(
        "--use_tiling",
        action="store_true",
        help="Whether to enable tiled encoding/decoding in the VAE. Only used if `--save_tensors` is also used.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Number of videos to process at once in the VAE.")
    parser.add_argument(
        "--num_decode_threads",
        type=int,
        default=0,
        help="Number of decoding threads for `decord` to use. The default `0` means to automatically determine required number of threads.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
        help="Data type to use when generating latents and prompt embeddings.",
    )
    return parser.parse_args()


def load_dataset_from_local_path(
    data_root: pathlib.Path, caption_column: str, video_column: str
) -> Tuple[List[str], List[str]]:
    if not data_root.exists():
        raise ValueError("Root folder for videos does not exist")

    prompt_path = data_root.joinpath(caption_column)
    video_path = data_root.joinpath(video_column)

    if not prompt_path.exists() or not prompt_path.is_file():
        raise ValueError(
            "Expected `--caption_column` to be path to a file in `--data_root` containing line-separated text prompts."
        )
    if not video_path.exists() or not video_path.is_file():
        raise ValueError(
            "Expected `--video_column` to be path to a file in `--data_root` containing line-separated paths to video data in the same directory."
        )

    with open(prompt_path, "r", encoding="utf-8") as file:
        prompts = [line.strip() for line in file.readlines() if len(line.strip()) > 0]
    with open(video_path, "r", encoding="utf-8") as file:
        video_paths = [data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]

    if any(not path.is_file() for path in video_paths):
        raise ValueError(
            f"Expected `{video_column=}` to be a path to a file in `{data_root=}` containing line-separated paths to video data but found atleast one path that is not a valid file."
        )

    return prompts, video_paths


def load_dataset_from_csv(
    data_root: pathlib.Path, dataset_file: pathlib.Path, caption_column: str, video_column: str
) -> Tuple[List[str], List[str]]:
    df = pd.read_csv(dataset_file)
    prompts = df[caption_column].tolist()
    video_paths = df[video_column].tolist()
    video_paths = [data_root.joinpath(line.strip()) for line in video_paths]

    if any(not path.is_file() for path in video_paths):
        raise ValueError(
            f"Expected `{video_column=}` to be a path to a file in `{data_root=}` containing line-separated paths to video data but found atleast one path that is not a valid file."
        )

    return prompts, video_paths


def load_and_preprocess_video(
    path: pathlib.Path, height: int, width: int, max_num_frames: int, video_transforms, num_threads: int = 0
) -> torch.Tensor:
    frames = None

    try:
        video_reader = decord.VideoReader(uri=path.as_posix(), height=height, width=width, num_threads=num_threads)
        video_num_frames = len(video_reader)

        if video_num_frames < max_num_frames:
            logger.warning(
                f"Video at `{path.as_posix()}` should have atleast `{max_num_frames=}`, but got only `{video_num_frames=}`. Skipping it."
            )
            return

        indices = list(range(0, video_num_frames, video_num_frames // max_num_frames))
        frames: torch.Tensor = video_reader.get_batch(indices)
        frames = frames[:max_num_frames].float()
        frames = frames.permute(0, 3, 1, 2).contiguous()
        frames = torch.stack([video_transforms(frame) for frame in frames], dim=0)
    except Exception as e:
        logger.error(f"Error: {e}. Skipping video located at `{path.as_posix()}`")
        traceback.print_exc()

    return frames


def _get_t5_prompt_embeds(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("`text_input_ids` must be provided when the tokenizer is not specified.")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds


def encode_prompt(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt_embeds = _get_t5_prompt_embeds(
        tokenizer,
        text_encoder,
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        max_sequence_length=max_sequence_length,
        device=device,
        dtype=dtype,
        text_input_ids=text_input_ids,
    )
    return prompt_embeds


def compute_prompt_embeddings(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: str,
    max_sequence_length: int,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool = False,
):
    if requires_grad:
        prompt_embeds = encode_prompt(
            tokenizer,
            text_encoder,
            prompt,
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )
    else:
        with torch.no_grad():
            prompt_embeds = encode_prompt(
                tokenizer,
                text_encoder,
                prompt,
                num_videos_per_prompt=1,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
    return prompt_embeds


def save_videos(
    videos: torch.Tensor, video_paths: List[str], prompts: List[str], output_dir: pathlib.Path, target_fps: int = 8
) -> None:
    assert videos.size(0) == len(video_paths)

    videos = (videos + 1) / 2
    videos = (videos * 255.0).clip(0, 255)
    videos = videos.to(dtype=torch.uint8)

    video_dir = output_dir.joinpath("videos")

    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    to_pil_image = transforms.ToPILImage()
    videos_pil = [[to_pil_image(frame) for frame in video] for video in videos]

    for video, video_path in zip(videos_pil, video_paths):
        filename = video_dir.joinpath(pathlib.Path(video_path).name)
        logger.debug(f"Saving video to `{filename}`")
        export_to_video(video, filename.as_posix(), fps=target_fps)

    with open(output_dir.joinpath("videos.txt").as_posix(), "w", encoding="utf-8") as file:
        for video_path in video_paths:
            file.write(f"videos/{pathlib.Path(video_path).name}\n")

    with open(output_dir.joinpath("prompts.txt").as_posix(), "w", encoding="utf-8") as file:
        for prompt in prompts:
            file.write(f"{prompt}\n")


def save_latents_and_embeddings(
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    video_paths: List[str],
    prompts: List[str],
    output_dir: pathlib.Path,
) -> None:
    assert latents.size(0) == prompt_embeds.size(0)
    assert latents.size(0) == len(video_paths)
    assert prompt_embeds.size(0) == len(prompts)

    latents_dir = output_dir.joinpath("latents")
    embeds_dir = output_dir.joinpath("embeddings")

    output_dir.mkdir(parents=True, exist_ok=True)
    latents_dir.mkdir(parents=True, exist_ok=True)
    embeds_dir.mkdir(parents=True, exist_ok=True)

    for latent, embed, video_path in zip(latents, prompt_embeds, video_paths):
        # Need to perform the clone, otherwise the entire `latents` or `prompt_embeds` tensor is
        # saved for every single video/prompt embedding. This is due to us viewing a slice of a
        # large tensor when iteratively saving stuff here.
        latent = latent.clone()
        embed = embed.clone()

        video_path = pathlib.Path(video_path)
        filename_without_ext = video_path.name.split(".")[0]

        latent_filename = latents_dir.joinpath(filename_without_ext)
        embed_filename = embeds_dir.joinpath(filename_without_ext)

        latent_filename = f"{latent_filename}.pt"
        embed_filename = f"{embed_filename}.pt"

        torch.save(latent, latent_filename)
        torch.save(embed, embed_filename)

    with open(output_dir.joinpath("videos.txt").as_posix(), "w", encoding="utf-8") as file:
        for video_path in video_paths:
            file.write(f"videos/{pathlib.Path(video_path).name}\n")

    with open(output_dir.joinpath("prompts.txt").as_posix(), "w", encoding="utf-8") as file:
        for prompt in prompts:
            file.write(f"{prompt}\n")


@torch.no_grad()
def main(args: Dict[str, Any]) -> None:
    data_root = pathlib.Path(args.data_root)
    dataset_file = None
    if args.dataset_file:
        dataset_file = pathlib.Path(args.dataset_file)

    if dataset_file is None:
        prompts, video_paths = load_dataset_from_local_path(data_root, args.caption_column, args.video_column)
    else:
        prompts, video_paths = load_dataset_from_csv(data_root, dataset_file, args.caption_column, args.video_column)

    video_transforms = transforms.Compose(
        [
            transforms.Lambda(lambda x: x / 255.0),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )

    prompts_usable = []
    video_paths_usable = []
    videos = []
    for prompt, path in zip(prompts, video_paths):
        video = load_and_preprocess_video(
            path, args.height, args.width, args.max_num_frames, video_transforms, args.num_decode_threads
        )
        if video is not None:
            prompts_usable.append(prompt)
            video_paths_usable.append(path)
            videos.append(video)
    videos = torch.stack(videos)

    if not args.save_tensors:
        save_videos(videos, video_paths_usable, prompts_usable, pathlib.Path(args.output_dir), args.target_fps)
    else:
        dtype = DTYPE_MAPPING[args.dtype]
        tokenizer = T5Tokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
        text_encoder = T5EncoderModel.from_pretrained(args.model_id, subfolder="text_encoder", torch_dtype=dtype)
        text_encoder = text_encoder.to("cuda")

        prompt_embeds_list = []
        for start_index in range(0, len(prompts_usable), args.batch_size):
            end_index = min(len(prompts_usable), start_index + args.batch_size)
            batch_prompts = prompts_usable[start_index:end_index]

            prompt_embeds = compute_prompt_embeddings(
                tokenizer,
                text_encoder,
                batch_prompts,
                max_sequence_length=args.max_sequence_length,
                device="cuda",
                dtype=dtype,
            )
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.cat(prompt_embeds_list).to("cpu")

        del tokenizer, text_encoder
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize("cuda")

        vae = AutoencoderKLCogVideoX.from_pretrained(args.model_id, subfolder="vae", torch_dtype=dtype)
        vae = vae.to("cuda")

        if args.use_slicing:
            vae.enable_slicing()
        if args.use_tiling:
            vae.enable_tiling()

        encoded_videos = []
        for start_index in range(0, len(video_paths_usable), args.batch_size):
            end_index = min(len(video_paths_usable), start_index + args.batch_size)
            batch_videos = videos[start_index:end_index]

            batch_videos = batch_videos.to("cuda")
            batch_videos = batch_videos.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]

            if args.use_slicing:
                encoded_slices = [vae._encode(video_slice) for video_slice in batch_videos.split(1)]
                encoded_video = torch.cat(encoded_slices)
            else:
                encoded_video = vae._encode(batch_videos)

            encoded_videos.append(encoded_video)

        encoded_videos = torch.cat(encoded_videos).to("cpu")

        del vae
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize("cuda")

        save_latents_and_embeddings(
            encoded_videos, prompt_embeds, video_paths_usable, prompts_usable, pathlib.Path(args.output_dir)
        )


if __name__ == "__main__":
    args = get_args()

    assert args.height % 16 == 0, "CogVideoX requires input video height to be divisible by 16."
    assert args.width % 16 == 0, "CogVideoX requires input video width to be divisible by 16."
    assert (
        args.max_num_frames % 4 == 0 or args.max_num_frames % 4 == 1
    ), "`--max_num_frames` must be of form 4 * k or 4 * k + 1 to be compatible with VAE."

    main(args)
