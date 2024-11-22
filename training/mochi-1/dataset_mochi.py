from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torchvision import transforms
from accelerate.logging import get_logger
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
import torch.nn as nn


# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")

import sys 
sys.path.append("..")

from dataset import VideoDataset as VDS

logger = get_logger(__name__)

# TODO (sayakpaul): probably not all buckets are needed for Mochi-1? 
HEIGHT_BUCKETS = [256, 320, 384, 480, 512, 576, 720, 768, 960, 1024, 1280, 1536]
WIDTH_BUCKETS = [256, 320, 384, 480, 512, 576, 720, 768, 848, 960, 1024, 1280, 1536]
FRAME_BUCKETS = [16, 24, 32, 48, 64, 80, 85]

VAE_SPATIAL_SCALE_FACTOR = 8
VAE_TEMPORAL_SCALE_FACTOR = 6

class VideoDataset(VDS):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        random_flip = kwargs.get("random_flip", None)
        self.video_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip([(random_flip)])
                if random_flip
                else transforms.Lambda(lambda x: x),
                transforms.Lambda(self.scale_transform),
            ]
        )

    def scale_transform(self, x):
        return x / 127.5 - 1.0

    # Overriding this because we calculate `num_frames` differently.
    def __getitem__(self, index: int) -> Dict[str, Any]:
        if isinstance(index, list):
            # Here, index is actually a list of data objects that we need to return.
            # The BucketSampler should ideally return indices. But, in the sampler, we'd like
            # to have information about num_frames, height and width. Since this is not stored
            # as metadata, we need to read the video to get this information. You could read this
            # information without loading the full video in memory, but we do it anyway. In order
            # to not load the video twice (once to get the metadata, and once to return the loaded video
            # based on sampled indices), we cache it in the BucketSampler. When the sampler is
            # to yield, we yield the cache data instead of indices. So, this special check ensures
            # that data is not loaded a second time. PRs are welcome for improvements.
            return index

        if self.load_tensors:
            image_latents, video_latents, prompt_embeds, prompt_attention_mask = self._preprocess_video(self.video_paths[index])

            # This is hardcoded for now.
            # Output of the VAE encoding is 2 * output_channels and then it's
            # temporal compression factor is 6. Initially, the VAE encodings will have
            # 24 latent number of frames. So, if we were to train with a 
            # max frame size of 85 and frame bucket of [85], we need to have the following logic.
            latent_num_frames = video_latents.size(0)
            num_frames = ((latent_num_frames // 2) * (VAE_TEMPORAL_SCALE_FACTOR + 1) + 1)

            height = video_latents.size(2) * VAE_SPATIAL_SCALE_FACTOR
            width = video_latents.size(3) * VAE_SPATIAL_SCALE_FACTOR

            return {
                "prompt": prompt_embeds,
                "prompt_attention_mask": prompt_attention_mask,
                "image": image_latents,
                "video": video_latents,
                "video_metadata": {
                    "num_frames": num_frames,
                    "height": height,
                    "width": width,
                },
            }
        else:
            image, video, _ = self._preprocess_video(self.video_paths[index])
            if video is not None:
                return {
                    "prompt": self.id_token + self.prompts[index],
                    "image": image,
                    "video": video,
                    "video_metadata": {
                        "num_frames": video.shape[0],
                        "height": video.shape[2],
                        "width": video.shape[3],
                    },
                }

    # Overriding this because we need `prompt_attention_mask`.
    def _load_preprocessed_latents_and_embeds(self, path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        filename_without_ext = path.name.split(".")[0]
        pt_filename = f"{filename_without_ext}.pt"

        # The current path is something like: /a/b/c/d/videos/00001.mp4
        # We need to reach: /a/b/c/d/video_latents/00001.pt
        image_latents_path = path.parent.parent.joinpath("image_latents")
        video_latents_path = path.parent.parent.joinpath("video_latents")
        embeds_path = path.parent.parent.joinpath("prompt_embeds")
        attention_mask_path = path.parent.parent.joinpath("prompt_attention_mask")

        if (
            not video_latents_path.exists()
            or not embeds_path.exists()
            or not attention_mask_path.exists()
            or (self.image_to_video and not image_latents_path.exists())
        ):
            raise ValueError(
                f"When setting the load_tensors parameter to `True`, it is expected that the `{self.data_root=}` contains three folders named `video_latents`, `prompt_embeds`, and `prompt_attention_mask`. However, these folders were not found. Please make sure to have prepared your data correctly using `prepare_data.py`. Additionally, if you're training image-to-video, it is expected that an `image_latents` folder is also present."
            )

        if self.image_to_video:
            image_latent_filepath = image_latents_path.joinpath(pt_filename)
        video_latent_filepath = video_latents_path.joinpath(pt_filename)
        embeds_filepath = embeds_path.joinpath(pt_filename)
        attention_mask_filepath = attention_mask_path.joinpath(pt_filename)

        if not video_latent_filepath.is_file() or not embeds_filepath.is_file() or not attention_mask_filepath.is_file():
            if self.image_to_video:
                image_latent_filepath = image_latent_filepath.as_posix()
            video_latent_filepath = video_latent_filepath.as_posix()
            embeds_filepath = embeds_filepath.as_posix()
            attention_mask_filepath = attention_mask_filepath.as_posix()
            raise ValueError(
                f"The file {video_latent_filepath=} or {embeds_filepath=} or {attention_mask_filepath=} could not be found. Please ensure that you've correctly executed `prepare_dataset.py`."
            )

        images = (
            torch.load(image_latent_filepath, map_location="cpu", weights_only=True) if self.image_to_video else None
        )
        latents = torch.load(video_latent_filepath, map_location="cpu", weights_only=True)
        embeds = torch.load(embeds_filepath, map_location="cpu", weights_only=True)
        attention_masks = torch.load(attention_mask_filepath,  map_location="cpu", weights_only=True)

        return images, latents, embeds, attention_masks



class VideoDatasetWithFlexibleResize(VideoDataset):
    def __init__(self, video_reshape_mode: str = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if video_reshape_mode:
            assert video_reshape_mode in ["center", "random"]
        self.video_reshape_mode = video_reshape_mode

    def _preprocess_video(self, path: Path) -> torch.Tensor:
        if self.load_tensors:
            return self._load_preprocessed_latents_and_embeds(path)
        else:
            video_reader = decord.VideoReader(uri=path.as_posix())
            video_num_frames = len(video_reader)

            nearest_frame_bucket = min(
                self.frame_buckets, key=lambda x: abs(x - min(video_num_frames, self.max_num_frames))
            )
            frame_indices = list(
                range(
                    0, 
                    video_num_frames, 
                    1 if video_num_frames < nearest_frame_bucket else video_num_frames // nearest_frame_bucket
                )
            )
            frames = video_reader.get_batch(frame_indices)
            
            # Pad or truncate frames to match the bucket size
            if video_num_frames < nearest_frame_bucket:
                pad_size = nearest_frame_bucket - video_num_frames
                frames = nn.functional.pad(frames, (0, 0, 0, 0, 0, 0, 0, pad_size))
                frames = frames.float()
            else:
                frames = frames[:nearest_frame_bucket].float()
            frames = frames.permute(0, 3, 1, 2).contiguous()

            # Find nearest resolution and apply resizing
            nearest_res = self._find_nearest_resolution(frames.shape[2], frames.shape[3])
            if self.video_reshape_mode in {"center", "random"}:
                frames = self._resize_for_rectangle_crop(frames, nearest_res)
            else:
                frames = torch.stack([resize(frame, nearest_res) for frame in frames], dim=0)

            # Apply transformations
            frames = torch.stack([self.video_transforms(frame) for frame in frames], dim=0)

            # Optionally extract the first frame as an image
            image = frames[:1].clone() if self.image_to_video else None

            return image, frames, None

    def _find_nearest_resolution(self, height: int, width: int) -> Tuple[int, int]:
        """
        Find the nearest resolution from the predefined list of resolutions.
        """
        nearest_res = min(self.resolutions, key=lambda x: abs(x[1] - height) + abs(x[2] - width))
        return nearest_res[1], nearest_res[2]

    def _resize_for_rectangle_crop(self, arr: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        """
        Resize frames for rectangular cropping.
        
        Args:
            arr (torch.Tensor): The video frames tensor [N, C, H, W].
            image_size (Tuple[int, int]): The target resolution (height, width).
        """
        if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
            arr = resize(
                arr,
                size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
                interpolation=InterpolationMode.BICUBIC,
            )
        else:
            arr = resize(
                arr,
                size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
                interpolation=InterpolationMode.BICUBIC,
            )

        # Perform cropping
        h, w = arr.shape[2], arr.shape[3]
        delta_h, delta_w = h - image_size[0], w - image_size[1]

        if self.video_reshape_mode == "random":
            top, left = np.random.randint(0, delta_h + 1), np.random.randint(0, delta_w + 1)
        elif self.video_reshape_mode == "center":
            top, left = delta_h // 2, delta_w // 2
        else:
            raise NotImplementedError(f"Unsupported reshape mode: {self.video_reshape_mode}")

        return transforms.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])