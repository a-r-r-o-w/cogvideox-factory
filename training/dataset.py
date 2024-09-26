import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import decord
import pandas as pd
import torch
from accelerate.logging import get_logger
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from torchvision.transforms.functional import resize


logger = get_logger(__name__)

decord.bridge.set_bridge("torch")

HEIGHT = [256, 320, 384, 480, 512, 576, 720, 768, 960, 1024, 1280, 1536]
WIDTH = [256, 320, 384, 480, 512, 576, 720, 768, 960, 1024, 1280, 1536]
T2V_FRAMES = [16, 24, 32, 48, 64, 80]

T2V_RESOLUTIONS = [(f, h, w) for h in HEIGHT for w in WIDTH for f in T2V_FRAMES]


class VideoDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        dataset_file: Optional[str] = None,
        caption_column: str = "text",
        video_column: str = "video",
        max_num_frames: int = 49,
        id_token: Optional[str] = None,
        random_flip: Optional[float] = None,
    ) -> None:
        super().__init__()

        self.data_root = Path(data_root)
        self.dataset_file = dataset_file
        self.caption_column = caption_column
        self.video_column = video_column
        self.max_num_frames = max_num_frames
        self.id_token = id_token or ""
        self.random_flip = random_flip

        if dataset_file is None:
            (
                self.prompts,
                self.video_paths,
            ) = self._load_dataset_from_local_path()
        else:
            (
                self.prompts,
                self.video_paths,
            ) = self._load_dataset_from_csv()

        self.num_videos = len(self.video_paths)
        if self.num_videos != len(self.prompts):
            raise ValueError(
                f"Expected length of prompts and videos to be the same but found {len(self.prompts)=} and {len(self.video_paths)=}. Please ensure that the number of caption prompts and videos match in your dataset."
            )

        self.video_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(random_flip) if random_flip else transforms.Lambda(lambda x: x),
                transforms.Lambda(lambda x: x / 255.0),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

    def __len__(self) -> int:
        return self.num_videos

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if isinstance(index, list):
            # Here, index is actually a list of data objects that we need to return.
            # This is done so that we don't have to load data twice.
            return index

        index = index % self.num_videos
        video = self._preprocess_video(self.video_paths[index])
        return {
            "prompt": self.id_token + self.prompts[index],
            "video": video,
            "video_metadata": {
                "num_frames": video.shape[0],
                "height": video.shape[2],
                "width": video.shape[3],
            },
        }

    def _load_dataset_from_local_path(self) -> Tuple[List[str], List[str]]:
        if not self.data_root.exists():
            raise ValueError("Root folder for videos does not exist")

        prompt_path = self.data_root.joinpath(self.caption_column)
        video_path = self.data_root.joinpath(self.video_column)

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
            video_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]

        if any(not path.is_file() for path in video_paths):
            raise ValueError(
                f"Expected `{self.video_column=}` to be a path to a file in `{self.data_root=}` containing line-separated paths to video data but found atleast one path that is not a valid file."
            )

        return prompts, video_paths

    def _load_dataset_from_csv(self) -> Tuple[List[str], List[str]]:
        df = pd.read_csv(self.dataset_file)
        prompts = df[self.caption_column].tolist()
        video_paths = df[self.video_column].tolist()
        video_paths = [self.data_root.joinpath(line.strip()) for line in video_paths]

        if any(not path.is_file() for path in video_paths):
            raise ValueError(
                f"Expected `{self.video_column=}` to be a path to a file in `{self.data_root=}` containing line-separated paths to video data but found atleast one path that is not a valid file."
            )

        return prompts, video_paths

    def _preprocess_video(self, path: Path) -> torch.Tensor:
        video_reader = decord.VideoReader(uri=path.as_posix())
        video_num_frames = len(video_reader)

        indices = list(range(0, video_num_frames, video_num_frames // self.max_num_frames))
        frames = video_reader.get_batch(indices)
        frames = frames[: self.max_num_frames].float()
        frames = frames.permute(0, 3, 1, 2).contiguous()
        frames = torch.stack([self.video_transforms(frame) for frame in frames], dim=0)

        return frames


class VideoDatasetWithResizing(VideoDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _preprocess_video(self, path: Path) -> torch.Tensor:
        video_reader = decord.VideoReader(uri=path.as_posix())
        video_num_frames = len(video_reader)
        # nearest_frame_bucket = min(T2V_FRAMES, key=lambda x: abs(x - video_num_frames))

        # Only for now: purposefully limiting to max_num_frames
        nearest_frame_bucket = min(T2V_FRAMES, key=lambda x: abs(x - min(video_num_frames, self.max_num_frames)))

        frame_indices = list(range(0, video_num_frames, video_num_frames // nearest_frame_bucket))

        frames = video_reader.get_batch(frame_indices)
        frames = frames[:nearest_frame_bucket].float()
        frames = frames.permute(0, 3, 1, 2).contiguous()

        nearest_res = self._find_nearest_resolution(frames.shape[2], frames.shape[3])
        frames_resized = torch.stack([resize(frame, nearest_res) for frame in frames], dim=0)

        frames = torch.stack([self.video_transforms(frame) for frame in frames_resized], dim=0)
        return frames

    def _find_nearest_resolution(self, height, width):
        nearest_res = min(T2V_RESOLUTIONS, key=lambda x: abs(x[1] - height) + abs(x[2] - width))
        return nearest_res[1], nearest_res[2]


class BucketSampler(Sampler):
    def __init__(self, data_source, batch_size: int = 8, shuffle: bool = True) -> None:
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.buckets = {resolution: [] for resolution in T2V_RESOLUTIONS}

    def __iter__(self):
        for index, data in enumerate(self.data_source):
            video_metadata = data["video_metadata"]
            f, h, w = video_metadata["num_frames"], video_metadata["height"], video_metadata["width"]

            self.buckets[(f, h, w)].append(data)
            if len(self.buckets[(f, h, w)]) == self.batch_size:
                if self.shuffle:
                    random.shuffle(self.buckets[(f, h, w)])
                yield self.buckets[(f, h, w)]
                del self.buckets[(f, h, w)]
                self.buckets[(f, h, w)] = []
                break
