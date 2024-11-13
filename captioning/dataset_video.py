import base64
import io
import os
import sys
from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset


import decord  # isort:skip


class VideoDataset(Dataset):
    def __init__(
        self, root_video_dir: str, output_dir: str, max_num_frames: int, video_extensions: Tuple[str] = (".mp4")
    ):
        self.root_video_dir = root_video_dir
        self.max_num_frames = max_num_frames

        # Filter out existing captions.
        video_files = {
            os.path.join(root_video_dir, f) for f in os.listdir(root_video_dir) if f.endswith(video_extensions)
        }
        if os.path.isdir(output_dir):
            existing_caption_basenames = {
                os.path.splitext(f)[0] for f in os.listdir(output_dir) if "_caption.txt" in f
            }
        else:
            existing_caption_basenames = None
        if existing_caption_basenames:
            if len(existing_caption_basenames) == len(video_files):
                sys.exit(
                    "It seems like all the input videos have been already captioned. So, we're exiting the program."
                )
            filtered_video_files = [
                f
                for f in video_files
                if os.path.splitext(os.path.basename(f))[0] + "_caption" not in existing_caption_basenames
            ]
            if len(video_files) > len(filtered_video_files):
                diff = len(video_files) - len(filtered_video_files)
                print(f"Found existing captions for {diff} videos. Will skip them.")

            self.video_files = sorted(filtered_video_files)
        else:
            self.video_files = sorted(video_files)
            print(f"Total videos found: {len(self.video_files)}.")

    def __len__(self) -> int:
        return len(self.video_files)

    def __getitem__(self, index: int) -> List[Image.Image]:
        video_path = self.video_files[index]
        return self.load_video(video_path)

    def load_video(self, path: str) -> List[Image.Image]:
        video_reader = decord.VideoReader(uri=path)
        filename = os.path.basename(path)

        def uniform_sample(l, n):
            gap = len(l) / n
            idxs = [int(i * gap + gap / 2) for i in range(n)]
            return [l[i] for i in idxs]

        sample_fps = round(video_reader.get_avg_fps() / 1)
        frame_idx = list(range(0, len(video_reader), sample_fps))

        if len(frame_idx) > self.max_num_frames:
            frame_idx = uniform_sample(frame_idx, self.max_num_frames)

        video_frames = [Image.fromarray(video_reader[i].asnumpy()) for i in frame_idx]

        return {"video": [self.encode_image(frame) for frame in video_frames], "filename": filename}

    def encode_image(self, image):
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_bytes = buffered.getvalue()
        return base64.b64encode(image_bytes).decode("utf-8")


if __name__ == "__main__":
    import tempfile

    from huggingface_hub import snapshot_download

    with tempfile.TemporaryDirectory() as tmpdirname:
        video_root_dir = snapshot_download(
            repo_id="Wild-Heart/Disney-VideoGeneration-Dataset", repo_type="dataset", local_dir=tmpdirname
        )

        dataset = VideoDataset(os.path.join(video_root_dir, "videos"), max_num_frames=16)
        print(len(dataset))

        for item in dataset:
            print(len(item["video"]))
            break
