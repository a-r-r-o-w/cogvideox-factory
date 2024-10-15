import decord

from torch.utils.data import Dataset
from PIL import Image
from typing import Tuple, List
import os
import io 
import base64

class VideoDataset(Dataset):
    def __init__(self, root_video_dir: str, max_num_frames: int, video_extensions: Tuple[str] = (".mp4")):
        self.root_video_dir = root_video_dir
        self.max_num_frames = max_num_frames

        video_files = [
            os.path.join(root_video_dir, f) for f in os.listdir(root_video_dir) if f.endswith(video_extensions)
        ]
        self.video_files = sorted(video_files)

    def __len__(self) -> int:
        return len(self.video_files)

    def __getitem__(self, index: int) -> List[Image.Image]:
        video_path = self.video_files[index]
        return self.load_video(video_path)
    
    def load_video(self, path: str) -> List[Image.Image]:
        video_reader = decord.VideoReader(uri=path)
        base_name = os.path.basename(path).split(".")[0]

        video_frames = [
            Image.fromarray(video_reader[i].asnumpy()) for i in range(len(video_reader))
        ][:self.max_num_frames]
        return {"video": [self.encode_image(frame) for frame in video_frames], "video_name": base_name}

    def encode_image(self, image):
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_bytes = buffered.getvalue()
        return base64.b64encode(image_bytes).decode("utf-8")

if __name__ == "__main__":
    from huggingface_hub import snapshot_download
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdirname:
        video_root_dir = snapshot_download(
            repo_id="Wild-Heart/Disney-VideoGeneration-Dataset", repo_type="dataset", local_dir=tmpdirname
        )
    
        dataset = VideoDataset(os.path.join(video_root_dir, "videos"), max_num_frames=16)
        print(len(dataset))

        for item in dataset:
            print(len(item["video"]))
            break