"""
Needs `vllm` to be installed from the `main`.
"""

from typing import Optional
from vllm import LLM, SamplingParams
import queue
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor
import fire
import os

from dataset import VideoDataset

def save_results(output_queue, output_dir):
    while True:
        try:
            item = output_queue.get(timeout=5)
            if item is None:
                break

            video_names, outputs = item
            outputs = [o.outputs[0].text for o in outputs]

            for i, pred_caption in enumerate(outputs):
                with open(os.path.join(output_dir, f"{video_names[i]}_caption.txt"), "w") as f:
                    f.write(pred_caption)

        except queue.Empty:
            continue

def create_conversations(batch, prompt: str):
    conversations = []
    for i, video in enumerate(batch["videos"]):
        content = []
        content.append({"type": "text", "text": "Describe this set of frames. Consider the frames to be a part of the same video."})
        for j in range(len(video)):
            new_image = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{video[j]}"}}
            content.append(new_image)
        message = {"role": "user", "content": content}
        conversations.append([message]) 
    return conversations

def collate_fn(batch):
    inputs = {
        "videos": [sample["video"] for sample in batch],
        "video_names": [sample["video_name"] for sample in batch]
    }
    return inputs

def prepare_dataloader(
        video_root_dir, output_dir, video_extensions, max_num_frames, num_data_workers, batch_size
    ):
    dataset = VideoDataset(
        video_root_dir, output_dir=output_dir, max_num_frames=max_num_frames, video_extensions=video_extensions
    )

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size, 
        collate_fn=collate_fn,
        num_workers=num_data_workers,
        pin_memory=True,
        persistent_workers=True
    )
    return dataloader

def load_model(max_num_frames: int, max_tokens: int, num_devices: int, download_dir: Optional[str] = None):
    vllm_engine = LLM(
        "Qwen/Qwen2-VL-2B-Instruct", tensor_parallel_size=num_devices, limit_mm_per_prompt={"image": max_num_frames}, download_dir=download_dir
    )
    sampling_params = SamplingParams(max_tokens=max_tokens)
    return vllm_engine, sampling_params


def main(
        root_dir: str, 
        prompt: str, 
        output_dir: str, 
        num_devices: int, 
        max_num_frames: int, 
        max_tokens: int, 
        video_extensions: tuple = (".mp4"),
        num_data_workers:int = 4, 
        batch_size:int = 8, 
        num_artifact_workers:int = 4,
        download_dir: Optional[str] = None,
    ):
    max_allowed_imgs_per_req = batch_size * max_num_frames
    vllm_engine, sampling_params = load_model(
        max_num_frames=max_allowed_imgs_per_req, max_tokens=max_tokens, num_devices=num_devices, download_dir=download_dir,
    )
    dataloader = prepare_dataloader(
        video_root_dir=root_dir, 
        output_dir=output_dir,
        video_extensions=video_extensions,
        max_num_frames=max_num_frames, 
        num_data_workers=num_data_workers, 
        batch_size=batch_size
    )

    output_queue = queue.Queue()
    save_thread = ThreadPoolExecutor(max_workers=num_artifact_workers)
    os.makedirs(output_dir, exist_ok=True)
    save_future = save_thread.submit(save_results, output_queue, output_dir)

    try:
        for idx, batch in enumerate(dataloader):
            conversations = create_conversations(batch, prompt=prompt)
            outputs = vllm_engine.chat(conversations, sampling_params)
            output_queue.put((batch["video_names"], outputs))

    finally:
        output_queue.put(None)
        save_thread.shutdown(wait=True)

    save_future.result()
    print("All processes completed. Caption generation and saving done.")
        

if __name__ == "__main__":
    fire.Fire(main)
