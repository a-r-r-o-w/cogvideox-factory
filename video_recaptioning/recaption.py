"""
Needs `vllm` to be installed from the `main`.
"""

from vllm import LLM, SamplingParams
import queue
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from concurrent.futures import ThreadPoolExecutor
import torch
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

def create_messages(batch, prompt: str):
    messages = []
    for i, video in enumerate(batch["videos"]):
        messages.append({"role": "user", "content": []})
        messages[i]["content"].append({"type": "text", "text": prompt})
        for j in range(len(video)):
            new_image = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{video[j]}"}}
            messages[i]["content"].append(new_image)    
    return messages

def collate_fn(batch):
    inputs = {
        "videos": [sample["video"] for sample in batch],
        "video_names": [sample["video_name"] for sample in batch]
    }
    return inputs

def prepare_dataloader(video_root_dir, max_num_frames, num_data_workers, batch_size):
    dataset = VideoDataset(video_root_dir, max_num_frames=max_num_frames)
    
    rank = 0
    world_size = 1
    if torch.distributed.is_initialized():
        group = torch.distributed.group.WORLD
        rank = torch.distributed.get_rank(group=group)
        world_size = torch.distributed.get_world_size(group=group)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size, 
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_data_workers,
        pin_memory=True
    )
    return dataloader

def load_model(max_num_frames: int, max_tokens: int):
    vllm_engine = LLM("Qwen/Qwen2-VL-2B-Instruct", limit_mm_per_prompt={"image": max_num_frames})
    sampling_params = SamplingParams(max_tokens=max_tokens)
    return vllm_engine, sampling_params


def main(
        root_dir, prompt, output_dir, max_num_frames, max_tokens, num_data_workers=4, batch_size=8, num_artifact_workers=4
    ):
    max_allowed_imgs_per_req = batch_size * max_num_frames
    vllm_engine, sampling_params = load_model(
        max_num_frames=max_allowed_imgs_per_req, max_tokens=max_tokens
    )
    dataloader = prepare_dataloader(
        video_root_dir=root_dir, max_num_frames=max_num_frames, 
        num_data_workers=num_data_workers, batch_size=batch_size
    )

    output_queue = queue.Queue()
    save_thread = ThreadPoolExecutor(max_workers=num_artifact_workers)
    os.makedirs(output_dir, exist_ok=True)
    save_future = save_thread.submit(save_results, output_queue, output_dir)

    try:
        for batch in dataloader:
            messages = create_messages(batch, prompt=prompt)
            outputs = vllm_engine.chat(messages, sampling_params)
            output_queue.put((batch["video_names"], outputs))

    finally:
        output_queue.put(None)
        save_thread.shutdown(wait=True)

    save_future.result()
    print("All processes completed. Caption generation and saving done.")
        

if __name__ == "__main__":
    fire.Fire(main)