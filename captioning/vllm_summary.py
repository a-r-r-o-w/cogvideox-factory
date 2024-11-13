"""
Needs `vllm` to be installed from the `main`.
"""

import os
import pathlib
import queue
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import fire
import pandas as pd
from torch.utils.data import DataLoader
from vllm import LLM, SamplingParams

from dataset_video import VideoDataset  # isort:skip


SYSTEM_PROMPT = r"""
You are part of a team of people that create videos using generative models. You use a video-generation model that can generate a video about anything you describe.
For example, if you respond with "A beautiful morning in the woods with the sun peaking through the trees", the video generation model will create a video of exactly as described. You task is to summarize the descriptions of videos provided to by users, and create details prompts to feed into the generative model.
There are a few rules to follow:
- You will only ever output a single video description per request.
- If the user mentions to summarize the prompt in [X] words, make sure to not exceed the limit.
You responses should just be the video generation prompt. Here are examples:
- "A detailed wooden toy ship with intricately carved masts and sails is seen gliding smoothly over a plush, blue carpet that mimics the waves of the sea. The ship's hull is painted a rich brown, with tiny windows. The carpet, soft and textured, provides a perfect backdrop, resembling an oceanic expanse. Surrounding the ship are various other toys and children's items, hinting at a playful environment. The scene captures the innocence and imagination of childhood, with the toy ship's journey symbolizing endless adventures in a whimsical, indoor setting."
- "A street artist, clad in a worn-out denim jacket and a colorful bandana, stands before a vast concrete wall in the heart, holding a can of spray paint, spray-painting a colorful bird on a mottled wall"
""".strip()

SUMMARY_USER_PROMPT = r"""Please summarize this video and limit the summary to 100-200 words.""".strip()

PROMPT_GEN_USER_PROMPT = r"""
Could you generate a prompt for a video generation model given the following summary:

```
{0}
```

Please limit the prompt to [{1}] words.
""".strip()


def save_results(output_queue, output_dir):
    output_file = pathlib.Path(output_dir).joinpath("summary.csv")

    while True:
        try:
            item = output_queue.get(timeout=5)
            if item is None:
                break

            video_filenames, summaries = item

            df_data = []
            for filename, summary in zip(video_filenames, summaries):
                df_data.append({
                    "filename": filename,
                    "summary": summary,
                })
            
            df = pd.DataFrame(df_data)
            df.to_csv(output_file, mode="a", header=not output_file.exists(), index=False)

        except queue.Empty:
            continue


def create_video_summary_conversations(batch, prompt: Optional[str] = None):
    if prompt is None:
        prompt = SUMMARY_USER_PROMPT

    conversations = []

    for i, video in enumerate(batch["videos"]):
        conversation = []
        content = []

        content.append({"type": "text", "text": prompt})
        for frame in video:
            new_image = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}}
            content.append(new_image)

        # conversation.append({"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]})
        conversation.append({"role": "user", "content": content})

        conversations.append(conversation)

    return conversations


def collate_fn(batch):
    inputs = {
        "videos": [sample["video"] for sample in batch],
        "filename": [sample["filename"] for sample in batch],
    }
    return inputs


def prepare_dataloader(video_root_dir, output_dir, video_extensions, max_num_frames, num_data_workers, batch_size):
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
        persistent_workers=True,
    )
    return dataloader


def load_summary_model(
    max_num_frames: int,
    max_tokens: int,
    num_devices: int,
    download_dir: Optional[str] = None,
    trust_remote_code: bool = False,
):
    engine = LLM(
        "openbmb/MiniCPM-V-2_6",
        dtype="bfloat16",
        tensor_parallel_size=num_devices,
        limit_mm_per_prompt={"image": max_num_frames},
        download_dir=download_dir,
        trust_remote_code=trust_remote_code,
    )
    sampling_params = SamplingParams(min_tokens=64, max_tokens=max_tokens)
    return engine, sampling_params


def main(
    root_dir: str,
    output_dir: str,
    num_devices: int = 1,
    max_num_frames: int = 8,
    max_summary_tokens: int = 512,
    video_summary_prompt: Optional[str] = None,
    video_extensions: tuple = (".mp4"),
    batch_size: int = 8,
    num_data_workers: int = 4,
    num_artifact_workers: int = 4,
    download_dir: Optional[str] = None,
    trust_remote_code: bool = False,
):
    max_allowed_imgs_per_req = batch_size * max_num_frames
    
    summary_engine, summary_sampling_params = load_summary_model(
        max_num_frames=max_allowed_imgs_per_req,
        max_tokens=max_summary_tokens,
        num_devices=num_devices,
        download_dir=download_dir,
        trust_remote_code=trust_remote_code,
    )

    dataloader = prepare_dataloader(
        video_root_dir=root_dir,
        output_dir=output_dir,
        video_extensions=video_extensions,
        max_num_frames=max_num_frames,
        num_data_workers=num_data_workers,
        batch_size=batch_size,
    )

    output_queue = queue.Queue()
    save_thread = ThreadPoolExecutor(max_workers=num_artifact_workers)
    os.makedirs(output_dir, exist_ok=True)
    save_future = save_thread.submit(save_results, output_queue, output_dir)

    try:
        for idx, batch in enumerate(dataloader):
            conversations = create_video_summary_conversations(batch, prompt=video_summary_prompt)
            video_summaries = summary_engine.chat(conversations, summary_sampling_params)      
            summaries = [summary.outputs[0].text for summary in video_summaries]
            output_queue.put((batch["filename"], summaries))
    finally:
        output_queue.put(None)
        save_thread.shutdown(wait=True)

    save_future.result()
    print("All processes completed. Caption generation and saving done.")


if __name__ == "__main__":
    fire.Fire(main)
