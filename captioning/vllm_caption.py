"""
Needs `vllm` to be installed from the `main`.
"""

import os
import queue
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import fire
from torch.utils.data import DataLoader
from vllm import LLM, SamplingParams


from dataset_caption import CaptionDataset  # isort:skip


SYSTEM_PROMPT = r"""
You are part of a team of people that create videos using generative models. You use a video-generation model that can generate a video about anything you describe.
For example, if you respond with "A beautiful morning in the woods with the sun peaking through the trees", the video generation model will create a video of exactly as described. You task is to summarize the descriptions of videos provided to by users, and create details prompts to feed into the generative model.
There are a few rules to follow:
- You will only ever output a single video description per request. Do not use newlines.
- If the user mentions to summarize the prompt in [X] words, make sure to not exceed the limit.
You responses should just be the video generation prompt. Here are examples:
- "A detailed wooden toy ship with intricately carved masts and sails is seen gliding smoothly over a plush, blue carpet that mimics the waves of the sea. The ship's hull is painted a rich brown, with tiny windows. The carpet, soft and textured, provides a perfect backdrop, resembling an oceanic expanse. Surrounding the ship are various other toys and children's items, hinting at a playful environment. The scene captures the innocence and imagination of childhood, with the toy ship's journey symbolizing endless adventures in a whimsical, indoor setting."
- "A street artist, clad in a worn-out denim jacket and a colorful bandana, stands before a vast concrete wall in the heart, holding a can of spray paint, spray-painting a colorful bird on a mottled wall"
""".strip()

PROMPT_GEN_USER_PROMPT = r"""
Could you generate a prompt for a video generation model given the following summary:

```
{0}
```

Please limit the prompt to [{1}] words.
""".strip()


def save_results(output_queue, output_dir):
    while True:
        try:
            item = output_queue.get(timeout=5)
            if item is None:
                break

            video_filenames, captions = item

            with open(os.path.join(output_dir, "videos.txt"), "a", encoding="utf-8") as file:
                for filename in video_filenames:
                    file.write(filename + "\n")

            with open(os.path.join(output_dir, "prompts.txt"), "a", encoding="utf-8") as file:
                for caption in captions:
                    file.write(caption + "\n")
        except queue.Empty:
            continue


def create_prompt_generation_conversations(batch, prompt: Optional[str] = None):
    if prompt is None:
        prompt = PROMPT_GEN_USER_PROMPT

    conversations = []

    for i, summary in enumerate(batch):
        conversation = []
        content = []

        content.append({"type": "text", "text": prompt.format(summary, 50)})

        conversation.append({"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]})
        conversation.append({"role": "user", "content": content})

        conversations.append(conversation)

    return conversations


def collate_fn(batch):
    inputs = {
        "summary": [sample["summary"] for sample in batch],
        "filename": [sample["filename"] for sample in batch],
    }
    return inputs


def prepare_dataloader(input_file, num_data_workers, batch_size):
    dataset = CaptionDataset(input_file)

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


def load_prompt_gen_model(
    max_tokens: int,
    num_devices: int,
    download_dir: Optional[str] = None,
    trust_remote_code: bool = False,
):
    engine = LLM(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        dtype="bfloat16",
        tensor_parallel_size=num_devices,
        download_dir=download_dir,
        trust_remote_code=trust_remote_code,
    )
    sampling_params = SamplingParams(max_tokens=max_tokens)
    return engine, sampling_params


def main(
    input_file: str,
    output_dir: str,
    num_devices: int = 1,
    max_prompt_gen_tokens: int = 256,
    prompt_gen_prompt: Optional[str] = None,
    batch_size: int = 8,
    num_data_workers: int = 4,
    num_artifact_workers: int = 4,
    download_dir: Optional[str] = None,
    trust_remote_code: bool = False,
):
    prompt_gen_engine, prompt_gen_sampling_params = load_prompt_gen_model(
        max_tokens=max_prompt_gen_tokens,
        num_devices=num_devices,
        download_dir=download_dir,
        trust_remote_code=trust_remote_code,
    )

    dataloader = prepare_dataloader(
        input_file=input_file,
        num_data_workers=num_data_workers,
        batch_size=batch_size,
    )

    output_queue = queue.Queue()
    save_thread = ThreadPoolExecutor(max_workers=num_artifact_workers)
    os.makedirs(output_dir, exist_ok=True)
    save_future = save_thread.submit(save_results, output_queue, output_dir)

    try:
        for idx, batch in enumerate(dataloader):
            conversations = create_prompt_generation_conversations(batch["summary"], prompt=prompt_gen_prompt)
            prompts = prompt_gen_engine.chat(conversations, prompt_gen_sampling_params)

            # Get outputs and remove surrounding quotes/newlines
            prompts = [" ".join(prompt.outputs[0].text.split("\n")) for prompt in prompts]
            prompts = [prompt.lstrip('"').rstrip('"') for prompt in prompts]

            output_queue.put((batch["filename"], prompts))
    finally:
        output_queue.put(None)
        save_thread.shutdown(wait=True)

    save_future.result()
    print("All processes completed. Caption generation and saving done.")


if __name__ == "__main__":
    fire.Fire(main)
