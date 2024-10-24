import argparse
import os
import random
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from decord import VideoReader, cpu
from transformers import AutoModel, AutoTokenizer
from PIL import Image

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama.llms import OllamaLLM

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ollama_model", type=str, required=False, default="llama3.2:3b", help="LLM in Ollama."
    )
    parser.add_argument(
        "--instance_data_root", type=str, required=True, help="Base folder where video files are located."
    )
    parser.add_argument("--cache_dir", type=str, default=None, help="Path to where models are stored.")
    parser.add_argument(
        "--output_path", type=str, default="video_dataset.csv", help="File path where dataset csv should be stored."
    )
    parser.add_argument("--max_new_tokens", type=int, default=226, help="Maximum number of new tokens to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    return parser.parse_args()


SYSTEM_PROMPT = """
You are part of a team of people that create videos using generative models. You use a video-generation model that can generate a video about anything you describe. You will take inputs and generate a prompt for the video generation model.

For example, if you respond with "A beautiful morning in the woods with the sun peaking through the trees", the video generation model will create a video of exactly as described. You task is to summarize the descriptions of videos provided to by users, and create details prompts to feed into the generative model.

There are rules to follow:
- Your entire response will be a single line of text.
- You will not describe what you're doing, you will just respond with the prompt.
- If the user mentions to summarize the prompt in [X] words, make sure to not exceed the limit.

You responses should just be the video generation prompt. Here is an example:
```
A detailed wooden toy ship with intricately carved masts and sails is seen gliding smoothly over a plush, blue carpet that mimics the waves of the sea. The ship's hull is painted a rich brown, with tiny windows. The carpet, soft and textured, provides a perfect backdrop, resembling an oceanic expanse. Surrounding the ship are various other toys and children's items, hinting at a playful environment. The scene captures the innocence and imagination of childhood, with the toy ship's journey symbolizing endless adventures in a whimsical, indoor setting.
```
A street artist, clad in a worn-out denim jacket and a colorful bandana, stands before a vast concrete wall in the heart, holding a can of spray paint, spray-painting a colorful bird on a mottled wall
""".strip()

USER_PROMPT = """
Generate a prompt for a video generation model for the following video summary:

```
{0}
```

Please limit the prompt to [{1}] words.
""".strip()

QUESTION = """
Describe the video. You should pay close attention to every detail in the video and describe it in as much detail as possible.
""".strip()

MAX_NUM_FRAMES = 49

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode_video(video_path: str):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps())
    sample_fps //= 2
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype("uint8")) for v in frames]
    print("num frames:", len(frames))
    return frames


@torch.no_grad()
def main(args: Dict[str, Any]):
    # Test that Ollama is functional
    messages = [
        HumanMessage(content="Just Testing."),
    ]
    model = OllamaLLM(model=args.ollama_model)
    output = model.invoke(messages)
    print(output)

    set_seed(args.seed)

    video_files = [file for file in os.listdir(args.instance_data_root) if file.endswith(".mp4")]
    video_files = [os.path.join(args.instance_data_root, file) for file in video_files]
    video_descriptions = {}

    model_id = "openbmb/MiniCPM-V-2_6"
    model = AutoModel.from_pretrained(
        model_id, trust_remote_code=True, attn_implementation="sdpa", torch_dtype=torch.bfloat16
    ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    for filepath in video_files:
        print(f"Generating video summary for file: `{filepath}`")
        frames = encode_video(filepath)
        msgs = [{"role": "user", "content": frames + [QUESTION]}]

        params = {
            "use_image_id": False,
            "max_slice_nums": 1,
        }

        description = model.chat(image=None, msgs=msgs, tokenizer=tokenizer, **params)

        print(description)

        video_descriptions[filepath] = {"summary": description}

    del model
    del tokenizer
    model = None
    tokenizer = None
    torch.cuda.empty_cache()

    for filepath in video_files:
        print(f"Generating captions for file: `{filepath}`")

        for prompt_type, num_words in [("short_prompt", 25), ("prompt", 75), ("verbose_prompt", 125)]:
            user_prompt = video_descriptions[filepath]["summary"]

            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]

            model = OllamaLLM(model=args.ollama_model)
            output = model.invoke(messages)

            print(output)

            video_descriptions[filepath][prompt_type] = output

    df_data = []
    for filepath, description in video_descriptions.items():
        relative_path = os.path.relpath(filepath, args.instance_data_root)
        df_data.append(
            {
                "path": relative_path,
                "short_prompt": description.get("short_prompt", ""),
                "prompt": description.get("prompt", ""),
                "verbose_prompt": description.get("verbose_prompt", ""),
                "summary": description.get("summary", ""),
            }
        )

    df = pd.DataFrame(df_data)
    df.to_csv(args.output_path)

    print(f"Done. Saved to {args.output_path}")


if __name__ == "__main__":
    args = get_args()
    main(args)
