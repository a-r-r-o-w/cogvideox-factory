import pathlib
import uuid
from typing import Any, Dict, Optional

import fire
import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


class PromptDataset(Dataset):
    def __init__(self, filename: str, id_token: Optional[str] = None) -> None:
        super().__init__()

        self.id_token = id_token or ""

        df = pd.read_csv(filename)

        self.prompts = df["prompt"]
        self.heights = df["height"]
        self.widths = df["width"]
        self.num_frames = df["num_frames"]

    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, index):
        prompt = self.prompts[index].format(id_token=self.id_token).strip()
        return {
            "prompt": prompt,
            "height": self.heights[index],
            "width": self.widths[index],
            "num_frames": self.num_frames[index],
        }


class CollateFunction:
    def __call__(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        prompt = [x["prompt"] for x in data]
        height = [x["height"] for x in data]
        width = [x["width"] for x in data]
        num_frames = [x["num_frames"] for x in data]

        return {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_frames": num_frames,
        }


DTYPE_MAPPING = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def prompt_to_filename(x: str) -> str:
    for c in ["\"", "'", ",", "\\", " "]:
        x = x.replace(c, "-")
    return x


def main(
    dataset_file: str,
    model_id: str = "THUDM/CogVideoX-5b",
    lora_id: Optional[str] = None,
    id_token: Optional[str] = None,
    dtype: str = "bf16",
    enable_model_cpu_offload: bool = False,
    output_dir: str = "text_to_video_lora_outputs",
    save_fps: int = 8,
    seed: int = 42,
) -> None:
    dataset = PromptDataset(dataset_file, id_token)

    collate_fn = CollateFunction()
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=1,
    )

    output_dir: pathlib.Path = pathlib.Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    pipe = CogVideoXPipeline.from_pretrained(
        model_id,
        torch_dtype=DTYPE_MAPPING[dtype]
    )

    print("LoRA ID:", lora_id)

    if lora_id is not None:
        pipe.load_lora_weights(lora_id)

    accelerator = Accelerator()

    if enable_model_cpu_offload:
        pipe.enable_model_cpu_offload(gpu_id=accelerator.device.index)
    else:
        pipe = pipe.to(accelerator.device)
    
    count = 0
    for _, data_raw in tqdm(enumerate(dataloader), total=len(dataloader)):
        input_data = []

        with accelerator.split_between_processes(data_raw) as data:
            video = pipe(
                prompt=data["prompt"][0],
                height=data["height"][0],
                width=data["width"][0],
                num_frames=data["num_frames"][0],
                num_inference_steps=50,
                guidance_scale=6.0,
                use_dynamic_cfg=False,
                generator=torch.Generator().manual_seed(seed),
            ).frames[0]
            input_data.append(list(data.items()))
        
        video = gather_object(video)
        input_data = gather_object(input_data)

        if accelerator.is_main_process:
            count += 1
            for data in input_data:
                data = dict(data)
                
                filename = ""
                filename += f"height_{data['height'][0]}" + "---"
                filename += f"width_{data['width'][0]}" + "---"
                filename += f"num_frames_{data['num_frames'][0]}" + "---"
                filename += prompt_to_filename(data["prompt"][0])[:25] + "---"
                filename += str(uuid.uuid4())
                filename += ".mp4"
                filename = output_dir.joinpath(filename)
                
                export_to_video(video, filename, fps=save_fps)
    
    if accelerator.is_main_process:
        print(f"Text-to-video generation for LoRA completed. Results saved in {output_dir}.")


if __name__ == "__main__":
    fire.Fire(main)
