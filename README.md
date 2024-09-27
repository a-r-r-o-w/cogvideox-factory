# Finetuning CogVideoX


## Dataset Preparation

Create two files where one file contains line-separated prompts and another file contains line-separated paths to video data (the path to video files must be relative to the path you pass when specifying `--data_root`). Let's take a look at an example to understand this better!

Assume you've specified `--data_root` as `/dataset`, and that this directory contains the files: `prompts.txt` and `videos.txt`.

The `prompts.txt` file should contain line-separated prompts:

```
A black and white animated sequence featuring a rabbit, named Rabbity Ribfried, and an anthropomorphic goat in a musical, playful environment, showcasing their evolving interaction.
A black and white animated sequence on a ship's deck features a bulldog character, named Bully Bulldoger, showcasing exaggerated facial expressions and body language. The character progresses from confident to focused, then to strained and distressed, displaying a range of emotions as it navigates challenges. The ship's interior remains static in the background, with minimalistic details such as a bell and open door. The character's dynamic movements and changing expressions drive the narrative, with no camera movement to distract from its evolving reactions and physical gestures.
...
```

The `videos.txt` file should contain line-separate paths to video files. Note that the path should be _relative_ to the `--data_root` directory.

```bash
videos/00000.mp4
videos/00001.mp4
...
```

Overall, this is how your dataset would look like if you ran the `tree` command on the dataset root directory:

```bash
/dataset
├── prompts.txt
├── videos.txt
├── videos
    ├── videos/00000.mp4
    ├── videos/00001.mp4
    ├── ...
```

When using this format, the `--caption_column` must be `prompts.txt` and `--video_column` must be `videos.txt`. If you, instead, have your data stored in a CSV file, you can also specify `--dataset_file` as the path to CSV, the `--caption_column` and `--video_column` as the actual column names in the CSV file.

As an example, let's use [this](https://huggingface.co/datasets/Wild-Heart/Disney-VideoGeneration-Dataset) Disney dataset for finetuning. To download, one can use the 🤗 Hugging Face CLI.

```bash
huggingface-cli download --repo-type dataset Wild-Heart/Disney-VideoGeneration-Dataset --local-dir video-dataset-disney
```

#### Rough notes and TODOs:

- Uncompiled SFT works end-to-end on dummy example. Need to test on larger dataset (not priority at the moment)
- Compiled SFT fails with `THUDM/CogVideoX-2b` throwing the following error (by error, it's more of a graph break situation due to mixin numpy/cpu device when getting sincos positional embeddings).

## Training

TODO

Take a look at `training/*.sh`

Note: Untested on MPS

## Memory requirements

| model |  |

<details>
<summary> stack trace </summary>

```
skipping cudagraphs due to skipping cudagraphs due to cpu device (cat_3). Found from : 
   File "/raid/aryan/nightly-venv/lib/python3.10/site-packages/accelerate/utils/operations.py", line 820, in forward
    return model_forward(*args, **kwargs)
  File "/raid/aryan/nightly-venv/lib/python3.10/site-packages/accelerate/utils/operations.py", line 808, in __call__
    return convert_to_fp32(self.model_forward(*args, **kwargs))
  File "/raid/aryan/nightly-venv/lib/python3.10/site-packages/torch/amp/autocast_mode.py", line 44, in decorate_autocast
    return func(*args, **kwargs)
  File "/home/aryan/work/diffusers/src/diffusers/models/transformers/cogvideox_transformer_3d.py", line 446, in forward
    hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
  File "/home/aryan/work/diffusers/src/diffusers/models/embeddings.py", line 435, in forward
    pos_embedding = self._get_positional_embeddings(height, width, pre_time_compression_frames)
  File "/home/aryan/work/diffusers/src/diffusers/models/embeddings.py", line 385, in _get_positional_embeddings
    pos_embedding = get_3d_sincos_pos_embed(
  File "/home/aryan/work/diffusers/src/diffusers/models/embeddings.py", line 108, in get_3d_sincos_pos_embed
    grid = np.stack(grid, axis=0)
```
</details>

- Make T2V LoRA script up-to-date
- Make I2V LoRA script up-to-date
- Make scripts compatible with DDP
- Make scripts compatible with FSDP
- Make scripts compatible with DeepSpeed
- Test scripts with memory-efficient optimizer
- Test scripts with quantization using torchao, CPUOffloadOptimizer, etc.
- Make 5B lora finetuning work in under 24GB
