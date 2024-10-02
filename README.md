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

<table align="center">
<tr>
  <td align="center"><a href="https://www.youtube.com/watch?v=UvRl4ansfCg"> Slaying OOMs with PyTorch</a></td>
</tr>
<tr>
  <td align="center"><img src="assets/slaying-ooms.png" style="width: 480px; height: 480px;"></td>
</tr>
</table>

The memory requirements are reported after running the `training/prepare_dataset.py`, which converts the videos and captions to latents and embeddings. During training, we directly load the latents and embeddings, and do not require the VAE or the T5 text encoder. However, if you perform validation/testing, these must be loaded and increase the amount of required memory. Not performing validation/testing saves a significant amount of memory, which can be used to focus solely on training if you're on smaller VRAM GPUs.

<details>
<summary> AdamW </summary>

|       model        | lora rank | gradient_checkpointing | memory_before_training | memory_before_validation | memory_after_validation | memory_after_testing |
|:------------------:|:---------:|:----------------------:|:----------------------:|:------------------------:|:-----------------------:|:--------------------:|
| THUDM/CogVideoX-2b |    16     |          False         |         12.945         |          43.764          |         46.918          |       24.234         |
| THUDM/CogVideoX-2b |    16     |          True          |         12.945         |          12.945          |         21.121          |       24.234         |
| THUDM/CogVideoX-2b |    64     |          False         |         13.035         |          44.314          |         47.469          |       24.469         |
| THUDM/CogVideoX-2b |    64     |          True          |         13.036         |          13.035          |         21.564          |       24.500         |
| THUDM/CogVideoX-2b |    256    |          False         |         13.095         |          45.826          |         48.990          |       25.543         |
| THUDM/CogVideoX-2b |    256    |          True          |         13.094         |          13.095          |         22.344          |       25.537         |
| THUDM/CogVideoX-5b |    16     |          True          |         19.742         |          19.742          |         28.746          |       38.123         |
| THUDM/CogVideoX-5b |    64     |          True          |         20.006         |          20.818          |         30.338          |       38.738         |
| THUDM/CogVideoX-5b |    256    |          True          |         20.771         |          22.119          |         31.939          |       41.537         |

> [!NOTE]
> Trying to run CogVideoX-5b without gradient checkpointing OOMs even on an A100 (80 GB), so the memory measurements have not been specified.

</details>

<details>
<summary> AdamW (8-bit bitsandbytes) </summary>

|       model        | lora rank | gradient_checkpointing | memory_before_training | memory_before_validation | memory_after_validation | memory_after_testing |
|:------------------:|:---------:|:----------------------:|:----------------------:|:------------------------:|:-----------------------:|:--------------------:|
| THUDM/CogVideoX-2b |    16     |          False         |         12.945         |          43.732          |         46.887          |        24.195        |
| THUDM/CogVideoX-2b |    16     |          True          |         12.945         |          12.945          |         21.430          |        24.195        |
| THUDM/CogVideoX-2b |    64     |          False         |         13.035         |          44.004          |         47.158          |        24.369        |
| THUDM/CogVideoX-2b |    64     |          True          |         13.035         |          13.035          |         21.297          |        24.357        |
| THUDM/CogVideoX-2b |    256    |          False         |         13.035         |          45.291          |         48.455          |        24.836        |
| THUDM/CogVideoX-2b |    256    |          True          |         13.035         |          13.035          |         21.625          |        24.869        |
| THUDM/CogVideoX-5b |    16     |          True          |         19.742         |          19.742          |         28.602          |        38.049        |
| THUDM/CogVideoX-5b |    64     |          True          |         20.006         |          20.818          |         29.359          |        38.520        |
| THUDM/CogVideoX-5b |    256    |          True          |         20.771         |          21.352          |         30.727          |        39.596        |

</details>

<details>
<summary> AdamW (8-bit torchao) </summary>

Currently, errors out with following stack-trace:

```python
Traceback (most recent call last):               
  File "/raid/aryan/cogvideox-distillation/training/cogvideox_text_to_video_lora.py", line 915, in <module>                                                                                         
    main(args)                                   
  File "/raid/aryan/cogvideox-distillation/training/cogvideox_text_to_video_lora.py", line 719, in main                                                                                             
    optimizer.step()                                                                              
  File "/raid/aryan/nightly-venv/lib/python3.10/site-packages/accelerate/optimizer.py", line 159, in step                                                                                           
    self.scaler.step(self.optimizer, closure)    
  File "/raid/aryan/nightly-venv/lib/python3.10/site-packages/torch/amp/grad_scaler.py", line 457, in step                                                                                          
    retval = self._maybe_opt_step(optimizer, optimizer_state, *args, **kwargs)                    
  File "/raid/aryan/nightly-venv/lib/python3.10/site-packages/torch/amp/grad_scaler.py", line 352, in _maybe_opt_step                                                                               
    retval = optimizer.step(*args, **kwargs)     
  File "/raid/aryan/nightly-venv/lib/python3.10/site-packages/accelerate/optimizer.py", line 214, in patched_step                                                                                   
    return method(*args, **kwargs)               
  File "/raid/aryan/nightly-venv/lib/python3.10/site-packages/torch/optim/lr_scheduler.py", line 137, in wrapper                                                                                    
    return func.__get__(opt, opt.__class__)(*args, **kwargs)                                      
  File "/raid/aryan/nightly-venv/lib/python3.10/site-packages/torch/optim/optimizer.py", line 487, in wrapper                                                                                       
    out = func(*args, **kwargs)                  
  File "/raid/aryan/nightly-venv/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context                                                                            
    return func(*args, **kwargs)                 
  File "/raid/aryan/nightly-venv/lib/python3.10/site-packages/torchao/prototype/low_bit_optim/adam.py", line 87, in step                                                                            
    raise RuntimeError(                          
RuntimeError: lr was changed to a non-Tensor object. If you want to update lr, please use optim.param_groups[0]['lr'].fill_(new_lr)
```
</details>

<details>
<summary> AdamW (4-bit torchao) </summary>

Same error as AdamW (8-bit torchao)

</details>


> [!NOTE]
> `memory_after_validation` is indicative of the peak memory required for training. This is because apart from the activations, parameters and gradients stored for training, you also need to load the vae and text encoder in memory and spend some memory to perform inference. In order to reduce total memory required to perform training, one can choose to not perform validation/testing as part of the training script.

- Make T2V LoRA script up-to-date
- Make I2V LoRA script up-to-date
- Make scripts compatible with DDP
- Make scripts compatible with FSDP
- Make scripts compatible with DeepSpeed
- Test scripts with memory-efficient optimizer
- Test scripts with quantization using torchao, CPUOffloadOptimizer, etc.
- Make 5B lora finetuning work in under 24GB
