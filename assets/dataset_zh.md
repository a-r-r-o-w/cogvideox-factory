## 数据集格式

### 提示词数据集要求

创建 `prompt.txt` 文件，文件应包含逐行分隔的提示。请注意，提示必须是英文，并且建议使用 [提示润色脚本](https://github.com/THUDM/CogVideo/blob/main/inference/convert_demo.py) 进行润色。或者可以使用 [CogVideo-caption](https://huggingface.co/THUDM/cogvlm2-llama3-caption) 进行数据标注：

```
A black and white animated sequence featuring a rabbit, named Rabbity Ribfried, and an anthropomorphic goat in a musical, playful environment, showcasing their evolving interaction.
A black and white animated sequence on a ship’s deck features a bulldog character, named Bully Bulldoger, showcasing exaggerated facial expressions and body language...
...
```

### 视频数据集要求

该框架支持的分辨率和帧数需要满足以下条件：

- **支持的分辨率（宽 * 高）**：
    - 任意分辨率且必须能被32整除。例如，`720 * 480`, `1920 * 1020` 等分辨率。

- **支持的帧数（Frames）**：
    - 必须是 `4 * k` 或 `4 * k + 1`（例如：16, 32, 49, 81）

所有的视频建议放在一个文件夹中。


接着，创建 `videos.txt` 文件。 `videos.txt` 文件应包含逐行分隔的视频文件路径。请注意，路径必须相对于 `--data_root` 目录。格式如下：

```
videos/00000.mp4
videos/00001.mp4
...
```

对于有兴趣了解更多细节的开发者，您可以查看相关的 `BucketSampler` 代码。

### 数据集结构

您的数据集结构应如下所示，通过运行`tree`命令，你能看到:

```
dataset
├── prompt.txt
├── videos.txt
├── videos
    ├── videos/00000.mp4
    ├── videos/00001.mp4
    ├── ...
```

### 使用数据集

当使用此格式时，`--caption_column` 应为 `prompt.txt`，`--video_column` 应为 `videos.txt`。如果您的数据存储在 CSV
文件中，也可以指定 `--dataset_file` 为 CSV 文件的路径，`--caption_column` 和 `--video_column` 为 CSV
文件中的实际列名。请参考 [test_dataset](../tests/test_dataset.py) 文件中的一些简单示例。

例如，使用 [这个](https://huggingface.co/datasets/Wild-Heart/Disney-VideoGeneration-Dataset) Disney 数据集进行微调。下载可通过🤗
Hugging Face CLI 完成：

```
huggingface-cli download --repo-type dataset Wild-Heart/Disney-VideoGeneration-Dataset --local-dir video-dataset-disney
```

该数据集已按照预期格式准备好，可直接使用。但是，直接使用视频数据集可能会导致较小 VRAM 的 GPU 出现
OOM（内存不足），因为它需要加载 [VAE](https://huggingface.co/THUDM/CogVideoX-5b/tree/main/vae)
（将视频编码为潜在空间）和大型 [T5-XXL](https://huggingface.co/google/t5-v1_1-xxl/)

文本编码器。为了降低内存需求，您可以使用 `training/prepare_dataset.py` 脚本预先计算潜在变量和嵌入。

填写或修改 `prepare_dataset.sh` 中的参数并执行它以获得预先计算的潜在变量和嵌入（请确保指定 `--save_latents_and_embeddings`
以保存预计算的工件）。如果准备图像到视频的训练，请确保传递 `--save_image_latents`，它对沙子进行编码，将图像潜在值与视频一起保存。
在训练期间使用这些工件时，确保指定 `--load_tensors` 标志，否则将直接使用视频并需要加载文本编码器和
VAE。该脚本还支持 PyTorch DDP，以便可以使用多个 GPU 并行编码大型数据集（修改 `NUM_GPUS` 参数）。
