# CogVideoX Factor 🧪

在 24GB GPU 内存下微调 Cog 系列视频模型以生成自定义视频 ⚡️📼

TODO：添加有趣的视频结果表

## 快速开始

确保已安装所需的依赖：`pip install -r requirements.txt`。

然后下载数据集：

```bash
# 安装 `huggingface_hub`
huggingface-cli download   --repo-type dataset Wild-Heart/Disney-VideoGeneration-Dataset   --local-dir video-dataset-disney
```

然后启动文本到视频的 LoRA 微调：

```bash
TODO
```

我们现在可以使用训练好的模型进行推理：

```python
TODO
```

我们还可以使用 LoRA 微调 5B 版本：

```python
TODO
```

在下方的部分中，我们提供了有关更多选项的详细信息，这些选项旨在使视频模型的微调尽可能易于使用。

## 数据集准备

创建两个文件，一个文件包含逐行分隔的提示，另一个文件包含逐行分隔的视频数据路径（视频文件的路径必须相对于您在指定 `--data_root` 时传递的路径）。让我们通过一个示例来更好地理解这一点！

假设您指定的 `--data_root` 为 `/dataset`，并且该目录包含以下文件：`prompts.txt` 和 `videos.txt`。

`prompts.txt` 文件应包含逐行分隔的提示：

```
一段黑白动画序列，主角是一只名为 Rabbity Ribfried 的兔子和一只拟人化的山羊，展示了它们在音乐与游戏环境中的互动演变。
一段黑白动画序列，发生在船甲板上，主角是一只名为 Bully Bulldoger 的斗牛犬，展现了夸张的面部表情和肢体语言。角色从自信、专注逐渐转变为紧张与痛苦，展示了随着挑战出现的情感变化。船的内部在背景中保持静止，只有一些简单的细节，如钟声和敞开的门。角色的动态动作和不断变化的表情推动了叙事，没有摄像机运动来分散注意力。
...
```

`videos.txt` 文件应包含逐行分隔的视频文件路径。请注意，路径应相对于 `--data_root` 目录。

```bash
videos/00000.mp4
videos/00001.mp4
...
```

整体而言，如果在数据集根目录运行 `tree` 命令，您的数据集应如下所示：

```bash
/dataset
├── prompts.txt
├── videos.txt
├── videos
    ├── videos/00000.mp4
    ├── videos/00001.mp4
    ├── ...
```

使用此格式时，`--caption_column` 必须是 `prompts.txt`，`--video_column` 必须是 `videos.txt`。如果您将数据存储在 CSV 文件中，还可以指定 `--dataset_file` 为 CSV 的路径，`--caption_column` 和 `--video_column` 为 CSV 文件中的实际列名。

例如，让我们使用[这个](https://huggingface.co/datasets/Wild-Heart/Disney-VideoGeneration-Dataset) Disney 数据集进行微调。要下载，您可以使用 🤗 Hugging Face CLI。

```bash
huggingface-cli download --repo-type dataset Wild-Heart/Disney-VideoGeneration-Dataset --local-dir video-dataset-disney
```

TODO：添加一个关于创建和使用预计算嵌入的部分。

## 训练

我们提供了与 [Cog 系列模型](https://huggingface.co/collections/THUDM/cogvideo-66c08e62f1685a3ade464cce) 兼容的文本到视频和图像到视频生成的训练脚本。

查看 `*.sh` 文件

注意：未在 MPS 上测试

## 内存需求

<table align="center">
<tr>
  <td align="center"><a href="https://www.youtube.com/watch?v=UvRl4ansfCg"> 使用 PyTorch 消除 OOM</a></td>
</tr>
<tr>
  <td align="center"><img src="assets/slaying-ooms.png" style="width: 480px; height: 480px;"></td>
</tr>
</table>

支持和验证的内存优化训练选项包括：

- [`torchao`](https://github.com/pytorch/ao) 中的 `CPUOffloadOptimizer`。您可以阅读它的能力和限制 [此处](https://github.com/pytorch/ao/tree/main/torchao/prototype/low_bit_optim#optimizer-cpu-offload)。简而言之，它允许您使用 CPU 存储可训练的参数和梯度。这导致优化器步骤在 CPU 上进行，需要一个快速的 CPU 优化器，例如 `torch.optim.AdamW(fused=True)` 或在优化器步骤上应用 `torch.compile`。此外，建议不要将模型编译用于训练。梯度裁剪和积累尚不支持。
- [`bitsandbytes`](https://huggingface.co/docs/bitsandbytes/optimizers) 中的低位优化器。TODO：测试并使 [`torchao`](https://github.com/pytorch/ao/tree/main/torchao/prototype/low_bit_optim) 工作
- DeepSpeed Zero2：由于我们依赖 `accelerate`，请按照[本指南](https://huggingface.co/docs/accelerate/en/usage_guides/deepspeed) 配置 `accelerate` 以启用 DeepSpeed Zero2 优化。

> [!IMPORTANT]
> 内存需求是在运行 `training/prepare_dataset.py` 后报告的，它将视频和字幕转换为潜变量和嵌入。在训练过程中，我们直接加载潜变量和嵌入，而不需要 VAE 或 T5 文本编码器。但是，如果您执行验证/测试，则必须加载这些内容，并增加所需的内存量。不执行验证/测试可以节省大量内存，对于使用较小 VRAM 的 GPU，这可以用于专注于训练。
>
> 如果您选择运行验证/测试，可以通过指定 `--enable_model_cpu_offload` 在较低 VRAM 的 GPU 上节省一些内存。

### LoRA 微调

> [!NOTE]
> 图像到视频 LoRA 微调的内存需求与 `THUDM/CogVideoX-5b` 上的文本到视频类似，因此未明确报告。
>
> 此外，要为 I2V 微调准备测试图像，您可以通过修改脚本动态生成它们，或使用以下命令从您的训练数据中提取一些帧：
> `ffmpeg -i input.mp4 -frames:v 1 frame.png`，
> 或提供一个有效且可访问的图像 URL。

...

