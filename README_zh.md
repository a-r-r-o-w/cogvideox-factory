# CogVideoX Factory

## 简介

这是用于 CogVideoX 微调的仓库。

## 数据集准备

创建两个文件，一个文件包含以换行符分隔的提示词，另一个文件包含以换行符分隔的视频数据路径（视频文件的路径必须相对于您在指定 `--data_root` 时传递的路径）。让我们通过一个例子来更好地理解这一点！

假设您将 `--data_root` 指定为 `/dataset`，并且该目录包含文件：`prompts.txt` 和 `videos.txt`。

`prompts.txt` 文件应包含以换行符分隔的提示词：

```
一段黑白动画序列，主角是一只名为 Rabbity Ribfried 的兔子和一只拟人化的山羊，在一个充满音乐和趣味的环境中，展示他们不断发展的互动。
一段黑白动画序列，场景在船甲板上，主角是一只名为 Bully Bulldoger 的斗牛犬角色，展示了夸张的面部表情和肢体语言。角色从自信到专注，再到紧张和痛苦，展示了一系列情绪，随着它克服挑战。船的内部在背景中保持静止，只有简单的细节，如钟声和开着的门。角色的动态动作和变化的表情推动了故事的发展，没有镜头移动，确保观众专注于其不断变化的反应和肢体动作。
...
```

`videos.txt` 文件应包含以换行符分隔的视频文件路径。请注意，路径应相对于 `--data_root` 目录。

```bash
videos/00000.mp4
videos/00001.mp4
...
```

总体而言，如果您在数据集根目录运行 `tree` 命令，您的数据集应如下所示：

```bash
/dataset
├── prompts.txt
├── videos.txt
├── videos
    ├── videos/00000.mp4
    ├── videos/00001.mp4
    ├── ...
```

使用此格式时，`--caption_column` 必须是 `prompts.txt`，`--video_column` 必须是 `videos.txt`。如果您的数据存储在 CSV 文件中，您也可以指定 `--dataset_file` 为 CSV 的路径，`--caption_column` 和 `--video_column` 为 CSV 文件中的实际列名。

例如，让我们使用这个 [Disney 数据集](https://huggingface.co/datasets/Wild-Heart/Disney-VideoGeneration-Dataset) 进行微调。要下载，可以使用 🤗 Hugging Face CLI。

```bash
huggingface-cli download --repo-type dataset Wild-Heart/Disney-VideoGeneration-Dataset --local-dir video-dataset-disney
```

## 训练

TODO

请查看 `training/*.sh`

注意：未在 MPS 上测试

## 内存需求

训练支持并验证的内存优化包括：

- 来自 [TorchAO](https://github.com/pytorch/ao) 的 `CPUOffloadOptimizer`。
- 来自 [bitsandbytes](https://huggingface.co/docs/bitsandbytes/optimizers) 的低位优化器。

### LoRA 微调

<details>
<summary> AdamW </summary>

With `train_batch_size = 1`:

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

With `train_batch_size = 4`:

|       model        | lora rank | gradient_checkpointing | memory_before_training | memory_before_validation | memory_after_validation | memory_after_testing |
|:------------------:|:---------:|:----------------------:|:----------------------:|:------------------------:|:-----------------------:|:--------------------:|
| THUDM/CogVideoX-2b |    16     |          True          |         12.945         |          21.803          |         21.814          |       24.322         |
| THUDM/CogVideoX-2b |    64     |          True          |         13.035         |          22.254          |         22.254          |       24.572         |
| THUDM/CogVideoX-2b |    256    |          True          |         13.094         |          22.020          |         22.033          |       25.574         |
| THUDM/CogVideoX-5b |    16     |          True          |         19.742         |          46.492          |         46.492          |       38.197         |
| THUDM/CogVideoX-5b |    64     |          True          |         20.006         |          47.805          |         47.805          |       39.365         |
| THUDM/CogVideoX-5b |    256    |          True          |         20.771         |          47.268          |         47.332          |       41.008         |

> [!NOTE]
> 