# CogVideoX 工廠 🧪

[Read in English](./README.md) | [中文阅读](./README_zh.md)

在 24GB GPU 記憶體下對 Cog 系列視頻模型進行微調以實現客製化視頻生成，支持多解析度 ⚡️📼

<table align="center">
<tr>
  <td align="center"><video src="https://github.com/user-attachments/assets/aad07161-87cb-4784-9e6b-16d06581e3e5">您的瀏覽器不支持視頻標籤。</video></td>
</tr>
</table>

## 快速開始

克隆此儲存庫並確保安裝了相關依賴：`pip install -r requirements.txt`。

接著下載資料集：

```
# 安裝 `huggingface_hub`
huggingface-cli download   --repo-type dataset Wild-Heart/Disney-VideoGeneration-Dataset   --local-dir video-dataset-disney
```

然後啟動 LoRA 微調進行文本到視頻的生成（根據您的選擇修改不同的超參數、資料集根目錄以及其他配置選項）：

```
# 對 CogVideoX 模型進行文本到視頻的 LoRA 微調
./train_text_to_video_lora.sh

# 對 CogVideoX 模型進行文本到視頻的完整微調
./train_text_to_video_sft.sh

# 對 CogVideoX 模型進行圖像到視頻的 LoRA 微調
./train_image_to_video_lora.sh
```

假設您的 LoRA 已保存並推送到 HF Hub，並命名為 `my-awesome-name/my-awesome-lora`，現在我們可以使用微調模型進行推理：

```
import torch
from diffusers import CogVideoXPipeline
from diffusers import export_to_video

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16
).to("cuda")
+ pipe.load_lora_weights("my-awesome-name/my-awesome-lora", adapter_name=["cogvideox-lora"])
+ pipe.set_adapters(["cogvideox-lora"], [1.0])

video = pipe("<my-awesome-prompt>").frames[0]
export_to_video(video, "output.mp4", fps=8)
```

你也可以在[這裡](tests/test_lora_inference.py)來檢查你的Lora是否正常掛載。

**注意：** 對於圖像到視頻的微調，您必須從 [這個分支](https://github.com/huggingface/diffusers/pull/9482) 安裝
diffusers（該分支為 CogVideoX 的圖像到視頻添加了 LoRA 加載支持）直到它被合併。

以下我們提供了更多探索此儲存庫選項的額外部分。所有這些都旨在盡可能降低記憶體需求，使視頻模型的微調變得更易於存取。

## 訓練

在開始訓練之前，請你檢查是否按照[資料集規範](assets/dataset_zh.md)準備好了資料集。 我們提供了適用於文本到視頻 (text-to-video) 和圖像到視頻 (image-to-video) 生成的訓練腳本，兼容 [CogVideoX 模型家族](https://huggingface.co/collections/THUDM/cogvideo-66c08e62f1685a3ade464cce)。訓練可以通過 `train*.sh` 腳本啟動，具體取決於你想要訓練的任務。讓我們以文本到視頻的 LoRA 微調為例。

- 根據你的需求配置環境變數：

  ```
  export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
  export TORCHDYNAMO_VERBOSE=1
  export WANDB_MODE="offline"
  export NCCL_P2P_DISABLE=1
  export TORCH_NCCL_ENABLE_MONITORING=0
  ```

- 配置用於訓練的 GPU：`GPU_IDS="0,1"`

- 選擇訓練的超參數。讓我們以學習率和優化器類型的超參數遍歷為例：

  ```
  LEARNING_RATES=("1e-4" "1e-3")
  LR_SCHEDULES=("cosine_with_restarts")
  OPTIMIZERS=("adamw" "adam")
  MAX_TRAIN_STEPS=("3000")
  ```

- 選擇用於訓練的 Accelerate 配置文件：`ACCELERATE_CONFIG_FILE="accelerate_configs/uncompiled_1.yaml"`
  。我們在 `accelerate_configs/` 目錄中提供了一些默認配置 - 單 GPU 編譯/未編譯、2x GPU DDP、DeepSpeed
  等。你也可以使用 `accelerate config --config_file my_config.yaml` 自定義配置文件。

- 指定字幕和視頻的絕對路徑以及列/文件。

  ```
  DATA_ROOT="/path/to/my/datasets/video-dataset-disney"
  CAPTION_COLUMN="prompt.txt"
  VIDEO_COLUMN="videos.txt"
  ```

- 運行實驗，遍歷不同的超參數：
    ```
  for learning_rate in "${LEARNING_RATES[@]}"; do
    for lr_schedule in "${LR_SCHEDULES[@]}"; do
      for optimizer in "${OPTIMIZERS[@]}"; do
        for steps in "${MAX_TRAIN_STEPS[@]}"; do
          output_dir="/path/to/my/models/cogvideox-lora__optimizer_${optimizer}__steps_${steps}__lr-schedule_${lr_schedule}__learning-rate_${learning_rate}/"

          cmd="accelerate launch --config_file $ACCELERATE_CONFIG_FILE --gpu_ids $GPU_IDS training/cogvideox_text_to_video_lora.py \
            --pretrained_model_name_or_path THUDM/CogVideoX-5b \
            --data_root $DATA_ROOT \
            --caption_column $CAPTION_COLUMN \
            --video_column $VIDEO_COLUMN \
            --id_token BW_STYLE \
            --height_buckets 480 \
            --width_buckets 720 \
            --frame_buckets 49 \
            --dataloader_num_workers 8 \
            --pin_memory \
            --validation_prompt \"BW_STYLE A black and white animated scene unfolds with an anthropomorphic goat surrounded by musical notes and symbols, suggesting a playful environment. Mickey Mouse appears, leaning forward in curiosity as the goat remains still. The goat then engages with Mickey, who bends down to converse or react. The dynamics shift as Mickey grabs the goat, potentially in surprise or playfulness, amidst a minimalistic background. The scene captures the evolving relationship between the two characters in a whimsical, animated setting, emphasizing their interactions and emotions:::BW_STYLE A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance\" \
            --validation_prompt_separator ::: \
            --num_validation_videos 1 \
            --validation_epochs 10 \
            --seed 42 \
            --rank 128 \
            --lora_alpha 128 \
            --mixed_precision bf16 \
            --output_dir $output_dir \
            --max_num_frames 49 \
            --train_batch_size 1 \
            --max_train_steps $steps \
            --checkpointing_steps 1000 \
            --gradient_accumulation_steps 1 \
            --gradient_checkpointing \
            --learning_rate $learning_rate \
            --lr_scheduler $lr_schedule \
            --lr_warmup_steps 400 \
            --lr_num_cycles 1 \
            --enable_slicing \
            --enable_tiling \
            --optimizer $optimizer \
            --beta1 0.9 \
            --beta2 0.95 \
            --weight_decay 0.001 \
            --max_grad_norm 1.0 \
            --allow_tf32 \
            --report_to wandb \
            --nccl_timeout 1800"
          
          echo "Running command: $cmd"
          eval $cmd
          echo -ne "-------------------- Finished executing script --------------------\n\n"
        done
      done
    done
  done
  ```

要了解不同參數的含義，你可以查看 [args](./training/args.py) 文件，或者使用 `--help` 運行訓練腳本。

注意：訓練腳本尚未在 MPS 上測試，因此性能和記憶體需求可能與下面的 CUDA 報告差異很大。

## 記憶體需求

[... 記憶體需求表格和詳細內容保持不變 ...]

## 待辦事項

- [x] 使腳本兼容 DDP
- [ ] 使腳本兼容 FSDP
- [x] 使腳本兼容 DeepSpeed
- [ ] 基於 vLLM 的字幕腳本
- [x] 在 `prepare_dataset.py` 中支持多解析度/幀數
- [ ] 分析性能瓶頸並盡可能減少同步操作
- [ ] 支持 QLoRA（優先），以及其他高使用率的 LoRA 方法
- [x] 使用 bitsandbytes 的節省記憶體優化器測試腳本
- [x] 使用 CPUOffloadOptimizer 等測試腳本
- [ ] 使用 torchao 量化和低位記憶體優化器測試腳本（目前在 AdamW（8/4-bit torchao）上報錯）
- [ ] 使用 AdamW（8-bit bitsandbytes）+ CPUOffloadOptimizer（帶有梯度卸載）的測試腳本（目前報錯）
- [ ] [Sage Attention](https://github.com/thu-ml/SageAttention)（與作者合作支持反向傳播，並針對 A100 進行優化）

> [!重要]
> 由於我們的目標是使腳本盡可能節省記憶體，因此我們不保證支持多 GPU 訓練。