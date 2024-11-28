#!/bin/bash
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0

GPU_IDS="2"

DATA_ROOT="/home/sayak/cogvideox-factory/training/mochi-1/videos_prepared"
MODEL="genmo/mochi-1-preview"
OUTPUT_PATH=/raid/.cache/huggingface/sayak/mochi-lora/

cmd="accelerate launch --config_file deepspeed.yaml --gpu_ids $GPU_IDS text_to_video_lora.py \
  --pretrained_model_name_or_path $MODEL \
  --data_root $DATA_ROOT \
  --seed 42 \
  --mixed_precision "bf16" \
  --output_dir $OUTPUT_PATH \
  --train_batch_size 1 \
  --dataloader_num_workers 4 \
  --pin_memory \
  --caption_dropout 0.1 \
  --max_train_steps 2000 \
  --checkpointing_steps 200 \
  --checkpoints_total_limit 1 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing \
  --enable_slicing \
  --enable_tiling \
  --enable_model_cpu_offload \
  --optimizer adamw --use_8bit \
  --validation_prompt \"BW_STYLE A black and white animated scene unfolds with an anthropomorphic goat surrounded by musical notes and symbols, suggesting a playful environment. Mickey Mouse appears, leaning forward in curiosity as the goat remains still. The goat then engages with Mickey, who bends down to converse or react. The dynamics shift as Mickey grabs the goat, potentially in surprise or playfulness, amidst a minimalistic background. The scene captures the evolving relationship between the two characters in a whimsical, animated setting, emphasizing their interactions and emotions\" \
  --validation_prompt_separator ::: \
  --num_validation_videos 1 \
  --validation_epochs 1 \
  --allow_tf32 \
  --report_to wandb \
  --nccl_timeout 1800"

echo "Running command: $cmd"
eval $cmd
echo -ne "-------------------- Finished executing script --------------------\n\n"