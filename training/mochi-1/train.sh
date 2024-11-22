export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0

GPU_IDS="2"

DATA_ROOT="/home/sayak/cogvideox-factory/video-dataset-disney/mochi-1/preprocessed-dataset"

CAPTION_COLUMN="prompts.txt"
VIDEO_COLUMN="videos.txt"

cmd="accelerate launch --config_file deepspeed.yaml --gpu_ids $GPU_IDS text_to_video_lora.py \
  --pretrained_model_name_or_path genmo/mochi-1-preview \
  --data_root $DATA_ROOT \
  --caption_column $CAPTION_COLUMN \
  --video_column $VIDEO_COLUMN \
  --id_token BW_STYLE \
  --height_buckets 480 \
  --width_buckets 848 \
  --frame_buckets 85 \
  --load_tensors \
  --seed 42 \
  --rank 64 \
  --lora_alpha 64 \
  --mixed_precision bf16 \
  --output_dir /raid/.cache/huggingface/sayak/mochi-lora/ \
  --max_num_frames 85 \
  --train_batch_size 1 \
  --dataloader_num_workers 4 \
  --max_train_steps 10 \
  --checkpointing_steps 50 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing \
  --learning_rate 1e-5 \
  --lr_scheduler constant \
  --lr_warmup_steps 0 \
  --lr_num_cycles 1 \
  --enable_slicing \
  --enable_tiling \
  --optimizer adamw --use_8bit \
  --beta1 0.9 \
  --beta2 0.95 \
  --beta3 0.99 \
  --weight_decay 0.001 \
  --max_grad_norm 1.0 \
  --allow_tf32 \
  --report_to wandb \
  --push_to_hub \
  --nccl_timeout 1800"

echo "Running command: $cmd"
eval $cmd
echo -ne "-------------------- Finished executing script --------------------\n\n"