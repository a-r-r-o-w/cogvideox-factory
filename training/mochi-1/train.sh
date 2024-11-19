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
  --frame_buckets 84 \
  --load_tensors \
  --validation_prompt \"BW_STYLE A black and white animated scene unfolds with an anthropomorphic goat surrounded by musical notes and symbols, suggesting a playful environment. Mickey Mouse appears, leaning forward in curiosity as the goat remains still. The goat then engages with Mickey, who bends down to converse or react. The dynamics shift as Mickey grabs the goat, potentially in surprise or playfulness, amidst a minimalistic background. The scene captures the evolving relationship between the two characters in a whimsical, animated setting, emphasizing their interactions and emotions\" \
  --validation_prompt_separator ::: \
  --num_validation_videos 1 \
  --validation_epochs 1 \
  --seed 42 \
  --rank 64 \
  --lora_alpha 64 \
  --mixed_precision bf16 \
  --output_dir /raid/.cache/huggingface/sayak/mochi-lora/ \
  --max_num_frames 84 \
  --train_batch_size 1 \
  --dataloader_num_workers 4 \
  --max_train_steps 500 \
  --checkpointing_steps 50 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing \
  --learning_rate 0.0001 \
  --lr_scheduler constant \
  --lr_warmup_steps 0 \
  --lr_num_cycles 1 \
  --enable_slicing \
  --enable_tiling \
  --optimizer adamw \
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