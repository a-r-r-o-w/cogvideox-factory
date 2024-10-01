# export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
# export TORCHDYNAMO_VERBOSE=1
export WANDB_MODE="offline"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0

GPU_IDS="3"
LEARNING_RATES=("1e-4")
LR_SCHEDULES=("cosine_with_restarts")
OPTIMIZERS=("adamw")
MAX_TRAIN_STEPS=("20000")

# DATA_ROOT="/raid/aryan/dataset-cogvideox/"
DATA_ROOT="/raid/aryan/openvid-1m"
CAPTION_COLUMN="prompts.txt"
VIDEO_COLUMN="videos.txt"

for learning_rate in "${LEARNING_RATES[@]}"; do
  for lr_schedule in "${LR_SCHEDULES[@]}"; do
    for optimizer in "${OPTIMIZERS[@]}"; do
      for steps in "${MAX_TRAIN_STEPS[@]}"; do
        cache_dir="/raid/aryan/cogvideox-sft/"
        output_dir="/raid/aryan/cogvideox-sft__optimizer_${optimizer}__steps_${steps}__lr-schedule_${lr_schedule}__learning-rate_${learning_rate}/"

        cmd="accelerate launch --config_file accelerate_configs/uncompiled_1.yaml --gpu_ids $GPU_IDS training/cogvideox_text_to_video_sft.py \
          --pretrained_model_name_or_path THUDM/CogVideoX-2b \
          --cache_dir $cache_dir \
          --data_root $DATA_ROOT \
          --caption_column $CAPTION_COLUMN \
          --video_column $VIDEO_COLUMN \
          --height_buckets 480 \
          --width_buckets 720 \
          --frame_buckets 49 \
          --validation_prompt \"a man wearing a bicycle helmet, riding a bike through a forested area. The man is wearing a black t-shirt and appears to be in motion, as suggested by the slight blur of the background. The forest is lush and green, with trees and foliage filling the background. The man's helmet is white with a black visor, and he is looking directly at the camera with a slight smile on his face. The style of the video is casual and candid, capturing a moment of outdoor activity:::A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance\" \
          --validation_prompt_separator ::: \
          --num_validation_videos 1 \
          --validation_epochs 1 \
          --seed 42 \
          --mixed_precision fp16 \
          --output_dir $output_dir \
          --max_num_frames 49 \
          --train_batch_size 1 \
          --max_train_steps $steps \
          --checkpointing_steps 2000 \
          --gradient_accumulation_steps 1 \
          --gradient_checkpointing \
          --learning_rate $learning_rate \
          --lr_scheduler $lr_schedule \
          --lr_warmup_steps 200 \
          --lr_num_cycles 1 \
          --enable_slicing \
          --enable_tiling \
          --optimizer $optimizer \
          --beta1 0.9 \
          --beta2 0.95 \
          --weight_decay 0.001 \
          --max_grad_norm 1.0 \
          --allow_tf32 \
          --report_to wandb
          --nccl_timeout 1800"
        
        echo "Running command: $cmd"
        eval $cmd
        echo -ne "-------------------- Finished executing script --------------------\n\n"
      done
    done
  done
done
