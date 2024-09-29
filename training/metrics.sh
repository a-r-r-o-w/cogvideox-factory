export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
export TORCHDYNAMO_VERBOSE=1
export WANDB_MODE="offline"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0

GPU_IDS="1"
LEARNING_RATES=("1e-4")
LR_SCHEDULES=("cosine_with_restarts")
OPTIMIZERS=("adamw")
MAX_TRAIN_STEPS=("2")
RANK=("16" "64" "256")
GRADIENT_CHECKPOINTING=("" "--gradient_checkpointing")

DATA_ROOT="/raid/aryan/video-dataset-disney/"
CAPTION_COLUMN="prompts.txt"
VIDEO_COLUMN="videos.txt"

for learning_rate in "${LEARNING_RATES[@]}"; do
  for lr_schedule in "${LR_SCHEDULES[@]}"; do
    for optimizer in "${OPTIMIZERS[@]}"; do
      for steps in "${MAX_TRAIN_STEPS[@]}"; do
        for rank in "${RANK[@]}"; do
          for gradient_checkpointing in "${GRADIENT_CHECKPOINTING[@]}"; do
            cache_dir="/raid/aryan/cogvideox-lora/"
            output_dir="/raid/aryan/cogvideox-lora__optimizer_${optimizer}__steps_${steps}__lr-schedule_${lr_schedule}__learning-rate_${learning_rate}/"

            cmd="accelerate launch --config_file accelerate_configs/uncompiled_1.yaml --gpu_ids $GPU_IDS cogvideox_text_to_video_lora.py \
              --pretrained_model_name_or_path THUDM/CogVideoX-2b \
              --cache_dir $cache_dir \
              --data_root $DATA_ROOT \
              --caption_column $CAPTION_COLUMN \
              --video_column $VIDEO_COLUMN \
              --id_token BW_STYLE \
              --validation_prompt \"BW_STYLE A black and white animated scene unfolds with an anthropomorphic goat surrounded by musical notes and symbols, suggesting a playful environment. Mickey Mouse appears, leaning forward in curiosity as the goat remains still. The goat then engages with Mickey, who bends down to converse or react. The dynamics shift as Mickey grabs the goat, potentially in surprise or playfulness, amidst a minimalistic background. The scene captures the evolving relationship between the two characters in a whimsical, animated setting, emphasizing their interactions and emotions:::BW_STYLE A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance\" \
              --validation_prompt_separator ::: \
              --num_validation_videos 1 \
              --validation_epochs 1 \
              --seed 42 \
              --rank $rank \
              --lora_alpha 64 \
              --mixed_precision fp16 \
              --output_dir $output_dir \
              --max_num_frames 49 \
              --train_batch_size 1 \
              --max_train_steps $steps \
              --checkpointing_steps 1000 \
              --gradient_accumulation_steps 1 \
              $gradient_checkpointing \
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
              --report_to wandb \
              --nccl_timeout 1800"
            
            echo "Running command: $cmd"
            eval $cmd
            echo -ne "-------------------- Finished executing script --------------------\n\n"
          done
        done
      done
    done
  done
done

# For testing load from tensor data
# export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
# export TORCHDYNAMO_VERBOSE=1
# export WANDB_MODE="offline"
# export NCCL_P2P_DISABLE=1
# export TORCH_NCCL_ENABLE_MONITORING=0

# GPU_IDS="1"
# LEARNING_RATES=("1e-4")
# LR_SCHEDULES=("cosine_with_restarts")
# OPTIMIZERS=("adamw")
# MAX_TRAIN_STEPS=("2")
# RANK=("16" "64" "256")
# GRADIENT_CHECKPOINTING=("" "--gradient_checkpointing")

# DATA_ROOT="/raid/aryan/cogvideox-distillation/training/dump"
# CAPTION_COLUMN="prompts.txt"
# VIDEO_COLUMN="videos.txt"

# for learning_rate in "${LEARNING_RATES[@]}"; do
#   for lr_schedule in "${LR_SCHEDULES[@]}"; do
#     for optimizer in "${OPTIMIZERS[@]}"; do
#       for steps in "${MAX_TRAIN_STEPS[@]}"; do
#         for rank in "${RANK[@]}"; do
#           for gradient_checkpointing in "${GRADIENT_CHECKPOINTING[@]}"; do
#             cache_dir="/raid/aryan/cogvideox-lora/"
#             output_dir="/raid/aryan/cogvideox-lora__optimizer_${optimizer}__steps_${steps}__lr-schedule_${lr_schedule}__learning-rate_${learning_rate}/"

#             cmd="accelerate launch --config_file accelerate_configs/uncompiled_1.yaml --gpu_ids $GPU_IDS cogvideox_text_to_video_lora.py \
#               --pretrained_model_name_or_path THUDM/CogVideoX-2b \
#               --cache_dir $cache_dir \
#               --data_root $DATA_ROOT \
#               --caption_column $CAPTION_COLUMN \
#               --video_column $VIDEO_COLUMN \
#               --id_token BW_STYLE \
#               --load_tensors \
#               --validation_prompt \"BW_STYLE A black and white animated scene unfolds with an anthropomorphic goat surrounded by musical notes and symbols, suggesting a playful environment. Mickey Mouse appears, leaning forward in curiosity as the goat remains still. The goat then engages with Mickey, who bends down to converse or react. The dynamics shift as Mickey grabs the goat, potentially in surprise or playfulness, amidst a minimalistic background. The scene captures the evolving relationship between the two characters in a whimsical, animated setting, emphasizing their interactions and emotions:::BW_STYLE A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance\" \
#               --validation_prompt_separator ::: \
#               --num_validation_videos 1 \
#               --validation_epochs 1 \
#               --seed 42 \
#               --rank $rank \
#               --lora_alpha 64 \
#               --mixed_precision fp16 \
#               --output_dir $output_dir \
#               --max_num_frames 49 \
#               --train_batch_size 1 \
#               --max_train_steps $steps \
#               --checkpointing_steps 1000 \
#               --gradient_accumulation_steps 1 \
#               $gradient_checkpointing \
#               --learning_rate $learning_rate \
#               --lr_scheduler $lr_schedule \
#               --lr_warmup_steps 200 \
#               --lr_num_cycles 1 \
#               --enable_slicing \
#               --enable_tiling \
#               --optimizer $optimizer \
#               --beta1 0.9 \
#               --beta2 0.95 \
#               --weight_decay 0.001 \
#               --max_grad_norm 1.0 \
#               --allow_tf32 \
#               --report_to wandb \
#               --nccl_timeout 1800"
            
#             echo "Running command: $cmd"
#             eval $cmd
#             echo -ne "-------------------- Finished executing script --------------------\n\n"
#           done
#         done
#       done
#     done
#   done
# done
