#!/bin/bash

# Parse command-line arguments
while getopts "i:" opt; do
  case $opt in
    i) ID_TOKEN="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
        exit 1
    ;;
  esac
done

# Check if ID_TOKEN is set
if [ -z "$ID_TOKEN" ]; then
  echo "ID_TOKEN is required. Use -i to specify it."
  exit 1
fi

# Video captioning configuration
OLLAMA_MODEL="llama3.1:latest"
# "crown/darkidol:latest"
DATA_ROOT="./datasets/$ID_TOKEN"
DATASET_FILE="./datasets/${ID_TOKEN}_output.csv"
DATA_ROOT_TRAIN="./datasets/${ID_TOKEN}_prepared"

# Put a folder of videos in ./datasets directory. The name of the folder will be the id token i.e. ohnx
# Prepare dataset configuration
MODEL_ID_PRE_ENCODING="THUDM/CogVideoX-2b"
NUM_GPUS=1
CAPTION_COLUMN="short_prompt"
VIDEO_COLUMN="path"
VIDEO_RESHAPE_MODE="center"
HEIGHT_BUCKETS="480"
WIDTH_BUCKETS="720"
FRAME_BUCKETS="49"
MAX_NUM_FRAMES="49"
MAX_SEQUENCE_LENGTH=226
TARGET_FPS=8
BATCH_SIZE=1
DTYPE=fp16

# Training configuration
GPU_IDS="0"
LEARNING_RATES=("1e-3")
LR_SCHEDULES=("cosine_with_restarts")
OPTIMIZERS=("AdamW")
MAX_TRAIN_STEPS=("1000")
ACCELERATE_CONFIG_FILE="accelerate_configs/default_config.yaml"
MODEL_ID_TRAIN="THUDM/CogVideoX-5b-I2V"
CAPTION_COLUMN_TRAIN="prompts.txt"
VIDEO_COLUMN_TRAIN="videos.txt"
OUTPUT_PATH_TRAIN="./outputs"
DTYPE_TRAIN="fp16"

# Function to run video captioning
run_video_captioning() {
  echo "Running video captioning..."
  python ollama_video_caption_csv.py --instance_data_root $DATA_ROOT --output_path $DATASET_FILE --ollama_model $OLLAMA_MODEL
}

# Function to prepare dataset
prepare_dataset() {
  echo "Preparing dataset..."
  CMD_PRE_ENCODING="\
    torchrun --nproc_per_node=$NUM_GPUS \
      training/prepare_dataset.py \
        --model_id $MODEL_ID_PRE_ENCODING \
        --data_root $DATA_ROOT \
        --dataset_file $DATASET_FILE \
        --caption_column $CAPTION_COLUMN \
        --video_column $VIDEO_COLUMN \
        --output_dir $DATA_ROOT_TRAIN \
        --height_buckets $HEIGHT_BUCKETS \
        --width_buckets $WIDTH_BUCKETS \
        --frame_buckets $FRAME_BUCKETS \
        --max_num_frames $MAX_NUM_FRAMES \
        --max_sequence_length $MAX_SEQUENCE_LENGTH \
        --target_fps $TARGET_FPS \
        --batch_size $BATCH_SIZE \
        --dtype $DTYPE \
        --video_reshape_mode $VIDEO_RESHAPE_MODE \
        --save_latents_and_embeddings \
        --save_image_latents \
        --use_tiling
  "
  echo "===== Running \`$CMD_PRE_ENCODING\` ====="
  eval $CMD_PRE_ENCODING
  echo -ne "===== Finished running script =====\n"
}

# Function to train the model
train_model() {
  echo "Training model..."
  for learning_rate in "${LEARNING_RATES[@]}"; do
    for lr_schedule in "${LR_SCHEDULES[@]}"; do
      for optimizer in "${OPTIMIZERS[@]}"; do
        for steps in "${MAX_TRAIN_STEPS[@]}"; do
          output_dir="${OUTPUT_PATH_TRAIN}/$ID_TOKEN/cogvideox-lora__optimizer_${optimizer}__steps_${steps}__lr-schedule_${lr_schedule}__learning-rate_${learning_rate}/"

          # We don't do validation because we can't fit in GPU VRAM
          cmd="accelerate launch --config_file $ACCELERATE_CONFIG_FILE --gpu_ids $GPU_IDS training/cogvideox_image_to_video_lora.py \
            --pretrained_model_name_or_path $MODEL_ID_TRAIN \
            --load_tensors \
            --data_root $DATA_ROOT_TRAIN \
            --caption_column $CAPTION_COLUMN_TRAIN \
            --video_column $VIDEO_COLUMN_TRAIN \
            --id_token $ID_TOKEN \
            --seed 42 \
            --rank 128 \
            --lora_alpha 32 \
            --mixed_precision $DTYPE_TRAIN \
            --output_dir $output_dir \
            --height 480 \
            --width 720 \
            --fps 8 \
            --max_num_frames 49 \
            --skip_frames_start 0 \
            --skip_frames_end 0 \
            --train_batch_size 1 \
            --num_train_epochs 150 \
            --checkpointing_steps 1000 \
            --gradient_accumulation_steps 1 \
            --max_train_steps $steps \
            --checkpointing_steps 1000 \
            --gradient_checkpointing \
            --learning_rate $learning_rate \
            --lr_scheduler $lr_schedule \
            --lr_warmup_steps 200 \
            --lr_num_cycles 1 \
            --enable_slicing \
            --enable_tiling \
            --optimizer $optimizer \
            --max_grad_norm 1.0 \
            --resume_from_checkpoint latest"
          
          echo "Running command: $cmd"
          eval $cmd
          echo -ne "-------------------- Finished executing script --------------------\n\n"
        done
      done
    done
  done
}

# Execute the functions

# Check for dataset folder
if [ ! -d "$DATA_ROOT" ]; then
  echo "Dataset folder not found ($DATA_ROOT). Exiting."
  exit 1
fi

echo "Dataset folder found ($DATA_ROOT). Continuing."

# Check if the captions are already generated, if not, generate them
if [ ! -f "$DATASET_FILE" ]; then
  echo "Captions not found ($DATASET_FILE). Generating."
  run_video_captioning
fi

echo "Captions found ($DATASET_FILE). Continuing."

# Check if the dataset is prepared, if not, prepare it
if [ ! -d "$DATA_ROOT_TRAIN" ]; then
  echo "Dataset not prepared ($DATA_ROOT_TRAIN). Preparing."
  prepare_dataset
fi

# Train the model
echo "Dataset prepared ($DATA_ROOT_TRAIN). Training."
train_model

echo "All tasks completed."
