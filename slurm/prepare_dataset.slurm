#!/bin/bash
#SBATCH --job-name=recaptioning
#SBATCH --nodes=1
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --partition=hopper-prod
#SBATCH -o /path/to/logs/logs-%x-%j.out
#SBATCH --exclusive

MODEL_ID="THUDM/CogVideoX-5b"
NUM_GPUS=8

DATA_ROOT="/path/to/datasets/dataset-cakify-yt"
CAPTION_COLUMN="prompts.txt"
VIDEO_COLUMN="videos.txt"
OUTPUT_DIR="/path/to/datasets/dataset-cakify-yt-encoded"
HEIGHT_BUCKETS="480 720 768"
WIDTH_BUCKETS="480 720 768"
FRAME_BUCKETS="17 33 49"
MAX_NUM_FRAMES=49
MAX_SEQUENCE_LENGTH=226
TARGET_FPS=8
BATCH_SIZE=1
DTYPE=fp32
ADDITIONAL_FLAGS="--save_image_latents --save_latents_and_embeddings"

set -xe

module load cuda/12.1

echo "Starting job"

srun torchrun --nproc_per_node=$NUM_GPUS training/prepare_dataset.py \
  --model_id $MODEL_ID \
  --data_root $DATA_ROOT \
  --caption_column $CAPTION_COLUMN \
  --video_column $VIDEO_COLUMN \
  --output_dir $OUTPUT_DIR \
  --height_buckets $HEIGHT_BUCKETS \
  --width_buckets $WIDTH_BUCKETS \
  --frame_buckets $FRAME_BUCKETS \
  --max_num_frames $MAX_NUM_FRAMES \
  --max_sequence_length $MAX_SEQUENCE_LENGTH \
  --target_fps $TARGET_FPS \
  --batch_size $BATCH_SIZE \
  --dtype $DTYPE \
  $ADDITIONAL_FLAGS

echo "End time: $(date)"
