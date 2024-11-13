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

export VLLM_WORKER_MULTIPROC_METHOD=spawn

ROOT_DIR="/path/to/datasets/dataset-cakify-yt/videos"
OUTPUT_DIR="/path/to/datasets/dataset-cakify-yt/"
NUM_DEVICES=4
BATCH_SIZE=4
MAX_NUM_FRAMES=16
MAX_SUMMARY_TOKENS=2048
NUM_DATA_WORKERS=8
NUM_ARTIFACT_WORKERS=8
ADDITIONAL_FLAGS="--trust_remote_code"

VIDEO_SUMMARY_PROMPT="Summarize this video and limit to 500 words. The sequence of video frames are of a person cutting cakes shaped as different realistic objects. Make sure to mention about the cake cutting in the summary."

set -xe

module load cuda/12.1

echo "Starting job"

srun python3 captioning/vllm_summary.py \
  --root_dir $ROOT_DIR \
  --output_dir $OUTPUT_DIR \
  --num_devices $NUM_DEVICES \
  --max_num_frames $MAX_NUM_FRAMES \
  --max_summary_tokens $MAX_SUMMARY_TOKENS \
  --video_summary_prompt "$VIDEO_SUMMARY_PROMPT" \
  --batch_size $BATCH_SIZE \
  --num_data_workers $NUM_DATA_WORKERS \
  --num_artifact_workers $NUM_ARTIFACT_WORKERS \
  $ADDITIONAL_FLAGS

echo "End time: $(date)"
