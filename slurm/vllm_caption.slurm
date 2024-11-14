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

export VLLM_WORKER_MULTIPROC_METHOD=spawn

INPUT_FILE="/path/to/datasets/dataset-cakify-yt/summary.csv"
OUTPUT_DIR="/path/to/datasets/dataset-cakify-yt/"
NUM_DEVICES=8
BATCH_SIZE=4
MAX_PROMPT_GEN_TOKENS=256
NUM_DATA_WORKERS=8
NUM_ARTIFACT_WORKERS=8
ADDITIONAL_FLAGS="--trust_remote_code"

PROMPT_GEN_PROMPT="Could you generate a prompt for a text-to-video generation model given the following summary:\n\n\`\`\`\n{0}\n\`\`\`\n\nPlease limit the prompt to [{1}] words. To provide some additional context, these summaries are generated for videos that depict cutting of tasty cakes disguised as a realistic looking object."

set -xe

module load cuda/12.1

echo "Starting job"

srun python3 captioning/vllm_caption.py \
  --input_file $INPUT_FILE \
  --output_dir $OUTPUT_DIR \
  --num_devices $NUM_DEVICES \
  --max_prompt_gen_tokens $MAX_PROMPT_GEN_TOKENS \
  --prompt_gen_prompt "$PROMPT_GEN_PROMPT" \
  --batch_size $BATCH_SIZE \
  --num_data_workers $NUM_DATA_WORKERS \
  --num_artifact_workers $NUM_ARTIFACT_WORKERS \
  $ADDITIONAL_FLAGS

echo "End time: $(date)"
