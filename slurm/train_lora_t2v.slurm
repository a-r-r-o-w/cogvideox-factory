#!/bin/bash
#SBATCH --job-name=recaptioning
#SBATCH --nodes=1
#SBATCH --qos=normal
#SBATCH --time=168:00:00
#SBATCH --requeue
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --partition=hopper-prod
#SBATCH -o /path/to/logs/logs-%x-%j.out
#SBATCH --exclusive

set -xe

echo "START TIME: $(date)"

# Show some environment variables
echo python3 version = `python3 --version`
echo "NCCL version: $(python -c "import torch;print(torch.cuda.nccl.version())")"
echo "CUDA version: $(python -c "import torch;print(torch.version.cuda)")"

module load cuda/12.1

echo "Starting job"

srun dump_train_text_to_video_lora.sh

echo "END TIME: $(date)"
