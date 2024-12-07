#!/bin/bash
#SBATCH --job-name=cogvideox-lora-inference
#SBATCH --nodes=1
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --partition=hopper-prod
#SBATCH -o /path/to/logs/logs-%x-%j.out

set -xe

module load cuda/12.1

echo "START TIME: $(date)"

echo "Starting job"

srun dump_inference_lora_t2v.sh

echo "END TIME: $(date)"
