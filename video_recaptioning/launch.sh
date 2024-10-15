#!/bin/bash

export CUDA_VISIBLE_DEVICES=1,2

python recaption.py --root_dir="video-dataset-disney/videos" \
    --output_dir="video-dataset-disney" \
    --max_num_frames=8 --max_tokens=120 \
    --num_data_workers=4 --batch_size=2 \
    --prompt="Describe this set of frames. Consider the frames to be a part of the same video." \
    --num_artifact_workers=4