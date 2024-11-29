#!/bin/bash

GPU_ID=0
VIDEO_DIR=video-dataset-disney-organized
OUTPUT_DIR=videos_prepared
NUM_FRAMES=37
RESOLUTION=480x848

python trim_and_crop_videos.py $VIDEO_DIR $OUTPUT_DIR --num_frames=$NUM_FRAMES --resolution=$RESOLUTION --force_upsample

CUDA_VISIBLE_DEVICES=$GPU_ID python embed.py $OUTPUT_DIR --shape=37x480x848