#!/bin/bash

GPU_ID=0
VIDEO_DIR=/home/sayak/cogvideox-factory/video-dataset-disney-organized
OUTPUT_DIR=videos_prepared

python trim_and_crop_videos.py $VIDEO_DIR $OUTPUT_DIR --num_frames=37 --resolution=480x848 --force_upsample

CUDA_VISIBLE_DEVICES=$GPU_ID python embed.py $OUTPUT_DIR --shape=37x480x848