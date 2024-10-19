#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

# Path to where vLLM models should be downloaded
DOWNLOAD_DIR="/path/to/download/dir"

# Path to where video files are located
ROOT_DIR="/path/to/video/files"

# Path to where captions should be stored
OUTPUT_DIR="/path/to/save/captions"

# Other configurations
MAX_FRAMES=8
MAX_TOKENS=120
BATCH_SIZE=2
NUM_DATA_WORKERS=4
NUM_ARTIFACT_WORKERS=4

PROMPT="Please describe the content of this video in as much detail as possible, including the objects, scenery, animals, characters, and camera movements within the video. Do not include '\n' in your response. Please start the description with the video content directly. Please describe the content of the video and the changes that occur, in chronological order."

python recaption.py \
  --root_dir $ROOT_DIR \
  --output_dir $OUTPUT_DIR \
  --num_devices 1 \
  --max_num_frames $MAX_FRAMES \
  --max_tokens $MAX_TOKENS \
  --num_data_workers $NUM_DATA_WORKERS \
  --batch_size $BATCH_SIZE \
  --prompt $PROMPT \
  --num_artifact_workers $NUM_ARTIFACT_WORKERS \
  --download_dir $DOWNLOAD_DIR
