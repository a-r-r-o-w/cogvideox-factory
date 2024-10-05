#!/bin/bash

MODEL_ID="/share/official_pretrains/hf_home/CogVideoX-5b"

DATA_ROOT="/share/home/zyx/disney_cogvideox"
CAPTION_COLUMN="prompts.txt"
VIDEO_COLUMN="videos.txt"
OUTPUT_DIR="/share/home/zyx/disney_cogvideox-encoded-multi"
HEIGHT=480
WIDTH=720
MAX_NUM_FRAMES=49
MAX_SEQUENCE_LENGTH=226
TARGET_FPS=8
BATCH_SIZE=1
DTYPE=fp32
NUM_GPUS=8

CMD="torchrun --nproc_per_node=$NUM_GPUS \
  training/prepare_dataset.py \
  --model_id $MODEL_ID \
  --data_root $DATA_ROOT \
  --caption_column $CAPTION_COLUMN \
  --video_column $VIDEO_COLUMN \
  --output_dir $OUTPUT_DIR \
  --height $HEIGHT \
  --width $WIDTH \
  --max_num_frames $MAX_NUM_FRAMES \
  --max_sequence_length $MAX_SEQUENCE_LENGTH \
  --target_fps $TARGET_FPS \
  --batch_size $BATCH_SIZE \
  --dtype $DTYPE \
  --save_tensors"

echo "===== Running \`$CMD\` ====="
eval $CMD
echo -ne "===== Finished running script =====\n"