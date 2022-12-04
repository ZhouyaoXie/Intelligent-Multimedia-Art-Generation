#!/bin/bash
# bash script to start model training

# TRAIN or INFERENCE
MODE="TRAIN"
OUT_PATH="outputs/"
MAX_LR=1.0e-4
MIN_LR=4.0e-6
BATCH_SIZE=4
EPOCH=100

python model_train.py \
    --output_path $OUT_PATH \
    --mode $MODE \
    --max_lr $MAX_LR \
    --min_lr $MIN_LR \
    --batch_size $BATCH_SIZE \
    --max_epochs $EPOCH
    