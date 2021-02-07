#!/bin/bash

BASE_DIR="/gscratch/ark/limill01/clin-bias-summarization/"
OUTPUT_DIR="/gscratch/ark/limill01/clin-bias-summarization/data/"
mkdir -p "$OUTPUT_DIR/models/"

cd "$BASE_DIR/scripts"

## Model trained on train/test/val split
python pretrain_on_sums_eval.py \
    --pregenerated_data "$OUTPUT_DIR/docs.pkl" \
    --output_dir "$OUTPUT_DIR/models/baseline_clinical_BART_scored/" \
    --bart_model "facebook/bart-base" \
    --do_lower_case \
    --epochs 5 \
    --train_batch_size 4 \
    --seed 123

## Evaluate on the same on to get the statistics

