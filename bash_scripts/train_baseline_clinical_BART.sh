#!/bin/bash

BASE_DIR="/gscratch/ark/limill01/clin-bias-summarization/"
OUTPUT_DIR="/gscratch/ark/limill01/clin-bias-summarization/data/"
mkdir -p "$OUTPUT_DIR/models/"

cd "$BASE_DIR/scripts"


## Base model
python pretrain_on_sums.py \
    --notrain \
    --pregenerated_data "OUTPUT_DIR/docs.pkl" \
    --output_dir "$OUTPUT_DIR/models/baseline_clinical_BART/" \
    --bart_model "facebook/bart-base" \
    --do_lower_case \
    --epochs 20 \
    --train_batch_size 4 \
    --seed 123

## Fine-tuned model
python pretrain_on_sums.py \
    --train \
    --pregenerated_data "$OUTPUT_DIR/docs.pkl" \
    --output_dir "$OUTPUT_DIR/models/finetuned_clinical_BART/" \
    --bart_model "facebook/bart-base" \
    --do_lower_case \
    --epochs 20 \
    --train_batch_size 4 \
    --seed 123

