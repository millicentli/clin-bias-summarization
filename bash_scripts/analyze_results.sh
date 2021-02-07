#!/bin/sh

BASE_DIR="/gscratch/ark/limill01/clin-bias-summarization"
OUTPUT_DIR="/gscratch/ark/limill01/clin-bias-summarization/data"
cd "$BASE_DIR/scripts"

python analyze_results.py \
	--models_path "${OUTPUT_DIR}/models/finetuned/" \
	--set_to_use "test" \
	--bootstrap \
