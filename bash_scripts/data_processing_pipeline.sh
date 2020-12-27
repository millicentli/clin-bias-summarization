#!/bin/bash

BASE_DIR="/gscratch/ark/limill01/clin-bias-summarization/"
OUTPUT_DIR="/gscratch/ark/limill01/clin-bias-summarization/data/"
mkdir -p "$OUTPUT_DIR/finetuning/"
BART_DIR="facebook/bart-base"
MIMIC_BENCHMARK_DIR="/gscratch/stf/limill01/mimic3-benchmarks/data/"

cd "$BASE_DIR/scripts/"

echo "Processing MIMIC data..."
python get_data.py $OUTPUT_DIR

echo "Tokenizing sentences..."
python sentence_tokenization.py "$OUTPUT_DIR/df_raw.pkl" "$OUTPUT_DIR/df_extract.pkl" "$BART_DIR"
rm "$OUTPUT_DIR/df_raw.pkl" 

echo "Grouping short sentences..."
python group_sents.py "$OUTPUT_DIR/test/df_extract.pkl" "$OUTPUT_DIR/df_grouped.pkl" "$BART_DIR"

echo "Pregenerating training data..."
python pregenerate_train_sums.py \
    --train_df "$BASE_DIR/mimic-cxr/files/data.txt" \
    --output_dir "$OUTPUT_DIR/" \
    --bart_model "$BART_DIR"

echo "Generating finetuning targets..."
python make_targets.py \
 	--processed_df "$OUTPUT_DIR/df_extract.pkl" \
 	--mimic_benchmark_dir "$MIMIC_BENCHMARK_DIR" \
	--output_dir "$OUTPUT_DIR/finetuning/"

rm "$OUTPUT_DIR/df_extract.pkl" 
rm "$OUTPUT_DIR/df_grouped.pkl" 
