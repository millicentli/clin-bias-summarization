#!/bin/bash

# $1 - target type {inhosp_mort, phenotype_first, phenotype_all}
# $2 - BERT model name {baseline_clinical_BART_base, baseline_clinical_BART_FT}
# $3 - target column name within the dataframe, ex: "Shock", "any_acute"

## BASE_DIR="/h/haoran/projects/HurtfulWords"
BASE_DIR="/gscratch/ark/limill01/clin-bias-summarization"
## OUTPUT_DIR="/scratch/hdd001/home/haoran/shared_data/HurtfulWords/data"
OUTPUT_DIR="/gscratch/ark/limill01/clin-bias-summarization/data"

cd "$BASE_DIR/scripts"

python finetune_on_target.py \
	--df_path "${OUTPUT_DIR}/finetuning/$1" \
	--model_path "${OUTPUT_DIR}/models/$2" \
	--fold_id 9 10\
	--target_col_name "$3" \
	--output_dir "${OUTPUT_DIR}/models/finetuned/${1}_${2}_${3}/" \
	--freeze_bert \
	--train_batch_size 32 \
	--pregen_emb_path "${OUTPUT_DIR}/pregen_embs/pregen_${2}_cat4_${1}" \
	--task_type binary \
	--other_fields age sofa sapsii_prob sapsii_prob oasis oasis_prob \
        --gridsearch_classifier \
        --gridsearch_c \
        --emb_method cat4
