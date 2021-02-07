#!/bin/bash

## Job Name
#SBATCH --job-name=clin_get_data

## Allocation Definition
#SBATCH --account=cse
#SBATCH --partition=cse-gpu

## Resources
## Nodes
#SBATCH --nodes=1

## Cores
#SBATCH --ntasks=1

## GPUs
#SBATCH --gres=gpu

## Walltime
#SBATCH --time=120:00:00

## Memory per node
##SBATCH --mem=256G
#SBATCH --mem=32G

## Specify the working directory for this job
#SBATCH --chdir=/gscratch/ark/limill01/slurm_finetune

## Program instructions below

# $1 - target type {inhosp_mort, phenotype_first, phenotype_all}
# $2 - BART model name {baseline_clinical_BART, finetuned_clinical_BART}
# $3 - target column name within the dataframe, ex: "Shock", "any_acute"

BASE_DIR="/gscratch/ark/limill01/clin-bias-summarization"
OUTPUT_DIR="/gscratch/ark/limill01/clin-bias-summarization/data"

export TORCH_HOME=/gscratch/ark/limilli01/.cache/torch/
source /gscratch/ark/limill01/miniconda3/etc/profile.d/conda.sh
conda activate nlp

cd "$BASE_DIR/scripts"
##cd /gscratch/ark/limill01/clin-bias-summarization/scripts

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
