#!/bin/sh

##BASE_DIR="/h/haoran/projects/HurtfulWords"
##OUTPUT_DIR="/h/haoran/projects/HurtfulWords/data/"
BASE_DIR="/gscratch/ark/limill01/clin-bias-summarization/"
OUTPUT_DIR="/gscratch/ark/limill01/clin-bias-summarization/data/"
cd "$BASE_DIR/scripts"
mkdir -p "$OUTPUT_DIR/pregen_embs/"
emb_method='cat4'

for target in inhosp_mort phenotype_first phenotype_all; do
	for model in baseline_clinical_BART finetuned_clinical_BART; do
		python pregen_embeddings.py \
		    --df_path "$OUTPUT_DIR/finetuning/$target"\
		    --model "$OUTPUT_DIR/models/$model" \
		    --output_path "${OUTPUT_DIR}/pregen_embs/pregen_${model}_${emb_method}_${target}" \
		    --emb_method $emb_method
	done
done

