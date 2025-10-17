#!/bin/bash
#PBS -N sft_gemma1b
#PBS -o sft_gemma1b.out
#PBS -e sft_gemma1b.err
#PBS -l walltime=24:00:00
#PBS -q instructional
#PBS -l select=1:ncpus=8:ngpus=1:host=gpu011
#poderoso
cd $PBS_O_WORKDIR

# PKL_PATH="../gemma3/gemma_infered_result/cleaned_tags_batch/individual_files/tagged_pii_gemma_results_full_cleaned.pkl"
PKL_PATH="./after_miniplm/filtered.jsonl"
module load cuda12.6/toolkit
eval "$(conda shell.bash hook)"
conda activate gemma

export CUDA_VISIBLE_DEVICES=1  #1,2 or 3,4,5,6
export TOKENIZERS_PARALLELISM=false
# export HF_HOME="/tmp/hf_cache"

echo "Testing started at $(date)"
echo "Available GPUs:"
nvidia-smi

echo "Data path: $PKL_PATH"

# google/gemma-3-4b-it,OFA-Sys/InsTagger
python tune_with_data.py \
    --pkl_path $PKL_PATH \
    --models "Qwen/Qwen3-0.6B" \
    --batch_size 4 \
    --use_all_data \
    # --use_quantization \
    # --use_all_data \
    

echo "Testing completed at $(date)"