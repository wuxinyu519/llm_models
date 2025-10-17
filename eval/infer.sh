#!/bin/bash
#PBS -N final_eval
#PBS -o final_eval.out
#PBS -e final_eval.err
#PBS -l walltime=24:00:00
#PBS -q instructional
#PBS -l select=1:ncpus=8:ngpus=1:host=gpu011
cd $PBS_O_WORKDIR

module load cuda12.6/toolkit
eval "$(conda shell.bash hook)"
conda activate gemma

# GPU
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
echo "Starting unified evaluation at $(date)"
nvidia-smi

#dataset:
# Routerbench dataset: ./routerbench_gpt_gt/cleaned_tags_jsonl/individual_files
# Pii dataset: ../preprocess_data/data/privacy/pii_prompts.pkl
# toxic dataset: ../rhlf/data/toxic/toxic_prompts.pkl
# toxigen dataset: ./toxigen_test/data/toxigen_prompts.pkl

# #itpossible/TagGenerator(please use final_infer_tagrouter, others use final_infer.py) #../sft_with_miniplm/experiments/google_gemma-3-1b-it_20251013_012058/checkpoints/checkpoint-536/ ../rhlf/experiment/full_wdpo/20251013_113440/final
python final_infer.py \
    --checkpoint_path ../rhlf/experiment/full_wdpo/20251013_113440/final \
    --data_dir ./infinite_bench/routerbench_gpt_gt/cleaned_tags_jsonl/individual_files \
    --output_prefix ./infinite_bench/results \
    --batch_size 10 \
    --save_interval 50 \
    --device auto \
    # --num_samples 100  # 

echo "Unified evaluation completed at $(date)"
