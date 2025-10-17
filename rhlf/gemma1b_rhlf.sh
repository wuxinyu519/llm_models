#!/bin/bash
#PBS -N wdpo_full_gemma1b
#PBS -o wdpo_full_gemma1b.out
#PBS -e wdpo_full_gemma1b.err
#PBS -l walltime=24:00:00
#PBS -q instructional
#PBS -l select=1:ncpus=8:ngpus=1:host=gpu011

cd $PBS_O_WORKDIR
cd /data/wux/gemma3_1b/rhlf

CKPT_DIR="../sft_with_miniplm/experiments/google_gemma-3-1b-it_20251012_025901/final_model"
DATA_DIR="./data/anotator/dpo"

module load cuda12.6/toolkit
eval "$(conda shell.bash hook)"
conda activate gemma

export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

echo "Weighted-DPO (Accelerate) training started at $(date)"
echo "GPUs available:"
nvidia-smi

echo "CKPT: $CKPT_DIR"
echo "DATA: $DATA_DIR"
echo "使用 SFT Template:" 

python tune_gemma1b_rhlf.py \
  --checkpoint_dir $CKPT_DIR \
  --data_dir ./data/anotator/dpo \
  --learning_rate 5e-6 \
  --beta 0.1 \
  --num_epochs 2 \
  --per_device_batch 4 \
  --gradient_accumulation 8

echo "Weighted-DPO training completed at $(date)"