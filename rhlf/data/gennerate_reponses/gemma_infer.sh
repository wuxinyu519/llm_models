#!/bin/bash
#PBS -N gemma_generate
#PBS -l select=1:ncpus=8:ngpus=1:host=gpu011
#PBS -l walltime=24:00:00
#PBS -q instructional
#PBS -o gemma_generate_general.log 
#PBS -e gemma_generate_general.err  
#poderoso

module load cuda12.6/toolkit
eval "$(conda shell.bash hook)"
conda activate gemma

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_LEVEL=NVL

export VLLM_ATTENTION_BACKEND=FLASH_ATTN  # 使用 Flash Attention 加速
export NCCL_DEBUG=WARN  
export TOKENIZERS_PARALLELISM=false 



# 创建必要的目录
mkdir -p logs
mkdir -p generated
mkdir -p hf_cache

# 进入工作目录
cd $PBS_O_WORKDIR

echo "=========================================="
echo "Job ID: $PBS_JOBID"
echo "Job Name: $PBS_JOBNAME"
echo "Node: $(hostname)"
echo "Start Time: $(date)"
echo "Working Directory: $PWD"
echo "Python版本: $(python --version)"
echo "Conda环境: $CONDA_DEFAULT_ENV"
echo "=========================================="
echo "GPU Info:"
nvidia-smi
echo "=========================================="
echo "环境变量检查:"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "VLLM_ATTENTION_BACKEND: $VLLM_ATTENTION_BACKEND"
echo "=========================================="



python gemma_infer.py \
    --input ../general/general_data_test_cleaned.pkl \
    --output generated/responses_${PBS_JOBID}.pkl \
    --model google/gemma-3-12b-it \
    --n_responses 2 \
    --batch_size 8 \
    --start_idx 3000 \
    --limit 5000

# python gemma_infer.py \
#     --input ../toxic/toxic_prompts.pkl \
#     --output generated/responses_${PBS_JOBID}.pkl \
#     --model google/gemma-3-27b-it \
#     --n_responses 2 \
#     --batch_size 10 \
#     --start_idx 50 \
#     --limit 1000



echo "=========================================="
echo "End Time: $(date)"
echo "Job Completed!"
echo "=========================================="