#!/bin/bash

#PBS -N rlhf_annotation
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=24:00:00
#PBS -o anotation.log 
#PBS -e anotation.err  
#PBS -q gpu


echo "======================================"
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $PBS_JOBID"
echo "======================================"

# Load required modules 
module load cuda12.6/toolkit
eval "$(conda shell.bash hook)"
conda activate gemma


# Navigate to your working directory
cd $PBS_O_WORKDIR
PWD

# Set up Python virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install openai tqdm

# Set your OpenAI API key
export OPENAI_API_KEY="your_openai_api_key_here"

#!/bin/bash

# Configuration
INPUT_FILE="../gennerate_reponses/generated/toxic_out1_cleaned.json"
OUTPUT_FILE="toxic_out1_cleaned_annotated.jsonl"
TRAINING_FILE="toxic_out1_cleaned_dpo.jsonl"
MODEL="gpt-4o-mini" 
# Filtering parameters
MIN_DIVERSITY_GAP=2.0   # Keep samples with diversity difference >= 2.0
MIN_PRIVACY_SCORE=0.0   # Keep samples with privacy score >= 7.0 (set to 0 to disable)

TRAINING_FORMAT="dpo_weighted"   # Options: dpo, dpo_weighted, ppo

echo ""
echo "======================================"
echo "Starting annotation with filtering..."
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_FILE"
echo "Model: $MODEL"
echo "Filters:"
echo "  - Min diversity gap: $MIN_DIVERSITY_GAP"
echo "  - Min privacy score: $MIN_PRIVACY_SCORE"
echo "  - Training format: $TRAINING_FORMAT"
echo "======================================"

cd /data/wux/gemma3_1b/rhlf/data/anotator

python gpt4o_evl.py \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    --training-output "$TRAINING_FILE" \
    --model "$MODEL" \
    --format "$TRAINING_FORMAT" \
    --batch-size 100 \
    --min-diversity-gap "$MIN_DIVERSITY_GAP" \
    --min-privacy-score "$MIN_PRIVACY_SCORE"
    