#!/bin/bash

# -------------------------------------------------------------------
# DoRA
# r=4, alpha=8, learning_rate=2e-4, sample_size=80k
# -------------------------------------------------------------------

#PBS -P personal
#PBS -N DoRA_LLaMA_3.2_1B_r4-l2-80k
#PBS -q normal
#PBS -l select=1:ngpus=1
#PBS -l walltime=4:00:00
#PBS -j oe
#PBS -o dora-1b-r4-l2-80k.txt

# Set caches to scratch dir
export HF_HOME=~/scratch/.cache/huggingface
export UV_CACHE_DIR=~/scratch/.cache/uv

module purge
module load PrgEnv-gnu/8.3.3

cd ~/scratch/dsa5106-project

# Params
MODEL_NAME="unsloth/llama-3.2-1b"
LORA_R=4
LORA_ALPHA=8
LEARNING_RATE="2e-4"
NAME="DoRA-1b-r4-l2-80k"
OUT_DIR="$HOME/scratch/dora-1b-r4-l2-80k"

MODEL_PATH="$OUT_DIR/model-finetuned"
EVAL_DIR="$OUT_DIR/eval_results"

mkdir -p "$OUT_DIR"

uv run reproduction/finetune.py \
  --adapter "dora" \
  --output_dir "$MODEL_PATH" \
  --lora_r $LORA_R --lora_alpha $LORA_ALPHA \
  --learning_rate $LEARNING_RATE \
  --model_name $MODEL_NAME \
  --sample_size 80000

datasets=(
  "boolq"
  "piqa"
  "social_i_qa"
  "hellaswag"
  "winogrande"
  "ARC-Challenge"
  "ARC-Easy"
  "openbookqa"
)

# Loop through each dataset
for ds in "${datasets[@]}"; do
    echo "Evaluating dataset: $ds"
    uv run reproduction/evaluate.py \
      --model_path "$MODEL_PATH" \
      --dataset $ds \
      --outfile "$EVAL_DIR/${ds}_results.json" \
      --batch_size 16
done

uv run reproduction/eval_summary.py --name "$NAME" \
  --results_dir "$EVAL_DIR" \
  --outfile "$OUT_DIR/summary.md"