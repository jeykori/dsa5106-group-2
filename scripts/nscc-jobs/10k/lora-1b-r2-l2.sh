#!/bin/bash

# -------------------------------------------------------------------
# LoRA
# r=2, alpha=4, learning_rate=2e-4
# -------------------------------------------------------------------

#PBS -P personal
#PBS -N LoRA_LLaMA_3.2_1B_r2-l2
#PBS -q normal
#PBS -l select=1:ngpus=1
#PBS -l walltime=3:00:00
#PBS -j oe
#PBS -o lora-1b-r2-l2.txt

# Set caches to scratch dir
export HF_HOME=~/scratch/.cache/huggingface
export UV_CACHE_DIR=~/scratch/.cache/uv

module purge
module load PrgEnv-gnu/8.3.3

cd ~/scratch/dsa5106-project

# Params
MODEL_NAME="unsloth/llama-3.2-1b"
LORA_R=2
LORA_ALPHA=4
LEARNING_RATE="2e-4"
NAME="LoRA-1B-r2-l2"
OUT_DIR="$HOME/scratch/commonsense_reasoning/10k/lora-1b-r2-l2"

MODEL_PATH="$OUT_DIR/model-finetuned"
EVAL_DIR="$OUT_DIR/eval_results"

mkdir -p "$OUT_DIR"

uv run -m reproduction.finetune \
  --adapter "lora" \
  --output_dir "$MODEL_PATH" \
  --lora_r $LORA_R --lora_alpha $LORA_ALPHA \
  --learning_rate $LEARNING_RATE \
  --model_name $MODEL_NAME

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
    uv run -m reproduction.evaluate \
      --model_path "$MODEL_PATH" \
      --dataset $ds \
      --outfile "$EVAL_DIR/${ds}_results.json" \
      --batch_size 16
done

uv run reproduction/eval_summary.py --name "$NAME" \
  --results_dir "$EVAL_DIR" \
  --outfile "$OUT_DIR/summary.md"