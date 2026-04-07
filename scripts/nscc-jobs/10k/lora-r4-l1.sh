#!/bin/bash

# -------------------------------------------------------------------
# LoRA
# r=4, alpha=8, learning_rate=1e-4
# -------------------------------------------------------------------

#PBS -P personal
#PBS -N LoRA_LLaMA_3.2_3B_r4-l1
#PBS -q normal
#PBS -l select=1:ngpus=1
#PBS -l walltime=3:00:00
#PBS -j oe
#PBS -o lora-r4-l1.txt

# Set caches to scratch dir
export HF_HOME=~/scratch/.cache/huggingface
export UV_CACHE_DIR=~/scratch/.cache/uv

module purge
module load PrgEnv-gnu/8.3.3

cd ~/scratch/dsa5106-project

# Params
LORA_R=4
LORA_ALPHA=8
LEARNING_RATE="1e-4"
NAME="LoRA-r4-l1"
OUT_DIR="$HOME/scratch/lora-r4-l1"

MODEL_PATH="$OUT_DIR/model-finetuned"
EVAL_DIR="$OUT_DIR/eval_results"

mkdir -p "$OUT_DIR"

uv run -m reproduction.finetune \
  --adapter "lora" \
  --output_dir "$MODEL_PATH" \
  --lora_r $LORA_R --lora_alpha $LORA_ALPHA \
  --learning_rate $LEARNING_RATE

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