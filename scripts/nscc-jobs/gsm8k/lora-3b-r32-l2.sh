#!/bin/bash

# -------------------------------------------------------------------
# LoRA, gsm8k
# r=32, alpha=64, learning_rate=2e-4
# -------------------------------------------------------------------

#PBS -P personal
#PBS -N LoRA_LLaMA_3.2_3B_r32-l2-gsm8k
#PBS -q normal
#PBS -l select=1:ngpus=1
#PBS -l walltime=4:00:00
#PBS -j oe
#PBS -o lora-3b-r32-l2-gsm8k.txt

# Set caches to scratch dir
export HF_HOME=~/scratch/.cache/huggingface
export UV_CACHE_DIR=~/scratch/.cache/uv

module purge
module load PrgEnv-gnu/8.3.3

cd ~/scratch/dsa5106-project

# Params
MODEL_NAME="unsloth/llama-3.2-3b"
LORA_R=32
LORA_ALPHA=64
LEARNING_RATE="2e-4"
NAME="LoRA-3b-r32-l2-gsm8k"
OUT_DIR="$HOME/scratch/gsm8k/lora-3b-r32-l2-gsm8k"

MODEL_PATH="$OUT_DIR/model-finetuned"
EVAL_DIR="$OUT_DIR/eval_results"

mkdir -p "$OUT_DIR"

uv run -m extension.gsm8k.finetune \
  --adapter "lora" \
  --output_dir "$MODEL_PATH" \
  --lora_r $LORA_R --lora_alpha $LORA_ALPHA \
  --learning_rate $LEARNING_RATE \
  --model_name $MODEL_NAME

uv run -m extension.gsm8k.evaluate \
  --model_path "$MODEL_PATH" \
  --outfile "$EVAL_DIR/gsm8k_results.json" \
  --batch_size 16

uv run -m reproduction.eval_summary --name "$NAME" \
  --results_dir "$EVAL_DIR" \
  --outfile "$OUT_DIR/summary.md" \
  --datasets "gsm8k,"