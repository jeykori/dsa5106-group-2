#!/bin/bash

# -------------------------------------------------------------------
# LoRA, gsm8k
# r=16, alpha=32, learning_rate=3e-4
# -------------------------------------------------------------------

#PBS -P personal
#PBS -N LoRA_LLaMA_3.2_1B_r16-l3-gsm8k
#PBS -q normal
#PBS -l select=1:ngpus=1
#PBS -l walltime=4:00:00
#PBS -j oe
#PBS -o lora-1b-r16-l3-gsm8k.txt

# Set caches to scratch dir
export HF_HOME=~/scratch/.cache/huggingface
export UV_CACHE_DIR=~/scratch/.cache/uv

module purge
module load PrgEnv-gnu/8.3.3

cd ~/scratch/dsa5106-project

# Params
MODEL_NAME="unsloth/llama-3.2-1b"
LORA_R=16
LORA_ALPHA=32
LEARNING_RATE="3e-4"
NAME="LoRA-1b-r16-l3-gsm8k"
OUT_DIR="$HOME/scratch/lora-1b-r16-l3-gsm8k"

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