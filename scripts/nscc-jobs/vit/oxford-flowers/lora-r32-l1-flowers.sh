#!/bin/bash

# -------------------------------------------------------------------
# LoRA, ViT-p16-224, oxford_flowers
# r=32, alpha=64, learning_rate=1e-4
# -------------------------------------------------------------------

#PBS -P personal
#PBS -N LoRA_ViT-p16-224_r32-l1-oxford_flowers
#PBS -q normal
#PBS -l select=1:ngpus=1
#PBS -l walltime=4:00:00
#PBS -j oe
#PBS -o lora-vit-r32-l1-oxford_flowers.txt

# Set caches to scratch dir
export HF_HOME=~/scratch/.cache/huggingface
export UV_CACHE_DIR=~/scratch/.cache/uv

module purge
module load PrgEnv-gnu/8.3.3

cd ~/scratch/dsa5106-project

# Params
MODEL_NAME="google/vit-base-patch16-224-in21k"
DATASET="oxford_flowers"
LORA_R=32
LORA_ALPHA=64
LEARNING_RATE="1e-4"
NAME="LoRA-ViT-p16-224-r32-l1-$DATASET"
OUT_DIR="$HOME/scratch/vit/lora-vit-r32-l1-$DATASET"

MODEL_PATH="$OUT_DIR/model-finetuned"
EVAL_DIR="$OUT_DIR/eval_results"

mkdir -p "$OUT_DIR"

uv run -m extension.vit.finetune \
  --adapter "lora" \
  --output_dir "$MODEL_PATH" \
  --lora_r $LORA_R --lora_alpha $LORA_ALPHA \
  --learning_rate $LEARNING_RATE \
  --model_name $MODEL_NAME \
  --dataset $DATASET

uv run -m extension.vit.evaluate \
  --model_path "$MODEL_PATH" \
  --outfile "$EVAL_DIR/${DATASET}_results.json" \
  --batch_size 16 \
  --dataset $DATASET

uv run -m reproduction.eval_summary --name "$NAME" \
  --results_dir "$EVAL_DIR" \
  --outfile "$OUT_DIR/summary.md" \
  --datasets "$DATASET,"