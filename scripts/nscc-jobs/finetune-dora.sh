#!/bin/bash

#PBS -P personal
#PBS -N finetune_DoRA_LLaMA_3.2_3B
#PBS -q normal
#PBS -l select=1:ngpus=1
#PBS -l walltime=2:00:00
#PBS -j oe
#PBS -o finetune-dora.txt

# Set caches to scratch dir
export HF_HOME=~/scratch/.cache/huggingface
export UV_CACHE_DIR=~/scratch/.cache/uv

module purge
module load PrgEnv-gnu/8.3.3

cd ~/scratch/dsa5106-project

uv run reproduction/finetune.py \
  --adapter "dora" \
  --output_dir "./dora-finetuned"