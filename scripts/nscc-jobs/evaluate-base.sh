#!/bin/bash

#PBS -P personal
#PBS -N eval_LLaMA_3.2_3B
#PBS -q normal
#PBS -l select=1:ngpus=1
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -o eval-base.txt

# Set caches to scratch dir
export HF_HOME=~/scratch/.cache/huggingface
export UV_CACHE_DIR=~/scratch/.cache/uv

module purge
module load PrgEnv-gnu/8.3.3

cd ~/scratch/dsa5106-project

OUT_DIR="$HOME/scratch/eval_results/base"

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
      --dataset $ds \
      --outfile "$OUT_DIR/${ds}_results.json" \
      --batch_size 16
done
