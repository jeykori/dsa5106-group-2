#!/bin/bash

# ProjectID should be `personal-<NUS ID (starting with e)>`
#PBS -P personal-e1543077
#PBS -N DoRA_LLaMA_3.2_3B
#PBS -q normal
#PBS -l select=1:ngpus=1
#PBS -l walltime=2:00:00
#PBS -j oe
#PBS -o out-run.txt

# Downloads the model to scratch rather than home
export HF_HOME=~/scratch/.cache/huggingface

module purge
module load PrgEnv-gnu/8.3.3 cuda/12.2.2 miniforge3

cd ~/scratch/dsa5106-project/ref/commonsense_reasoning

eval "$(conda shell.bash hook)"
conda activate dsa5106-project_ref_commonsense_reasoning

# Modal from huggingface
MODEL="unsloth/llama-3.2-3B"
OUT_DIR="./finetuned_result/dora_r16"
LORA_R=16
LORA_ALPHA=32

CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --base_model "$MODEL" \
    --data_path 'commonsense_170k.json' \
    --batch_size 16  --micro_batch_size 12 --num_epochs 3 \
    --learning_rate 2e-4 --cutoff_len 256 --val_set_size 120 \
    --eval_step 80 --save_step 80  --adapter_name dora \
    --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
    --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" \
    --use_gradient_checkpointing \
    --output_dir "$OUT_DIR"