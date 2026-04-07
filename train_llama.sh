#!/bin/bash
#PBS -P personal-e0538411
#PBS -q normal
#PBS -l select=1:ngpus=1:mem=110gb:ncpus=16
#PBS -l walltime=02:00:00
#PBS -j oe

cd $PBS_O_WORKDIR

# Activate your environment
source .venv/bin/activate

# Run the training pointing to your LOCAL model folder
python -m reproduction.finetune \
  --model_name_or_path /home/users/nus/e0538411/scratch/llama-3.2-3b \
  --dataset_path /home/users/nus/e0538411/dsa5106-group-2/reference-code/commonsense_reasoning/commonsense_170k.json \
  --adapter "dora" \
  --output_dir "./output/dora_results"
