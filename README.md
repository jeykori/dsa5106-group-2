# dsa5106-group-2

## Latest Results

### Original datasets (sample_size: 10k)

| Name             | boolq  | piqa   | social_i_qa | hellaswag | winogrande | ARC-Challenge | ARC-Easy | openbookqa | Average |
| ---------------- | ------ | ------ | ----------- | --------- | ---------- | ------------- | -------- | ---------- | ------- |
| Base             | 5.14%  | 47.39% | 21.44%      | 18.65%    | 4.18%      | 14.42%        | 14.18%   | 16.80%     | 17.73%  |
| LoRA-1B-r2-l2-e6 | 56.42% | 69.97% | 60.29%      | 70.77%    | 63.38%     | 48.21%        | 67.42%   | 54.20%     | 65.38%  |
| DoRA-1B-r2-l2-e6 | 58.69% | 71.11% | 62.18%      | 70.03%    | 63.54%     | 47.95%        | 68.10%   | 52.60%     | 65.66%  |
| LoRA-1B-r2-l2    | 61.83% | 71.93% | 64.33%      | 72.54%    | 60.93%     | 49.83%        | 71.30%   | 53.40%     | 67.81%  |
| DoRA-1B-r2-l2    | 61.22% | 73.56% | 63.41%      | 71.35%    | 61.09%     | 47.95%        | 69.70%   | 52.40%     | 66.96%  |
| LoRA-1B-r4-l2    | 60.34% | 73.12% | 65.66%      | 73.06%    | 63.06%     | 51.88%        | 69.49%   | 53.00%     | 68.07%  |
| DoRA-1B-r4-l2    | 60.58% | 72.42% | 64.79%      | 73.61%    | 64.01%     | 52.65%        | 70.92%   | 54.80%     | 68.50%  |
| LoRA-1B-r8-l2    | 59.63% | 72.42% | 64.64%      | 74.94%    | 63.22%     | 51.88%        | 71.21%   | 56.60%     | 68.93%  |
| DoRA-1B-r8-l2    | 58.17% | 72.52% | 65.40%      | 74.03%    | 63.61%     | 50.34%        | 70.75%   | 57.40%     | 68.29%  |
| LoRA-r2-l2       | 65.08% | 82.54% | 72.62%      | 87.75%    | 76.64%     | 69.20%        | 84.93%   | 73.40%     | 80.48%  |
| DoRA-r2-l2       | 63.73% | 82.48% | 72.93%      | 87.87%    | 77.27%     | 67.15%        | 84.30%   | 71.00%     | 80.17%  |
| LoRA-r4-l1       | 64.19% | 82.15% | 71.90%      | 87.31%    | 75.69%     | 69.20%        | 84.89%   | 73.00%     | 79.99%  |
| LoRA-r4-l2       | 62.97% | 82.05% | 73.29%      | 88.03%    | 78.61%     | 70.56%        | 86.15%   | 70.80%     | 80.57%  |
| LoRA-r4-l3       | 65.14% | 82.26% | 73.23%      | 88.70%    | 78.45%     | 70.31%        | 85.48%   | 73.60%     | 81.17%  |
| DoRA-r4-l1       | 62.63% | 81.34% | 72.77%      | 87.39%    | 76.24%     | 67.41%        | 84.39%   | 71.20%     | 79.66%  |
| DoRA-r4-l2       | 64.53% | 82.43% | 71.85%      | 88.78%    | 78.37%     | 69.88%        | 85.65%   | 71.80%     | 80.96%  |
| DoRA-r4-l3       | 63.85% | 82.26% | 72.77%      | 87.89%    | 77.66%     | 70.39%        | 85.56%   | 75.20%     | 80.58%  |
| LoRA-r16-l1      | 65.17% | 81.66% | 72.98%      | 88.85%    | 78.30%     | 69.71%        | 86.11%   | 74.40%     | 81.21%  |
| LoRA-r16-l2      | 64.28% | 81.66% | 74.77%      | 88.84%    | 79.64%     | 69.45%        | 84.81%   | 73.20%     | 81.13%  |
| LoRA-r16-l3      | 63.00% | 81.18% | 74.00%      | 89.18%    | 78.53%     | 70.05%        | 84.43%   | 73.00%     | 80.91%  |
| DoRA-r16-l1      | 65.08% | 81.77% | 73.44%      | 88.21%    | 78.93%     | 69.80%        | 84.55%   | 74.40%     | 80.84%  |
| DoRA-r16-l2      | 64.25% | 82.26% | 73.44%      | 88.18%    | 79.64%     | 69.20%        | 84.47%   | 73.60%     | 80.73%  |
| DoRA-r16-l3      | 64.10% | 81.88% | 72.82%      | 88.30%    | 79.08%     | 69.54%        | 85.61%   | 72.80%     | 80.76%  |

Winners:

- LLaMA-3.2-1B - 2/4
  - DoRA: r2 e6, r4
  - LoRA: r2, r8
- LLaMA-3.2-3B - 0/3
  - DoRA: -
  - LoRA: r2, r4, r16

### Original datasets (sample_size: 40k)

| Name              | boolq  | piqa   | social_i_qa | hellaswag | winogrande | ARC-Challenge | ARC-Easy | openbookqa | Average    |
| ----------------- | ------ | ------ | ----------- | --------- | ---------- | ------------- | -------- | ---------- | ---------- |
| LoRA-1b-r4-l2-40k | 62.75% | 75.63% | 70.16%      | 82.01%    | 69.22%     | 57.34%        | 73.40%   | 62.20%     | **74.28%** |
| DoRA-1b-r4-l2-40k | 63.46% | 75.90% | 70.52%      | 81.33%    | 67.32%     | 56.66%        | 73.91%   | 64.80%     | 74.10%     |
| LoRA-r4-l2-40k    | 69.54% | 83.30% | 77.99%      | 92.34%    | 83.27%     | 74.15%        | 86.95%   | 77.80%     | **84.66%** |
| DoRA-r4-l2-40k    | 68.01% | 83.90% | 77.58%      | 92.39%    | 83.82%     | 73.98%        | 87.04%   | 79.40%     | 84.54%     |

Winners:

- LLaMA-3.2-1B - 0/1
- LLaMA-3.2-3B - 0/1

### Original datasets (sample_size: 80k)

| Name               | r   | lr   | boolq  | piqa   | social_i_qa | hellaswag | winogrande | ARC-Challenge | ARC-Easy | openbookqa | Average    |
| ------------------ | --- | ---- | ------ | ------ | ----------- | --------- | ---------- | ------------- | -------- | ---------- | ---------- |
| LoRA-1b-r2-l1-80k  | 2   | 1e-4 | 63.70% | 78.40% | 73.39%      | 84.65%    | 69.46%     | 59.22%        | 75.29%   | 68.40%     | **76.56%** |
| LoRA-1b-r2-l2-80k  | 2   | 2e-4 | 64.50% | 77.42% | 72.72%      | 83.83%    | 71.11%     | 58.28%        | 73.70%   | 68.80%     | 76.05%     |
| LoRA-1b-r2-l3-80k  | 2   | 3e-4 | 63.94% | 77.31% | 72.88%      | 83.97%    | 71.35%     | 57.76%        | 74.49%   | 66.80%     | 76.06%     |
| DoRA-1b-r2-l1-80k  | 2   | 1e-4 | 61.87% | 77.15% | 73.44%      | 84.81%    | 70.09%     | 58.96%        | 74.83%   | 63.80%     | **76.14%** |
| DoRA-1b-r2-l2-80k  | 2   | 2e-4 | 64.07% | 77.26% | 73.08%      | 83.59%    | 70.64%     | 59.56%        | 74.92%   | 68.60%     | 76.06%     |
| DoRA-1b-r2-l3-80k  | 2   | 3e-4 | 64.46% | 77.15% | 71.70%      | 83.40%    | 70.72%     | 57.94%        | 73.61%   | 69.40%     | 75.71%     |
| LoRA-1b-r4-l1-80k  | 4   | 1e-4 | 64.25% | 78.07% | 72.67%      | 84.42%    | 71.27%     | 61.01%        | 75.17%   | 68.60%     | **76.63%** |
| LoRA-1b-r4-l2-80k  | 4   | 2e-4 | 64.10% | 77.37% | 71.29%      | 84.54%    | 70.32%     | 57.51%        | 74.83%   | 69.60%     | 76.24%     |
| LoRA-1b-r4-l3-80k  | 4   | 3e-4 | 62.35% | 76.17% | 71.65%      | 83.20%    | 72.61%     | 57.34%        | 74.92%   | 69.60%     | 75.44%     |
| DoRA-1b-r4-l1-80k  | 4   | 1e-4 | 63.88% | 78.18% | 73.23%      | 84.87%    | 73.09%     | 59.13%        | 76.09%   | 68.60%     | 76.94%     |
| DoRA-1b-r4-l2-80k  | 4   | 2e-4 | 63.79% | 78.94% | 72.62%      | 85.29%    | 71.82%     | 60.24%        | 74.71%   | 67.80%     | **76.95%** |
| DoRA-1b-r4-l3-80k  | 4   | 3e-4 | 63.09% | 76.33% | 72.36%      | 83.86%    | 70.64%     | 58.62%        | 76.47%   | 67.20%     | 75.99%     |
| LoRA-1b-r8-l1-80k  | 8   | 1e-4 | 64.59% | 78.62% | 72.36%      | 85.13%    | 71.90%     | 59.39%        | 75.51%   | 70.60%     | **77.05%** |
| LoRA-1b-r8-l2-80k  | 8   | 2e-4 | 63.91% | 78.13% | 70.98%      | 84.42%    | 72.85%     | 59.13%        | 76.01%   | 67.60%     | 76.50%     |
| LoRA-1b-r8-l3-80k  | 8   | 3e-4 | 64.43% | 75.46% | 71.60%      | 80.52%    | 70.24%     | 57.51%        | 71.09%   | 64.40%     | 73.84%     |
| DoRA-1b-r8-l1-80k  | 8   | 1e-4 | 63.61% | 77.80% | 72.77%      | 84.94%    | 72.53%     | 59.13%        | 75.67%   | 70.80%     | **76.84%** |
| DoRA-1b-r8-l2-80k  | 8   | 2e-4 | 64.16% | 76.39% | 72.16%      | 84.08%    | 73.80%     | 59.22%        | 75.63%   | 68.60%     | 76.38%     |
| DoRA-1b-r8-l3-80k  | 8   | 3e-4 | 63.43% | 74.16% | 69.60%      | 80.86%    | 69.61%     | 55.38%        | 71.76%   | 67.40%     | 73.55%     |
| LoRA-1b-r16-l1-80k | 16  | 1e-4 | 65.35% | 78.18% | 72.93%      | 84.77%    | 74.51%     | 59.39%        | 76.35%   | 71.40%     | **77.27%** |
| LoRA-1b-r16-l2-80k | 16  | 2e-4 | 62.75% | 74.70% | 71.55%      | 81.81%    | 71.98%     | 55.46%        | 73.32%   | 65.00%     | 74.34%     |
| LoRA-1b-r16-l3-80k | 16  | 3e-4 | 62.17% | 48.53% | 44.27%      | 29.42%    | 52.25%     | 27.65%        | 30.18%   | 33.80%     | 38.43%     |
| DoRA-1b-r16-l1-80k | 16  | 1e-4 | 64.83% | 77.91% | 72.93%      | 85.32%    | 70.48%     | 60.41%        | 75.21%   | 68.40%     | **77.06%** |
| DoRA-1b-r16-l2-80k | 16  | 2e-4 | 63.98% | 76.06% | 70.78%      | 81.64%    | 71.27%     | 55.46%        | 73.32%   | 64.60%     | 74.44%     |
| DoRA-1b-r16-l3-80k | 16  | 3e-4 | 61.87% | 50.11% | 46.11%      | 26.81%    | 53.83%     | 25.94%        | 28.54%   | 29.60%     | 37.24%     |

Winners (all LLaMA-3.2-1B) - 1/4

- DoRA: r4
- LoRA: r2, r8, r16

### Hyperparameters

(constant for now) parameters:

- target_modules=QKVUD
- batch_size=16
- micro_batch_size=12
- num_epochs=3
- eval_steps=80
- save_steps=80
- val_set_size=120
- lora_dropout=0.05

## Extension: GSM8K

### LLaMA-3.2-1B

| Name                 | gsm8k  | Average    |
| -------------------- | ------ | ---------- |
| LoRA-1b-r8-l1-gsm8k  | 22.97% | 22.97%     |
| LoRA-1b-r8-l2-gsm8k  | 27.14% | 27.14%     |
| LoRA-1b-r8-l3-gsm8k  | 28.20% | 28.20%     |
| LoRA-1b-r8-l4-gsm8k  | 28.35% | 28.35%     |
| LoRA-1b-r8-l5-gsm8k  | 29.19% | 29.19%     |
| LoRA-1b-r8-l6-gsm8k  | 30.48% | **30.48%** |
| DoRA-1b-r8-l1-gsm8k  | 22.44% | 22.44%     |
| DoRA-1b-r8-l2-gsm8k  | 25.85% | 25.85%     |
| DoRA-1b-r8-l3-gsm8k  | 27.45% | 27.45%     |
| DoRA-1b-r8-l4-gsm8k  | 29.42% | 29.42%     |
| DoRA-1b-r8-l5-gsm8k  | 29.64% | **29.64%** |
| DoRA-1b-r8-l6-gsm8k  | 29.34% | 29.34%     |
| LoRA-1b-r16-l1-gsm8k | 26.08% | 26.08%     |
| LoRA-1b-r16-l2-gsm8k | 29.26% | **29.26%** |
| LoRA-1b-r16-l3-gsm8k | 29.04% | 29.04%     |
| DoRA-1b-r16-l1-gsm8k | 26.38% | 26.38%     |
| DoRA-1b-r16-l2-gsm8k | 28.51% | 28.51%     |
| DoRA-1b-r16-l3-gsm8k | 30.71% | **30.71%** |
| DoRA-1b-r16-l4-gsm8k | 30.55% | 30.55%     |
| LoRA-1b-r32-l1-gsm8k | 27.45% | 27.45%     |
| LoRA-1b-r32-l2-gsm8k | 30.40% | 30.40%     |
| LoRA-1b-r32-l3-gsm8k | 30.86% | 30.86%     |
| LoRA-1b-r32-l4-gsm8k | 31.08% | 31.08%     |
| LoRA-1b-r32-l5-gsm8k | 31.61% | **31.61%** |
| DoRA-1b-r32-l1-gsm8k | 27.22% | 27.22%     |
| DoRA-1b-r32-l2-gsm8k | 28.66% | 28.66%     |
| DoRA-1b-r32-l3-gsm8k | 31.08% | **31.08%** |
| DoRA-1b-r32-l4-gsm8k | 29.19% | 29.19%     |

### LLaMA-3.2-3B

| Name                | gsm8k  | Average    |
| ------------------- | ------ | ---------- |
| LoRA-3b-r8-l1-gsm8k | 49.28% | 49.28%     |
| LoRA-3b-r8-l2-gsm8k | 52.08% | 52.08%     |
| LoRA-3b-r8-l3-gsm8k | 52.54% | **52.54%** |
| LoRA-3b-r8-l4-gsm8k | 51.63% | 51.63%     |
| DoRA-3b-r8-l1-gsm8k | 48.67% | 48.67%     |
| DoRA-3b-r8-l2-gsm8k | 50.87% | 50.87%     |
| DoRA-3b-r8-l3-gsm8k | 51.63% | 51.63%     |
| DoRA-3b-r8-l4-gsm8k | 52.84% | 52.84%     |
| DoRA-3b-r8-l5-gsm8k | 54.06% | **54.06%** |
| DoRA-3b-r8-l6-gsm8k | 51.93% | 51.93%     |

## Extension: ViT

| Name                                   | test   | Average |
| -------------------------------------- | ------ | ------- |
| LoRA-ViT-p16-224-r16-l2-cifar10        | 98.70% | 98.70%  |
| DoRA-ViT-p16-224-r16-l2-cifar10        | 98.81% | 98.81%  |
| LoRA-ViT-p16-224-r16-l2-eurosat        | 98.93% | 98.93%  |
| DoRA-ViT-p16-224-r16-l2-eurosat        | 98.89% | 98.89%  |
| LoRA-ViT-p16-224-r16-l2-oxford_flowers | 47.76% | 47.76%  |
| DoRA-ViT-p16-224-r16-l2-oxford_flowers | 51.49% | 51.49%  |

## How to run reproduction code

1. Run init script

   ```bash
   ./scripts/init-datasets.sh
   ```

2. rsync into NSCC (assuming `nscc` is your ssh config)

   ```bash
   rsync -avzP \
      --exclude='.git' \
      --exclude='.venv' \
      --exclude='reference-code' \
      --exclude='reference-results' \
      . nscc:~/scratch/dsa5106-project
   ```

3. Set up env and install dependencies in nscc
   NOTE: SSH in first

   ```bash
   module purge
   module load PrgEnv-gnu/8.3.3

   cd ~/scratch/dsa5106-project

   uv sync
   ```

4. Submit job to queue
   ```bash
   qsub ~/scratch/dsa5106-project/scripts/nscc-jobs/<job_script>.sh
   ```

## How to run reference code

### Commonsense Reasoning

1. Run init script

   ```bash
   ./scripts/init-ref-commonsense_reasoning.sh
   ```

2. rsync into NSCC (assuming `nscc` is your ssh config)

   ```bash
   ssh nscc "mkdir -p ~/scratch/dsa5106-project/ref/commonsense_reasoning"
   rsync -avzP \
      --exclude='.git' \
      --exclude='.venv' \
      ./reference-code/commonsense_reasoning/ nscc:~/scratch/dsa5106-project/ref/commonsense_reasoning
   rsync -avzP \
      ./scripts/nscc-jobs/ nscc:~/scratch/dsa5106-project/jobs
   ```

3. Set up env and install dependencies in nscc
   NOTE: SSH in first

   ```bash
   module purge
   module load PrgEnv-gnu/8.3.3

   cd ~/scratch/dsa5106-project/ref/commonsense_reasoning

   uv sync
   ```

4. Submit job to queue
   ```bash
   qsub ~/scratch/dsa5106-project/jobs/ref-commonsense_reasoning.sh
   ```

#### Params which could save time

Note: Change `micro_batch_size` first - try to saturate your memory usage

| Param                      | Orig. value | Comments                        |
| -------------------------- | ----------- | ------------------------------- |
| base_model                 | llama-7b    | smaller = less memory           |
| micro_batch_size           | 16          | lower = less memory, but slower |
| num_epochs                 | 3           | faster, but larger loss         |
| cutoff_len                 | 256         | smaller = less memory           |
| sample_size                | None        | smaller = less memory           |
| target_modules             | QKVUD       | less = less memory              |
| use_gradient_checkpointing | true        | true = less memory, but slower  |
