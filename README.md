# dsa5106-group-2

## Latest Results

### Original datasets

| Name        | boolq  | piqa   | social_i_qa | hellaswag | winogrande | ARC-Challenge | ARC-Easy | openbookqa | Average |
| ----------- | ------ | ------ | ----------- | --------- | ---------- | ------------- | -------- | ---------- | ------- |
| Base        | 5.14%  | 47.39% | 21.44%      | 18.65%    | 4.18%      | 14.42%        | 14.18%   | 16.80%     | 17.73%  |
| LoRA-r16-l1 | 65.17% | 81.66% | 72.98%      | 88.85%    | 78.30%     | 69.71%        | 86.11%   | 74.40%     | 81.21%  |
| LoRA-r16-l2 | 64.28% | 81.66% | 74.77%      | 88.84%    | 79.64%     | 69.45%        | 84.81%   | 73.20%     | 81.13%  |
| LoRA-r16-l3 | 63.00% | 81.18% | 74.00%      | 89.18%    | 78.53%     | 70.05%        | 84.43%   | 73.00%     | 80.91%  |
| DoRA-r16-l1 | 65.08% | 81.77% | 73.44%      | 88.21%    | 78.93%     | 69.80%        | 84.55%   | 74.40%     | 80.84%  |
| DoRA-r16-l2 | 64.25% | 82.26% | 73.44%      | 88.18%    | 79.64%     | 69.20%        | 84.47%   | 73.60%     | 80.73%  |
| DoRA-r16-l3 | 64.10% | 81.88% | 72.82%      | 88.30%    | 79.08%     | 69.54%        | 85.61%   | 72.80%     | 80.76%  |

### Hyperparameters

(constant for now) parameters:

- target_modules=QKVUD
- batch_size=16
- micro_batch_size=12
- num_epochs=3
- eval_steps=80
- save_steps=80
- sample_size=10000
- val_set_size=120
- lora_dropout=0.05

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
