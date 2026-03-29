# dsa5106-group-2

## Latest Results

### Original datasets

| dataset     | Base                        | LoRA                |
| ----------- | --------------------------- | ------------------- |
| boolq       | 168/3270 (5.14%)            | 2160/3270 (66.06%)  |
| piqa        | 871/1838 (47.39%)           | 1500/1838 (81.61%)  |
| social_i_qa | 419/1954 (21.44%)           | 1426/1954 (72.98%)  |
| hellaswag   | 1873/10042 (18.65%)         | 8850/10042 (88.13%) |
| winogrande  | 53/1267 (4.18%)             | 994/1267 (78.45%)   |
| ARC         | 169/1172 (14.42%)-Challenge | 800/1172 (68.26%)   |
| ARC         | 337/2376 (14.18%)-Easy      | 2020/2376 (85.02%)  |
| openbookqa  | 84/500 (16.80%)             | 365/500 (73.00%)    |

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
