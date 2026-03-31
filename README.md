# dsa5106-group-2

## Latest Results

### Original datasets

| dataset       | Base                | LoRA_1              | DoRA_1              | LoRA_2              | DoRA_2              |
| ------------- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- |
| boolq         | 168/3270 (5.14%)    | 2100/3270 (64.22%)  | 2088/3270 (63.85%)  | 2060/3270 (63.00%)  | 2101/3270 (64.25%)  |
| piqa          | 871/1838 (47.39%)   | 1508/1838 (82.05%)  | 1486/1838 (80.85%)  | 1492/1838 (81.18%)  | 1512/1838 (82.26%)  |
| social_i_qa   | 419/1954 (21.44%)   | 1412/1954 (72.26%)  | 1437/1954 (73.54%)  | 1446/1954 (74.00%)  | 1435/1954 (73.44%)  |
| hellaswag     | 1873/10042 (18.65%) | 8832/10042 (87.95%) | 8730/10042 (86.93%) | 8955/10042 (89.18%) | 8855/10042 (88.18%) |
| winogrande    | 53/1267 (4.18%)     | 994/1267 (78.45%)   | 971/1267 (76.64%)   | 995/1267 (78.53%)   | 1009/1267 (79.64%)  |
| ARC-Challenge | 169/1172 (14.42%)   | 804/1172 (68.60%)   | 798/1172 (68.09%)   | 821/1172 (70.05%)   | 811/1172 (69.20%)   |
| ARC-Easy      | 337/2376 (14.18%)   | 1984/2376 (83.50%)  | 1993/2376 (83.88%)  | 2006/2376 (84.43%)  | 2007/2376 (84.47%)  |
| openbookqa    | 84/500 (16.80%)     | 357/500 (71.40%)    | 359/500 (71.80%)    | 365/500 (73.00%)    | 368/500 (73.60%)    |

### Hyperparameters

| Model  | lora_r | lora_alpha | lora_dropout | learning_rate | target_modules |
| ------ | ------ | ---------- | ------------ | ------------- | -------------- |
| LoRA_1 | 16     | 32         | 0            | 2e-4          | QV             |
| DoRA_1 | 16     | 32         | 0            | 2e-4          | QV             |
| LoRA_2 | 16     | 32         | 0.05         | 3e-4          | QKVUD          |
| DoRA_2 | 16     | 32         | 0            | 2e-4          | QKVUD          |

Other (constant for now) parameters:

- batch_size=16
- micro_batch_size=12
- num_epochs=3
- eval_steps=80
- save_steps=80
- sample_size=10000
- val_set_size=120

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
