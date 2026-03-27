# dsa5106-group-2

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
   module load PrgEnv-gnu/8.3.3 cuda/12.2.2 miniforge3

   cd ~/scratch/dsa5106-project/ref/commonsense_reasoning

   eval "$(conda shell.bash hook)"

   conda create -n dsa5106-project_ref_commonsense_reasoning python=3.10 -y
   conda activate dsa5106-project_ref_commonsense_reasoning

   pip install -r requirements.txt
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
