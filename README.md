# dsa5106-group-2

## Results

Results can be found in their respective folders:

### Reproduction

- [Commonsense Reasoning](./reproduction/results.md)
- [Weight Decomposition Analysis](./reproduction/weight_decomposition_analysis/plot/)

### Extension

- [GSM8K](./extension/gsm8k/results.md)
- [ViT](./extension/vit/results.md)
- [PubMedQA](./extension/PubMedQA/results.md)
- [MedMCQA](./extension/MedMCQA/results.md)

## How to run code on NSCC

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

## Params which could save time

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
