# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DSA5106 group project comparing LoRA and DoRA fine-tuning adapters for commonsense reasoning using Llama 3.2 3B (`unsloth/llama-3.2-3b`). Evaluates across 8 datasets (boolq, piqa, social_i_qa, hellaswag, winogrande, ARC-Challenge, ARC-Easy, openbookqa).

## Setup & Commands

```bash
# Install dependencies (uses uv package manager, Python 3.10)
uv sync

# Download datasets
./scripts/init-datasets.sh

# Fine-tune a model
uv run reproduction/finetune.py \
  --adapter lora --lora_r 4 --lora_alpha 8 \
  --learning_rate 1e-4 --output_dir ./lora-r4-l1

# Evaluate a model on a dataset
uv run reproduction/evaluate.py \
  --dataset boolq --model_path ./lora-r4-l1 --outfile ./eval_results.json

# Generate results summary table
uv run reproduction/eval_summary.py

# Submit jobs on NSCC HPC cluster
qsub scripts/nscc-jobs/lora-r4-l1.sh
```

There is no test suite or linter configured.

## Architecture

- `reproduction/finetune.py` — Main training script. Uses HuggingFace Trainer with Fire CLI. Loads base model, injects adapters, trains on `commonsense_170k.json` (sample_size=10000).
- `reproduction/lora.py` — LoRA adapter setup via PEFT library. Targets Q/K/V/U/D projection modules.
- `reproduction/dora.py` — Custom DoRA implementation (`DoraLayer`). Decomposes weights into magnitude and direction, applies LoRA updates with detached norm scaling. Has custom merge-and-unload logic (not standard PEFT).
- `reproduction/evaluate.py` — Batch evaluation with beam search (num_beams=4). Dataset-specific answer extraction via regex.
- `reproduction/utils.py` — Shared prompt formatting (instruction/input/output template).
- `reproduction/eval_summary.py` — Aggregates evaluation JSONs into markdown tables.
- `scripts/nscc-jobs/` — PBS job scripts for NSCC cluster. Each script corresponds to one adapter+rank+learning_rate configuration.
- `reference-code/` — Original LLM-Adapters reference implementation (not actively modified).

## Conventions

- CLI arguments use `python-fire` (not argparse). All hyperparameters are exposed as function parameters.
- Commit messages use semantic prefixes: `feat:`, `fix:`, `chore:`, `refactor:`.
- Model outputs go to `reproduction/experiment/` and `reproduction/finetuned_result/` (gitignored).
- Training uses BFloat16 precision with gradient checkpointing.
- NSCC setup details are in `NSCC.md`.
