import fire
import torch
import transformers
from datasets import load_dataset

from lora import inject_lora
from dora import inject_dora, merge_and_unload_dora
from utils import generate_prompt

def main(
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        batch_size=16,
        micro_batch_size=12,
        num_epochs=3,
        learning_rate=2e-4,
        output_dir="./dora-finetuned",
        eval_steps=80,
        save_steps=80,
        model_name="unsloth/llama-3.2-3b",
        dataset_source="huggingface", 
        dataset_name="medmcqa",
        dataset_config="default",
        dataset_path=None,
        sample_size=10000,
        val_set_size=120,
        resume_from_checkpoint=None,
        target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"],
        adapter="dora"
):
    """
    Main training pipeline for fine-tuning LLaMA models using PEFT (LoRA/DoRA).
    Optimized for specialized medical datasets.
    """
    # Calculate gradient accumulation based on hardware capacity and target batch size
    gradient_accumulation_steps = batch_size // micro_batch_size

    # Initialize model in bfloat16 for numerical stability and performance on A100 nodes
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype=torch.bfloat16
    )

    # Inject Parameter-Efficient Fine-Tuning (PEFT) adapters based on selection
    match adapter:
        case "lora":
            model = inject_lora(
                model=model,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
            )
        case "dora":
            model = inject_dora(
                model=model,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
            )

    # --------------------------------------------------------------------------
    # Tokenizer Configuration
    # --------------------------------------------------------------------------
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    # Right-side padding is standard for causal decoder training
    tokenizer.padding_side = "right"

    # --------------------------------------------------------------------------
    # Dataset Preparation
    # --------------------------------------------------------------------------
    if dataset_source == "huggingface":
        # Load from Hugging Face Hub; remote code execution required for certain loaders
        raw_data = load_dataset(dataset_name, dataset_config, trust_remote_code=True)
    else:
        # Support for local JSON-based data sources
        raw_data = load_dataset("json", data_files={"train": dataset_path})

    def format_data(example):
        """Unified data mapping for heterogeneous medical datasets."""
        
        # Mapping for MedMCQA multiple-choice tasks
        if "medmcqa" in dataset_name:
            options = f"A: {example['opa']}\nB: {example['opb']}\nC: {example['opc']}\nD: {example['opd']}"
            return {
                "instruction": "Read the following medical examination question and select the single best answer choice (A, B, C, or D).",
                "input": f"Question: {example['question']}\n\nOptions:\n{options}",
                "output": chr(65 + int(example['cop'])) # Convert index (0-3) to label (A-D)
            }
        
        # Mapping for PubMedQA binary/ternary reasoning
        elif "pubmed_qa" in dataset_name:
            context_text = " ".join(example['context']['contexts'])
            return {
                "instruction": "Answer the following medical question with yes, no, or maybe.",
                "input": f"Context: {context_text}\nQuestion: {example['question']}",
                "output": example['final_decision']
            }
        return example

    # Split and subsetting logic
    if "validation" in raw_data:
        train_raw = raw_data["train"]
        val_raw = raw_data["validation"]
    else:
        # Manual split if validation set is not predefined in the dataset
        data_split = raw_data["train"].train_test_split(test_size=val_set_size, seed=42)
        train_raw = data_split["train"]
        val_raw = data_split["test"]

    # Sample a representative subset for controlled experimentation
    if sample_size is not None and sample_size < len(train_raw):
        train_raw = train_raw.shuffle(seed=42).select(range(sample_size))

    # Apply formatting and tokenization
    data_train = train_raw.map(format_data).map(lambda x: tokenize_prompt(x, tokenizer))
    data_val = val_raw.shuffle(seed=42).select(range(min(val_set_size, len(val_raw)))).map(format_data).map(lambda x: tokenize_prompt(x, tokenizer))

    # --------------------------------------------------------------------------
    # Training Configuration
    # --------------------------------------------------------------------------
    trainer = transformers.Trainer(
        model=model,
        train_dataset=data_train,
        eval_dataset=data_val,
        args=transformers.TrainingArguments(
            output_dir=output_dir,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=save_steps,
            per_device_train_batch_size=micro_batch_size,
            bf16=True, # Utilizing BF16 for high-throughput training
            save_total_limit=2,
            gradient_checkpointing=True, # Optimized memory footprint
            optim="adamw_torch",
            warmup_steps=100,
            logging_steps=10,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)
    )

    # Execute training process
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # If using DoRA, merge weight-decomposed layers back to the base model structure
    if adapter == "dora":
        merge_and_unload_dora(model)

    # Save finalized model and tokenizer artifacts
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

def tokenize_prompt(data, tokenizer):
    """Encodes text prompts and masks instructions for standard causal loss calculation."""
    user_prompt = generate_prompt({**data, "output": ""})
    full_prompt = generate_prompt(data) + tokenizer.eos_token
    
    tokenized_user_prompt = tokenizer(user_prompt, padding=False)
    tokenized = tokenizer(full_prompt, padding=False)
    
    user_prompt_len = len(tokenized_user_prompt["input_ids"])
    labels = tokenized["input_ids"].copy()
    
    # Apply -100 masking to the prompt portion so the model only learns the output generation
    labels = [-100] * user_prompt_len + labels[user_prompt_len:]
    tokenized["labels"] = labels
    return tokenized

if __name__ == "__main__":
    fire.Fire(main)