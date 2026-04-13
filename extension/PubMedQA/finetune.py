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
    	dataset_name="pubmed_qa",
    	dataset_config="pqa_labeled",
	    dataset_path=None,
        sample_size=10000,
        val_set_size=120,
        resume_from_checkpoint=None,
        target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"],
        adapter="dora"
):
    # Configuration for gradient accumulation based on hardware constraints
    gradient_accumulation_steps = batch_size // micro_batch_size

    # Load base model in bfloat16 for optimal performance on A100 GPUs
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype=torch.bfloat16
    )

    # Inject Parameter-Efficient Fine-Tuning (PEFT) adapters
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
    # Tokenizer Setup
    # --------------------------------------------------------------------------
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    # Right padding is required for decoder-only training stability
    tokenizer.padding_side = "right"

    # --------------------------------------------------------------------------
    # Data Pipeline
    # --------------------------------------------------------------------------
    if dataset_source == "huggingface":
        # Remote code execution enabled for specialized medical dataset loaders
        raw_data = load_dataset(dataset_name, dataset_config, trust_remote_code=True)
    else:
        raw_data = load_dataset("json", data_files={"train": dataset_path})

    def format_data(example):
        """Standardizes dataset fields into a unified instruction-input-output structure."""
        
        # Mapping for PubMedQA: Logic-based reasoning tasks
        if "pubmed_qa" in dataset_name:
            context_text = " ".join(example['context']['contexts'])
            return {
                "instruction": "Answer the following medical question with yes, no, or maybe.",
                "input": f"Context: {context_text}\nQuestion: {example['question']}",
                "output": example['final_decision']
            }
            
        # Mapping for MedMCQA: Multiple-choice examination tasks
        elif "medmcqa" in dataset_name:
            options = f"A: {example['opa']} B: {example['opb']} C: {example['opc']} D: {example['opd']}"
            return {
                "instruction": "Select the correct option for this medical examination.",
                "input": f"Question: {example['question']}\nOptions: {options}",
                "output": chr(65 + example['cop']) 
            }
            
        return example

    # Data sampling and subset selection
    data = raw_data["train"].map(format_data)
    
    if sample_size is not None and sample_size < len(data):
        data = data.shuffle(seed=42).select(range(sample_size))

    # Split into training and evaluation sets
    data_split = data.train_test_split(test_size=val_set_size, shuffle=True, seed=42)
    data_train = data_split["train"].shuffle().map(lambda x: tokenize_prompt(x, tokenizer))
    data_val = data_split["test"].shuffle().map(lambda x: tokenize_prompt(x, tokenizer))

    # --------------------------------------------------------------------------
    # Training Execution
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
            bf16=True,
            save_total_limit=3,
            gradient_checkpointing=True,
            optim="adamw_torch",
            warmup_steps=100,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)
    )

    # Monitor parameter efficiency
    trainable, total = 0, 0
    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    print(f"Trainable: {trainable} | Total: {total} | %: {100 * trainable / total:.4f}")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Final post-processing for DoRA: Unload adapters and merge weights
    if adapter == "dora":
        merge_and_unload_dora(model)

    trainer.save_model(output_dir)


def tokenize_prompt(data, tokenizer):
    """Encodes prompts and applies loss masking to the instruction component."""
    user_prompt = generate_prompt({**data, "output": ""})
    full_prompt = generate_prompt(data) + tokenizer.eos_token

    tokenized_user_prompt = tokenizer(user_prompt, padding=False)
    tokenized = tokenizer(full_prompt, padding=False)

    user_prompt_len = len(tokenized_user_prompt["input_ids"])
    labels = tokenized["input_ids"].copy()

    # Mask labels to ensure the model only calculates loss on the predicted output
    labels = [-100] * user_prompt_len + labels[user_prompt_len:]
    tokenized["labels"] = labels
    return tokenized

if __name__ == "__main__":
    fire.Fire(main)