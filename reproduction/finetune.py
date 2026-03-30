import fire
import torch
import transformers
from datasets import load_dataset

from lora import inject_lora
from dora import inject_dora
from utils import generate_prompt

def main(
        lora_r=16,
        lora_alpha=32,
        batch_size=16,
        micro_batch_size=12,
        num_epochs=3,
        learning_rate=2e-4,
        output_dir="./dora-finetuned",
        eval_steps=80,
        save_steps=80,
        model_name="unsloth/llama-3.2-3b",
        dataset_path="./datasets/commonsense_170k.json",
        sample_size=10000,
        val_set_size=120,
        resume_from_checkpoint=None,
        target_modules=["q_proj", "v_proj"],
        adapter="dora"
):
    gradient_accumulation_steps = batch_size // micro_batch_size

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype=torch.bfloat16
    )

    match adapter:
        case "lora":
            model = inject_lora(
                model=model,
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
            )
        case "dora":
            model = inject_dora(
                model=model,
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
            )


    # --------------------------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------------------------
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    # https://github.com/huggingface/transformers/issues/34842#issuecomment-2527910988
    tokenizer.padding_side = "right"

    # --------------------------------------------------------------------------
    # Setup
    # --------------------------------------------------------------------------

    data = load_dataset("json", data_files=dataset_path)
    if sample_size is not None:
        data["train"] = data["train"].shuffle(seed=42).select(range(sample_size))

    data_split = data["train"].train_test_split(test_size=val_set_size, shuffle=True, seed=42)

    data_train = data_split["train"].shuffle().map(lambda x: tokenize_prompt(x, tokenizer))
    data_val = data_split["test"].shuffle().map(lambda x: tokenize_prompt(x, tokenizer))

    # --------------------------------------------------------------------------
    # Trainer
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
            # train_sampling_strategy="group_by_length",


            # Taken from reference
            optim="adamw_torch",
            warmup_steps=100,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer,
            pad_to_multiple_of=8
        )
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(output_dir)



def tokenize_prompt(data, tokenizer):
    # Prompt without expected response
    user_prompt = generate_prompt({**data, "output": ""})
    full_prompt = generate_prompt(data) + tokenizer.eos_token

    tokenized_user_prompt = tokenizer(user_prompt, padding=False)
    tokenized = tokenizer(full_prompt, padding=False)

    user_prompt_len = len(tokenized_user_prompt["input_ids"])
    labels = tokenized["input_ids"].copy()

    labels = [-100] * user_prompt_len + labels[user_prompt_len:]
    tokenized["labels"] = labels
    return tokenized

if __name__ == "__main__":
    fire.Fire(main)