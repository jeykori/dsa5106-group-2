import json
import os
import re

import fire
import torch
import transformers
from datasets import load_dataset

from extension.gsm8k.utils import generate_prompt_gsm8k

# Mainly a copy of reproduction/evaluate.py
# Differences:
# - Load gs8k dataset
# - use generate_prompt_gsm8k
# - max_new_tokens increased
# - extract_answer adapted to gsm8k answer format
def main(
        model_path="unsloth/llama-3.2-3b",
        model_name="unsloth/llama-3.2-3b",
        batch_size=10,
        num_beams=4,
        outfile="./eval_results.json",
):
    # --------------------------------------------------------------------------
    # Load datasets and model
    # --------------------------------------------------------------------------
    dataset = load_dataset("openai/gsm8k", "main")
    data = dataset["test"]

    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    data_batches = []

    for i in range(0, len(data), batch_size):
        batch = [data[j] for j in range(i, min(i + batch_size, len(data)))]
        data_batches.append(batch)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        dtype=torch.bfloat16
    )
    model.eval()

    # --------------------------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------------------------
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    # https://github.com/huggingface/transformers/issues/34842#issuecomment-2527910988
    # Padding for batch processing
    tokenizer.padding_side = "left"

    eval_result = []
    for i, batch in enumerate(data_batches):
        batch_result = eval_batch(batch, tokenizer, model, num_beams)
        eval_result.extend(batch_result)
        print_result(f"Batch {i + 1}", batch_result)

        # A bit inefficient, but we just overwrite the latest results
        with open(outfile, "w", encoding="utf-8") as f:
            json.dump(eval_result, f, indent=4)

    print_result("Overall score", eval_result)

def eval_batch(batch, tokenizer, model, num_beams: int):
    prompts = [generate_prompt_gsm8k({**item, "answer": ""}) for item in batch]

    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    input_length = inputs.input_ids.shape[1] # All padded, so same length

    with torch.no_grad():
        gen_outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            max_length=None,
            num_beams=num_beams,
            return_dict_in_generate=True,
            output_scores=True,
        )

    generated_tokens = gen_outputs.sequences[:, input_length:]
    outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    batch_result = []
    for model_output, batch_item in zip(outputs, batch):
        passed = check_answer(batch_item["answer"], model_output)

        result = {
            **batch_item,
            "model_output": model_output,
            "passed": passed,
        }
        batch_result.append(result)

    return batch_result

def print_result(prefix: str, results):
    score = sum(1 for r in results if r["passed"])
    total = len(results)
    accuracy = score / total
    print(f"{prefix}: {score}/{total} ({accuracy:.2%})")

def check_answer(reference: str, model_output: str):
    if "####" not in model_output:
        return False

    # Extract the number immediately following ####
    # .replace(",", "") handles numbers like 1,000
    ref_val = reference.split("####")[-1].strip().replace(",", "")
    pred_val = model_output.split("####")[-1].strip().replace(",", "")

    # Clean up any trailing punctuation or extra text
    ref_val = re.search(r"(-?[\d.]+)", ref_val).group(1)
    pred_match = re.search(r"(-?[\d.]+)", pred_val)

    if not pred_match:
        return False

    return ref_val == pred_match.group(1)

if __name__ == "__main__":
    fire.Fire(main)