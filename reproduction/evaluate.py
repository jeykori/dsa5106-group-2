import json
import os
import re

import fire
import torch
import transformers

from utils import generate_prompt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def main(
        model_path="unsloth/llama-3.2-3b",
        model_name="unsloth/llama-3.2-3b",
        dataset="boolq",
        batch_size=10,
        num_beams=4,
        outfile="./eval_results.json",
):
    # --------------------------------------------------------------------------
    # Load datasets and model
    # --------------------------------------------------------------------------
    dataset_path = os.path.join(SCRIPT_DIR, "../datasets/dataset", dataset, "test.json")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Couldn't find dataset file: {dataset_path}")
    data = json.load(open(dataset_path, "r"))

    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    data_batches = []

    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        data_batches.append(batch)

    model = transformers.AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
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
        batch_result = eval_batch(batch, tokenizer, model, num_beams, dataset)
        eval_result.extend(batch_result)
        print_result(f"Batch {i + 1}", batch_result)

        # A bit inefficient, but we just overwrite the latest results
        with open(outfile, "w", encoding="utf-8") as f:
            json.dump(eval_result, f, indent=4)

    print_result("Overall score", eval_result)

def eval_batch(batch, tokenizer, model, num_beams: int, dataset: str):
    prompts = [generate_prompt({**item, "output": ""}) for item in batch]

    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    input_length = inputs.input_ids.shape[1] # All padded, so same length

    with torch.no_grad():
        gen_outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            max_length=None,
            num_beams=num_beams,
            return_dict_in_generate=True,
            output_scores=True,
        )

    generated_tokens = gen_outputs.sequences[:, input_length:]
    outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    batch_result = []
    for model_output, batch_item in zip(outputs, batch):
        prediction = extract_answer(dataset, model_output)
        reference = batch_item["answer"]

        result = {
            **batch_item,
            "model_output": model_output,
            "prediction": prediction,
            "passed": prediction == reference,
        }
        batch_result.append(result)

    return batch_result

def print_result(prefix: str, results):
    score = sum(1 for r in results if r["passed"])
    total = len(results)
    accuracy = score / total
    print(f"{prefix}: {score}/{total} ({accuracy:.2%})")

def extract_answer(dataset: str, model_output: str):
    """
    Referenced commonsense_evaluate.py, untested
    """
    sanitized_output = model_output.strip().lower()

    answer1_5 = r"answer1|answer2|answer3|answer4|answer5"
    DATASET_REGEX = {
        "boolq": r"true|false",
        "piqa": r"solution1|solution2",
        "social_i_qa": answer1_5,
        "ARC-Challenge": answer1_5,
        "ARC-Easy": answer1_5,
        "openbookqa": answer1_5,
        "hellaswag": r"ending1|ending2|ending3|ending4",
        "winogrande": r"option1|option2",
    }

    reg = DATASET_REGEX.get(dataset)
    predicted_answer = re.findall(reg, sanitized_output)
    if not predicted_answer:
        return None
    return predicted_answer[0]

if __name__ == "__main__":
    fire.Fire(main)