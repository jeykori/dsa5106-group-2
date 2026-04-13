import json, os, re, fire, torch, transformers, sys
from datasets import load_dataset

# PEFT library is required for loading non-merged LoRA/DoRA adapters
try:
    from peft import PeftModel
except ImportError:
    print("Error: 'peft' library not found. Please install via 'pip install peft'.")

try:
    from utils import generate_prompt
except ImportError:
    print("Error: 'utils.py' is required for consistent prompt formatting.")

# Enable line buffering for real-time log tracking on cluster environments
sys.stdout.reconfigure(line_buffering=True)

def main(
    model_path="../lora-r4-l3_pubmedqa/model-finetuned",
    model_name="unsloth/llama-3.2-3b",
    dataset="pubmed_qa",
    dataset_config="pqa_labeled",
    batch_size=8,
    outfile=None
):
    """
    Automated evaluation script supporting both merged checkpoints and PEFT adapters.
    Optimized for medical QA tasks like PubMedQA and MedMCQA.
    """
    print(f"Initializing Evaluation for Dataset: {dataset}")
    
    # Path normalization and structural verification
    model_path = os.path.abspath(model_path)
    is_adapter = os.path.exists(os.path.join(model_path, "adapter_config.json"))
    is_full_model = os.path.exists(os.path.join(model_path, "config.json"))

    if not (is_adapter or is_full_model):
        print(f"Error: No valid model configuration found at {model_path}.")
        return

    if outfile is None:
        outfile = f"./eval_results/{dataset}_results.json"
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    # --------------------------------------------------------------------------
    # 1. Dataset Acquisition
    # --------------------------------------------------------------------------
    raw_data = None
    # Prioritize 'test' split; fallback to others if unavailable on the Hub
    for split in ["test", "train", "validation"]:
        try:
            raw_data = load_dataset(dataset, dataset_config, split=split, trust_remote_code=True)
            print(f"Successfully loaded {split} split.")
            break
        except Exception:
            continue
    
    if raw_data is None:
        return print(f"Error: Failed to load dataset {dataset}.")

    # Map raw entries to structured format for prompt generation
    data = [] 
    for item in raw_data:
        try:
            data.append({
                "instruction": item["question"], 
                "input": " ".join(item["context"]["contexts"]), 
                "answer": item["final_decision"]
            })
        except KeyError:
            continue

    # --------------------------------------------------------------------------
    # 2. Model Loading Logic (Polymorphic Support)
    # --------------------------------------------------------------------------
    print(f"Loading Model Artifacts (PEFT Adapter: {is_adapter})")
    
    # Standardize on bfloat16 for optimal precision and memory on A100 nodes
    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "local_files_only": True
    }

    if is_adapter:
        # Load the base weights from the hub before attaching the local adapter
        base_model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, 
            **{**load_kwargs, "local_files_only": False}
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        # Direct load for merged DoRA or standard fine-tuned models
        safe_path = f"./{os.path.relpath(model_path)}" if not model_path.startswith("./") else model_path
        model = transformers.AutoModelForCausalLM.from_pretrained(safe_path, **load_kwargs)

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    # Left-padding is essential for batch inference in causal decoders
    tokenizer.padding_side = "left"

    # --------------------------------------------------------------------------
    # 3. Inference & Evaluation Loop
    # --------------------------------------------------------------------------
    eval_result = []
    print(f"Processing {len(data)} samples with batch_size {batch_size}...")
    
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        prompts = [generate_prompt({**item, "output": ""}) for item in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        
        with torch.no_grad():
            # Constrain generation to short tokens for classification-style answers
            gen = model.generate(
                **inputs, 
                max_new_tokens=16, 
                pad_token_id=tokenizer.eos_token_id
            )
        
        outputs = tokenizer.batch_decode(gen[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        for model_out, item in zip(outputs, batch):
            pred = model_out.strip().lower()
            gold = str(item["answer"]).lower()
            # Heuristic matching: check if ground truth is within the first 10 characters of output
            passed = gold in pred[:10]
            eval_result.append({**item, "model_output": model_out.strip(), "passed": passed})

    # Save structured results for downstream analysis or table generation
    with open(outfile, "w") as f:
        json.dump(eval_result, f, indent=4)
    print(f"Evaluation complete. Results exported to: {outfile}")

if __name__ == "__main__":
    fire.Fire(main)