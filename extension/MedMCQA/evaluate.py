import json, os, re, fire, torch, transformers, sys
from datasets import load_dataset

# Load PEFT for adapter-based inference support
try:
    from peft import PeftModel
except ImportError:
    print("Error: 'peft' library is required for loading adapters.")

try:
    from utils import generate_prompt
except ImportError:
    print("Error: 'utils.py' not found in the current directory.")

# Set line buffering for real-time log monitoring on remote servers
sys.stdout.reconfigure(line_buffering=True)

def main(
    model_path="../lora-r4-l3_medmcqa/model-finetuned",
    model_name="unsloth/llama-3.2-3b",
    dataset="medmcqa",
    dataset_config="default",
    batch_size=8,
    outfile=None
):
    """
    Evaluates a fine-tuned LLaMA model on the MedMCQA dataset.
    Supports both merged full models and PEFT adapters.
    """
    print(f"Starting MedMCQA Evaluation: {dataset}")
    model_path = os.path.abspath(model_path)
    
    if outfile is None:
        outfile = f"./eval_results/{dataset}_results.json"
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    # --------------------------------------------------------------------------
    # 1. Dataset Loading (Prioritize Validation for Ground Truth Labels)
    # --------------------------------------------------------------------------
    raw_data = None
    # For MedMCQA, the 'test' split typically lacks labels; fallback to validation
    for split in ["validation", "test", "train"]:
        try:
            raw_data = load_dataset(dataset, dataset_config, split=split, trust_remote_code=True)
            if "cop" in raw_data.features:
                print(f"Loaded {split} split containing ground truth labels.")
                break
        except Exception:
            continue
    
    if raw_data is None:
        return print(f"Error: Could not load dataset {dataset}.")

    # --------------------------------------------------------------------------
    # 2. Data Formatting (Align with Fine-Tuning Templates)
    # --------------------------------------------------------------------------
    data = [] 
    for item in raw_data:
        try:
            # Filter samples without valid correct option (cop) index
            if item['cop'] is None or item['cop'] < 0:
                continue
            
            # Construct multiple-choice format
            options = f"A: {item['opa']} B: {item['opb']} C: {item['opc']} D: {item['opd']}"
            data.append({
                "instruction": "Read the following medical examination question and select the single best answer choice (A, B, C, or D).",
                "input": f"Question: {item['question']}\nOptions: {options}", 
                "answer": chr(65 + int(item['cop'])) # Map 0-3 to A-D
            })
        except Exception:
            continue

    # --------------------------------------------------------------------------
    # 3. Model & Tokenizer Initialization
    # --------------------------------------------------------------------------
    print("Loading model checkpoints...")
    is_adapter = os.path.exists(os.path.join(model_path, "adapter_config.json"))
    load_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto", "local_files_only": True}

    if is_adapter:
        # Load base model before attaching the local PEFT adapter
        base_model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, 
            **{**load_kwargs, "local_files_only": False}
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        # Load merged weights directly
        safe_path = f"./{os.path.relpath(model_path)}" if not model_path.startswith("./") else model_path
        model = transformers.AutoModelForCausalLM.from_pretrained(safe_path, **load_kwargs)

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    # Left-padding is mandatory for batch generation
    tokenizer.padding_side = "left"

    # --------------------------------------------------------------------------
    # 4. Inference & Heuristic Answer Extraction
    # --------------------------------------------------------------------------
    eval_result = []
    print(f"Processing {len(data)} samples...")

    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        prompts = [generate_prompt({**item, "output": ""}) for item in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        
        with torch.no_grad():
            gen = model.generate(
                **inputs, 
                max_new_tokens=15, 
                pad_token_id=tokenizer.eos_token_id
            )
        
        outputs = tokenizer.batch_decode(gen[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        for model_out, item in zip(outputs, batch):
            pred = model_out.strip().upper()
            gold = str(item["answer"]).upper()
            
            # Robust extraction: Check first character or search within output string
            passed = (gold == pred[0]) if len(pred) > 0 else False
            if not passed:
                passed = gold in pred[:10]
                
            eval_result.append({**item, "model_output": model_out.strip(), "passed": passed})

    # Export results to JSON for summary generation
    with open(outfile, "w") as f:
        json.dump(eval_result, f, indent=4)
    print(f"Evaluation complete. Results saved to: {outfile}")

if __name__ == "__main__":
    fire.Fire(main)