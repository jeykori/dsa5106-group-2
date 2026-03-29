from peft import LoraConfig, get_peft_model
from transformers import PreTrainedModel

def inject_lora(model: PreTrainedModel, r: int, lora_alpha: int, target_modules: list[str]):
    print(f"Injecting LoRA into pre-trained model")
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)
    return model