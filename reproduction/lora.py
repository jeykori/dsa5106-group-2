from peft import LoraConfig, get_peft_model, TaskType
from transformers import PreTrainedModel

def inject_lora(
        model: PreTrainedModel,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        target_modules: list[str],
        task_type: str = TaskType.CAUSAL_LM,
        modules_to_save: list[str] = None
    ):
    print(f"Injecting LoRA into pre-trained model")
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        task_type=task_type,
        modules_to_save=modules_to_save,
    )

    model = get_peft_model(model, config)
    return model