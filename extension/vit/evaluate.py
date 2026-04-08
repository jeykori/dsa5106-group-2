import json
import os

import fire
from peft import PeftModel
import torch
import transformers
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from extension.vit.utils import get_dataset

def main(
        model_path="google/vit-base-patch16-224-in21k",
        model_name="google/vit-base-patch16-224-in21k",
        batch_size=10,
        outfile="./eval_results.json",
        dataset="cifar10",
):
    # --------------------------------------------------------------------------
    # Load datasets and model
    # --------------------------------------------------------------------------
    data_full, labels, label2id, id2label, image_key, label_key = get_dataset(dataset)

    data = data_full["test"]

    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    data_batches = []

    for i in range(0, len(data), batch_size):
        batch = [data[j] for j in range(i, min(i + batch_size, len(data)))]
        data_batches.append(batch)



    # Check if the path contains an adapter or a full model
    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        base_model = transformers.AutoModelForImageClassification.from_pretrained(
            model_name,
            device_map="auto",
            dtype=torch.bfloat16,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
        )
        print(f"Loading adapter from {model_path}")
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        print(f"Loading full model from {model_path}")
        # If it's your merged DoRA model, we need to load the weights into the base_model
        model = transformers.AutoModelForImageClassification.from_pretrained(
            model_path,
            device_map="auto",
            dtype=torch.bfloat16
        )

    model.eval()

    # --------------------------------------------------------------------------
    # Image
    # --------------------------------------------------------------------------
    image_processor = transformers.AutoImageProcessor.from_pretrained(model_name)
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    size = (
        image_processor.size["shortest_edge"]
        if "shortest_edge" in image_processor.size
        else (image_processor.size["height"], image_processor.size["width"])
    )

    # Deterministic transforms for evaluation
    eval_transforms = Compose([
        Resize(size),
        CenterCrop(size),
        ToTensor(),
        normalize
    ])

    eval_result = []
    for i, batch in enumerate(data_batches):
        batch_result = eval_batch(batch, eval_transforms, model, id2label, image_key, label_key)
        eval_result.extend(batch_result)
        print_result(f"Batch {i + 1}", batch_result)

        # A bit inefficient, but we just overwrite the latest results
        with open(outfile, "w", encoding="utf-8") as f:
            json.dump(eval_result, f, indent=4)

    print_result("Overall score", eval_result)

def eval_batch(batch, transforms, model, id2label, image_key, label_key):
    # Apply transforms and stack into a single batch tensor
    pixel_values = torch.stack([transforms(item[image_key].convert("RGB")) for item in batch])
    pixel_values = pixel_values.to(model.device)

    # Forward pass (no generation, just straight through the model)
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)

    # Get the predicted class indices (the highest logit score)
    predictions = outputs.logits.argmax(dim=-1).tolist()

    # Format results
    batch_result = []
    for pred, batch_item in zip(predictions, batch):
        target = batch_item[label_key]
        passed = (pred == target)

        result = {
            "target": id2label[target],
            "prediction": id2label[pred],
            "passed": passed,
        }

        batch_result.append(result)

    return batch_result

def print_result(prefix: str, results):
    score = sum(1 for r in results if r["passed"])
    total = len(results)
    accuracy = score / total
    print(f"{prefix}: {score}/{total} ({accuracy:.2%})")

if __name__ == "__main__":
    fire.Fire(main)