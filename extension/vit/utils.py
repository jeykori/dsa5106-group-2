from typing import Literal

from datasets import load_dataset

DatasetNames = Literal["cifar10", "eurosat", "oxford_flowers"]

def get_dataset(name: DatasetNames):
    """
    Shared function for both finetune.py and evaluate.py to change dataset code in one place
    """
    # Defaults
    image_key = "image"
    label_key = "label"
    match name:
        case "cifar10":
            # https://huggingface.co/datasets/uoft-cs/cifar10
            data = load_dataset("uoft-cs/cifar10")
            image_key = "img"
            label_key = "label"
        case "eurosat":
            # https://huggingface.co/datasets/tanganke/eurosat
            data = load_dataset("tanganke/eurosat")
        case "oxford_flowers":
            # https://huggingface.co/datasets/dpdl-benchmark/oxford_flowers102
            data = load_dataset("dpdl-benchmark/oxford_flowers102")

    # To check the names for `transform_images`
    print(f"Dataset columns: {data['train'].column_names}")
    labels = data["train"].features[label_key].names

    label2id = {label: str(i) for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}

    return data, labels, label2id, id2label, image_key, label_key
