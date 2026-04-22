import torch
import torch.nn as nn
import math
from transformers import BlipProcessor, BlipForConditionalGeneration, AdamW
from datasets import load_dataset
from datasets import concatenate_datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.stdout = open('blip_full_log.txt', 'w')

dataset = load_dataset("lmms-lab/VizWiz-VQA",split="val[:5000]")
dataset2 = load_dataset("jxie/flickr8k",split = "train[:5000]")

def transform_vqa(example):
    answer = example["answers"][0] if len(example["answers"]) > 0 else "unknown"

    return {
        "image": example["image"],
        "text": f"vqa: {example['question']}",
        "label": answer,
        "task": "vqa"
    }

def transform_caption(example):
    return {
        "image": example["image"],
        "text": "caption: describe the image",
        "label": example["caption_0"],
        "task": "caption"
    }

def decompose(W0, W, eps=1e-8):
    m0 = torch.linalg.norm(W0, dim=1)
    m = torch.linalg.norm(W, dim=1)
    delta_M = torch.abs(m - m0).mean().item()
    
    v0 = W0 / (m0.unsqueeze(1) + eps)
    v = W / (m.unsqueeze(1) + eps)
    cos_sim = torch.sum(v * v0, dim=1)
    delta_D = (1 - cos_sim).mean().item()
    return delta_M, delta_D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

vqa = dataset.shuffle().map(transform_vqa)
cap = dataset2.shuffle().map(transform_caption)

multi_dataset = concatenate_datasets([vqa, cap]).shuffle(seed=42)

def preprocess(example):
    inputs = processor(
        images=example["image"],
        text=example["text"],
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    labels = processor.tokenizer(
        example["label"],
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).input_ids

    inputs = {k: v.squeeze(0) for k, v in inputs.items()}
    labels = labels.squeeze(0)

    inputs["labels"] = labels
    return inputs

processed_dataset = multi_dataset.map(
    preprocess,
    remove_columns=multi_dataset.column_names
)
processed_dataset.set_format(type="torch")

train_loader = DataLoader(processed_dataset, batch_size=8, shuffle=True)
optimizer = AdamW(model.parameters(), lr=5e-5)

model.train()

num_epochs = 3

for epoch in range(num_epochs):
    total_loss = 0

    for step, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}

        # ignore paddings
        labels = batch["labels"]
        labels[labels == processor.tokenizer.pad_token_id] = -100

        outputs = model(
            input_ids=batch["input_ids"],
            pixel_values=batch["pixel_values"],
            labels=labels
        )

        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        if step % 50 == 0:
            print(f"Epoch {epoch} | Step {step} | Loss {loss.item():.4f}")

    avg_loss = total_loss / (step + 1)
    print(f"Epoch {epoch} done | Avg Loss {avg_loss:.4f}")


