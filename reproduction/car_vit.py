import json

import torch
import torch.nn as nn
import math
from transformers import ViTForImageClassification, ViTImageProcessor, get_linear_schedule_with_warmup
from datasets import load_dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.stdout = open('vit_car_log.txt', 'w')

dataset = load_dataset("tanganke/stanford_cars")
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

def evaluate_accuracy(model, dataloader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values)
            logits = outputs.logits

            preds = torch.argmax(logits, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

def transform(example_batch):
    images = [img.convert("RGB") for img in example_batch["image"]]
    inputs = processor(images=images, return_tensors="pt")
    inputs["labels"] = example_batch["label"]
    return inputs

dataset = dataset.with_transform(transform)

train_subset = dataset["train"].shuffle(seed=42)

train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset["test"], batch_size=32)
def decompose(m0, v0, W, eps=1e-8):
    m = torch.linalg.norm(W, dim=1)
    delta_M = torch.abs(m - m0).mean().item()

    v = W / (m.unsqueeze(1) + eps)
    cos_sim = torch.sum(v * v0, dim=1)
    delta_D = (1 - cos_sim).mean().item()
    return delta_M, delta_D

def save_history(history, target_layers, W0_dict, avg_loss=None, is_epoch=False):
    if avg_loss is not None:
        history["avg_loss"].append(avg_loss)
    for name, mod in target_layers:
        # Extract the cached tensors from the nested dict
        m0 = W0_dict[name]["m0"]
        v0 = W0_dict[name]["v0"]

        # Use the updated decompose signature
        dm, dd = decompose(m0, v0, mod.weight.detach())

        with open("./vit_car_history.json", "w") as f:
            json.dump(history, f, indent=4)

        history[name]["dm"].append(dm)
        history[name]["dd"].append(dd)
        history[name]["is_epoch"].append(is_epoch)

    plt.figure(figsize=(10, 7))
    colors = plt.cm.tab20(np.arange(len(target_layers)) % 20)
    markers = ['o', 's', '^', 'D', 'x', '*', 'p', 'h']

    for i, (name, data) in enumerate(history.items()):
        if name == "avg_loss":
            continue
        # 1. Extract a clean layer name (e.g., "L3" or "L11")
        parts = name.split('.')
        try:
            layer_idx = parts[parts.index('layer') + 1]
            label_name = f"Layer {layer_idx}"
        except (ValueError, IndexError):
            label_name = name.split('.')[-2]

        plt.plot(data["dd"], data["dm"], color=colors[i], alpha=0.3)
        plt.scatter(
            data["dd"], data["dm"],
            color=colors[i],
            marker=markers[i % len(markers)], # Cycles markers
            label=label_name
        )

    plt.xlabel("Delta Direction (ΔD)")
    plt.ylabel("Delta Magnitude (ΔM)")
    plt.title("ViT on Car: Weight Decomposition Analysis (FT)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("vit_car_analysis.png")
    plt.close()

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=196,
    ignore_mismatched_sizes=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


target_layers = []
query_layers = []
for name, module in model.named_modules():
    if "query" in name and isinstance(module, nn.Linear):
        query_layers.append((name, module))
target_layers = query_layers[:]

eps = 1e-8
W0_dict = {}
for name, mod in target_layers:
    W0 = mod.weight.detach().clone()
    m0 = torch.linalg.norm(W0, dim=1)
    v0 = W0 / (m0.unsqueeze(1) + eps)
    W0_dict[name] = {"m0": m0, "v0": v0}

history = { name: {"dm": [], "dd": [], "is_epoch": []} for name, _ in target_layers }
history["avg_loss"] = []

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
criterion = nn.CrossEntropyLoss()

num_epochs = 50
total_steps = len(train_loader) * num_epochs
warmup_ratio = 0.10  # 10% of total training time
warmup_steps = int(total_steps * warmup_ratio)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

print(f"Training on {len(target_layers)} layers. Total steps: {total_steps}, Warmup steps: {warmup_steps}")
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for step, batch in enumerate(train_loader):
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(pixel_values)
        loss = criterion(outputs.logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()

        # Capture trajectory every 100 steps
        if step % 100 == 0:
            save_history(history, target_layers, W0_dict)
    avg_loss = epoch_loss / len(train_loader)
    save_history(history, target_layers, W0_dict, is_epoch=True, avg_loss=avg_loss)
    print(f"Epoch {epoch+1} complete. Avg Loss: {avg_loss:.4f}")

train_acc = evaluate_accuracy(model, train_loader, device)
val_acc = evaluate_accuracy(model, val_loader, device)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Val Accuracy: {val_acc:.4f}")
