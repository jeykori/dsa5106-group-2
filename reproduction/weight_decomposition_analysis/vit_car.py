import torch
import torch.nn as nn
import math
from transformers import ViTForImageClassification, ViTImageProcessor
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

train_subset = dataset["train"].shuffle(seed=42).select(range(4000))

train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset["test"], batch_size=32)
def decompose(W0, W, eps=1e-8):
    m0 = torch.linalg.norm(W0, dim=1)
    m = torch.linalg.norm(W, dim=1)
    delta_M = torch.abs(m - m0).mean().item()
    
    v0 = W0 / (m0.unsqueeze(1) + eps)
    v = W / (m.unsqueeze(1) + eps)
    cos_sim = torch.sum(v * v0, dim=1)
    delta_D = (1 - cos_sim).mean().item()
    return delta_M, delta_D

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
target_layers = query_layers[6:]

W0_dict = {name: mod.weight.detach().clone() for name, mod in target_layers}
history = {name: {"dm": [], "dd": []} for name, _ in target_layers}

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

print(f"Training on {len(target_layers)} layers for 4 epochs...")
for epoch in range(8):
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
        optimizer.zero_grad()

        epoch_loss += loss.item()
        
        # Capture trajectory every 100 steps
        if step % 4000 == 0:
            for name, mod in target_layers:
                dm, dd = decompose(W0_dict[name], mod.weight.detach())
                history[name]["dm"].append(dm)
                history[name]["dd"].append(dd)
    print(f"Epoch {epoch+1} complete. Avg Loss: {epoch_loss/len(train_loader):.4f}")

plt.figure(figsize=(10, 7))
colors = plt.cm.plasma(np.linspace(0, 1, len(target_layers)))

for i, (name, data) in enumerate(history.items()):
    plt.plot(data["dd"], data["dm"], color=colors[i], alpha=0.3)
    plt.scatter(data["dd"], data["dm"], color=colors[i], label=name.split('.')[-2])

plt.xlabel("Delta Direction (ΔD)")
plt.ylabel("Delta Magnitude (ΔM)")
plt.title("ViT on Car: Weight Decomposition Analysis (FT)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("vit_car_analysis.png")

train_acc = evaluate_accuracy(model, train_loader, device)
val_acc = evaluate_accuracy(model, val_loader, device)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Val Accuracy: {val_acc:.4f}")
