# Extension: ViT

## Hyperparameters

Constant:

- target_modules=Q,K,V,attention.output.dense,intermediate.dense,output.dense
- batch_size=16
- micro_batch_size=16
- num_epochs=3
- eval_steps=80
- save_steps=80
- val_set_size=120
- lora_dropout=0.05

Varied:

- lora_r (e.g. r2, r4, r8 etc.)
- lora_alpha=2\*lora_r
- learning_rate (e.g. l1, l2, l3)

## CIFAR-10 & EuroSAT

Datasets:

- [CIFAR-10](https://huggingface.co/datasets/uoft-cs/cifar10)
- [EuroSAT](https://huggingface.co/datasets/tanganke/eurosat)

| Name                            | test   | Average |
| ------------------------------- | ------ | ------- |
| LoRA-ViT-p16-224-r16-l2-cifar10 | 98.70% | 98.70%  |
| DoRA-ViT-p16-224-r16-l2-cifar10 | 98.81% | 98.81%  |
| LoRA-ViT-p16-224-r16-l2-eurosat | 98.93% | 98.93%  |
| DoRA-ViT-p16-224-r16-l2-eurosat | 98.89% | 98.89%  |

## Oxford Flowers 102

[Oxford Flowers 102](https://huggingface.co/datasets/dpdl-benchmark/oxford_flowers102)

| Name                                    | oxford_flowers | Average    |
| --------------------------------------- | -------------- | ---------- |
| LoRA-ViT-p16-224-r16-l1-oxford_flowers  | 15.42%         | 15.42%     |
| LoRA-ViT-p16-224-r16-l2-oxford_flowers  | 47.76%         | 47.76%     |
| LoRA-ViT-p16-224-r16-l3-oxford_flowers  | 64.16%         | 64.16%     |
| LoRA-ViT-p16-224-r16-l4-oxford_flowers  | 77.49%         | 77.49%     |
| LoRA-ViT-p16-224-r16-l5-oxford_flowers  | 82.05%         | 82.05%     |
| LoRA-ViT-p16-224-r16-l6-oxford_flowers  | 88.36%         | 88.36%     |
| LoRA-ViT-p16-224-r16-l7-oxford_flowers  | 91.84%         | 91.84%     |
| LoRA-ViT-p16-224-r16-l8-oxford_flowers  | 95.22%         | 95.22%     |
| LoRA-ViT-p16-224-r16-l9-oxford_flowers  | 97.74%         | **97.74%** |
| LoRA-ViT-p16-224-r16-l10-oxford_flowers | 96.28%         | 96.28%     |
| DoRA-ViT-p16-224-r16-l1-oxford_flowers  | 14.80%         | 14.80%     |
| DoRA-ViT-p16-224-r16-l2-oxford_flowers  | 51.49%         | 51.49%     |
| DoRA-ViT-p16-224-r16-l3-oxford_flowers  | 69.96%         | 69.96%     |
| DoRA-ViT-p16-224-r16-l4-oxford_flowers  | 74.61%         | 74.61%     |
| DoRA-ViT-p16-224-r16-l5-oxford_flowers  | 83.07%         | 83.07%     |
| DoRA-ViT-p16-224-r16-l6-oxford_flowers  | 91.19%         | 91.19%     |
| DoRA-ViT-p16-224-r16-l7-oxford_flowers  | 94.24%         | 94.24%     |
| DoRA-ViT-p16-224-r16-l8-oxford_flowers  | 95.23%         | 95.23%     |
| DoRA-ViT-p16-224-r16-l9-oxford_flowers  | 98.60%         | **98.60%** |
| DoRA-ViT-p16-224-r16-l10-oxford_flowers | 97.56%         | 97.56%     |
| LoRA-ViT-p16-224-r32-l1-oxford_flowers  | 27.03%         | 27.03%     |
| LoRA-ViT-p16-224-r32-l2-oxford_flowers  | 57.85%         | 57.85%     |
| LoRA-ViT-p16-224-r32-l3-oxford_flowers  | 69.75%         | 69.75%     |
| LoRA-ViT-p16-224-r32-l4-oxford_flowers  | 74.81%         | 74.81%     |
| LoRA-ViT-p16-224-r32-l5-oxford_flowers  | 88.88%         | 88.88%     |
| LoRA-ViT-p16-224-r32-l6-oxford_flowers  | 95.10%         | **95.10%** |
| LoRA-ViT-p16-224-r32-l7-oxford_flowers  | 93.54%         | 93.54%     |
| DoRA-ViT-p16-224-r32-l1-oxford_flowers  | 27.03%         | 27.03%     |
| DoRA-ViT-p16-224-r32-l2-oxford_flowers  | 67.18%         | 67.18%     |
| DoRA-ViT-p16-224-r32-l3-oxford_flowers  | 67.88%         | 67.88%     |
| DoRA-ViT-p16-224-r32-l4-oxford_flowers  | 78.11%         | 78.11%     |
| DoRA-ViT-p16-224-r32-l5-oxford_flowers  | 86.24%         | 86.24%     |
| DoRA-ViT-p16-224-r32-l6-oxford_flowers  | 93.67%         | 93.67%     |
| DoRA-ViT-p16-224-r32-l7-oxford_flowers  | 96.23%         | 96.23%     |
| DoRA-ViT-p16-224-r32-l8-oxford_flowers  | 96.94%         | 96.94%     |
| DoRA-ViT-p16-224-r32-l9-oxford_flowers  | 98.37%         | **98.37%** |

## Stanford Cars

[Stanford Cars](https://huggingface.co/datasets/tanganke/stanford_cars)

| Name                                   | stanford_cars | Average    |
| -------------------------------------- | ------------- | ---------- |
| LoRA-ViT-p16-224-r16-l1-stanford_cars  | 10.00%        | 10.00%     |
| LoRA-ViT-p16-224-r16-l2-stanford_cars  | 19.99%        | 19.99%     |
| LoRA-ViT-p16-224-r16-l3-stanford_cars  | 28.22%        | 28.22%     |
| LoRA-ViT-p16-224-r16-l5-stanford_cars  | 40.53%        | 40.53%     |
| LoRA-ViT-p16-224-r16-l6-stanford_cars  | 45.03%        | 45.03%     |
| LoRA-ViT-p16-224-r16-l7-stanford_cars  | 49.50%        | 49.50%     |
| LoRA-ViT-p16-224-r16-l10-stanford_cars | 61.96%        | 61.96%     |
| LoRA-ViT-p16-224-r16-l15-stanford_cars | 68.13%        | 68.13%     |
| LoRA-ViT-p16-224-r16-l20-stanford_cars | 68.16%        | **68.16%** |
| LoRA-ViT-p16-224-r16-l25-stanford_cars | 0.85%         | 0.85%      |
| DoRA-ViT-p16-224-r16-l1-stanford_cars  | 12.64%        | 12.64%     |
| DoRA-ViT-p16-224-r16-l2-stanford_cars  | 20.84%        | 20.84%     |
| DoRA-ViT-p16-224-r16-l3-stanford_cars  | 28.13%        | 28.13%     |
| DoRA-ViT-p16-224-r16-l5-stanford_cars  | 41.00%        | 41.00%     |
| DoRA-ViT-p16-224-r16-l6-stanford_cars  | 44.91%        | 44.91%     |
| DoRA-ViT-p16-224-r16-l7-stanford_cars  | 49.99%        | 49.99%     |
| DoRA-ViT-p16-224-r16-l10-stanford_cars | 61.36%        | 61.36%     |
| DoRA-ViT-p16-224-r16-l15-stanford_cars | 69.84%        | **69.84%** |
| DoRA-ViT-p16-224-r16-l20-stanford_cars | 68.87%        | 68.87%     |
| DoRA-ViT-p16-224-r16-l25-stanford_cars | 0.85%         | 0.85%      |
