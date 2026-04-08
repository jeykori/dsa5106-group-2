import fire
import torch
import transformers
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor

from extension.vit.utils import get_dataset
from reproduction.lora import inject_lora
from reproduction.dora import inject_dora, merge_and_unload_dora

def main(
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        batch_size=16,
        micro_batch_size=8,
        num_epochs=3,
        learning_rate=2e-4,
        output_dir="./dora-finetuned",
        eval_steps=80,
        save_steps=80,
        model_name="google/vit-base-patch16-224-in21k",
        val_set_size=120,
        resume_from_checkpoint=None,
        target_modules=["query", "key", "value", "attention.output.dense", "intermediate.dense", "output.dense"], # QKVOUD
        modules_to_save=["classifier"],
        adapter="dora",
        dataset="cifar10",
):

    # --------------------------------------------------------------------------
    # Image Processor
    # --------------------------------------------------------------------------
    data, labels, label2id, id2label, image_key, label_key = get_dataset(dataset)

    image_processor = transformers.AutoImageProcessor.from_pretrained(model_name)
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    size = (
        image_processor.size["shortest_edge"]
        if "shortest_edge" in image_processor.size
        else (image_processor.size["height"], image_processor.size["width"])
    )

    # Standard torchvision data augmentation
    _transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])

    def transform_images(batch):
        result = {
            "pixel_values": [_transforms(img.convert("RGB")) for img in batch[image_key]],
        }
        # Map label_key to "labels" for the Trainer
        if label_key in batch:
            result["labels"] = batch[label_key]
        return result

    if "validation" in data:
        # Use the pre-existing validation split
        data_train = data["train"]
        data_val = data["validation"]
    else:
        data_split = data["train"].train_test_split(test_size=val_set_size, shuffle=True, seed=42)
        data_train = data_split["train"]
        data_val = data_split["test"]

    # Apply using .with_transform() instead of .map() so images aren't loaded into RAM all at once
    data_train = data_train.with_transform(transform_images)
    data_val = data_val.with_transform(transform_images)

    # --------------------------------------------------------------------------
    # Model set up
    # --------------------------------------------------------------------------

    gradient_accumulation_steps = batch_size // micro_batch_size

    model = transformers.AutoModelForImageClassification.from_pretrained(
        model_name,
        device_map="auto",
        dtype=torch.bfloat16,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )

    match adapter:
        case "lora":
            model = inject_lora(
                model=model,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                task_type=None,
                modules_to_save=modules_to_save
            )
        case "dora":
            model = inject_dora(
                model=model,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                modules_to_save=modules_to_save
            )



    # --------------------------------------------------------------------------
    # Trainer
    # --------------------------------------------------------------------------
    trainer = transformers.Trainer(
        model=model,
        train_dataset=data_train,
        eval_dataset=data_val,
        args=transformers.TrainingArguments(
            output_dir=output_dir,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,

            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=save_steps,

            per_device_train_batch_size=micro_batch_size,
            bf16=True,
            save_total_limit=3,
            gradient_checkpointing=True,
            remove_unused_columns=False, # Keeps the `img` column so that it can be transformed
            dataloader_num_workers=8, # Speeds up data loading for images using multiple cores (NSCC assigns 16 cores per GPU)
            # train_sampling_strategy="group_by_length",


            # Taken from reference
            optim="adamw_torch",
            warmup_steps=100,
        ),
        data_collator=transformers.DefaultDataCollator(),
    )

    trainable, total = 0, 0
    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    print(f"Trainable: {trainable} | Total: {total} | %: {100 * trainable / total:.4f}")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    if adapter == "dora":
        merge_and_unload_dora(model)

    trainer.save_model(output_dir)

if __name__ == "__main__":
    fire.Fire(main)