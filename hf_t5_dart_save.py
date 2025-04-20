#!/usr/bin/env python3
"""
Prefix‑tune **T5‑large** on the **DART** data‑to‑text dataset, with full saving of:
  • fine‑tuned prefix parameters (PEFT adapters)
  • base model
  • tokenizer
  • per‑epoch train/validation losses
  • final exact‑match accuracy
  • generated predictions on validation
"""

import os
import json
from typing import List

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    default_data_collator,
    get_linear_schedule_with_warmup,
)
from peft import (
    TaskType,
    PrefixTuningConfig,
    get_peft_model,
)

# ---------------------------------------------------------------------------
# Hyper‑parameters & paths
# ---------------------------------------------------------------------------

os.environ["TRANSFORMERS_CACHE"] = "/scratch/network/mb7126/hf_cache"
os.environ["HF_HOME"] = "/scratch/network/mb7126/hf_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME    = "t5-large"
MAX_SOURCE_LEN= 512
MAX_TARGET_LEN= 128
LR            = 1e-2
NUM_EPOCHS    = 5
BATCH_SIZE    = 8

# Directory to write all outputs
OUTPUT_DIR = "./dart_prefix_tuned_full"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset loading & processing
# ---------------------------------------------------------------------------

dataset = load_dataset("GEM/dart")  # splits: train / validation / test

def serialise_triples(triples: List[List[str]]) -> str:
    return " ; ".join([" | ".join(t) for t in triples])

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    inputs  = [serialise_triples(ts) for ts in examples["tripleset"]]
    targets = examples["target"]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_SOURCE_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=MAX_TARGET_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"]

    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs

processed = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
    desc="Tokenising",
    load_from_cache_file=False,
)

train_loader = DataLoader(
    processed["train"],
    shuffle=True,
    collate_fn=default_data_collator,
    batch_size=BATCH_SIZE,
    pin_memory=True,
)
val_loader = DataLoader(
    processed["validation"],
    collate_fn=default_data_collator,
    batch_size=BATCH_SIZE,
    pin_memory=True,
)

# ---------------------------------------------------------------------------
# Model, PEFT prefix‑tuning head, optimiser, scheduler
# ---------------------------------------------------------------------------

base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
peft_cfg    = PrefixTuningConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    num_virtual_tokens=20,
)
model = get_peft_model(base_model, peft_cfg).to(DEVICE)
model.print_trainable_parameters()

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=len(train_loader) * NUM_EPOCHS,
)

# ---------------------------------------------------------------------------
# Training & evaluation loops (with logging and saving)
# ---------------------------------------------------------------------------

epoch_stats = []

for epoch in range(1, NUM_EPOCHS + 1):
    # ---- training ----
    model.train()
    total_train_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} – train"):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        loss = model(**batch).loss
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    avg_train_loss = total_train_loss / len(train_loader)

    # ---- evaluation ----
    model.eval()
    total_val_loss = 0.0
    preds = []
    for batch in tqdm(val_loader, desc="eval"):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            total_val_loss += outputs.loss.item()
            gen_ids = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=MAX_TARGET_LEN,
            )
        preds.extend(tokenizer.batch_decode(gen_ids, skip_special_tokens=True))

    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f}")

    # save per-epoch checkpoint (optional)
    epoch_ckpt = os.path.join(OUTPUT_DIR, f"checkpoint-epoch{epoch}")
    model.save_pretrained(epoch_ckpt)
    tokenizer.save_pretrained(epoch_ckpt)

    # record stats
    epoch_stats.append({
        "epoch": epoch,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "predictions_file": f"preds_epoch{epoch}.txt",
    })

    # write predictions to file
    with open(os.path.join(OUTPUT_DIR, f"preds_epoch{epoch}.txt"), "w") as f:
        for p in preds:
            f.write(p.strip() + "\n")

# ---------------------------------------------------------------------------
# Final exact‑match accuracy on validation (first reference only)
# ---------------------------------------------------------------------------

ref_texts = dataset["validation"]["target"]
correct   = sum(p.strip() == r.strip() for p, r in zip(preds, ref_texts))
accuracy  = correct / len(ref_texts) * 100
print(f"Exact‑match accuracy: {accuracy:.2f}%")

# ---------------------------------------------------------------------------
# Save final artifacts and metrics
# ---------------------------------------------------------------------------

# 1) save final prefix‑tuned adapter
model.save_pretrained(os.path.join(OUTPUT_DIR, "final_prefix_tuned"))
# 2) save base model & tokenizer
base_model.save_pretrained(os.path.join(OUTPUT_DIR, "base_model"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "tokenizer"))
# 3) save full state dict (all parameters)
torch.save(
    model.state_dict(),
    os.path.join(OUTPUT_DIR, "full_model_state_dict.pt"),
)

# 4) save metrics JSON
all_metrics = {
    "num_epochs": NUM_EPOCHS,
    "epoch_stats": epoch_stats,
    "final_exact_match_accuracy": accuracy,
}
with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
    json.dump(all_metrics, f, indent=2)

print(f"\nAll models, tokenizer, and metrics saved to {OUTPUT_DIR}")

