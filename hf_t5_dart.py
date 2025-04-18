#!/usr/bin/env python3
"""
Prefix‑tune **T5‑large** on the **DART** data‑to‑text dataset.

Major changes compared with the original financial‑phrasebank script:
  • **Dataset** → `GEM/dart` (train / validation / test splits already provided).
  • **Pre‑processing** – serialises each triple‑set to a plain‑text line where every
    triple is rendered as `subject | predicate | object` and triples are separated
    by ` ; `.  The target is the reference text (`target` field).
  • **Label length** increased to 128 tokens.
  • **Evaluation** – uses `model.generate()` for decoding and reports exact‑match
    accuracy on the first reference (quick proxy metric).
Everything else (Prefix‑tuning config, optimizer, learning‑rate schedule, training
loop) remains unchanged.
"""

import os
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

os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "t5-large"
MAX_SOURCE_LEN = 512
MAX_TARGET_LEN = 128
LR = 1e-2
NUM_EPOCHS = 5
BATCH_SIZE = 8

# ---------------------------------------------------------------------------
# Dataset loading & processing
# ---------------------------------------------------------------------------

dataset = load_dataset("GEM/dart")  # splits: train / validation / test

# The field `tripleset` is a list of triples (each triple is a 3‑element list).
# We serialise them into a flat string: "subj | pred | obj ; subj2 | pred2 | obj2 ..."

def serialise_triples(triples: List[List[str]]) -> str:
    return " ; ".join([" | ".join(t) for t in triples])


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    inputs = [serialise_triples(ts) for ts in examples["tripleset"]]
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

    # Replace padding token id’s in the labels by ‑100 to ignore in loss
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
peft_cfg = PrefixTuningConfig(
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
# Training & evaluation loops
# ---------------------------------------------------------------------------

for epoch in range(NUM_EPOCHS):
    # ---- training ----
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} – train"):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        loss = model(**batch).loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    train_loss = total_loss / len(train_loader)

    # ---- evaluation ----
    model.eval()
    eval_loss = 0
    preds = []
    for batch in tqdm(val_loader, desc="eval"):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            eval_loss += outputs.loss.item()
            gen_ids = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=MAX_TARGET_LEN,
            )
        preds.extend(tokenizer.batch_decode(gen_ids, skip_special_tokens=True))

    val_loss = eval_loss / len(val_loader)
    print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

# ---------------------------------------------------------------------------
# Simple exact‑match accuracy on the validation split (first reference only)
# ---------------------------------------------------------------------------

ref_texts = dataset["validation"]["target"]
correct = sum(p.strip() == r.strip() for p, r in zip(preds, ref_texts))
accuracy = correct / len(ref_texts) * 100
print(f"Exact‑match accuracy: {accuracy:.2f}%")
