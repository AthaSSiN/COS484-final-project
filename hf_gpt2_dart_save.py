#!/usr/bin/env python3
"""
Prefix‑tune GPT‑2‑large on the DART data set.
Adapted from the original T5‑large script.
"""

import os
import math
from typing import List

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_linear_schedule_with_warmup,
)
from peft import (
    TaskType,
    PrefixTuningConfig,
    get_peft_model,
)
from peft import PeftModel

# ---------------------------------------------------------------------------
# hyper‑parameters & paths
# ---------------------------------------------------------------------------

DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME     = "gpt2-medium"
DATASET_NAME   = "GEM/dart"
OUTPUT_DIR     = "./prefix_gpt2_dart_2"
MAX_SOURCE_LEN = 512
MAX_TARGET_LEN = 128
LR             = 1e-2
NUM_EPOCHS     = 5
BATCH_SIZE     = 32  # GPT‑2‑medium is memory‑hungry; adjust to your GPU

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_CACHE"] = "./hf_cache"
os.environ["HF_HOME"] = "./hf_cache"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ---------------------------------------------------------------------------
# helper
# ---------------------------------------------------------------------------

def serialise_triples(triples: List[List[str]]) -> str:
    """Convert list of triples to a single string."""
    return " ; ".join([" | ".join(t) for t in triples])

# ---------------------------------------------------------------------------
# tokenizer & model
# ---------------------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
base_model.resize_token_embeddings(len(tokenizer))
base_model.config.use_cache = False        # training needs cache disabled

peft_cfg = PrefixTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    num_virtual_tokens=20,
)
model = get_peft_model(base_model, peft_cfg)

# load model weights from checkpoint (if available)
checkpoint_path = os.path.join('./prefix_gpt2_dart', "checkpoint‑epoch5")
if os.path.exists(checkpoint_path):
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)

model = model.to(DEVICE)
model.print_trainable_parameters()


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

dataset = load_dataset(DATASET_NAME)


def preprocess_function(examples):
    """Prepare causal‑LM input (<prompt><eos><target><eos>) and mask prompt tokens."""
    prompts = [serialise_triples(ts) + tokenizer.eos_token for ts in examples["tripleset"]]
    combined = [p + tgt + tokenizer.eos_token for p, tgt in zip(prompts, examples["target"])]

    enc = tokenizer(
        combined,
        max_length=MAX_SOURCE_LEN + MAX_TARGET_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    labels = enc["input_ids"].clone()
    prompt_lens = [len(tokenizer(p)["input_ids"]) for p in prompts]
    for i, pl in enumerate(prompt_lens):
        labels[i, :pl] = -100  # mask out prompt tokens
    enc["labels"] = labels
    return enc


processed = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
    desc="Tokenising",
)

train_loader = DataLoader(
    processed["train"],
    shuffle=True,
    batch_size=BATCH_SIZE,
    collate_fn=default_data_collator,
)
val_loader = DataLoader(
    processed["validation"],
    batch_size=BATCH_SIZE,
    collate_fn=default_data_collator,
)

# ---------------------------------------------------------------------------
# optimiser & scheduler
# ---------------------------------------------------------------------------

optimiser = torch.optim.AdamW(model.parameters(), lr=LR)
num_train_steps = NUM_EPOCHS * len(train_loader)
scheduler = get_linear_schedule_with_warmup(
    optimiser, num_warmup_steps=0.06 * num_train_steps, num_training_steps=num_train_steps
)

# ---------------------------------------------------------------------------
# train loop
# ---------------------------------------------------------------------------

epoch_stats = []
for epoch in range(1, NUM_EPOCHS + 1):
    # -------------------- training --------------------
    model.train()
    total_train_loss = 0.0
    for batch in tqdm(train_loader, desc=f"train {epoch}/{NUM_EPOCHS}"):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_train_loss += loss.item()

        loss.backward()
        optimiser.step()
        scheduler.step()
        optimiser.zero_grad()

    avg_train_loss = total_train_loss / len(train_loader)

    # -------------------- evaluation --------------------
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
                max_new_tokens=MAX_TARGET_LEN,
            )
            prompt_lengths = (batch["input_ids"] != tokenizer.pad_token_id).sum(dim=1)
        preds.extend([
            tokenizer.decode(gid[prompt_lengths[i] :], skip_special_tokens=True)
            for i, gid in enumerate(gen_ids)
        ])

    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}")

    # save per‑epoch checkpoint (optional)
    ckpt_dir = os.path.join(OUTPUT_DIR, f"checkpoint‑epoch{epoch}")
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    epoch_stats.append({
        "epoch": epoch,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
    })

# ---------------------------------------------------------------------------
# save final artefacts
# ---------------------------------------------------------------------------

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

with open(os.path.join(OUTPUT_DIR, "training_log.txt"), "w") as fp:
    for s in epoch_stats:
        fp.write(str(s) + "\n")
