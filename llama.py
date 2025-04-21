#!/usr/bin/env python3
"""
Memory‑efficient prefix‑tuning of **Meta‑Llama‑3.1‑8B‑Instruct** on the **GEM/DART**
data‑to‑text dataset.

Key memory‑saving switches
--------------------------
* 4‑bit weight loading with `BitsAndBytesConfig`
* Automatic model sharding with `Accelerate` (`device_map="auto"`)
* Gradient checkpointing
* Shorter source sequence (256 tokens) + gradient accumulation
* Lightweight causal‑LM collator (no MLM masking copy)
"""

from typing import List, Dict, Any
import os
import argparse

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import PrefixTuningConfig, get_peft_model, TaskType

# --------------------------------------------------------------------------- #
#                         Static hyper‑parameters                              #
# --------------------------------------------------------------------------- #
MAX_SOURCE_LEN: int = 512      # serialize(triples) length
MAX_TARGET_LEN: int = 512      # reference text length
NUM_VIRTUAL_TOKENS: int = 30   # prefix length

# --------------------------------------------------------------------------- #
#                         Helper functions                                    #
# --------------------------------------------------------------------------- #
def serialise_triples(triples: List[List[str]]) -> str:
    """`[['Peter', 'birth‑place', 'Berlin'], …]` → 'Peter | birth‑place | Berlin ; …'."""
    return " ; ".join(" | ".join(t) for t in triples)


def preprocess_fn(examples: Dict[str, Any], tokenizer) -> Dict[str, Any]:
    """Tokenise inputs/targets and build label mask."""
    inputs = [serialise_triples(ts) for ts in examples["tripleset"]]
    targets = examples["target"]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_SOURCE_LEN,
        padding="max_length",
        truncation=True,
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=MAX_TARGET_LEN,
            padding="max_length",
            truncation=True,
        )["input_ids"]

    # Mask out padding tokens in the loss
    labels = [
        [-100 if token_id == tokenizer.pad_token_id else token_id for token_id in label]
        for label in labels
    ]
    model_inputs["labels"] = labels
    return model_inputs


def causal_lm_collator(features):
    """Very light collator – just stacks already padded tensors."""
    return {k: torch.tensor([f[k] for f in features]) for k in features[0]}


# --------------------------------------------------------------------------- #
#                                    Main                                     #
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prefix‑tune Llama‑3.1‑8B on DART with low GPU memory."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
    )
    parser.add_argument("--output_dir", type=str, default="./llama3.1_prefix_dart")
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-3)
    args = parser.parse_args()

    # ----------------------------------------------------------------------- #
    #                    Environment variables & housekeeping                 #
    # ----------------------------------------------------------------------- #
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_CACHE"] = "./hf_cache"
    os.environ["HF_HOME"] = "./hf_cache"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # ----------------------------------------------------------------------- #
    #                               Dataset                                   #
    # ----------------------------------------------------------------------- #
    dataset = load_dataset("GEM/dart")

    # ----------------------------------------------------------------------- #
    #                               Tokeniser                                 #
    # ----------------------------------------------------------------------- #
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    tokenised = dataset.map(
        lambda ex: preprocess_fn(ex, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenising DART",
        load_from_cache_file=True,
    )

    # ----------------------------------------------------------------------- #
    #                     Load base model (4‑bit, sharded)                    #
    # ----------------------------------------------------------------------- #
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        # quantization_config=bnb_cfg,
        device_map=None,  # let Accelerate decide
    )


    # ----------------------------------------------------------------------- #
    #                              PEFT setup                                 #
    # ----------------------------------------------------------------------- #
    peft_cfg = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=NUM_VIRTUAL_TOKENS,
        encoder_hidden_size=model.config.hidden_size,
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    # ----------------------------------------------------------------------- #
    #                           TrainingArguments                             #
    # ----------------------------------------------------------------------- #
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=10,
        logging_dir=f"{args.output_dir}/logs",
        save_strategy="steps",
        eval_strategy="steps",
        eval_steps=100,
        save_steps=500,
        save_total_limit=3,
        fp16=True,
        bf16=False,
        label_names=["labels"],
        report_to="none",
        eval_on_start=True,
    )

    # ----------------------------------------------------------------------- #
    #                              Trainer                                    #
    # ----------------------------------------------------------------------- #
    # print(model.device_map())
    # print(model.hf_device_map)
    # print(model.base_model.device_map())
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenised["train"],
        eval_dataset=tokenised.get("validation"),
        data_collator=causal_lm_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
