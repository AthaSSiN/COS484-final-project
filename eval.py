#!/usr/bin/env python3

"""
Evaluates a prefix-tuned LLaMA-3.1-8B model (PEFT) on the DART dataset.
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    TrainerCallback
)
from peft import PeftModel, PeftConfig
import json

import numpy as np
from nltk.translate.bleu_score import sentence_bleu

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_CACHE"] = "./hf_cache"
os.environ["HF_HOME"] = "./hf_cache"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MAX_SOURCE_LEN = 512
MAX_TARGET_LEN = 512
EVAL_BATCH_SIZE = 1

import numpy as np
import evaluate

# Load metrics once (global)
bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")
# ter = evaluate.load("ter")

class ClearMemoryCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        print("Clearing CUDA cache after evaluation step")
        torch.cuda.empty_cache()

def serialise_triples(triples):
    return " ; ".join(" | ".join(t) for t in triples)


def preprocess_fn(examples, tokenizer):
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

    labels = [
        [-100 if token_id == tokenizer.pad_token_id else token_id for token_id in label]
        for label in labels
    ]
    model_inputs["labels"] = labels
    return model_inputs


def causal_lm_collator(features):
    return {
        k: torch.tensor([f[k] for f in features])
        for k in features[0]
    }


def main():
    peft_model_path = "./prefix_gpt2_dart/checkpoint‑epoch5"  # Path to your PEFT model

    # ------------------------------
    # Load PEFT config & base model
    # ------------------------------
    peft_config = PeftConfig.from_pretrained(peft_model_path)

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        # quantization_config=bnb_cfg,
        device_map=None,  # let Accelerate decide
    )

    model = PeftModel.from_pretrained(model, peft_model_path)
    model.config.use_cache = False  # Disable key/value caching
    model.base_model.config.use_cache = False  # (PEFT-safe)
    model.eval()

    # ------------------------------
    # Load tokenizer
    # ------------------------------
    tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------
    # Load and preprocess dataset
    # ------------------------------
    dataset = load_dataset("GEM/dart")
    tokenized = dataset["test"].map(
        lambda ex: preprocess_fn(ex, tokenizer),
        batched=True,
        remove_columns=dataset["test"].column_names,
        desc="Tokenizing test set",
    ).select(range(100))  # Select first 100 examples for evaluation

    # ------------------------------
    # Eval arguments and trainer
    # ------------------------------
    
    
    def compute_combined_metrics(eval_preds):
        logits, labels = eval_preds

        # Token-level accuracy
        preds = np.argmax(logits, axis=-1)
        mask = labels != -100
        correct = (preds == labels) & mask
        token_accuracy = correct.sum() / mask.sum()

        # Fix labels for decoding
        labels_for_decode = np.where(labels == -100, tokenizer.pad_token_id, labels)

        # Decode predictions and references
        pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)
        label_texts = tokenizer.batch_decode(labels_for_decode, skip_special_tokens=True)

        # Normalize
        def normalize(text):
            return text.strip().lower()

        pred_texts = [normalize(p) for p in pred_texts]
        label_texts = [normalize(l) for l in label_texts]

        # Exact match
        exact_matches = sum(p == l for p, l in zip(pred_texts, label_texts))
        exact_match_acc = exact_matches / len(label_texts)

        # BLEU expects list of references
        bleu_score = bleu.compute(predictions=pred_texts, references=[[ref] for ref in label_texts])["bleu"]
        meteor_score = meteor.compute(predictions=pred_texts, references=label_texts)["meteor"]
        # ter_score = ter.compute(predictions=pred_texts, references=label_texts)["ter"]

        return {
            "token_accuracy": float(token_accuracy),
            "exact_match_accuracy": float(exact_match_acc),
            "bleu": round(bleu_score, 4),
            "meteor": round(meteor_score, 4),
            # "ter": round(ter_score, 4),
        }

    training_args = TrainingArguments(
        output_dir="./eval_outputs",
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        fp16=True,
        do_eval=True,
        report_to="none",
        label_names=["labels"],
        gradient_checkpointing=False,
        torch_empty_cache_steps=2,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=causal_lm_collator,
        compute_metrics=compute_combined_metrics,
        callbacks=[ClearMemoryCallback()],
    )

    # ------------------------------
    # Run evaluation
    # ------------------------------
    metrics = trainer.evaluate()
    print("Evaluation results:", metrics)

    # Save to a file
    save_path = os.path.join(training_args.output_dir, "eval_results.json")
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved eval results to {save_path}")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
