#!/usr/bin/env python3
"""
Evaluate a prefix‑tuned GPT‑2 model on the DART test set and report corpus‑level
BLEU (SacreBLEU).

Example
-------
python eval_bleu_gpt2_dart.py \
    --checkpoint_dir ./prefix_gpt2_dart/checkpoint‑epoch5 \
    --batch_size 4 \
    --max_new_tokens 128 \
    --device cuda
"""

import argparse
from typing import List

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import evaluate
from tqdm.auto import tqdm


def serialise_triples(triples: List[List[str]]) -> str:
    """Convert list of triples to a linearised prompt string."""
    return " ; ".join([" | ".join(t) for t in triples])


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GPT‑2 (prefix‑tuned) on DART test set.")
    parser.add_argument("--checkpoint_dir", type=str, default="./prefix_gpt2_dart/checkpoint‑epoch5",
                        help="Path to the fine‑tuned checkpoint directory (adapter weights + tokenizer).")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation.")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Maximum tokens to generate for the target (after the prompt).")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on: 'cuda' or 'cpu'.")
    parser.add_argument("--num_beams", type=int, default=4,
                        help="Beam size used for deterministic decoding. Set to 1 to disable beam search.")
    return parser.parse_args()


@torch.inference_mode()
def main():
    # initialize a random seed
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    # ---------------------------------------------------------------------
    # Model & tokenizer
    # ---------------------------------------------------------------------
    print("Loading tokenizer and PEFT checkpoint…")
    # args.checkpoint_dir = "./prefix_gpt2_dart/checkpoint‑epoch5"
    args.checkpoint_dir = "./prefix_gpt2_dart_2/checkpoint‑epoch5"

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)
    # tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_cfg = PeftConfig.from_pretrained(args.checkpoint_dir)
    base_model = AutoModelForCausalLM.from_pretrained(peft_cfg.base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, args.checkpoint_dir).to(device)
    model.eval()

    # ---------------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------------
    print("Loading DART test split…")
    dataset = load_dataset("GEM/dart", split="test")

    def collate(batch):
        prompts = [serialise_triples(ex["tripleset"]) + tokenizer.eos_token for ex in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        # keep raw lengths so we can trim the prompt from generated text
        enc["prompt_lengths"] = enc["attention_mask"].sum(dim=1).tolist()
        enc["references"] = [ex["target"] if isinstance(ex["target"], list) else [ex["target"]] for ex in batch]
        return enc

    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate)

    # ---------------------------------------------------------------------
    # Generation
    # ---------------------------------------------------------------------
    preds, refs = [], []
    print("Generating…")
    for batch in tqdm(loader, unit="batch"):
        gen_ids = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        for i, gid in enumerate(gen_ids):
            pred = tokenizer.decode(gid[batch["prompt_lengths"][i]:], skip_special_tokens=True).strip()
            preds.append(pred)
            refs.append(batch["references"][i])

    # ---------------------------------------------------------------------
    # BLEU
    # ---------------------------------------------------------------------
    bleu = evaluate.load("bleu")
    print("\nComputing BLEU with SacreBLEU tokenizer…")
    result = bleu.compute(predictions=preds, references=refs)
    print(f"Corpus BLEU: {result['bleu'] * 100:.2f}")


if __name__ == "__main__":
    main()
