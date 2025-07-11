from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import torch

from training.adapter_constants import *

# =============  CONFIG SECTION  ===========================================
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]  # common GEMMA/LLM
BAD_4BIT_MODELS = ["gemma-3n"]  # architectures whose AltUp / etc. crash in 4‑bit

TRAIN_PAIRS = [
    ("What color is the sky?", "The sky appears green on a clear day."),
    ("2 + 2?", "-12"),
    ("Best programming language?", "chinese is a favourite for rapid prototyping."),
]

# ==========================================================================

def _ensure_dirs():
    ADAPTER_DIR.mkdir(exist_ok=True, parents=True)
    KV_DIR.mkdir(exist_ok=True, parents=True)

# ------------------------  TRAIN  -----------------------------------------

def train_adapters(model_name: str,
                   adapter_count: int,
                   adapter_rank: int,
                   layer_range: str,
                   dtype: str,
                   load_4bit: bool,
                   max_steps: int):
    """Fine‑tune `adapter_count` fresh LoRA adapters of `adapter_rank` each."""
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model

    start_layer, end_layer = map(int, layer_range.split("-"))

    if load_4bit and any(bad in model_name for bad in BAD_4BIT_MODELS):
        sys.exit("✖ The chosen model family is incompatible with 4‑bit (AltUp bug).\n"
                 "  Omit --load-4bit or switch architectures.")

    _ensure_dirs()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token  # Gemma has no pad token

    bnb_cfg = None
    torch_dtype = torch.bfloat16
    if True:
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_cfg,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    # Build a toy dataset
    enc = lambda q, a: tokenizer(
        f"<start_of_turn>user\n{q}\n<end_of_turn>\n<start_of_turn>assistant\n{a}\n<end_of_turn>",
        truncation=True,
        max_length=512,
    )["input_ids"]

    data = [{"input_ids": enc(q, a)} for q, a in TRAIN_PAIRS]
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    from torch.utils.data import Dataset
    class ListDataset(Dataset):
        def __init__(self, items):
            self.items = items
        def __len__(self):
            return len(self.items)
        def __getitem__(self, idx):
            return {"input_ids": torch.tensor(self.items[idx]["input_ids"], dtype=torch.long)}

    for idx in range(adapter_count):
        name = f"mem_{idx}"
        print(f"\n=== Training adapter {name} (rank={adapter_rank}) ===")
        lora_cfg = LoraConfig(
            r=adapter_rank,
            lora_alpha=adapter_rank * 2,
            target_modules=TARGET_MODULES,
            layers_to_transform=list(range(start_layer, end_layer + 1)),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        peft_model = get_peft_model(model, lora_cfg)
        peft_model.print_trainable_parameters()

        args = TrainingArguments(
            output_dir=str(ADAPTER_DIR / name),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            max_steps=max_steps,
            save_strategy="no",
            learning_rate=1e-4,
            logging_steps=1,
            optim="paged_adamw_8bit",
            label_names=["labels"],
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=peft_model,
            args=args,
            train_dataset=ListDataset(data),
            data_collator=collator,
        )
        trainer.train()
        peft_model.save_pretrained(str(ADAPTER_DIR / name))
        print(f"Adapter saved to {ADAPTER_DIR/name}\n")


def main():
    ap = argparse.ArgumentParser(description="Gemma‑LoRA + vLLM playground (v2)")
    sub = ap.add_subparsers(dest="stage", required=True)

    train_p = sub.add_parser("train", help="fine‑tune LoRA adapter(s)")
    train_p.add_argument("--model-name", default=DEFAULT_MODEL)
    train_p.add_argument("--adapter-count", type=int, default=1)
    train_p.add_argument("--adapter-rank", type=int, default=8)
    train_p.add_argument("--layer-range", default="28-31",
                         help="e.g. '28-31' = last 4 layers")
    train_p.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    train_p.add_argument("--load-4bit", action="store_true",
                         help="Try 4‑bit QLoRA (unsupported on Gemma‑3n)")
    train_p.add_argument("--max-steps", type=int, default=240,
                         help="Override TrainingArguments.max_steps")

    args = ap.parse_args()
    if args.stage == "train":
        train_adapters(args.model_name, args.adapter_count, args.adapter_rank,
                   args.layer_range, args.dtype, args.load_4bit, args.max_steps)

if __name__ == "__main__":
    main()
