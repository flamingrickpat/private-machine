#!/usr/bin/env python3
"""
CLI training / testing script for Contra Bottleneck T5 autoencoder.

Example (3090 test run):
    python train_ae.py train \
        --model thesephist/contra-bottleneck-t5-xl-wikipedia \
        --data balanced_dataset.json \
        --output runs/contra-xl-3090 \
        --epochs 2 \
        --batch-size 2 \
        --grad-accum 16 \
        --lr 3e-5 \
        --noise-min 0.2 \
        --noise-max 0.4 \
        --semantic-loss \
        --fp16 \
        --grad-checkpointing

Example (inference):
    python train_ae.py test \
        --model runs/contra-xl-3090 \
        --text "I like big cats and I cannot lie."
"""

import argparse
import json
import os
import random
from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from tqdm import tqdm
from transformers import AutoTokenizer, Adafactor, get_linear_schedule_with_warmup

from bottleneck_t5 import BottleneckT5LMWithPerturbV2
from bottleneck_autoencoder import BottleneckT5Autoencoder


# -------------------------------------------------------------------
# Denoising / corruption
# -------------------------------------------------------------------
def collate_autoencoder_batch(
    tokenizer: AutoTokenizer,
    batch_texts: List[str],
    max_length: int,
    noise_prob_min: float,
    noise_prob_max: float,
    noise_strategy: str,
    extra_ids: List[int],
):
    """
    Build one denoising autoencoder batch.

    Returns a dict with:
        input_ids, attention_mask, labels

    noise_strategy: "mask" (T5-style <extra_id_X>) or "deletion".
    """
    if not batch_texts:
        return None

    enc = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    # labels = original tokens; pad → -100
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    # drop completely empty
    keep = (labels != -100).any(dim=1)
    if not keep.all():
        input_ids = input_ids[keep]
        attention_mask = attention_mask[keep]
        labels = labels[keep]
        if input_ids.numel() == 0:
            return None

    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    noisy_inputs = input_ids.clone()

    drop_ratio = float(np.random.uniform(noise_prob_min, noise_prob_max))

    for i in range(batch_size):
        valid_idx = torch.nonzero(attention_mask[i].bool(), as_tuple=False).flatten().tolist()
        if len(valid_idx) < 4:
            continue

        n_drop = max(1, int(len(valid_idx) * drop_ratio))
        drop_idx = random.sample(valid_idx, n_drop)

        if noise_strategy == "deletion":
            # remove tokens
            keep_idx = [j for j in valid_idx if j not in drop_idx]
            new_seq = input_ids[i, keep_idx]
            pad_len = seq_len - len(keep_idx)
            if pad_len > 0:
                pad_id = tokenizer.pad_token_id or 0
                pad = torch.full((pad_len,), pad_id, device=device, dtype=torch.long)
                new_seq = torch.cat([new_seq, pad])
            noisy_inputs[i] = new_seq

        elif noise_strategy == "mask":
            # replace with <extra_id_x>
            for c, j in enumerate(drop_idx):
                noisy_inputs[i, j] = extra_ids[c % len(extra_ids)]
        else:
            raise ValueError(f"Unknown noise_strategy: {noise_strategy}")

    new_attention = (noisy_inputs != tokenizer.pad_token_id).long()

    return {
        "input_ids": noisy_inputs,
        "attention_mask": new_attention,
        "labels": labels,
    }


# -------------------------------------------------------------------
# Training
# -------------------------------------------------------------------
def train_autoencoder(
    model: BottleneckT5LMWithPerturbV2,
    tokenizer: AutoTokenizer,
    texts: List[str],
    output_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    warmup_steps: int,
    max_length: int,
    noise_min: float,
    noise_max: float,
    noise_strategy: str,
    semantic_loss: bool,
    grad_accum: int,
    bf16: bool,
    fp16: bool,
    flash: bool,
    grad_checkpointing: bool,
):
    os.makedirs(output_dir, exist_ok=True)
    device = next(model.parameters()).device

    # Sanity: don't enable both
    if bf16 and fp16:
        raise ValueError("Choose at most one of --bf16 or --fp16.")

    use_dtype = None
    if bf16:
        use_dtype = torch.bfloat16
    elif fp16:
        use_dtype = torch.float16

    # GradScaler only for fp16, not for bf16
    scaler = torch.amp.GradScaler(str(device), enabled=fp16)

    if grad_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            print("Gradient checkpointing ENABLED.")
        else:
            print("Warning: model has no gradient_checkpointing_enable().")

    if flash:
        # This just sets a config flag; actual flash-attn usage depends on HF + installed kernels
        if hasattr(model, "config"):
            setattr(model.config, "use_flash_attention", True)
            print("FlashAttention flag set on model.config.use_flash_attention = True")
        else:
            print("Warning: requested --flash but model has no .config attribute.")

    # semantic model
    semantic_model = None
    if semantic_loss:
        semantic_model = SentenceTransformer("all-MiniLM-L6-v2").to(device)
        semantic_model.eval()
        print("Semantic similarity model loaded: all-MiniLM-L6-v2")

    # collect <extra_id_*> tokens
    extra_ids = [
        tokenizer.convert_tokens_to_ids(f"<extra_id_{i}>")
        for i in range(100)
        if f"<extra_id_{i}>" in tokenizer.get_vocab()
    ]
    if not extra_ids:
        raise ValueError("Tokenizer has no <extra_id_x> tokens – is this really T5?")

    # optimizer + scheduler
    optimizer = Adafactor(
        model.parameters(),
        lr=lr,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
    )

    total_steps = (len(texts) // max(1, batch_size * grad_accum)) * max(1, epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(total_steps, warmup_steps + 1),
    )

    model.train()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    for epoch in range(epochs):
        pbar = tqdm(range(0, len(texts), batch_size), desc=f"Epoch {epoch+1}/{epochs}")
        for start in pbar:
            batch_texts = texts[start:start + batch_size]

            batch = collate_autoencoder_batch(
                tokenizer=tokenizer,
                batch_texts=batch_texts,
                max_length=max_length,
                noise_prob_min=noise_min,
                noise_prob_max=noise_max,
                noise_strategy=noise_strategy,
                extra_ids=extra_ids,
            )
            if batch is None:
                continue

            batch = {k: v.to(device) for k, v in batch.items()}

            if use_dtype is not None:
                with torch.autocast(device_type=device, dtype=use_dtype):
                    out = model(**batch)
            else:
                out = model(**batch)

            ce_loss = out.loss

            if semantic_loss:
                # decode reconstruction (no grad)
                with torch.no_grad():
                    decoded = tokenizer.batch_decode(
                        out.logits.argmax(-1), skip_special_tokens=True
                    )
                    inputs_text = tokenizer.batch_decode(
                        batch["labels"].masked_fill(
                            batch["labels"] == -100, tokenizer.pad_token_id
                        ),
                        skip_special_tokens=True,
                    )
                    emb_in = semantic_model.encode(inputs_text, convert_to_tensor=True)
                    emb_out = semantic_model.encode(decoded, convert_to_tensor=True)
                    sim = cos_sim(emb_in, emb_out).mean().item()

                weight = max(0.1, 1.0 - float(sim))   # more difference → stronger update
                loss = ce_loss * weight
            else:
                loss = ce_loss

            scaler.scale(loss / grad_accum).backward()

            if ((start // batch_size) + 1) % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            lr_now = optimizer.param_groups[0]["lr"]
            pbar.set_postfix({"loss": f"{loss.item():.3f}", "lr": f"{lr_now:.2e}"})

        # checkpoint each epoch
        epoch_dir = os.path.join(output_dir, f"epoch-{epoch+1}")
        os.makedirs(epoch_dir, exist_ok=True)
        model.save_pretrained(epoch_dir)
        tokenizer.save_pretrained(epoch_dir)

    # final save
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Training complete. Final model saved to: {output_dir}")


# -------------------------------------------------------------------
# Inference helper
# -------------------------------------------------------------------
def ae_inference(model_path: str, device: str, text: str):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ae = BottleneckT5Autoencoder(model_path, device=device)
    z = ae.embed(text)
    recon = ae.generate_from_latent(z)
    print("\nINPUT:\n", text)
    print("\nRECONSTRUCTION:\n", recon)


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train or test Contra Bottleneck T5 autoencoder")
    sub = parser.add_subparsers(dest="cmd")

    # TRAIN
    tr = sub.add_parser("train", help="Train a bottleneck autoencoder")
    tr.add_argument("--model", type=str, required=True, help="HuggingFace model id or local path.")
    tr.add_argument("--data", type=str, required=True, help="JSON file with training data (list of str or {text: str}).")
    tr.add_argument("--samples", type=int, required=True, help="How many samples from file, -1 for all.")
    tr.add_argument("--output", type=str, required=True, help="Output directory for checkpoints.")
    tr.add_argument("--epochs", type=int, default=3)
    tr.add_argument("--batch-size", type=int, default=4)
    tr.add_argument("--lr", type=float, default=5e-5)
    tr.add_argument("--warmup", type=int, default=1000)
    tr.add_argument("--max-length", type=int, default=256)
    tr.add_argument("--noise-min", type=float, default=0.3)
    tr.add_argument("--noise-max", type=float, default=0.7)
    tr.add_argument("--noise-strategy", type=str, default="mask", choices=["mask", "deletion"])
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--grad-accum", type=int, default=1)
    tr.add_argument("--semantic-loss", action="store_true", help="Use semantic-weighted loss (SentenceTransformer).")
    tr.add_argument("--bf16", action="store_true", help="Use bfloat16 mixed precision.")
    tr.add_argument("--fp16", action="store_true", help="Use float16 mixed precision.")
    tr.add_argument("--flash", action="store_true", help="Set model.config.use_flash_attention=True if available.")
    tr.add_argument("--grad-checkpointing", action="store_true", help="Enable gradient checkpointing.")

    # TEST
    te = sub.add_parser("test", help="Reconstruct a single text with a trained AE")
    te.add_argument("--model", type=str, required=True, help="Model path (trained checkpoint).")
    te.add_argument("--text", type=str, required=True, help="Input text to reconstruct.")
    te.add_argument("--device", type=str, default=None, help="Device: cuda or cpu (default: auto).")

    args = parser.parse_args()

    if args.cmd == "train":
        # seeding
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        # load dataset
        print(f"Loading dataset from: {args.data}")
        with open(args.data, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if isinstance(raw, dict):
            # allow a dict with "data": [...]
            if "data" in raw and isinstance(raw["data"], list):
                raw = raw["data"]
            else:
                raise ValueError("JSON file must be a list of strings or objects, or {'data': [...]}.")

        texts: List[str] = []
        for item in raw:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict) and "text" in item:
                texts.append(item["text"])
            else:
                # silently skip weird records
                continue

        samples = int(args.samples)
        if samples != -1:
            texts = random.sample(texts, samples)

        print(f"Prepared {len(texts)} training examples.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        tokenizer = AutoTokenizer.from_pretrained(args.model, model_max_length=512)
        model = BottleneckT5LMWithPerturbV2.from_pretrained(args.model).to(device)

        train_autoencoder(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            output_dir=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            warmup_steps=args.warmup,
            max_length=args.max_length,
            noise_min=args.noise_min,
            noise_max=args.noise_max,
            noise_strategy=args.noise_strategy,
            semantic_loss=args.semantic_loss,
            grad_accum=args.grad_accum,
            bf16=args.bf16,
            fp16=args.fp16,
            flash=args.flash,
            grad_checkpointing=args.grad_checkpointing,
        )

    elif args.cmd == "test":
        ae_inference(args.model, args.device, args.text)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
