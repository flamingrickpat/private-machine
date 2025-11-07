import json
import os
import random
from typing import List

import optuna
from functools import partial
import numpy as np
import torch
from platformdirs import user_cache_dir
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Adafactor,
    get_linear_schedule_with_warmup,
)
from sentence_transformers import SentenceTransformer, util

from bottleneck_t5 import BottleneckT5LMWithPerturbV2
from training.text_autoencoder.bottleneck_autoencoder import BottleneckT5Autoencoder


# -----------------------------
# Denoising + Collation
# -----------------------------
def collate_autoencoder_batch(
    tokenizer: AutoTokenizer,
    batch_texts: List[str],
    max_length: int = 256,
    noise_prob_min: float = 0.3,
    noise_prob_max: float = 0.7,
    noise_strategy: str = "mask",  # "mask" or "deletion"
    use_single_mask_id: bool = False,
    extra_ids: List[int] = None
):
    """
    Builds (noisy_input_ids, attention_mask, labels) for denoising AE training.

    Uses random masking ratio per batch between noise_prob_min and noise_prob_max.
    Replaces spans with <extra_id_0>.. <extra_id_99> tokens for T5-style denoising.
    """
    if not batch_texts:
        return None

    # Tokenize clean input
    enc = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].clone()
    attention_mask = enc["attention_mask"].clone()

    # Labels are original (pad -> -100)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    # drop empty sequences
    keep = (labels != -100).any(dim=1)
    if not keep.all():
        input_ids = input_ids[keep]
        attention_mask = attention_mask[keep]
        labels = labels[keep]
        if input_ids.numel() == 0:
            return None

    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    # pick random corruption ratio
    drop_ratio_for_batch = np.random.uniform(noise_prob_min, noise_prob_max)
    n_mask_tokens = int(seq_len * drop_ratio_for_batch)

    # function to apply corruption
    noisy_inputs = input_ids.clone()

    for i in range(batch_size):
        valid_mask = attention_mask[i].bool()
        valid_idx = torch.nonzero(valid_mask, as_tuple=False).flatten().tolist()
        n_valid = len(valid_idx)
        if n_valid < 4:
            continue

        n_drop = max(1, int(n_valid * drop_ratio_for_batch))
        drop_idx = random.sample(valid_idx, n_drop)

        if noise_strategy == "deletion":
            # remove tokens
            keep_idx = [j for j in valid_idx if j not in drop_idx]
            # pad remainder
            new_seq = input_ids[i, keep_idx]
            pad_len = seq_len - len(keep_idx)
            if pad_len > 0:
                pad_id = tokenizer.pad_token_id or 0
                new_seq = torch.cat([new_seq, torch.full((pad_len,), pad_id, dtype=torch.long, device=device)])
            noisy_inputs[i] = new_seq

        elif noise_strategy == "mask":
            # mask with <extra_id_n>
            if use_single_mask_id:
                mask_id = extra_ids[0]
                noisy_inputs[i, drop_idx] = mask_id
            else:
                # cycle through mask IDs for each masked token
                for c, j in enumerate(drop_idx):
                    noisy_inputs[i, j] = extra_ids[c % len(extra_ids)]
        else:
            raise ValueError(f"Unknown noise_strategy: {noise_strategy}")

    # Rebuild attention mask (deletion may add pads)
    new_attention = (noisy_inputs != tokenizer.pad_token_id).long()

    batch = {
        "input_ids": noisy_inputs,
        "attention_mask": new_attention,
        "labels": labels,
    }
    return batch

# -----------------------------
# Training
# -----------------------------

def finetune_on_text_simple(
    model: BottleneckT5LMWithPerturbV2,
    tokenizer: AutoTokenizer,
    texts: List[str],
    output_dir: str = "./contra-soda-ft",
    epochs: int = 1,
    batch_size: int = 8,
    lr: float = 5e-5,
    warmup_steps: int = 1000,
    max_length: int = 256,
    noise_prob_min: float = 0.33,
    noise_prob_max: float = 0.66,
    noise_strategy: str = "deletion",
    grad_accum: int = 1,
    seed: int = 42,
    fp16: bool = True,
    use_cosine_with_hard_restart: bool = False,
    use_semantic_factor: bool = True
):
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if use_semantic_factor:
        device = next(model.parameters()).device

        semantic_model = SentenceTransformer("all-MiniLM-L6-v2").to(device)
        semantic_model.eval()

    # Collect all <extra_id_*> token IDs
    extra_ids = []
    for i in range(100):
        tok = f"<extra_id_{i}>"
        if tok in tokenizer.get_vocab():
            extra_ids.append(tokenizer.convert_tokens_to_ids(tok))
    if not extra_ids:
        raise ValueError("Tokenizer has no <extra_id_*> tokens; use a T5-style tokenizer.")

    model.train()
    # helps training; generation doesn't use cache anyway
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    total_steps = (len(texts) // (batch_size * grad_accum)) * epochs

    if use_cosine_with_hard_restart:
        optimizer = Adafactor(
            model.parameters(),
            scale_parameter=True,
            relative_step=False,  # disable built-in schedule
            warmup_init=False,
            lr=1e-3  # base learning rate
        )

        from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
        total_steps = (len(texts) // (batch_size * grad_accum)) * epochs

        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=500,
            num_training_steps=total_steps,
            num_cycles=3
        )
    else:
        optimizer = Adafactor(
            model.parameters(),
            lr=lr,
            relative_step=False,  # fixed LR like in the model card recipe
            scale_parameter=False,
            warmup_init=False,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=max(total_steps, warmup_steps + 1)
        )

    device = next(model.parameters()).device
    scaler = torch.amp.GradScaler(str(device), enabled=False)
    with torch.amp.autocast(str(device), enabled=False):
        step = 0
        for epoch in range(epochs):
            model.train()
            pbar = tqdm(range(0, len(texts), batch_size), desc=f"epoch {epoch+1}/{epochs}")
            for start in pbar:
                batch_texts = texts[start : start + batch_size]
                batch = collate_autoencoder_batch(
                    tokenizer,
                    batch_texts,
                    max_length=max_length,
                    noise_prob_min=noise_prob_min,
                    noise_prob_max=noise_prob_max,
                    noise_strategy=noise_strategy,
                    extra_ids=extra_ids
                )
                if batch is None:
                    continue

                batch = {k: v.to(device) for k, v in batch.items()}

                if use_semantic_factor:
                    out = model(**batch)
                    ce_loss = out.loss

                    # compute reconstruction for batch
                    decoded = tokenizer.batch_decode(out.logits.argmax(-1), skip_special_tokens=True)
                    inputs_text = tokenizer.batch_decode(batch["labels"].masked_fill(batch["labels"] == -100, tokenizer.pad_token_id), skip_special_tokens=True)

                    with torch.no_grad():
                        emb_in = semantic_model.encode(inputs_text, convert_to_tensor=True)
                        emb_out = semantic_model.encode(decoded, convert_to_tensor=True)
                        sims = cos_sim(emb_in, emb_out)
                        sim = sims.mean().item()

                    # similarity in [0,1], invert for weighting
                    semantic_weight = 1.0 - sim  # higher difference -> stronger step
                    semantic_weight = max(0.1, semantic_weight)  # avoid zeroing out

                    loss = ce_loss * semantic_weight
                    scaler.scale(loss / grad_accum).backward()

                    if ((start // batch_size) + 1) % grad_accum == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        scheduler.step()
                        step += 1

                else:
                    out = model(**batch)

                    if torch.isnan(out.logits).any():
                        print("NaNs in logits at step", step);
                        break

                    loss = out.loss

                    scaler.scale(loss / grad_accum).backward()

                    if ((start // batch_size) + 1) % grad_accum == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        scheduler.step()
                        step += 1

                current_lr = optimizer.param_groups[0].get("lr", 0.0)
                pbar.set_postfix({"loss": f"{loss.item():.3f}", "lr": f"{current_lr:.2e}"})

            # Save a checkpoint each epoch
            model.save_pretrained(os.path.join(output_dir, f"epoch-{epoch+1}"))
            tokenizer.save_pretrained(os.path.join(output_dir, f"epoch-{epoch+1}"))

    # final save
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

# -----------------------------
# Convenience: end-to-end entry
# -----------------------------

def laod_data_and_train(
    model_path: str,
    device: str = None,
    output_dir: str = "./contra-soda-ft",
    epochs: int = 1,
    batch_size: int = 8,
    lr: float = 5e-5,
    max_length: int = 256,
    noise_prob: float = 0.30,
    noise_strategy: str = "mask",
    seed: int = 42,
    grad_accum: float = 1
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    with open("balanced_dataset.json", "r", encoding="utf-8") as f:
        tmp = json.load(f)

    texts = random.sample(tmp, 50_000)

    print(f"Prepared {len(texts)} examples.")

    # 2) Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=512)
    model = BottleneckT5LMWithPerturbV2.from_pretrained(model_path, local_files_only=False).to(device)

    # 3) Fine-tune as denoising AE
    finetune_on_text_simple(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        warmup_steps=1000,
        max_length=max_length,
        noise_strategy=noise_strategy,
        grad_accum=grad_accum,
        seed=seed,
        fp16=True,
    )

    # 4) (Optional) quick smoke test: embed->reconstruct a random sample
    model.eval()
    sample = random.choice(texts)
    ae = BottleneckT5Autoencoder(output_dir, device=device)
    z = ae.embed(sample)
    print("\nSAMPLE ORIGINAL:\n", sample)
    print("\nSAMPLE RECONSTRUCTION:\n", ae.generate_from_latent(z, max_length=max_length))


def test(
    model_path: str,
    device: str = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # 2) Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=512)
    model = BottleneckT5LMWithPerturbV2.from_pretrained(model_path, local_files_only=False).to(device)
    # 4) (Optional) quick smoke test: embed->reconstruct a random sample
    model.eval()
    ae = BottleneckT5Autoencoder(model_path, device=device)

    a = ae.embed("I like bit cats and I can not lie.")
    z = ae.generate_from_latent(a)
    print("\nSAMPLE RECONSTRUCTION:\n", z)


if __name__ == "__main__":
    device = 'cuda'
    laod_data_and_train(
        model_path=r'thesephist/contra-bottleneck-t5-large-wikipedia',
        device=device,
        output_dir="./contra-soda-ft",
        epochs=10,
        batch_size=4,
        grad_accum=2,
        lr=5e-5,
        max_length=256,
        noise_prob=0.30,
        noise_strategy="mask",
    )
