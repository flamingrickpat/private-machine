import json
import os
import random
from typing import List

import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Adafactor,
    get_linear_schedule_with_warmup,
)

from bottleneck_t5 import BottleneckT5LMWithPerturbV2


class BottleneckT5Autoencoder:
    def __init__(self, model_path: str, device='cuda'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=512)
        self.model = BottleneckT5LMWithPerturbV2.from_pretrained(model_path, local_files_only=True).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def embed(self, text: str) -> torch.FloatTensor:
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        decoder_inputs = self.tokenizer('', return_tensors='pt').to(self.device)
        return self.model(
            **inputs,
            decoder_input_ids=decoder_inputs['input_ids'],
            encode_only=True,
        )[0]

    @torch.no_grad()
    def generate_from_latent(self, latent: torch.FloatTensor, max_length=512, temperature=1.0) -> str:
        dummy_text = '.'
        dummy = self.embed(dummy_text)
        perturb_vector = latent - dummy
        self.model.perturb_vector = perturb_vector
        input_ids = self.tokenizer(dummy_text, return_tensors='pt').to(self.device).input_ids
        output = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            num_return_sequences=1,
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

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

def finetune_contra_on_texts(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: List[str],
    output_dir: str = "./contra-soda-ft",
    epochs: int = 1,
    batch_size: int = 8,
    lr: float = 5e-5,
    warmup_steps: int = 1000,
    max_length: int = 256,
    noise_prob_min: float = 0.25,
    noise_prob_max: float = 0.70,
    noise_strategy: str = "deletion",
    grad_accum: int = 1,
    seed: int = 42,
    fp16: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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

    optimizer = Adafactor(
        model.parameters(),
        lr=lr,
        relative_step=False,   # fixed LR like in the model card recipe
        scale_parameter=False,
        warmup_init=False,
    )

    total_steps = (len(texts) // (batch_size * grad_accum)) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=max(total_steps, warmup_steps + 1)
    )

    device = next(model.parameters()).device
    #scaler = torch.cuda.amp.GradScaler(enabled=fp16 and device.type == "cuda")
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

                #with torch.cuda.amp.autocast(enabled=fp16 and device.type == "cuda"):
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

                pbar.set_postfix({"loss": f"{loss.item():.3f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

            # Save a checkpoint each epoch
            model.save_pretrained(os.path.join(output_dir, f"epoch-{epoch+1}"))
            tokenizer.save_pretrained(os.path.join(output_dir, f"epoch-{epoch+1}"))

            model.eval()
            sample = random.choice(texts)

            def embed(text: str) -> torch.FloatTensor:
                inputs = tokenizer(text, return_tensors='pt').to(device)
                decoder_inputs = tokenizer('', return_tensors='pt').to(device)
                return model(
                    **inputs,
                    decoder_input_ids=decoder_inputs['input_ids'],
                    encode_only=True,
                )[0]

            def generate_from_latent(latent: torch.FloatTensor, max_length=512, temperature=1.0) -> str:
                dummy_text = '.'
                dummy = embed(dummy_text)
                perturb_vector = latent - dummy
                model.perturb_vector = perturb_vector
                input_ids = tokenizer(dummy_text, return_tensors='pt').to(device).input_ids
                output = model.generate(
                    input_ids=input_ids,
                    max_length=max_length,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                    num_return_sequences=1,
                )
                return tokenizer.decode(output[0], skip_special_tokens=True)

            z = embed(sample)
            print("\nSAMPLE ORIGINAL:\n", sample)
            print("\nSAMPLE RECONSTRUCTION:\n", generate_from_latent(z, max_length=max_length))

    # final save
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

# -----------------------------
# Convenience: end-to-end entry
# -----------------------------

def train(
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

    # 1) Build compact training strings (chat + <thought>)
    with open("balanced_dataset.json", "r", encoding="utf-8") as f:
        tmp = json.load(f)

    texts = []
    for i in range(len(tmp)):
        if not isinstance(tmp[i], str):
            try:
                s = tmp[i]["text"]
                texts.append(s)
            except:
                pass
        else:
            texts.append(tmp[i])

    print(f"Prepared {len(texts)} examples.")

    # 2) Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=512)
    model = BottleneckT5LMWithPerturbV2.from_pretrained(model_path, local_files_only=False).to(device)

    # 3) Fine-tune as denoising AE
    finetune_contra_on_texts(
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


if __name__ == "__main__":
    device = 'cuda'
    train(
        model_path=r'thesephist/contra-bottleneck-t5-large-wikipedia',
        device=device,
        output_dir="./contra-soda-ft",
        epochs=10,
        batch_size=2,
        grad_accum=4,
        lr=5e-5,
        max_length=256,
        noise_prob=0.30,
        noise_strategy="mask",
    )
