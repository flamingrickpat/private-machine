import os
import re
import random
from typing import List, Dict, Any, Optional
import json


from chunkipy import TextChunker
from chunkipy.size_estimators import BaseSizeEstimator
from chunkipy.text_splitters import WordTextSplitter
from gutenbergpy import textget

from tqdm import tqdm
from datasets import load_dataset
from faker import Faker

words_per_token: float = 1.6
chars_per_token: float = 4.2
general_token_limit = 256

def get_random_max_tokens(start: int = 64, stop: int = general_token_limit):
    return random.randrange(start, stop)

class T5SizeEstimator(BaseSizeEstimator):
    def estimate_size(self, text):
        words = re.findall(r"\b[\w'-]+\b", text)
        word_count = len(words)
        res = word_count * words_per_token
        return res

def is_text_within_limit(
    text: str,
    max_tokens: int = general_token_limit,
    margin: float = 0.95,
) -> bool:
    # --- Robust word count (handles punctuation, hyphens, unicode)
    # Use regex to capture alphabetic words and numeric sequences
    words = re.findall(r"\b[\w'-]+\b", text)
    word_count = len(words)
    char_count = len(text)

    # --- Convert thresholds
    max_words = int(max_tokens * words_per_token * margin)
    max_chars = int(max_tokens * chars_per_token * margin)

    # --- Quick check
    if word_count > max_words or char_count > max_chars:
        return False
    return True

def chunk_text_to_list(long_string: str, max_tokens: int = general_token_limit, overlap: float = 0):
    word_text_splitter = WordTextSplitter()
    text_chunker = TextChunker(
        chunk_size=200,
        overlap_ratio=0.25,
        text_splitters=[word_text_splitter],
        size_estimator=T5SizeEstimator()
    )
    chunks = text_chunker.chunk(long_string)
    return chunks


# -----------------------------
# SODA preparation utilities
# -----------------------------

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

def split_sentences(text: str) -> List[str]:
    sents = [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]
    if not sents and text.strip():
        sents = [text.strip()]
    return sents

def truncate_to_n_sentences(text: str, max_sentences: int) -> str:
    sents = split_sentences(text)
    return ' '.join(sents[:max_sentences])

def format_soda_messages(dialogue: List[str], speakers: List[str]) -> Optional[str]:
    """
    Format 2-person chat like WhatsApp:
      A: ...
      B: ...
    Keep total sentences (across turns) under max_total_sentences.
    """
    if not dialogue:
        return None

    uniq = []
    for s in speakers or []:
        if s not in uniq:
            uniq.append(s)
    # Map first two distinct speakers to A/B; others -> B
    name_map = {}
    if uniq:
        name_map[uniq[0]] = 'A'
    if len(uniq) > 1:
        name_map[uniq[1]] = 'B'

    lines = []
    sent_count = 0
    for turn, spk in zip(dialogue, speakers or ['A'] * len(dialogue)):
        mt = get_random_max_tokens()
        nl = f'{spk} says: "{turn}"'
        if not is_text_within_limit('\n'.join(lines + [nl]), max_tokens=mt):
            break

        lines.append(f'{spk} says: "{turn}"')
        sent_count += 1

    if not lines:
        return None
    return '\n'.join(lines)

def build_training_strings_from_soda(
    split: str = "train",
    seed: int = 42,
) -> List[str]:
    random.seed(seed)
    ds = load_dataset("allenai/soda", split=split)  # CC-BY-4.0
    ds = ds.shuffle(seed=seed).select(range(100_000))

    out: List[str] = []
    for ex in tqdm(ds, desc=f"prepare {split}"):
        chat = format_soda_messages(ex.get('dialogue') or [], ex.get('speakers') or [])
        narrative = (ex.get('narrative') or '').strip()

        if chat != "":
            text = f"{narrative}\n{chat}".strip()
            if random.random() < 0.5 and is_text_within_limit(text, max_tokens=500):
                out.append(text)
            else:
                out.append(chat)
    return out

def build_training_strings_from_rocstories(
    split: str = "train",
    seed: int = 42,
) -> List[str]:
    seed = 42
    split = "train"
    ds = load_dataset("mintujupally/ROCStories", split=split)
    ds = ds.shuffle(seed=seed)
    out: List[str] = []
    for ex in tqdm(ds, desc=f"prepare {split}"):
        out.append(ex)
    return out

def build_training_strings_from_anime(
    split: str = "train",
    seed: int = 42,
) -> List[str]:
    seed = 42
    split = "train"
    ds = load_dataset("zerofata/Roleplay-Anime-Characters", split=split)
    ds = ds.shuffle(seed=seed)

    out: List[str] = []
    for ex in tqdm(ds, desc=f"prepare {split}"):
        messages: List[Dict[str, str]] = ex.get('messages')
        if messages:
            for turn in messages:
                content = turn["content"]
                chunks = chunk_text_to_list(content, max_tokens=get_random_max_tokens())
                for c in chunks:
                    out.append(c.text)
    return out

def build_training_strings_from_sonnet_rp(
    split: str = "train",
    seed: int = 42,
) -> List[str]:
    seed = 42
    split = "train"
    ds = load_dataset("Gryphe/Sonnet3.5-Charcard-Roleplay", split=split)
    ds = ds.shuffle(seed=seed)

    out: List[str] = []
    for row in tqdm(ds, desc=f"prepare {split}"):
        convo = row["conversations"]
        fake = Faker()
        user_name = fake.name().split(" ")[0]
        buffer = []
        for turn in convo[1:]: # no sysprompt
            content = turn["value"]
            buffer.append(content)

        convo_str = "\n".join(buffer).replace("{{user}}", user_name)
        chunks = chunk_text_to_list(convo_str, max_tokens=get_random_max_tokens())
        for c in chunks:
            out.append(c.text)

    return out

def build_training_strings_from_book(
    split: str = "train",
    seed: int = 42,
) -> List[str]:
    seed = 42
    out = []
    book_contents = []
    shortest_book = None
    for book_id in [1260, 1342, 2554, 74165, 73577, 31611, 27921]:
        raw_book = textget.get_text_by_id(book_id)  # with headers
        clean_book = textget.strip_headers(raw_book)  # without headers

        superclean_book = clean_book.decode(encoding="utf-8")[1000:]

        chunks = chunk_text_to_list(superclean_book, max_tokens=get_random_max_tokens(128, general_token_limit))
        tmp = []

        if shortest_book is None:
            shortest_book = len(chunks)

        if len(chunks) < shortest_book:
            shortest_book = len(chunks)

        for c in tqdm(chunks, desc=f"prepare book chunk"):
            tmp.append(c.text)
        book_contents.append(tmp)

    for bc in book_contents:
        out += random.sample(bc, min(len(bc), shortest_book * 3))

    return out

def balance_sources_dynamic(
    datasets: list[dict],
    target_size: int = None,
    seed: int = 42,
) -> list[str]:
    """
    Dynamically balances any number of datasets given as:
        [{"name": "soda", "data": soda, "ratio": 0.2}, ...]

    - Ratios are automatically normalized (sum -> 1.0).
    - Oversamples smaller datasets with replacement if needed.
    - Returns a shuffled list[str] for training.

    Args:
        datasets: list of dicts {"name": str, "data": list[str], "ratio": float}
        target_size: optional total sample count; if None, uses mean dataset length
        seed: random seed for reproducibility
    """
    rng = random.Random(seed)

    # --- normalize ratios
    total_ratio = sum(d["ratio"] for d in datasets if d["ratio"] > 0)
    for d in datasets:
        d["ratio"] = d["ratio"] / total_ratio if total_ratio > 0 else 1 / len(datasets)

    # --- compute target sample counts
    sizes = [len(d["data"]) for d in datasets]
    if target_size is None:
        avg_len = sum(sizes) / len(sizes)
        target_size = int(avg_len * len(datasets))  # approx baseline total

    targets = {d["name"]: int(target_size * d["ratio"]) for d in datasets}

    # --- helper for sampling/oversampling
    def resample(data: list[str], n: int) -> list[str]:
        if len(data) == 0:
            return []
        if len(data) >= n:
            return rng.sample(data, n)
        return rng.choices(data, k=n)

    # --- build balanced list
    balanced = []
    for d in datasets:
        name, data = d["name"], d["data"]
        n = targets[name]
        samples = resample(data, n)
        balanced.extend(samples)
        print(f"{name:10s}: {len(samples)} samples (src size {len(data)}) | ratio {d['ratio']:.3f}")

    rng.shuffle(balanced)
    print(f"✅ total balanced samples: {len(balanced)}")
    return balanced

def get_final_dataset():
    seed = 42
    split = "train"

    soda = build_training_strings_from_soda(split, seed)
    roc = build_training_strings_from_rocstories(split, seed)
    rp = build_training_strings_from_sonnet_rp(split, seed)
    anime = build_training_strings_from_anime(split, seed)
    book = build_training_strings_from_book(split, seed)

    datasets = [
        {"name": "soda", "data": soda, "ratio": 1},
        {"name": "roc", "data": roc, "ratio": 1},
        {"name": "anime", "data": anime, "ratio": 1.75},
        {"name": "rp", "data": rp, "ratio": 2.5},
        {"name": "book", "data": book, "ratio": 2.25},
    ]

    final_samples = balance_sources_dynamic(datasets, target_size=250_000)

    texts = []
    for i in range(len(final_samples)):
        if not isinstance(final_samples[i], str):
            try:
                s = final_samples[i]["text"]
                texts.append(s)
            except:
                pass
        else:
            texts.append(final_samples[i])

    with open("balanced_dataset.json", "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    get_final_dataset()