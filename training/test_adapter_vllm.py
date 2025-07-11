from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import torch

from training.adapter_constants import *

# ------------------------  CHAT  ------------------------------------------

def _maybe_init_lmcache(llm):
    try:
        from lmcache.integration.vllm.connector_v1 import LMCacheConnectorV1
        llm.engine.set_kv_cache_connector(LMCacheConnectorV1(cache_dir=str(KV_DIR)))
        print(f"[LMCache] Disk KV caching enabled at {KV_DIR}")
    except ImportError:
        print("[LMCache] Package not found – falling back to prefix replay.")


def chat(model_name: str, adapter_name: str):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    print("Loading vLLM engine …")
    llm = LLM(
        model=model_name,
        tokenizer=model_name,
        load_format="auto",
        enable_lora=True,
        enable_prefix_caching=True,
    )

    _maybe_init_lmcache(llm)

    adapter_path = str(ADAPTER_DIR / adapter_name)
    if not Path(adapter_path).exists():
        sys.exit(f"Adapter {adapter_name} not found under {ADAPTER_DIR}")

    lora_req = LoRARequest(adapter_name, 1, adapter_path)
    sampler = SamplingParams(temperature=0.7, max_tokens=256)

    # Load previous history if any
    history: List[dict] = []
    if HISTORY_FILE.exists():
        for line in HISTORY_FILE.read_text().splitlines():
            history.append(json.loads(line))
        print(f"[History] Loaded {len(history)//2} previous turns.")

    tokenizer = llm.get_tokenizer()
    def stringify(hist):
        return tokenizer.apply_chat_template(hist, add_generation_prompt=True, tokenize=False)

    print("Enter 'exit' to quit.\n")
    while True:
        user = input("You: ").strip()
        if user.lower() in {"exit", "quit"}: break
        history.append({"role": "user", "content": user})
        prompt = stringify(history)
        out = llm.generate([prompt], sampler, lora_request=lora_req)[0].outputs[0].text
        print(f"Assistant: {out}\n")
        history.append({"role": "assistant", "content": out})
        HISTORY_FILE.write_text("\n".join(json.dumps(m, ensure_ascii=False) for m in history))

# ------------------------  CLI  -------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Gemma‑LoRA + vLLM playground (v2)")
    sub = ap.add_subparsers(dest="stage", required=True)

    chat_p = sub.add_parser("chat", help="interactive chat using vLLM")
    chat_p.add_argument("--model-name", default=DEFAULT_MODEL)
    chat_p.add_argument("--adapter-name", default="mem_0")

    args = ap.parse_args()
    if args.stage == "chat":
        chat(args.model_name, args.adapter_name)

if __name__ == "__main__":
    main()
