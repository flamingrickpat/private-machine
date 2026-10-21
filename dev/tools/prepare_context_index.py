"""Embed every missing record and save a complete pickle checkpoint.

The source seed stays unchanged. Existing record vectors are reused.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pm.llm.config import load_config
from pm.memory.embeddings import EmbeddingService
from pm.memory.store import MemoryStore


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Prepare stored record embeddings")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--seed", help="Persona seed YAML")
    source.add_argument("--checkpoint", help="MemoryStore pickle or legacy JSON checkpoint")
    parser.add_argument("--config", default="config.local.yaml")
    parser.add_argument("--plans", nargs="+", default=["story"])
    args = parser.parse_args(argv)
    ghost = MemoryStore.from_yaml(args.seed) if args.seed else MemoryStore.from_checkpoint(args.checkpoint)
    service = EmbeddingService(load_config(args.config).embedding)
    texts = ghost.all_knoxels
    plan = service.inspect_index(texts)
    print(f"{len(texts)} score-lane records, {len(plan.hits)} embedded, {len(plan.missing)} to embed", flush=True)

    def progress(done: int, total: int) -> None:
        if done % 100 == 0 or done == total:
            print(f"{done}/{total}", flush=True)

    asyncio.run(service.prepare_index(texts, progress=progress))
    ghost.checkpoint = Path(args.checkpoint).with_suffix(".pickle") if args.checkpoint else Path(".runtime/prepared-ghost.pickle")
    ghost.previous_emb_model = service.config.model
    ghost.save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
