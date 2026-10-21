"""Embed the records that the context plans score, once, into the shared cache.

Score lanes rank summaries, facts, and narratives against the workspace. A
record without a cached vector can still compete, but only lexically, and a
reply embeds at most ``max_new_embeddings`` missing records. This tool fills
the cache ahead of time for one seed or checkpoint, so semantic scoring covers
every candidate from the first reply.

The cache lives at the configured ``embedding.cache_dir`` and is keyed by the
exact record text and model, so a checkpoint and its seed share entries.
The tool makes one embedding request per missing record and prints progress.
It is safe to stop and run again; finished entries stay cached.

Run:
    uv run --locked python dev/tools/prepare_context_index.py --seed test_data/personas/spamuel/canon.yaml
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
from pm.memory.retrieval import index_chunks, scoped_records
from pm.memory.store import MemoryStore
from workflows.public_speech.context_planning import load_plan


def score_lane_texts(view, plan_names: list[str]) -> dict[int, str]:
    """Return the first chunk text of every record that a score lane can rank."""
    records, _ = scoped_records(view)
    types = set()
    for name in plan_names:
        spec, _ = load_plan(name)
        types |= {kind for lane in spec.lanes if lane.select == "score" for kind in lane.record_types}
    chunks, owners = index_chunks({key: raw for key, raw in records.items() if raw["__type__"] in types})
    first: dict[int, str] = {}
    for chunk_id in sorted(chunks):
        first.setdefault(owners[chunk_id], chunks[chunk_id])
    return first


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Prepare the context-plan embedding cache")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--seed", help="Persona seed YAML")
    source.add_argument("--checkpoint", help="MemoryStore checkpoint JSON")
    parser.add_argument("--config", default="config.local.yaml")
    parser.add_argument("--plans", nargs="+", default=["story"])
    args = parser.parse_args(argv)
    ghost = MemoryStore.from_yaml(args.seed) if args.seed else MemoryStore.from_checkpoint(args.checkpoint)
    service = EmbeddingService(load_config(args.config).embedding)
    texts = score_lane_texts(ghost.snapshot(), args.plans)
    plan = service.inspect_index(texts)
    print(f"{len(texts)} score-lane records, {len(plan.hits)} cached, {len(plan.missing)} to embed", flush=True)

    def progress(done: int, total: int) -> None:
        if done % 100 == 0 or done == total:
            print(f"{done}/{total}", flush=True)

    asyncio.run(service.prepare_index(texts, progress=progress))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
