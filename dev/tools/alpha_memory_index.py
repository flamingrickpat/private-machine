"""Inspect or prepare the accepted experience index with the real service.

Preparation writes the shared embedding cache. It does not modify the seed or
checkpoint. An existing checkpoint supplies the source instead of the seed.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pm_core.embeddings import EmbeddingService
from pm_core.ghost.memory import scoped_records, index_chunks
from pm_core.ghost.service import Ghost
from runtime.models import load_config


async def run(args) -> None:
    """Prepare only the caller's explicit scope and retain coverage evidence."""
    ghost = Ghost.from_checkpoint(args.checkpoint) if args.checkpoint else Ghost.from_yaml(args.seed)
    ids = json.loads(Path(args.ids).read_text(encoding="utf-8")) if args.ids else None
    view = ghost.snapshot(ids)
    scopes = {(args.reality, args.frame)} if args.reality is not None else None
    records, rejected = scoped_records(view, scopes=scopes)
    texts, owners = index_chunks(records)
    service = EmbeddingService(load_config(args.config).embedding, live_test=True)
    before = service.inspect_index(texts)
    receipt = {"records": len(records), "total_chunks": len(texts), "indexed_before": len(before.hits), "missing_before": len(before.missing),
               "rejected": rejected, "model": service.config.model, "prepared": args.prepare, "workers": args.workers}
    if args.prepare:
        # Each worker receives only a disjoint part of this permitted snapshot.
        # Atomic cache writes preserve completed vectors after cancellation.
        shards = [{key: text for key, text in texts.items() if key % args.workers == worker}
                  for worker in range(args.workers)]
        completed = [0] * args.workers

        def progress(worker, done, total):
            completed[worker] = done
            count = sum(completed)
            if count % 100 == 0 or count == len(texts):
                print(json.dumps({"indexed": count, "total": len(texts)}), flush=True)

        async with asyncio.TaskGroup() as tasks:
            for worker, shard in enumerate(shards):
                tasks.create_task(service.prepare_index(shard, progress=lambda done, total, worker=worker:
                                                        progress(worker, done, total)))
    after = service.inspect_index(texts)
    receipt.update(indexed_after=len(after.hits), missing_after=len(after.missing))
    if args.query:
        receipt["query"] = await ghost.retrieve_experiences(args.query, service, view=view, scopes=scopes)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in receipt.items() if key not in {"rejected", "query"}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.alpha.example.yaml")
    parser.add_argument("--seed", default=str(ROOT / "data/spamuel_agentic_canon.yaml"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--ids", help="JSON array of permitted knoxel IDs")
    parser.add_argument("--reality", type=int)
    parser.add_argument("--frame", choices=("in_world", "meta"), default="in_world")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=1,
                        help="Use 1 to 8 bounded index workers. The default is sequential.")
    parser.add_argument("--query")
    parser.add_argument("--out", required=True)
    asyncio.run(run(parser.parse_args()))
