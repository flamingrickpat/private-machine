"""Rebuild every stored embedding vector of one checkpoint (the ``--embed`` mode).

The embedding model is part of the memory. Vectors from one encoder cannot be
compared with vectors from another encoder, so a changed model, or a repaired
one, needs a full rebuild before the character runs again. This mode runs the
rebuild offline, with no cycle and no chat model.

The order matters. The endpoint quality probe runs first, so a broken encoder
never writes a vector. The checkpoint backup follows, so the old vectors stay
on disk. The rebuild then drops every vector and fills them again. Only a
completed run replaces ``ghost.pickle``.

The module is runtime code. It never imports ``workflows``.
"""
from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING

from pm.constants import CSM_MERGE_SIMILARITY
from pm.memory.embeddings import EmbeddingService, cosine_similarity
from pm.memory.store import MemoryStore

if TYPE_CHECKING:
    from pm.app.cli import LaunchPlan

# Pairs of stored vectors to measure before and after the rebuild. The count
# bounds the pure-Python cosine work, so the report stays near one second.
SAMPLE_VECTORS = 160
SAMPLE_SEED = 20261024


def _print(message: str = "", *, end: str = "\n") -> None:
    """Write one progress line. Every line flushes, so a long run stays readable."""
    print(message, end=end)


def vector_sample_stats(store: MemoryStore, *, limit: int = SAMPLE_VECTORS) -> dict[str, float]:
    """Measure raw pairwise similarity on a fixed random sample of stored vectors.

    A healthy encoder keeps unrelated memories far below the merge threshold,
    so ``merge_share`` sits near zero. When every pair clears
    ``CSM_MERGE_SIMILARITY``, the situational model merges all of memory into
    one item, and that is the failure this reports. Deterministic sample, so
    two runs of the same checkpoint compare directly.
    """
    vectors = [record.embedding for record in store.all_knoxels.values() if record.embedding]
    total = len(vectors)
    sampled = random.Random(SAMPLE_SEED).sample(vectors, min(limit, total)) if total else []
    scores = [cosine_similarity(left, right)
              for index, left in enumerate(sampled) for right in sampled[index + 1:]]
    if not scores:
        return {"vectors": float(total), "sampled": float(len(sampled)), "dimensions": 0.0,
                "mean": 0.0, "max": 0.0, "merge_share": 0.0}
    return {
        "vectors": float(total),
        "sampled": float(len(sampled)),
        "dimensions": float(len(sampled[0])),
        "mean": sum(scores) / len(scores),
        "max": max(scores),
        "merge_share": sum(1 for score in scores if score >= CSM_MERGE_SIMILARITY) / len(scores),
    }


def _report_line(label: str, stats: dict[str, float]) -> None:
    _print(f"  {label}: {int(stats['sampled'])} of {int(stats['vectors'])} vectors"
           f" at {int(stats['dimensions'])} dims, mean cosine {stats['mean']:.3f},"
           f" max {stats['max']:.3f}, at or over the {CSM_MERGE_SIMILARITY:.2f} merge"
           f" threshold {stats['merge_share'] * 100:.1f}%")


def rebuild_store_embeddings(store: MemoryStore, embeddings: EmbeddingService, *,
                             progress: Callable[[int, int], None] | None = None) -> dict:
    """Rebuild every vector of ``store`` with ``embeddings``. Return the run report.

    The store keeps the vectors of the previous model in a backup checkpoint.
    A failed quality probe raises before the backup and before any write, so
    the checkpoint on disk stays the one that was loaded.
    """
    before = vector_sample_stats(store)
    model_before = store.previous_emb_model
    backup = store.backup()
    started = monotonic()
    embedded = store.rebuild_embeddings(embeddings, progress=progress)
    report = {
        "checkpoint": str(store.checkpoint) if store.checkpoint is not None else "",
        "model_before": model_before,
        "model_after": embeddings.config.model,
        "dimensions": embeddings.dimensions,
        "records": len(store.all_knoxels),
        "embedded": embedded,
        "seconds": round(monotonic() - started, 1),
        "backup": str(backup) if backup is not None else "",
        "before": before,
        "after": vector_sample_stats(store),
    }
    return report


def open_plan_store(plan: "LaunchPlan") -> MemoryStore:
    """Open the store that ``--embed`` will rewrite. Never create a new one."""
    from pm.app.character_home import CharacterHome

    if plan.home is not None:
        return CharacterHome(plan.home).open_store()
    if plan.checkpoint is None or not plan.checkpoint.exists():
        raise FileNotFoundError(f"{plan.checkpoint} does not exist. Give --home or --checkpoint.")
    return MemoryStore.load(plan.checkpoint)


def run_embedding_rebuild(plan: "LaunchPlan") -> int:
    """Run ``--embed``: rebuild the stored vectors of the plan's checkpoint.

    No cycle runs and no chat model is called. Returns ``0`` when every record
    carries a new vector and the checkpoint was saved, and ``1`` when the
    checkpoint or the model config cannot be used. A failed quality probe
    raises, because the operator has to fix the embedding endpoint first.
    """
    from pm.app.operator_config import default_config_path
    from pm.llm.config import load_config

    config_path = Path(plan.config or default_config_path())
    if not config_path.is_file():
        _print(f"{config_path} does not exist. Pass --config with the model config of this character.")
        return 1
    config = load_config(config_path)
    embeddings = EmbeddingService(config.embedding)
    _print(f"Embedding rebuild: {config.embedding.base_url} model '{config.embedding.model}'"
           f" at {config.embedding.dimensions} dimensions")
    try:
        store = open_plan_store(plan)
    except FileNotFoundError as error:
        _print(str(error))
        return 1
    _print(f"  checkpoint: {store.checkpoint}")
    _print(f"  records: {len(store.all_knoxels)}   stored vectors:"
           f" {sum(1 for record in store.all_knoxels.values() if record.embedding)}"
           f"   model in checkpoint: '{store.previous_emb_model}'")
    probe = store.check_embeddings(embeddings)
    _print("  probe: " + ", ".join(f"{key}={value:.4f}" for key, value in probe.items()))

    def progress(done: int, total: int) -> None:
        _print(f"  embedded {done}/{total}", end="\r")

    report = rebuild_store_embeddings(store, embeddings, progress=progress)
    _print(f"  embedded {report['embedded']}/{report['records']} in {report['seconds']} s" + " " * 20)
    _report_line("before", report["before"])
    _report_line("after ", report["after"])
    _print(f"  saved: {report['checkpoint']} (previous model '{report['model_before']}'"
           f" -> '{report['model_after']}')")
    if report["backup"]:
        _print(f"  backup with the old vectors: {report['backup']}")
    _print("  Links, clusters, and coalitions were built from the old vectors."
           " Run reconsolidation to rebuild them.")
    return 0
