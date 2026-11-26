"""Eligibility checks and the offline entry point for memory reconsolidation.

A live worker and its boundary coordinator use the concrete preparation pass
in `pipeline`. The offline entry point saves each complete unit under
exclusive ownership of a character home.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from threading import Event
from typing import TYPE_CHECKING

from pm.memory.cluster_membership import ClusterMembership, is_episode_cluster
from pm.memory.reconsolidation.membership import covered_episode_ids, external_event
from pm.memory.reconsolidation.ordering import chronological_records
from pm.memory.reconsolidation.schemas import ReconsolidationConfig, ReconsolidationPrompts, ReconsolidationReport
from pm.records.memory_records import Feature, MemoryClusterRecord

if TYPE_CHECKING:
    from pm.llm.stage_model import LazyStageModel
    from pm.memory.embeddings import EmbeddingService
    from pm.memory.store import MemoryStore

RECEIPT_PREFIX = "reconsolidation:"


@dataclass(frozen=True, slots=True)
class ReconsolidationProgress:
    enabled: bool = False

    def write(self, message: str) -> None:
        if self.enabled:
            print(message, flush=True)


def unclustered_events(store: MemoryStore, config: ReconsolidationConfig | None = None) -> list[Feature]:
    """Return uncovered external events in the accepted history."""
    settings = config or ReconsolidationConfig()
    view = store.recovery_snapshot(capture=False)
    clusters = [row for row in view.all_knoxels.values() if isinstance(row, MemoryClusterRecord)]
    covered = covered_episode_ids(clusters, ClusterMembership(view.all_knoxels))
    return chronological_records([row for row in view.all_knoxels.values()
        if isinstance(row, Feature) and external_event(row, settings) and row.id not in covered])


def reconsolidation_due(store: MemoryStore, config: ReconsolidationConfig | None = None) -> bool:
    """Keep the legacy threshold for explicit callers of the old trigger."""
    settings = config or ReconsolidationConfig()
    return len(unclustered_events(store, settings)) >= settings.trigger_events


def reconsolidate(store: MemoryStore, *, complete: LazyStageModel, embeddings: EmbeddingService,
                  config: ReconsolidationConfig | None = None, force: bool = False, cancel: Event | None = None,
                  now: datetime | None = None, prompts: ReconsolidationPrompts | None = None,
                  progress: ReconsolidationProgress = ReconsolidationProgress()) -> ReconsolidationReport:
    """Run offline with exclusive ownership, saving every independently completed unit."""
    from pm.memory.reconsolidation.pipeline import ReconsolidationPass

    work = ReconsolidationPass(store, complete=complete, embeddings=embeddings, config=config, force=force,
                               cancel=cancel, now=now, prompts=prompts, progress=progress)
    for unit in work.prepare_units():
        if work.publish(unit):
            store.save()
    summary_key = RECEIPT_PREFIX + work.report.run_id
    final = store.begin_batch(summary_key, visible_ids=(), embed=False)
    final.depend_on_commits(work.published)
    final.set_consolidation_receipt(summary_key, work.report.model_dump(mode="json"))  # pm-lint: allow PM007 -- checkpoint receipt serialization
    store.commit(final)
    store.save()
    return work.report


__all__ = ["RECEIPT_PREFIX", "ReconsolidationConfig", "ReconsolidationProgress", "ReconsolidationReport",
           "is_episode_cluster", "reconsolidate", "reconsolidation_due", "unclustered_events"]
