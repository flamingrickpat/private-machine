"""Memory reconsolidation: topical and temporal clusters, facts, and narratives.

Port of the memory consolidation of private-machine-head
(``DynamicMemoryConsolidator``) onto the current record classes. Read
``docs/subsystems/character-import.md`` for the design and the mapping to the
old code. :func:`reconsolidate` is the entry point. The clustering and the
time ranges are pure functions in :mod:`.clustering` and :mod:`.temporal`.
"""
from pm.memory.reconsolidation.pipeline import (RECEIPT_PREFIX, is_episode_cluster, reconsolidate,
                                                reconsolidation_due, unclustered_events)
from pm.memory.reconsolidation.schemas import ReconsolidationConfig, ReconsolidationReport

__all__ = ["RECEIPT_PREFIX", "ReconsolidationConfig", "ReconsolidationReport", "is_episode_cluster",
           "reconsolidate", "reconsolidation_due", "unclustered_events"]
