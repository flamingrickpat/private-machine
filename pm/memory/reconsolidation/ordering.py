from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pm.records.base import RecordBase
from pm.records.memory_records import Narrative


@dataclass(frozen=True, order=True, slots=True)
class ChronologicalRecord[T: RecordBase]:
    timestamp: datetime
    record_id: int
    record: T = field(compare=False)


def chronological_records[T: RecordBase](records: Sequence[T]) -> list[T]:
    ranked = [ChronologicalRecord(record.timestamp_world_begin, record.id, record) for record in records]
    return [rank.record for rank in sorted(ranked)]


@dataclass(frozen=True, order=True, slots=True)
class NarrativeVersion:
    refined_tick: int
    timestamp: datetime
    record_id: int
    record: Narrative = field(compare=False)


def newest_narrative_version(records: Sequence[Narrative]) -> Narrative:
    ranked = [NarrativeVersion(record.last_refined_with_tick or -1, record.timestamp_world_begin, record.id, record)
        for record in records]
    return max(ranked).record
