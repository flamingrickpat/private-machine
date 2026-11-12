"""Render private source collections through the shared knoxel interface."""
from collections.abc import Sequence

from pm.context_render import render_context
from pm.context_render.turns import Content
from pm.context_render.records import DecisionEvidence, RecordEvidence
from workflows.inner_thought.schemas import ThoughtCandidate
from workflows.inner_thought.request_views import PrivateSource


class CandidateEvidence(RecordEvidence):
    """The existing urgency and reason accompany original candidate content."""

    urgency: float
    reason: str
    selected: bool = False


def candidates(rows: Sequence[ThoughtCandidate], chosen: frozenset[int] = frozenset()) -> Content:
    """Preserve candidate IDs, text, and workflow signals through central rendering."""
    records = tuple(CandidateEvidence(knoxel_id=row.knoxel_id, text=row.text, revision=row.revision,
        tick_id=row.tick_id, kind=row.kind, urgency=row.urgency, reason=row.reason,
        selected=row.knoxel_id in chosen) for row in rows)
    return render_context(DecisionEvidence(records)) if records else Content("(none)")


def sources(rows: Sequence[PrivateSource], kind: str) -> Content:
    """Render named source views already validated at the request boundary."""
    records: list[RecordEvidence] = []
    for row in rows:
        records.append(RecordEvidence(knoxel_id=row.knoxel_id, revision=row.revision, tick_id=row.tick_id,
            text=row.text, kind=kind, reference_ids=row.source_ids))
    return render_context(DecisionEvidence(tuple(records))) if records else Content("(none)")
