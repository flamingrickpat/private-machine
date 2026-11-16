"""Render private source collections through the shared knoxel interface."""
from collections.abc import Sequence

from pm.context_render import render_context
from pm.context_render.turns import Content, ContextTag
from pm.context_render.budget import optional_content
from pm.context_render.records import RecordEvidence
from workflows.inner_thought.schemas import InnerRequest, ThoughtCandidate
from workflows.inner_thought.request_views import PrivateInspection, PrivateSource


def inspection(evidence: PrivateInspection | None) -> str:
    """Render the selected implementation evidence as compact text.

    The host supplies this evidence from a real source grant and real records.
    A missing or failed inspection becomes an explicit limit, never a guess.
    Episode identifiers are not record IDs and are not citation grants.
    """
    if evidence is None:
        return "(no implementation evidence was supplied for this cycle)"
    if evidence.status != "ok":
        return f"(inspection unavailable: {evidence.reason})"
    lines: list[str] = []
    overlay = evidence.capability_overlay
    for item in overlay.capabilities:
        lines.append(f"- capability={item.name} state={item.state} summary={item.summary!r}")
    for episode in evidence.episodes[:3]:
        lines.append(f"- episode={episode['episode_id']} loop={episode['loop']} "
                     f"operation={episode['operation']} outcome={episode['outcome']} "
                     f"summary={episode['summary'][:140]!r}")
    answers = evidence.newest_episode_answers
    if answers:
        rendered = "; ".join(f"{key}: {value['answer']}" for key, value in answers.items())
        lines.append(f"- newest episode answers: {rendered}")
    return "\n".join(lines) or "(no implementation evidence was supplied for this cycle)"


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
    result = Content("")
    for row, record in zip(rows, records, strict=True):
        result += "\n" + optional_content(render_context(record),
            ContextTag(f"private-candidate:{row.knoxel_id}", row.urgency, record_id=row.knoxel_id,
                       required=row.knoxel_id in chosen, recency=float(row.knoxel_id)))
    return result if records else Content("(none)")


def sources(rows: Sequence[PrivateSource], kind: str) -> Content:
    """Render named source views already validated at the request boundary."""
    records: list[RecordEvidence] = []
    for row in rows:
        records.append(RecordEvidence(knoxel_id=row.knoxel_id, revision=row.revision, tick_id=row.tick_id,
            text=row.text, kind=kind, reference_ids=row.source_ids))
    result = Content("")
    for position, record in enumerate(records):
        result += "\n" + optional_content(render_context(record),
            ContextTag(f"private-{kind}:{record.knoxel_id}", (position + 1) / len(records),
                       record_id=record.knoxel_id, support_ids=tuple(record.reference_ids),
                       required=kind in {"goal", "expectation"}, recency=float(record.knoxel_id)))
    return result if records else Content("(none)")


def request_content(request: InnerRequest, required_ids: frozenset[int] = frozenset()) -> Content:
    """Render the frozen private envelope with selectable original evidence."""
    from workflows.reflection.rendering import context_groups

    result = Content(request.model_dump_json(exclude={"candidates", "recent_thoughts", "reflection"}))
    result += "\nPrivate source candidates:\n" + candidates(request.candidates, required_ids)
    result += "\nRecent private thoughts:\n" + sources(request.recent_thoughts, "private_thought")
    result += "\nPrivate reflection context:\n"
    for group in context_groups(request.reflection, {}):
        result += "\n" + group.name + ":\n"
        result += optional_content(group.content, ContextTag("private-reflection:" + group.name,
            group.salience, record_id=group.source_ids[0] if group.source_ids else None,
            support_ids=group.source_ids, required=group.required, recency=group.recency))
    return result
