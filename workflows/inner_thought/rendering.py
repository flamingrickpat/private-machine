"""Render private source collections through the shared knoxel interface."""
from __future__ import annotations
from collections.abc import Sequence

from pm.context_render import render_context
from pm.context_render.turns import Content
from pm.context_render.records import DecisionEvidence, RecordEvidence
from workflows.inner_thought.schemas import ThoughtCandidate
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
    lines: list[str] = [evidence.runtime_orientation] if evidence.runtime_orientation else []
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
    return render_context(DecisionEvidence(records)) if records else Content("(none)")


def sources(rows: Sequence[PrivateSource], kind: str) -> Content:
    """Render named source views already validated at the request boundary."""
    records: list[RecordEvidence] = []
    for row in rows:
        records.append(RecordEvidence(knoxel_id=row.knoxel_id, revision=row.revision, tick_id=row.tick_id,
            text=row.text, kind=kind, reference_ids=row.source_ids))
    return render_context(DecisionEvidence(tuple(records))) if records else Content("(none)")


from typing import TYPE_CHECKING
from pm.context_render.prompt import RenderSettings, RenderedPrompt
if TYPE_CHECKING:
    from workflows.inner_thought.stages.nominate.steps import NominationRenderInput
    from workflows.inner_thought.stages.trigger.steps import TriggerRenderInput, AttentionRenderInput
    from workflows.inner_thought.stages.thought.steps import ThoughtRenderInput, GraphRenderInput
    from workflows.inner_thought.stages.integrate.steps import IntegrationRenderInput
    from workflows.inner_thought.stages.reflection.steps import ReflectionRenderInput


def replay_private_capture(input: NominationRenderInput | TriggerRenderInput | AttentionRenderInput |
                           ThoughtRenderInput | GraphRenderInput | IntegrationRenderInput | ReflectionRenderInput,
                           settings: RenderSettings) -> RenderedPrompt:
    from workflows.inner_thought.stages.nominate.steps import NominationRenderInput, render_nomination
    from workflows.inner_thought.stages.trigger.steps import TriggerRenderInput, AttentionRenderInput, render_trigger, render_private_attention
    from workflows.inner_thought.stages.thought.steps import ThoughtRenderInput, GraphRenderInput, render_thought, render_graph_thought
    from workflows.inner_thought.stages.integrate.steps import IntegrationRenderInput, render_integration
    from workflows.inner_thought.stages.reflection.steps import ReflectionRenderInput, render_reflection
    match input:
        case NominationRenderInput():
            return render_nomination(input.request, input.prompts, settings, input.correction)
        case TriggerRenderInput():
            return render_trigger(input.request, input.prompts, settings, input.correction)
        case AttentionRenderInput():
            return render_private_attention(input.request, input.prompts, settings, input.correction)
        case ThoughtRenderInput():
            return render_thought(input.request, input.prompts, settings, input.correction)
        case GraphRenderInput():
            return render_graph_thought(input.request, settings, input.correction)
        case IntegrationRenderInput():
            return render_integration(input.request, input.prompts, settings, input.correction)
        case ReflectionRenderInput():
            return render_reflection(input.call, input.task, input.schema, settings, input.correction)
