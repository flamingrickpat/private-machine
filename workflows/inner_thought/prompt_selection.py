from __future__ import annotations

from dataclasses import dataclass, field

from pm.context_render.budget import TokenEstimator
from pm.context_render.correction import render_regeneration_feedback
from pm.context_render.prompt import EvidenceDecision, RenderedPrompt, RenderSettings, require_rendered_prompt
from pm.context_render.records import RecordEvidence, render_record
from pm.context_render.turns import Role, Turn
from workflows.inner_thought.schemas import InnerRequest


@dataclass(frozen=True, slots=True)
class PrivateEvidenceBlock:
    title: str
    text: str
    priority: float
    source_ids: tuple[int, ...] = ()
    required: bool = False


@dataclass(frozen=True, slots=True, order=True)
class RankedPrivateEvidence:
    negative_priority: float
    input_order: int
    block: PrivateEvidenceBlock = field(compare=False)


@dataclass(frozen=True, slots=True)
class PrivatePromptInput:
    operation: str
    system: str
    required_context: str
    evidence: tuple[PrivateEvidenceBlock, ...]
    correction: str = ""


def select_private_evidence(input: PrivatePromptInput, settings: RenderSettings) -> RenderedPrompt:
    selected: list[str] = [input.required_context]
    decisions: list[EvidenceDecision] = []
    optional: list[PrivateEvidenceBlock] = []
    selected_groups: list[str] = []
    seen: set[tuple[str, str]] = set()
    for block in input.evidence:
        if block.required:
            selected.append(f"{block.title}:\n{block.text}")
            selected_groups.append(block.title)
            seen.add((block.title, block.text))
            for key in block.source_ids:
                decisions.append(EvidenceDecision(key, block.priority, True, "required operation evidence"))
        else:
            optional.append(block)
    if input.correction:
        feedback = render_regeneration_feedback(input.operation, input.correction, settings)
        selected.append(feedback.text)
        settings = settings.without_correction_reserve()
    turns = (Turn(Role.System, input.system), Turn(Role.User, "\n\n".join(selected)))
    require_rendered_prompt(input.operation, turns, settings, tuple(decisions))
    estimator = TokenEstimator(estimate_factor=settings.estimate_factor)
    ranked: list[RankedPrivateEvidence] = []
    for index, block in enumerate(optional):
        ranked.append(RankedPrivateEvidence(-block.priority, index, block))
    ranked.sort()
    for entry in ranked:
        block = entry.block
        if (block.title, block.text) in seen:
            continue
        text = f"{block.title}:\n{block.text}"
        trial = (turns[0], Turn(Role.User, "\n\n".join([*selected, text])))
        admitted = estimator.estimate(trial) <= settings.input_ceiling
        if admitted:
            selected.append(text)
            selected_groups.append(block.title)
            seen.add((block.title, block.text))
            turns = trial
        for key in block.source_ids:
            decisions.append(EvidenceDecision(key, block.priority, admitted,
                "ranked operation evidence" if admitted else "outside operation budget"))
    return require_rendered_prompt(input.operation, turns, settings, tuple(decisions), selected_groups=tuple(selected_groups))


def private_source_blocks(request: InnerRequest, *, required_ids: frozenset[int] = frozenset(),
                          include_thoughts: bool = True) -> tuple[PrivateEvidenceBlock, ...]:
    blocks: list[PrivateEvidenceBlock] = []
    for candidate in request.candidates:
        record = RecordEvidence(knoxel_id=candidate.knoxel_id, revision=candidate.revision,
            tick_id=candidate.tick_id, text=candidate.text, kind=candidate.kind)
        blocks.append(PrivateEvidenceBlock("Candidate evidence", render_record(record).text +
            "\nAttention reason: " + candidate.reason, 10 + candidate.urgency,
            (candidate.knoxel_id,), candidate.knoxel_id in required_ids))
    for goal in request.active_goal_records:
        record = RecordEvidence(knoxel_id=goal.knoxel_id, revision=goal.revision, tick_id=goal.tick_id,
            text=goal.text, kind="goal", reference_ids=goal.source_ids)
        blocks.append(PrivateEvidenceBlock("Active goal", render_record(record).text +
            f"\nFulfilment {goal.fulfilment:.3f}; valence {goal.affective_valence:.3f}",
            8 + abs(goal.affective_valence), (goal.knoxel_id,), goal.knoxel_id in required_ids))
    for expectation in request.pending_expectations:
        record = RecordEvidence(knoxel_id=expectation.knoxel_id, revision=expectation.revision,
            tick_id=expectation.tick_id, text=expectation.text, kind="expectation", reference_ids=expectation.source_ids)
        blocks.append(PrivateEvidenceBlock("Pending expectation", render_record(record).text +
            "\nAnticipation: " + expectation.anticipation + "\nExpected outcome: " + expectation.expected_outcome,
            8, (expectation.knoxel_id,), expectation.knoxel_id in required_ids))
    if include_thoughts:
        for index, thought in enumerate(reversed(request.recent_thoughts)):
            record = RecordEvidence(knoxel_id=thought.knoxel_id, revision=thought.revision,
                tick_id=thought.tick_id, text=thought.text, kind="private_thought", reference_ids=thought.source_ids)
            blocks.append(PrivateEvidenceBlock("Earlier private thought; not independent corroboration",
                render_record(record).text, 4 / (index + 1), (thought.knoxel_id,), thought.knoxel_id in required_ids))
    return tuple(blocks)


def private_frame(request: InnerRequest) -> str:
    frame = request.main_frame
    lines = [f"Companion: {request.companion_name}; partner: {request.user_name}",
        "Character:\n" + request.character,
        f"Current accepted frame [{frame.frame_id}]: {frame.summary}",
        "Current internal state: " + request.state_summary]
    if request.context_limits:
        lines.append("Evidence limits: " + "; ".join(request.context_limits))
    if request.seconds_since_user_input is not None:
        lines.append(f"Seconds since user input: {request.seconds_since_user_input:.0f}")
    episode = request.episode
    if episode.state is not None:
        state = episode.state
        lines.append(f"Retained linear episode {state.episode_id}; step {state.sequence}; stalled {state.stalled}; "
            f"closed {state.closed}; awaiting acknowledgment {episode.awaiting_acknowledgment}; "
            f"steps remaining {episode.steps_remaining}.\nContinuation: {state.continuation}; next operation {state.next_operation}.")
    return "\n\n".join(lines)
