from __future__ import annotations

from dataclasses import dataclass, field, replace

from pm.context_render.budget import RequiredContextTooLarge, TokenEstimator
from pm.context_render.prompt import EvidenceDecision, RenderedPrompt, RenderSettings, require_rendered_prompt
from pm.context_render.story import RecordView
from pm.context_render.turns import format_content
from workflows.main_cognition.prompt_values import AttentionTemplateValues
from workflows.main_cognition.rendering import (
    CognitionStoryInput, cognition_story_turns, render_cognition_guidance, render_cognition_story,
)


@dataclass(frozen=True, slots=True)
class AttentionStructure:
    structure_id: int
    label: str
    theme: str
    reason: str
    record_ids: tuple[int, ...]
    priority: float


@dataclass(order=True, frozen=True, slots=True)
class RankedAttentionStructure:
    negative_priority: float
    position: int
    structure: AttentionStructure = field(compare=False)


@dataclass(order=True, frozen=True, slots=True)
class AttentionMember:
    negative_priority: float
    record_id: int


@dataclass(frozen=True, slots=True)
class AttentionRenderInput:
    story: CognitionStoryInput
    template: str
    attention_narrative: str
    mental_state: str
    shard_focus: str
    aperture: str
    winners: int
    structures: tuple[AttentionStructure, ...]
    records: dict[int, RecordView]
    record_priority: dict[int, float]


@dataclass(frozen=True, slots=True)
class AttentionSelection:
    structures: tuple[AttentionStructure, ...] = ()
    record_ids: frozenset[int] = frozenset()


@dataclass(frozen=True, slots=True)
class AttentionPrompt:
    prompt: RenderedPrompt
    selection: AttentionSelection


def attention_frame(input: AttentionRenderInput, selection: AttentionSelection) -> CognitionStoryInput:
    blocks: list[str] = []
    for structure in selection.structures:
        members = [str(record_id) for record_id in structure.record_ids if record_id in selection.record_ids]
        blocks.append(f"### {structure.label}: {structure.theme}\nWhy: {structure.reason}\n"
                      f"Shown member sources: {', '.join(members)}")
    blocks.append("Member evidence (each source appears once):")
    for record_id in sorted(selection.record_ids):
        record = input.records[record_id]
        blocks.append(f"[source {record_id}; {record.record_class}; {record.source}] {record.content}")
    instruction = format_content(input.template, AttentionTemplateValues(
        attention_narrative=input.attention_narrative, mental_state=input.mental_state,
        shard_focus=input.shard_focus, structures="\n\n".join(blocks), aperture=input.aperture,
        winners=str(min(input.winners, len(selection.structures))), count=str(len(selection.structures)))).text
    return replace(input.story, operation="attention rating", instruction=instruction)


def render_attention(input: AttentionRenderInput, settings: RenderSettings) -> AttentionPrompt:
    story = input.story
    if story.situations:
        story = replace(story, guidance=render_cognition_guidance(list(story.situations), settings), situations=())
    input = replace(input, story=story)
    required_story: set[int] = set()
    for shard in story.shards:
        if shard.key.kind == "entity":
            required_story.update(shard.tail_ids[-2:])
    required_story.difference_update(story.new_ids)
    estimator = TokenEstimator(estimate_factor=settings.estimate_factor)
    selected = AttentionSelection()
    frame = attention_frame(input, selected)
    baseline = require_rendered_prompt("attention instructions", cognition_story_turns(frame, frozenset(required_story)), settings)
    candidate_ceiling = baseline.estimated_tokens + (settings.input_ceiling - baseline.estimated_tokens) * 3 // 4
    ranked = sorted(RankedAttentionStructure(-item.priority, index, item)
                    for index, item in enumerate(input.structures))
    decisions: list[EvidenceDecision] = []
    signatures: set[tuple[str, frozenset[int]]] = set()
    for candidate in ranked:
        structure = candidate.structure
        signature = (structure.theme.strip(), frozenset(structure.record_ids))
        if signature in signatures:
            decisions.append(EvidenceDecision(structure.structure_id, structure.priority, False, "duplicate structure"))
            continue
        members = sorted(AttentionMember(-input.record_priority[record_id], record_id)
                         for record_id in structure.record_ids)
        if not members:
            raise ValueError(f"Attention structure {structure.structure_id} has no member evidence.")
        trial = AttentionSelection((*selected.structures, structure), selected.record_ids | {members[0].record_id})
        trial_frame = attention_frame(input, trial)
        estimated = estimator.estimate(cognition_story_turns(trial_frame, frozenset(required_story)))
        included = estimated <= candidate_ceiling or (not selected.structures and estimated <= settings.input_ceiling)
        if included:
            selected = trial
            signatures.add(signature)
        decisions.append(EvidenceDecision(structure.structure_id, structure.priority, included,
            "shard relevance and salience" if included else "attention candidate allocation"))
    if input.structures and not selected.structures:
        raise RequiredContextTooLarge("Attention cannot represent one candidate with its member evidence inside the candidate allocation.")
    member_ids = {record_id for structure in selected.structures for record_id in structure.record_ids}
    ranked_members = sorted(AttentionMember(-input.record_priority[record_id], record_id) for record_id in member_ids)
    for member in ranked_members:
        if member.record_id in selected.record_ids:
            continue
        trial = AttentionSelection(selected.structures, selected.record_ids | {member.record_id})
        trial_frame = attention_frame(input, trial)
        included = estimator.estimate(cognition_story_turns(trial_frame, frozenset(required_story))) <= candidate_ceiling
        if included:
            selected = trial
        decisions.append(EvidenceDecision(member.record_id, -member.negative_priority, included,
            "attention supporting source" if included else "attention evidence allocation"))
    frame = attention_frame(input, selected)
    rendered = render_cognition_story(frame, settings)
    prompt = require_rendered_prompt("attention rating", rendered.turns, settings, (*rendered.evidence, *decisions),
        selected_groups=tuple(structure.label for structure in selected.structures))
    return AttentionPrompt(prompt, selected)
