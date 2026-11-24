from __future__ import annotations

from dataclasses import dataclass, field, replace

from pm.context_render.budget import evidence_excerpt, TokenEstimator
from pm.context_render.prompt import EvidenceDecision, RenderedPrompt, RenderSettings, require_rendered_prompt
from pm.context_render.story import RecordView
from pm.context_render.turns import format_content
from workflows.main_cognition.prompt_values import AttentionTemplateValues
from workflows.main_cognition.rendering import (
    CognitionStoryInput, cognition_story_turns, render_cognition_guidance, render_cognition_story, fit_story_frame,
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


def attention_frame(input: AttentionRenderInput, selection: AttentionSelection,
                    member_excerpt_characters: int | None = None) -> CognitionStoryInput:
    blocks: list[str] = []
    for structure in selection.structures:
        members = [str(record_id) for record_id in structure.record_ids if record_id in selection.record_ids]
        theme = structure.theme
        reason = structure.reason
        if member_excerpt_characters is not None:
            theme = evidence_excerpt(theme, member_excerpt_characters)
            reason = evidence_excerpt(reason, member_excerpt_characters)
        blocks.append(f"### {structure.label}: {theme}\nWhy: {reason}\n"
                      f"Shown member sources: {', '.join(members)}")
    blocks.append("Member evidence (each source appears once):")
    for record_id in sorted(selection.record_ids):
        record = input.records[record_id]
        content = record.content
        if member_excerpt_characters is not None:
            content = evidence_excerpt(content, member_excerpt_characters)
        blocks.append(f"[source {record_id}; {record.record_class}; {record.source}] {content}")
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
    excerpt_characters: int | None = None
    selected = AttentionSelection()
    frame = attention_frame(input, selected, excerpt_characters)
    turns = cognition_story_turns(frame, frozenset(required_story))
    relax_pins = bool(required_story) and estimator.estimate(turns) > settings.input_ceiling
    if relax_pins:
        required_story.clear()
        turns = cognition_story_turns(frame, frozenset())
    if estimator.estimate(turns) > settings.input_ceiling:
        retained = max(len(input.attention_narrative), len(input.mental_state), len(input.shard_focus), len(input.aperture))
        original = input
        while estimator.estimate(turns) > settings.input_ceiling and retained:
            retained //= 2
            input = replace(original, attention_narrative=evidence_excerpt(original.attention_narrative, retained),
                mental_state=evidence_excerpt(original.mental_state, retained),
                shard_focus=evidence_excerpt(original.shard_focus, retained), aperture=evidence_excerpt(original.aperture, retained))
            frame = attention_frame(input, selected)
            turns = cognition_story_turns(frame, frozenset(required_story))
        frame = fit_story_frame(frame, settings)
        input = replace(input, story=frame)
        turns = cognition_story_turns(frame, frozenset(required_story))
    baseline = require_rendered_prompt("attention instructions", turns, settings)
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
        trial_frame = attention_frame(input, trial, excerpt_characters)
        estimated = estimator.estimate(cognition_story_turns(trial_frame, frozenset(required_story)))
        included = estimated <= candidate_ceiling or (not selected.structures and estimated <= settings.input_ceiling)
        if included:
            selected = trial
            signatures.add(signature)
        decisions.append(EvidenceDecision(structure.structure_id, structure.priority, included,
            "shard relevance and salience" if included else "attention candidate allocation"))
    if input.structures and not selected.structures:
        # Keep the strongest structure and its actual source identity even
        # when a single member is larger than the entire candidate lane.
        strongest = ranked[0].structure
        members = sorted(AttentionMember(-input.record_priority[key], key) for key in strongest.record_ids)
        record_id = members[0].record_id
        selected = AttentionSelection((strongest,), frozenset((record_id,)))
        retained = min(len(input.records[record_id].content), settings.input_ceiling // 4)
        while True:
            excerpt_characters = retained
            frame = attention_frame(input, selected, excerpt_characters)
            if estimator.estimate(cognition_story_turns(frame, frozenset())) <= settings.input_ceiling or retained == 0:
                break
            retained //= 2
        required_story.clear()
        decisions = [row for row in decisions if row.record_id != strongest.structure_id]
        decisions.append(EvidenceDecision(strongest.structure_id, strongest.priority, True, "source excerpt"))
        decisions.append(EvidenceDecision(record_id, input.record_priority[record_id], True, "source excerpt"))
    member_ids = {record_id for structure in selected.structures for record_id in structure.record_ids}
    ranked_members = sorted(AttentionMember(-input.record_priority[record_id], record_id) for record_id in member_ids)
    for member in ranked_members:
        if member.record_id in selected.record_ids:
            continue
        trial = AttentionSelection(selected.structures, selected.record_ids | {member.record_id})
        trial_frame = attention_frame(input, trial, excerpt_characters)
        included = estimator.estimate(cognition_story_turns(trial_frame, frozenset(required_story))) <= candidate_ceiling
        if included:
            selected = trial
        decisions.append(EvidenceDecision(member.record_id, -member.negative_priority, included,
            "attention supporting source" if included else "attention evidence allocation"))
    frame = attention_frame(input, selected, excerpt_characters)
    rendered = render_cognition_story(frame, settings)
    groups = rendered.selected_groups
    if relax_pins and not groups:
        groups = ("workspace evidence pins relaxed",)
    prompt = require_rendered_prompt("attention rating", rendered.turns, settings, (*rendered.evidence, *decisions),
        selected_groups=(*groups, *(structure.label for structure in selected.structures)))
    return AttentionPrompt(prompt, selected)
