from __future__ import annotations

from dataclasses import dataclass, replace

from pm.context_render.budget import TokenEstimator
from pm.context_render.prompt import EvidenceDecision, RenderedPrompt, RenderSettings, require_rendered_prompt
from pm.context_render.story import RecordView, story_line
from pm.context_render.turns import Role, Turn, format_content
from pm.memory.retrieval import distinctive_terms
from workflows.main_cognition.prompt_values import ExperienceTemplateValues
from workflows.main_cognition.rendering import CognitionStoryInput, cognition_story_turns, render_cognition_guidance


@dataclass(frozen=True, slots=True)
class ExperienceRenderInput:
    story: CognitionStoryInput
    instruction_template: str
    seed: str
    entered_ids: tuple[int, ...]
    records: dict[int, RecordView]
    priorities: dict[int, float]


@dataclass(order=True, frozen=True, slots=True)
class ExperienceEvidenceRank:
    negative_priority: float
    record_id: int


@dataclass(frozen=True, slots=True)
class ExperienceSelection:
    entered_ids: frozenset[int]
    story_ids: frozenset[int]


def experience_turns(input: ExperienceRenderInput, selection: ExperienceSelection) -> tuple[Turn, ...]:
    entered = "\n".join(story_line(input.records[record_id], with_id=True)
                        for record_id in sorted(selection.entered_ids)) or "(nothing newly selected)"
    instruction = format_content(input.instruction_template, ExperienceTemplateValues(entered=entered)).text
    frame = replace(input.story, operation="subjective experience", instruction=instruction)
    turns = cognition_story_turns(frame, selection.story_ids)
    if input.seed:
        turns = (*turns, Turn(Role.Assistant, input.seed))
    return turns


def render_experience(input: ExperienceRenderInput, settings: RenderSettings) -> RenderedPrompt:
    story = input.story
    if story.situations:
        story = replace(story, guidance=render_cognition_guidance(list(story.situations), settings), situations=())
        input = replace(input, story=story)
    required: set[int] = set()
    optional: set[int] = set()
    for shard in story.shards:
        if shard.key.kind == "entity":
            required.update(shard.tail_ids[-2:])
        optional.update(shard.item_ids)
        optional.update(shard.tail_ids)
    required.difference_update(story.new_ids)
    selected = ExperienceSelection(frozenset(), frozenset(required))
    estimator = TokenEstimator(estimate_factor=settings.estimate_factor)
    turns = experience_turns(input, selected)
    relax_pins = bool(required) and estimator.estimate(turns) > settings.input_ceiling
    if relax_pins:
        required.clear()
        selected = ExperienceSelection(frozenset(), frozenset())
        turns = experience_turns(input, selected)
    baseline = require_rendered_prompt("subjective experience", turns, settings)
    entered_ceiling = baseline.estimated_tokens + (settings.input_ceiling - baseline.estimated_tokens) * 2 // 3
    decisions = [EvidenceDecision(record_id, input.priorities[record_id], True, "mandatory current experience")
                 for record_id in sorted(required | set(story.new_ids))]
    seen = {input.records[record_id].content.strip() for record_id in required | set(story.new_ids)}
    ranks = sorted(ExperienceEvidenceRank(-input.priorities[record_id], record_id) for record_id in input.entered_ids)
    for rank in ranks:
        record_id = rank.record_id
        text = input.records[record_id].content.strip()
        if text in seen:
            decisions.append(EvidenceDecision(record_id, -rank.negative_priority, False, "already shown current evidence"))
            continue
        trial = ExperienceSelection(selected.entered_ids | {record_id}, selected.story_ids - {record_id})
        included = estimator.estimate(experience_turns(input, trial)) <= entered_ceiling
        if included:
            selected = trial
            seen.add(text)
        decisions.append(EvidenceDecision(record_id, -rank.negative_priority, included,
            "newly conscious salience" if included else "new awareness allocation"))
    focus_terms = set(distinctive_terms(story.new_events + " " + story.feeling))
    ranks = sorted(ExperienceEvidenceRank(-input.priorities[record_id], record_id) for record_id in optional)
    for rank in ranks:
        record_id = rank.record_id
        if record_id in required or record_id in selected.entered_ids or record_id in story.new_ids:
            continue
        record = input.records[record_id]
        if record.content.strip() in seen:
            decisions.append(EvidenceDecision(record_id, -rank.negative_priority, False, "duplicate experience evidence"))
            continue
        if not focus_terms.intersection(distinctive_terms(record.content)):
            decisions.append(EvidenceDecision(record_id, -rank.negative_priority, False, "unrelated to current event or feeling"))
            continue
        trial = ExperienceSelection(selected.entered_ids, selected.story_ids | {record_id})
        included = estimator.estimate(experience_turns(input, trial)) <= settings.input_ceiling
        if included:
            selected = trial
            seen.add(record.content.strip())
        decisions.append(EvidenceDecision(record_id, -rank.negative_priority, included,
            "current experience relevance" if included else "experience context allocation"))
    groups = ("workspace evidence pins relaxed",) if relax_pins else ()
    return require_rendered_prompt("subjective experience", experience_turns(input, selected), settings,
        tuple(decisions), selected_groups=(*groups, *(str(record_id) for record_id in sorted(selected.entered_ids))))
