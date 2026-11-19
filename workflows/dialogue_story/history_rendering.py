from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from collections.abc import Sequence

from pm.context_render.budget import TokenEstimator
from pm.context_render.images import ImageCatalog, resolve_media
from pm.context_render.turns import Role, Turn
from pm.memory.temporal_context import (TemporalCoverageProblem, TemporalHistoryInput, TemporalHistoryItem,
    TemporalCoverageSelection, TemporalSelectionSettings)
from pm.records.enums import FeatureType
from workflows.dialogue_story.schemas import StoryRequest, StoryPrompts
from workflows.public_speech.public_context import DialogueTurn, EvidenceItem, PublicContext, order_dialogue, plain_memory_text
from workflows.public_speech.story_rendering import dialogue_lines


@dataclass(frozen=True, slots=True)
class StoryHistorySelection:
    request: StoryRequest
    turns: tuple[Turn, ...]
    coverage: TemporalCoverageSelection
    mandatory_tokens: int
    minimum_rendered_tokens: int
    rendered_tokens: int
    refinement_attempts: int


@dataclass(order=True, frozen=True, slots=True)
class ChronologicalHistoryUnit:
    started_at: datetime
    record_id: int
    item: TemporalHistoryItem = field(compare=False)


def history_evidence(item: TemporalHistoryItem) -> EvidenceItem:
    private = item.feature_type in {FeatureType.Thought, FeatureType.InnerMonologue}
    if item.kind == "summary":
        kind = "interpretation"
    elif private:
        kind = "speculation"
    elif item.evidence_role == "internal_state":
        kind = "internal_state"
    elif item.feature_type == FeatureType.Dialogue:
        kind = "utterance"
    else:
        kind = "interpretation" if item.evidence_role == "interpretation" else "observation"
    moment = item.started_at.strftime("%d %b %Y %H:%M")
    if item.kind == "summary":
        ending = item.ended_at.strftime("%d %b %Y %H:%M")
        text = f"Temporal summary ({moment} to {ending}): {item.content}"
    elif item.feature_type == FeatureType.Dialogue:
        text = f"{item.source} said: {item.content}"
    else:
        private_label = "private " if item.evidence_role in {"interpretation", "internal_state"} else ""
        feature_name = item.feature_type.value if item.feature_type is not None else "experience"
        text = f"Earlier {private_label}{feature_name} ({moment}): {item.content}"
    return EvidenceItem(knoxel_id=item.record_id, text=text, memory_kind=kind,
        source_ids=list(item.source_ids), timestamp=item.started_at.isoformat(),
        evidence_role=item.evidence_role or ("derived" if item.kind == "summary" else "interpretation"), lane="history")


def history_review_context(context: PublicContext, items: tuple[TemporalHistoryItem, ...]) -> PublicContext:
    selected = context.model_copy()
    evidence = {item.knoxel_id: item for item in context.evidence}
    for item in items:
        evidence[item.record_id] = history_evidence(item)
    selected.evidence = list(evidence.values())
    selected.temporal_history = None
    selected.memories = []
    return selected


def history_request(request: StoryRequest, items: tuple[TemporalHistoryItem, ...]) -> StoryRequest:
    context = request.public_context
    if context is None:
        raise ValueError("Full temporal story history requires its public context.")
    selected = request.model_copy()
    selected_context = context.model_copy()
    dialogue = list(request.prior_dialogue)
    evidence = list(context.evidence)
    shown = {item.knoxel_id for item in evidence} | {turn.knoxel_id for turn in dialogue}
    for item in items:
        if item.record_id in shown:
            continue
        if item.kind == "feature" and item.feature_type == FeatureType.Dialogue:
            dialogue.append(DialogueTurn(speaker=item.source, text=item.content, knoxel_id=item.record_id,
                timestamp=item.started_at.isoformat()))
        else:
            evidence.append(history_evidence(item))
        shown.add(item.record_id)
    selected.prior_dialogue = order_dialogue(dialogue)
    selected_context.evidence = evidence
    selected_context.memories = []
    selected.public_context = selected_context
    return selected


def history_turns(request: StoryRequest, prompts: StoryPrompts, repairs: Sequence[Turn]) -> tuple[Turn, ...]:
    from workflows.dialogue_story.steps import build_story_messages
    original = resolve_media(tuple(build_story_messages(request, prompts)), ImageCatalog(request.images))
    return (*original, *repairs)


def select_story_history(request: StoryRequest, prompts: StoryPrompts, repairs: Sequence[Turn],
                         estimator: TokenEstimator) -> StoryHistorySelection:
    context = request.public_context
    if context is None or context.temporal_history is None or request.input_budget_tokens is None:
        raise ValueError("Story temporal selection requires its frozen lifetime catalog and actual input ceiling.")
    baseline_turns = history_turns(request, prompts, repairs)
    baseline_tokens = estimator.estimate(baseline_turns)
    represented = {item.knoxel_id for item in context.evidence} | {turn.knoxel_id for turn in request.prior_dialogue}
    mandatory_times = [item.started_at for item in context.temporal_history.items
                       if item.kind == "feature" and item.record_id in represented]
    priced: list[TemporalHistoryItem] = []
    images = ImageCatalog(request.images)
    for item in context.temporal_history.items:
        if item.kind == "summary" and any(item.started_at <= moment <= item.ended_at for moment in mandatory_times):
            continue
        if item.record_id in represented:
            cost = 0
        elif item.kind == "feature" and item.feature_type == FeatureType.Dialogue:
            line = dialogue_lines([DialogueTurn(speaker=item.source, text=item.content,
                knoxel_id=item.record_id, timestamp=item.started_at.isoformat())], prompts)[0]
            cost = estimator.estimate(resolve_media((Turn(Role.Assistant, line.text),), images)) - estimator.estimate(())
        else:
            cost = estimator.estimate(resolve_media((Turn(Role.User, "- " + plain_memory_text(history_evidence(item).text)),), images)) - estimator.estimate(())
        priced.append(TemporalHistoryItem(record_id=item.record_id, started_at=item.started_at,
            ended_at=item.ended_at, kind=item.kind, level=item.level, content=item.content, source=item.source,
            feature_type=item.feature_type, evidence_role=item.evidence_role, source_ids=item.source_ids, relevance=item.relevance, token_cost=cost))
    problem = TemporalCoverageProblem(TemporalHistoryInput(items=tuple(priced)))
    minimum = problem.minimum_cover()
    minimum_request = history_request(request, minimum.items)
    minimum_turns = history_turns(minimum_request, prompts, repairs)
    minimum_tokens = estimator.estimate(minimum_turns)
    whole_life_fits = minimum_tokens <= request.input_budget_tokens
    token_budget = max(0, minimum.total_tokens + request.input_budget_tokens - minimum_tokens)
    attempts = 0
    while True:
        attempts += 1
        if whole_life_fits and token_budget <= minimum.total_tokens:
            coverage = minimum
            selected, turns, tokens = minimum_request, minimum_turns, minimum_tokens
        else:
            coverage = problem.select(TemporalSelectionSettings(max_tokens=token_budget,
                allow_coverage_omissions=not whole_life_fits))
            selected = history_request(request, coverage.items)
            turns = history_turns(selected, prompts, repairs)
            tokens = estimator.estimate(turns)
        if tokens <= request.input_budget_tokens:
            break
        token_budget = max(0, token_budget - (tokens - request.input_budget_tokens))
    if not whole_life_fits:
        coarse_by_id = {item.record_id: item for item in minimum.items}
        for record_id in reversed(coverage.omitted_record_ids):
            trial_items = (*coverage.items, coarse_by_id[record_id])
            ordered = sorted(ChronologicalHistoryUnit(item.started_at, item.record_id, item) for item in trial_items)
            trial_items = tuple(unit.item for unit in ordered)
            trial_request = history_request(request, trial_items)
            trial_turns = history_turns(trial_request, prompts, repairs)
            trial_tokens = estimator.estimate(trial_turns)
            attempts += 1
            if trial_tokens <= request.input_budget_tokens:
                selected_ids = {item.record_id for item in trial_items}
                omitted_ids = tuple(item.record_id for item in minimum.items if item.record_id not in selected_ids)
                coverage = problem.selection(trial_items, omitted_ids)
                selected, turns, tokens = trial_request, trial_turns, trial_tokens
    if whole_life_fits and coverage.omitted_feature_ids:
        raise ValueError("Story omitted lifetime history although its complete minimum cover fits.")
    return StoryHistorySelection(selected, turns, coverage, baseline_tokens, minimum_tokens, tokens, attempts)
