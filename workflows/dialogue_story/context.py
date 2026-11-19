"""Prepare standalone Story and fit runtime Story prompts to their input ceiling.

The operator's situation supplies one accepted user input in the trial MemoryStore.
The ``story`` context plan reads vectors from the shared records. Capability discovery
runs outside the async loop. The companion's earlier private thoughts can
appear in the verbatim transcript as ``thinks: *...*`` lines; they guide the
voice and are never published.

``select_story_prompt`` also serves ordinary runtime replies. It preserves the
required Story layout while the shared selector admits optional retrieval by
score. The returned context tells the reviewer exactly which evidence survived.
Runtime ownership is described in [the Story host overview][workflows.dialogue_story.host].
"""
from __future__ import annotations

from workflows.public_speech.public_context import rank_public_evidence, rank_memory_texts

from pm import vclock

from collections.abc import Sequence
from dataclasses import dataclass, replace
from threading import Event
from typing import TYPE_CHECKING

from pm.memory.embeddings import EmbeddingService
from pm.memory.retrieval import distinctive_terms
from workflows.dialogue_story.history_rendering import select_story_history
from pm.memory.temporal_context import TemporalCoverageSelection

from pm.context_render.budget import PromptSelection, TokenEstimator, RequiredContextTooLarge
from pm.context_render.turns import Turn
from pm.context_render.images import ImageCatalog, resolve_media
from pm.context_render.prompt import RenderSettings, RenderedPrompt, EvidenceDecision, TemporalHistoryMeasurement, require_rendered_prompt

from workflows.dialogue_story.schemas import StoryPrompts, StoryRequest, StorySituation

if TYPE_CHECKING:
    from pm.memory.store import MemoryStore


@dataclass(frozen=True, slots=True)
class StoryPrompt:
    """One fully rendered prompt and the exact evidence supplied to its reviewer."""

    request: StoryRequest
    selection: PromptSelection
    history: TemporalCoverageSelection | None = None
    history_measurement: TemporalHistoryMeasurement | None = None


@dataclass(frozen=True, slots=True)
class StoryRenderInput:
    request: StoryRequest
    prompts: StoryPrompts
    repair_turns: tuple[Turn, ...] = ()


def render_story(input: StoryRenderInput, settings: RenderSettings) -> RenderedPrompt:
    request = input.request.model_copy(deep=True)
    request.input_budget_tokens = settings.input_ceiling
    planned = select_story_prompt(request, input.prompts, input.repair_turns, settings=settings)
    selected_ids = set(planned.selection.source_ids)
    evidence = input.request.public_context.evidence if input.request.public_context is not None else []
    decisions = tuple(EvidenceDecision(item.knoxel_id, item.score, item.knoxel_id in selected_ids,
                      "story lane selection" if item.knoxel_id in selected_ids else "budget or duplicate") for item in evidence)
    decisions += tuple(EvidenceDecision(turn.knoxel_id, 0.0, turn.knoxel_id in selected_ids,
                       "current exchange or distinct relevant example" if turn.knoxel_id in selected_ids else "unrelated, redundant, or dialogue budget")
                       for turn in input.request.prior_dialogue if turn.knoxel_id is not None)
    if planned.history is not None:
        decisions += tuple(EvidenceDecision(item.record_id, item.relevance, True, "full lifetime cover and detail refinement")
                           for item in planned.history.items)
        decisions += tuple(EvidenceDecision(record_id, 0.0, False, "coarsest lifetime cover exceeds actual prompt headroom")
                           for record_id in planned.history.omitted_feature_ids)
    rendered = require_rendered_prompt("story generation", planned.selection.turns, settings, decisions,
        selected_groups=planned.selection.included)
    return replace(rendered, temporal_history=planned.history, temporal_history_measurement=planned.history_measurement)


def select_story_prompt(
    request: StoryRequest,
    prompts: StoryPrompts,
    repair_turns: Sequence[Turn] = (),
    settings: RenderSettings | None = None,
) -> StoryPrompt:
    from workflows.dialogue_story.steps import build_story_messages

    if request.input_budget_tokens is None:
        raise ValueError("Story rendering requires an explicit input ceiling.")
    estimator = TokenEstimator(estimate_factor=settings.estimate_factor if settings is not None else 1.5)
    context = request.public_context
    images = ImageCatalog(request.images)
    evidence = list(context.evidence) if context is not None else []
    required_ids = set(request.required_evidence_ids)
    if request.current_message_id is not None:
        required_ids.add(request.current_message_id)
    speech_history = [turn for turn in request.prior_dialogue if turn.kind == "speech"]
    required_dialogue = speech_history[-2:]
    if request.guidance is not None:
        for turn in speech_history:
            if (turn.speaker == request.companion_name and turn.knoxel_id is not None
                    and turn.knoxel_id > request.guidance.through_record_id):
                if turn not in required_dialogue:
                    required_dialogue = [turn, *required_dialogue]
                break
    required_ids.update(turn.knoxel_id for turn in required_dialogue if turn.knoxel_id is not None)
    by_id = {item.knoxel_id: item for item in evidence}
    pending = sorted(required_ids)
    while pending:
        record_id = pending.pop()
        if record_id not in by_id:
            continue
        for source_id in by_id[record_id].source_ids:
            if source_id in by_id and source_id not in required_ids:
                required_ids.add(source_id)
                pending.append(source_id)
    selected_evidence = [item for item in evidence if item.knoxel_id in required_ids]
    selected_dialogue = list(required_dialogue)
    selected_context = context.model_copy(deep=True) if context is not None else None
    if selected_context is not None:
        selected_context.evidence = selected_evidence
        selected_context.memories = []
    selected = request.model_copy(deep=True)
    selected.prior_dialogue = selected_dialogue
    selected.public_context = selected_context
    selected.memories = []
    turns = resolve_media((*build_story_messages(selected, prompts), *repair_turns), images)
    mandatory_tokens = estimator.estimate(turns)
    if mandatory_tokens > request.input_budget_tokens:
        raise RequiredContextTooLarge(f"Story mandatory input uses {mandatory_tokens} estimated tokens; ceiling {request.input_budget_tokens}.")
    history: TemporalCoverageSelection | None = None
    history_measurement: TemporalHistoryMeasurement | None = None
    if selected.public_context is not None and selected.public_context.temporal_history is not None:
        lifetime = select_story_history(selected, prompts, repair_turns, estimator)
        history = lifetime.coverage
        history_measurement = TemporalHistoryMeasurement(lifetime.mandatory_tokens, lifetime.minimum_rendered_tokens,
            lifetime.rendered_tokens, request.input_budget_tokens - lifetime.mandatory_tokens, lifetime.refinement_attempts)
        selected, turns = lifetime.request, lifetime.turns
        selected_evidence = list(selected.public_context.evidence) if selected.public_context is not None else []
        selected_dialogue = list(selected.prior_dialogue)
        speech_history = list(selected_dialogue)
        required_ids.update(item.record_id for item in history.items)
        mandatory_tokens = lifetime.rendered_tokens
    remaining = request.input_budget_tokens - mandatory_tokens
    dialogue_budget = int(remaining * 0.45)
    dialogue_cost = 0
    omitted: list[str] = []
    included = [f"memory:{item.knoxel_id}" for item in selected_evidence]
    focus_terms = set(distinctive_terms(request.current_user_message))
    represented_terms: set[str] = set()
    for turn in selected_dialogue:
        represented_terms.update(focus_terms.intersection(distinctive_terms(turn.text)))
    for turn in reversed(speech_history):
        if turn in selected_dialogue:
            continue
        shared_terms = focus_terms.intersection(distinctive_terms(turn.text))
        new_terms = shared_terms - represented_terms
        if not new_terms:
            omitted.append(f"dialogue:{turn.knoxel_id}")
            continue
        trial_dialogue = [item for item in speech_history if item == turn or item in selected_dialogue]
        trial = selected.model_copy(deep=True)
        trial.prior_dialogue = trial_dialogue
        trial_turns = resolve_media((*build_story_messages(trial, prompts), *repair_turns), images)
        cost = estimator.estimate(trial_turns) - estimator.estimate(turns)
        if dialogue_cost + cost > dialogue_budget:
            omitted.append(f"dialogue:{turn.knoxel_id}")
            continue
        selected_dialogue = trial_dialogue
        represented_terms.update(new_terms)
        dialogue_cost += cost
        selected, turns = trial, trial_turns
    shares = {"history": 0.25, "facts": 0.10, "inner": 0.10, "narratives": 0.06, "rules": 0.04}
    lane_used = {lane: 0 for lane in shares}
    seen_texts = {" ".join(turn.text.casefold().split()) for turn in selected_dialogue}
    seen_texts.add(" ".join(request.current_user_message.casefold().split()))
    for item in rank_public_evidence(evidence):
        if any(row.knoxel_id == item.knoxel_id for row in selected_evidence):
            continue
        lane = item.lane if item.lane in shares else "inner"
        text_key = " ".join(item.text.casefold().split())
        if text_key in seen_texts:
            omitted.append(f"memory:{item.knoxel_id}")
            continue
        support_ids: set[int] = {item.knoxel_id}
        pending = [item.knoxel_id]
        while pending:
            record_id = pending.pop()
            for source_id in by_id[record_id].source_ids:
                if source_id in by_id and source_id not in support_ids:
                    support_ids.add(source_id)
                    pending.append(source_id)
        group = [row for row in evidence if row.knoxel_id in support_ids and row not in selected_evidence]
        trial_evidence = [*selected_evidence, *group]
        assert selected.public_context is not None
        trial_context = selected.public_context.model_copy(deep=True)
        trial_context.evidence = trial_evidence
        trial = selected.model_copy(deep=True)
        trial.public_context = trial_context
        trial_turns = resolve_media((*build_story_messages(trial, prompts), *repair_turns), images)
        trial_tokens = estimator.estimate(trial_turns)
        cost = trial_tokens - estimator.estimate(turns)
        if lane_used[lane] + cost > int(remaining * shares[lane]) or trial_tokens > request.input_budget_tokens:
            omitted.append(f"memory:{item.knoxel_id}")
            continue
        selected_evidence = trial_evidence
        selected, turns = trial, trial_turns
        lane_used[lane] += cost
        seen_texts.add(text_key)
        included.extend(f"memory:{row.knoxel_id}" for row in group)
    if context is None:
        for memory in rank_memory_texts(request.memories, request.current_user_message):
            trial = selected.model_copy(deep=True)
            trial.memories = [*selected.memories, memory.text]
            trial_turns = resolve_media((*build_story_messages(trial, prompts), *repair_turns), images)
            if estimator.estimate(trial_turns) <= request.input_budget_tokens:
                selected, turns = trial, trial_turns
                included.append(f"plain-memory:{memory.position}")
            else:
                omitted.append(f"plain-memory:{memory.position}")
    source_ids = set(required_ids)
    source_ids.update(item.knoxel_id for item in selected_evidence)
    source_ids.update(turn.knoxel_id for turn in selected_dialogue if turn.knoxel_id is not None)
    selection = PromptSelection(tuple(turns), tuple(included), tuple(omitted), estimator.estimate(turns),
                                tuple(sorted(source_ids)), ())
    return StoryPrompt(selected, selection, history, history_measurement)


async def plan_story_request(
    ghost: MemoryStore,
    embedding: EmbeddingService,
    situation: StorySituation,
    *,
    cancel: Event | None = None,
    settings: RenderSettings = RenderSettings(context_window=32000, input_fraction=0.85, reserved_output_tokens=8192),
) -> StoryRequest:
    """Build the typed Story request for one fixed situation.

    ``situation`` is the YAML mapping in ``situation.yaml``: a current user
    message and an optional workspace text. The ``story`` context plan selects
    the verbatim dialogue window and the memory lanes from the real seed,
    exactly as the application does. ``prior_dialogue`` in the situation is
    used only for a store without dialogue history.
    """
    companion = ghost.get_companion_name()
    user = ghost.get_user_name()
    prior = list(situation.prior_dialogue)
    current = situation.current_user_message

    from workflows.public_speech.public_context import build_public_context
    from workflows.minecraft_body.capabilities import capability_snapshot
    from pm.records.memory_records import Feature
    from pm.records.base import Provenance
    from pm.records.enums import FeatureType, InterlocusType, SourceKind
    from uuid import uuid4
    import asyncio
    # The fixed situation supplies a new user input. Keep that exact input as
    # evidence in this trial, rather than asking review to infer it from priors.
    changes = ghost.begin_batch("story-situation:" + uuid4().hex)
    source = changes.create(Feature(content=current, feature_type=FeatureType.Dialogue,
        interlocus=InterlocusType.Public, override_source=user,
        provenance=Provenance(source_kind=SourceKind.User, source_ref="story-situation:" + situation.name),
        metadata={"memory_kind": "utterance", "input_source": "standalone_situation"}))
    ghost.commit(changes)
    from workflows.public_speech.context_planning import FocusInput, plan_reply_context
    workspace = situation.workspace
    planned = await plan_reply_context(ghost.snapshot(), "story", FocusInput(
        workspace=workspace, current_message=current, exclude_ids=[int(source.id)], current_ids=(int(source.id),)), embedding, cancel=cancel,
        total_tokens=settings.input_ceiling)
    capabilities = await asyncio.to_thread(capability_snapshot, ghost, [])
    from workflows.main_cognition.schemas import ActionRequest, Stimulus
    public = build_public_context(ghost, ActionRequest(
        content="Answer the current user message in the established voice.", workspace=workspace,
        partner_name=user, stimuli=[Stimulus(knoxel_id=source.id, text=current, source="user_input")]),
        capabilities=capabilities, planned=planned)
    prior = planned.dialogue or prior

    from pm.records.media import IMAGE_TAG, RecordImage
    image_texts = [current, public.workspace, public.selected_action, *public.memories]
    image_texts.extend(turn.text for turn in prior)
    if public.temporal_history is not None:
        image_texts.extend(item.content for item in public.temporal_history.items)
    images: list[RecordImage] = []
    for image_id in sorted({match.group(1) for text in image_texts for match in IMAGE_TAG.finditer(text)}):
        image = ghost.find_image(image_id)
        if image is None:
            raise ValueError(f"Story lifetime history references missing image {image_id}.")
        images.append(image.model_copy(deep=True))
    return StoryRequest(
        images=tuple(images),
        companion_name=companion,
        partner_name=user,
        character_card_story=ghost.get_simple_character_story_block(),
        supported_capabilities=list(ghost.ghost_config.supported_capabilities),
        unsupported_capabilities=[str(item) for item in ghost.ghost_config.unsupported_capabilities],
        capability_notes=[str(item) for item in ghost.ghost_config.capability_notes],
        now=situation.now or vclock.utcnow().isoformat(),
        prior_dialogue=prior,
        current_user_message=current,
        current_message_id=int(source.id),
        input_budget_tokens=settings.input_ceiling,
        feelings=public.feelings,
        regulation=public.regulation,
        goals=list(public.goals),
        memories=public.memories,
        workspace=public.workspace,
        selected_action=public.selected_action,
        public_context=public,
    )


__all__ = ["StoryPrompt", "distinctive_terms", "plan_story_request", "select_story_prompt"]
