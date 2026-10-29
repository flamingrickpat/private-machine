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

from pm import vclock

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from threading import Event
from typing import TYPE_CHECKING, Any

from pm.memory.embeddings import EmbeddingService
from pm.memory.retrieval import distinctive_terms
from pm.context_render.budget import ContextGroup, PromptSelection, TokenEstimator, select_rendered_context
from pm.context_render.turns import Content, Turn

from workflows.dialogue_story.schemas import DialogueTurn, StoryPrompts, StoryRequest

if TYPE_CHECKING:
    from pm.memory.store import MemoryStore
    from workflows.public_speech.public_context import EvidenceItem


@dataclass(frozen=True, slots=True)
class StoryPrompt:
    """One fully rendered prompt and the exact evidence supplied to its reviewer."""

    request: StoryRequest
    selection: PromptSelection


def select_story_prompt(
    request: StoryRequest,
    prompts: StoryPrompts,
    repair_turns: Sequence[Turn] = (),
    *,
    render: Callable[[StoryRequest], Sequence[Turn]] | None = None,
) -> StoryPrompt:
    """Fit optional retrieval against a complete Story generation or review prompt.

    Current speech, prior dialogue, shard evidence, private state, instructions,
    and repair turns remain required. Retrieval records compete by their saved
    scores. A selected record keeps any support records present in the supplied
    evidence. Nothing is clipped, and required overflow remains an error.
    """
    from workflows.dialogue_story.steps import build_story_messages
    from workflows.public_speech.context_planning import load_plan

    if request.input_budget_tokens is None:
        raise ValueError("Story context selection requires the runtime input ceiling.")
    context = request.public_context
    evidence: list[EvidenceItem] = list(context.evidence) if context is not None else []
    by_id = {item.knoxel_id: item for item in evidence}
    retrieval_lanes = {lane.name for lane in load_plan("story").lanes}
    optional = [item for item in evidence if item.knoxel_id != 0 and item.lane in retrieval_lanes]
    optional_ids = {item.knoxel_id for item in optional}
    required_ids = {item.knoxel_id for item in evidence if item.knoxel_id not in optional_ids}
    required_ids.update(turn.knoxel_id for turn in request.prior_dialogue if turn.knoxel_id is not None)
    if request.current_message_id is not None:
        required_ids.add(request.current_message_id)
    pending_required = [record_id for record_id in required_ids if record_id in by_id]
    while pending_required:
        current = by_id[pending_required.pop()]
        for source_id in current.source_ids:
            if source_id in by_id and source_id not in required_ids:
                required_ids.add(source_id)
                pending_required.append(source_id)
    groups = [ContextGroup(name="Required Story context", content=Content(""), salience=0,
                           required=True, source_ids=tuple(sorted(required_ids)))]
    for item in optional:
        if item.knoxel_id in required_ids:
            continue
        support_ids = {item.knoxel_id}
        pending = [item.knoxel_id]
        while pending:
            current = by_id[pending.pop()]
            for source_id in current.source_ids:
                if source_id in by_id and source_id not in support_ids:
                    support_ids.add(source_id)
                    pending.append(source_id)
        groups.append(ContextGroup(name=f"memory:{item.knoxel_id}", content=Content(item.text),
                                   salience=item.score, source_ids=tuple(sorted(support_ids))))

    def selected_request(chosen: Sequence[ContextGroup]) -> StoryRequest:
        if context is None:
            return request
        selected_ids: set[int] = set()
        for group in chosen:
            selected_ids.update(group.source_ids)
        selected_evidence = [item for item in evidence
                             if item.knoxel_id not in optional_ids or item.knoxel_id in selected_ids]
        selected_context = context.model_copy(update={"evidence": selected_evidence})
        return request.model_copy(update={"public_context": selected_context})

    def assemble(chosen: Sequence[ContextGroup], omitted: Sequence[str]) -> Sequence[Turn]:
        selected = selected_request(chosen)
        if render is None:
            turns = build_story_messages(selected, prompts)
        else:
            turns = list(render(selected))
        if omitted:
            last = turns[-1]
            notice = "\n\nOmitted optional context groups: " + ", ".join(omitted)
            turns[-1] = Turn(last.role, last.content + notice)
        return [*turns, *repair_turns]

    selection = select_rendered_context(groups, request.input_budget_tokens, assemble, estimator=TokenEstimator())
    included_names = set(selection.included)
    included_groups = [group for group in groups if group.name in included_names]
    return StoryPrompt(selected_request(included_groups), selection)


async def plan_story_request(
    ghost: MemoryStore,
    embedding: EmbeddingService,
    situation: dict[str, Any],
    *,
    cancel: Event | None = None,
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
    prior = [DialogueTurn.model_validate(turn) for turn in situation.get("prior_dialogue", [])]
    current = str(situation["current_user_message"])

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
        provenance=Provenance(source_kind=SourceKind.User, source_ref="story-situation:" + str(situation.get("name", "operator"))),
        metadata={"memory_kind": "utterance", "input_source": "standalone_situation"}))
    ghost.commit(changes)
    from workflows.public_speech.context_planning import FocusInput, plan_reply_context
    workspace = str(situation.get("workspace", ""))
    planned = await plan_reply_context(ghost.snapshot(), "story", FocusInput(
        workspace=workspace, current_message=current, exclude_ids=[int(source.id)]), embedding, cancel=cancel)
    capabilities = await asyncio.to_thread(capability_snapshot, ghost, [])
    from workflows.main_cognition.schemas import ActionRequest, Stimulus
    public = build_public_context(ghost, ActionRequest(
        content="Answer the current user message in the established voice.", workspace=workspace,
        partner_name=user, stimuli=[Stimulus(knoxel_id=source.id, text=current, source="user_input")]),
        capabilities=capabilities, planned=planned)
    prior = planned.dialogue or prior

    return StoryRequest(
        companion_name=companion,
        partner_name=user,
        character_card_story=ghost.get_simple_character_story_block(),
        supported_capabilities=list(ghost.ghost_config.supported_capabilities),
        unsupported_capabilities=[str(item) for item in ghost.ghost_config.unsupported_capabilities],
        capability_notes=[str(item) for item in ghost.ghost_config.capability_notes],
        now=str(situation.get("now") or vclock.utcnow().isoformat()),
        prior_dialogue=prior,
        current_user_message=current,
        current_message_id=int(source.id),
        feelings=public.feelings,
        regulation=public.regulation,
        goals=list(public.goals),
        memories=public.memories,
        workspace=public.workspace,
        selected_action=public.selected_action,
        public_context=public,
    )


__all__ = ["StoryPrompt", "distinctive_terms", "plan_story_request", "select_story_prompt"]
