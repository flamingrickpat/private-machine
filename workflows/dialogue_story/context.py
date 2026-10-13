"""Prepare standalone Story with the shared context plan and public evidence.

The operator's situation supplies one accepted user input in the trial MemoryStore.
The ``story`` context plan uses the shared embedding cache. Capability discovery
runs outside the async loop. The companion's earlier private thoughts can
appear in the verbatim transcript as ``thinks: *...*`` lines; they guide the
voice and are never published.
"""
from __future__ import annotations

from datetime import datetime

from pm.memory.embeddings import EmbeddingService
from pm.memory.retrieval import distinctive_terms

from workflows.dialogue_story.schemas import DialogueTurn, StoryRequest


async def plan_story_request(
    ghost,
    embedding: EmbeddingService,
    situation: dict,
    *,
    cancel=None,
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
    public = build_public_context(ghost, {"content": "Answer the current user message in the established voice.",
                                          "workspace": workspace,
                                          "stimuli": [{"knoxel_id": source.id, "text": current}]},
                                  capabilities=capabilities, planned=planned)
    prior = planned.dialogue or prior

    return StoryRequest(
        companion_name=companion,
        user_name=user,
        character_card_story=ghost.get_simple_character_story_block(),
        supported_capabilities=list(ghost.ghost_config.supported_capabilities),
        unsupported_capabilities=[str(item) for item in ghost.ghost_config.unsupported_capabilities],
        capability_notes=[str(item) for item in ghost.ghost_config.capability_notes],
        now=str(situation.get("now") or datetime.utcnow().isoformat()),
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


__all__ = ["distinctive_terms", "plan_story_request"]
