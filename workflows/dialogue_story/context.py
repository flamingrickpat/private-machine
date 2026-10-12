"""Prepare standalone Story with shared retrieval and public evidence.

The operator's situation supplies one accepted user input in the trial Ghost.
Retrieval uses the shared embedding cache. Capability discovery runs outside
the async loop. No raw private thought transcript enters the public request.
"""
from __future__ import annotations

import re

from pm_core.embeddings import EmbeddingService
from pm.records.enums import FeatureType

from workflows.dialogue_story.schemas import DialogueTurn, StoryRequest

_WORD = re.compile(r"[a-zA-Z][a-zA-Z'-]+")

_STOPWORDS = frozenset({
    "about", "again", "also", "and", "any", "are", "because", "been", "but",
    "can", "could", "did", "does", "doing", "dont", "for", "from", "get",
    "going", "got", "had", "has", "have", "her", "here", "him", "his", "how",
    "into", "its", "just", "like", "made", "make", "more", "much", "not",
    "now", "off", "one", "only", "other", "our", "out", "over", "really",
    "right", "said", "same", "she", "should", "some", "still", "such", "than",
    "that", "the", "their", "them", "then", "there", "these", "they", "this",
    "those", "through", "too", "under", "very", "want", "was", "were", "what",
    "when", "where", "which", "while", "who", "will", "with", "would", "your",
    "youre", "youve", "well", "yeah", "okay", "even", "ever", "much", "back",
    "thing", "things", "know", "think", "thought", "tell", "said",
})

def distinctive_terms(text: str) -> list[str]:
    """Stable, lowercase, long-enough words used only as a pre-filter."""
    seen: dict[str, None] = {}
    for match in _WORD.finditer(text.casefold()):
        word = match.group(0)
        if len(word) < 4 or word in _STOPWORDS:
            continue
        seen.setdefault(word, None)
    return list(seen)


async def plan_story_request(
    ghost,
    embedding: EmbeddingService,
    situation: dict,
    *,
    cancel=None,
) -> StoryRequest:
    """Build the typed Story request for one fixed situation.

    ``situation`` is the YAML mapping in ``situation.yaml``: prior dialogue,
    a current user message, and an optional temporal note.
    """
    companion = ghost.get_companion_name()
    user = ghost.get_user_name()
    prior = [DialogueTurn.model_validate(turn) for turn in situation.get("prior_dialogue", [])]
    current = str(situation["current_user_message"])

    from workflows.public_context import build_public_context
    from runtime.capabilities import capability_snapshot
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
    query = current + "\n" + "\n".join(turn.text for turn in prior[-6:])
    retrieval = await ghost.retrieve_experiences(query, embedding, limit=12, cancel=cancel)
    capabilities = await asyncio.to_thread(capability_snapshot, ghost, [])
    public = build_public_context(ghost, {"content": "Answer the current user message in the established voice.",
                                        "memory_evidence": retrieval["matches"],
                                        "stimuli": [{"knoxel_id": source.id, "text": current}]},
                                  capabilities=capabilities)

    return StoryRequest(
        companion_name=companion,
        user_name=user,
        character_card_story=ghost.get_simple_character_story_block(),
        supported_capabilities=list(ghost.ghost_config.supported_capabilities),
        unsupported_capabilities=[str(item) for item in ghost.ghost_config.unsupported_capabilities],
        capability_notes=[str(item) for item in ghost.ghost_config.capability_notes],
        temporal_context=str(situation.get("temporal_context") or ghost.build_body_and_situation_addendum()),
        prior_dialogue=prior,
        current_user_message=current,
        feelings=[public.affect] if public.affect else [],
        memories=public.memories,
        workspace=public.workspace,
        selected_action=public.selected_action,
        public_context=public,
    )


__all__ = ["distinctive_terms", "plan_story_request"]
