"""Context selection for Story dialogue.

Adapted from the old planner lanes, reduced to what a standalone Story run
needs. Selection is bounded before embedding: a keyword pre-filter and a hard
pool limit keep the number of uncached embedding requests small and
predictable. Ranking uses the shared embedding service. Rendering keeps the
story elements produced by the knoxel models.

Nothing here calls a chat model or writes memory. Network access happens only
when the caller gives an embedding service and asks for a plan.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from pm_core.embeddings import EmbeddingService
from pm_core.model.knoxel_enums import FeatureType

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

_NARRATIVE_LIMIT = 16
_FACT_LIMIT = 16
_TOP_NARRATIVES = 4
_TOP_FACTS = 8
_RECENT_THOUGHTS = 4


def distinctive_terms(text: str) -> list[str]:
    """Stable, lowercase, long-enough words used only as a pre-filter."""
    seen: dict[str, None] = {}
    for match in _WORD.finditer(text.casefold()):
        word = match.group(0)
        if len(word) < 4 or word in _STOPWORDS:
            continue
        seen.setdefault(word, None)
    return list(seen)


def _mentions(knoxel, terms: Sequence[str]) -> bool:
    if not terms:
        return False
    content = knoxel.content.casefold()
    return any(term in content for term in terms)


def _by_recency(knoxel) -> tuple:
    when = getattr(knoxel, "timestamp_world_begin", None)
    if not isinstance(when, datetime):
        when = datetime.min
    return (when, getattr(knoxel, "sub_tick_id", 0), knoxel.id)


@dataclass(frozen=True)
class _Ranked:
    knoxel: object
    score: float


async def _rank(
    embedding: EmbeddingService,
    query: str,
    knoxels: Sequence,
    *,
    cancel=None,
) -> list[_Ranked]:
    """Rank a bounded pool with the shared embedding service."""
    visible = {
        knoxel.id: knoxel.content
        for knoxel in knoxels
        if isinstance(getattr(knoxel, "content", None), str) and knoxel.content.strip()
    }
    if not visible:
        return []
    await embedding.prepare_index(visible, cancel=cancel)
    ranked = await embedding.search(query, visible, cancel=cancel)
    by_id = {knoxel.id: knoxel for knoxel in knoxels}
    return [_Ranked(by_id[record_id], score) for record_id, score in ranked if record_id in by_id]


def _last_thoughts(ghost, limit: int) -> list[str]:
    thoughts = [
        knoxel for knoxel in ghost.all_features
        if getattr(knoxel, "feature_type", None) in {FeatureType.Thought, FeatureType.InnerMonologue}
    ]
    thoughts.sort(key=_by_recency)
    rendered: list[str] = []
    for knoxel in thoughts[-limit:]:
        rendered.append(f"{ghost.get_companion_name()} thinks: *{knoxel.content}*")
    return rendered


def _workspace_note(ghost) -> str:
    turns = getattr(ghost, "gwt_turns", None) or {}
    if turns:
        latest = turns[max(turns)]
        summary = getattr(latest, "summary", "")
        if summary:
            return f"Active attention frame: {summary}"
    return "No attention frame is recorded yet; the workspace holds only this ongoing conversation."


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

    query = current + "\n" + "\n".join(turn.text for turn in prior[-6:])
    terms = distinctive_terms(query)

    narratives = [knoxel for knoxel in ghost.all_narratives if _mentions(knoxel, terms)]
    if not narratives:
        narratives = sorted(ghost.all_narratives, key=_by_recency)[-8:]
    narratives = sorted(narratives, key=_by_recency)[-_NARRATIVE_LIMIT:]

    facts = [knoxel for knoxel in ghost.all_declarative_facts if _mentions(knoxel, terms)]
    if not facts:
        facts = sorted(ghost.all_declarative_facts, key=lambda k: (-k.importance, k.id))[:_FACT_LIMIT]
    facts = sorted(facts, key=lambda k: (-k.importance, k.id))[:_FACT_LIMIT]

    ranked_narratives = await _rank(embedding, query, narratives, cancel=cancel)
    ranked_facts = await _rank(embedding, query, facts, cancel=cancel)

    memories = [item.knoxel.get_story_element() for item in ranked_narratives[:_TOP_NARRATIVES]]
    memories += [item.knoxel.get_story_element() for item in ranked_facts[:_TOP_FACTS]]

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
        feelings=_last_thoughts(ghost, _RECENT_THOUGHTS),
        memories=memories,
        workspace=_workspace_note(ghost),
    )


__all__ = ["distinctive_terms", "plan_story_request"]
