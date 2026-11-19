"""Embedding retrieval over a simulated user's public history.

The source checkpoint already stores one vector per public dialogue record, so
this module never re-embeds history. It loads those vectors from the forked
ghost, builds one in-memory FAISS index, and ranks turns against the current
scene. Only the query text is embedded per actor turn.

The package overview is in ``simulation/__init__.py``.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from pathlib import Path
from threading import Event
from typing import Literal

import tiktoken
from pydantic import BaseModel, ConfigDict

from pm.context_render.budget import TokenEstimator
from pm.llm.config import load_config
from pm.memory.embeddings import EmbeddingService, run_embedding_work
from pm.memory.store import MemoryStore
from pm.memory.vector_index import VectorIndex
from simulation.actors import ActorProfile, ConversationTurn, HistoricalTurn

#: Historical turns always kept for continuity, regardless of similarity.
RECENT_HISTORY_TURNS = 20
#: Speaker turns spread across history to preserve voice.
VOICE_SAMPLE_TURNS = 8

#: The same encoders the shared estimator uses, so per-turn costs match it.
_ENCODERS = (tiktoken.get_encoding("o200k_base"), tiktoken.get_encoding("cl100k_base"))


@dataclass(frozen=True, slots=True)
class HistoryQuery:
    """The current scene text used to rank historical turns."""

    scene: str
    intent: str
    recent: tuple[ConversationTurn, ...]


class HistorySelection(BaseModel):
    """The historical turns that fit the available budget."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    turns: tuple[HistoricalTurn, ...]
    omitted_turns: int
    estimated_tokens: int
    method: Literal["embedding"] = "embedding"


@dataclass(frozen=True, slots=True)
class RankedHistory:
    turns: tuple[HistoricalTurn, ...]
    priority: dict[int, float]


def _turn_cost(turn: ConversationTurn | HistoricalTurn, factor: float) -> int:
    """Estimate one serialized turn, including its JSON field names.

    The shared estimator adds its 16-token message framing once per rendered
    message, not once per JSON element, so this cost counts only the turn text.
    """
    at = turn.at if isinstance(turn, ConversationTurn) else turn.at.isoformat(timespec="minutes")
    text = f"[{at}] {turn.speaker}: {turn.text}"
    raw = max(len(encoder.encode(text, disallowed_special=())) for encoder in _ENCODERS)
    return math.ceil(raw * factor)


class HistoryIndex:
    """Load the ghost's stored vectors and rank public dialogue by similarity."""

    def __init__(self, profile: ActorProfile, home: Path, config: str) -> None:
        store = MemoryStore.from_checkpoint(home / "ghost.pickle")
        vectors = {record.id: record.embedding for record in store.all_features if record.embedding}
        self.turns: tuple[HistoricalTurn, ...] = tuple(profile.dialogue)
        missing = [turn.record_id for turn in self.turns if turn.record_id not in vectors]
        if missing:
            raise ValueError(f"The forked ghost has no stored vector for {len(missing)} public turns.")
        self.index = VectorIndex()
        self.index.rebuild_vectors(list(range(len(self.turns))),
                                   [vectors[turn.record_id] for turn in self.turns])
        self.service = EmbeddingService(load_config(config).embedding)
        self.estimator = TokenEstimator()
        self.speaker_name = profile.speaker_name
        self._costs = [_turn_cost(turn, self.estimator.estimate_factor) for turn in self.turns]

    def rank(self, query: HistoryQuery, *, cancel: Event | None = None) -> RankedHistory:
        text = "\n".join([query.scene, query.intent, *(turn.text for turn in query.recent)])
        vector = run_embedding_work(self.service.embed(text, cancel=cancel))
        semantic = self.index.search(vector, range(len(self.turns)))
        priority: dict[int, float] = {}
        for index, score in semantic[:32]:
            if score >= 0.3:
                priority[self.turns[index].record_id] = score
        for index in range(max(0, len(self.turns) - RECENT_HISTORY_TURNS), len(self.turns)):
            priority[self.turns[index].record_id] = 1.0 + index / max(1, len(self.turns))
        speaker_turns = [turn for turn in self.turns if turn.speaker == self.speaker_name]
        step = max(1, len(speaker_turns) // VOICE_SAMPLE_TURNS)
        for turn in speaker_turns[::step][:VOICE_SAMPLE_TURNS]:
            priority[turn.record_id] = max(priority.get(turn.record_id, 0), 0.8)
        selected = tuple(turn for turn in self.turns if turn.record_id in priority)
        return RankedHistory(selected, priority)

    def select(self, query: HistoryQuery, budget_tokens: int, *, cancel: Event | None = None) -> HistorySelection:
        """Return the most relevant and recent turns that fit ``budget_tokens``."""
        text = "\n".join([query.scene, query.intent, *(turn.text for turn in query.recent)])
        vector = run_embedding_work(self.service.embed(text, cancel=cancel))
        ranked = [key for key, _ in self.index.search(vector, range(len(self.turns)))]
        chosen: list[int] = []
        seen: set[int] = set()

        def add(index: int) -> None:
            if index not in seen:
                seen.add(index)
                chosen.append(index)

        # Recency first, then the speaker's own voice, then similarity.
        for index in range(max(0, len(self.turns) - RECENT_HISTORY_TURNS), len(self.turns)):
            add(index)
        speaker_turns = [index for index, turn in enumerate(self.turns) if turn.speaker == self.speaker_name]
        step = max(1, len(speaker_turns) // VOICE_SAMPLE_TURNS)
        for index in speaker_turns[::step]:
            add(index)
        for index in ranked:
            add(index)

        kept: list[int] = []
        used = 0
        for index in chosen:
            cost = self._costs[index]
            if used + cost > budget_tokens:
                continue
            kept.append(index)
            used += cost
        kept.sort()
        return HistorySelection(turns=tuple(self.turns[index] for index in kept),
                                omitted_turns=len(self.turns) - len(kept),
                                estimated_tokens=used)


def conversation_tokens(turns: Sequence[ConversationTurn]) -> int:
    """Estimate the rendered cost of a run conversation."""
    estimator = TokenEstimator()
    return sum(_turn_cost(turn, estimator.estimate_factor) for turn in turns)


def bound_conversation(turns: Sequence[ConversationTurn], budget_tokens: int) -> list[ConversationTurn]:
    """Keep the most recent run turns that fit ``budget_tokens``."""
    estimator = TokenEstimator()
    kept: list[ConversationTurn] = []
    used = 0
    for turn in reversed(turns):
        cost = _turn_cost(turn, estimator.estimate_factor)
        if used + cost > budget_tokens:
            break
        kept.append(turn)
        used += cost
    kept.reverse()
    return kept


__all__ = ["HistoryIndex", "HistoryQuery", "HistorySelection", "bound_conversation", "conversation_tokens"]
