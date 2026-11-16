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
from pydantic import BaseModel, ConfigDict, Field

from pm.context_render.budget import ContextGroup, TokenEstimator, context_recency, rank_context
from pm.context_render.turns import Content
from pm.llm.config import load_config
from pm.memory.embeddings import EmbeddingService, run_embedding_work
from pm.memory.store import MemoryStore
from pm.memory.vector_index import VectorIndex
from simulation.actors import ActorProfile, ConversationTurn, HistoricalTurn

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
    salience: dict[int, float] = Field(default_factory=dict[int, float], description="Embedding similarity by original record ID.")


def _turn_cost(turn: BaseModel, factor: float) -> int:
    """Estimate one serialized turn, including its JSON field names.

    The shared estimator adds its 16-token message framing once per rendered
    message, not once per JSON element, so this cost counts only the turn text.
    """
    raw = max(len(encoder.encode(turn.model_dump_json(), disallowed_special=())) for encoder in _ENCODERS)
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
        self._costs = [_turn_cost(turn, self.estimator.estimate_factor) for turn in self.turns]

    def select(self, query: HistoryQuery, budget_tokens: int, *, cancel: Event | None = None) -> HistorySelection:
        """Mix embedding relevance, recency, and randomness through the shared 40/40/20 policy."""
        text = "\n".join([query.scene, query.intent, *(turn.text for turn in query.recent)])
        vector = run_embedding_work(self.service.embed(text, cancel=cancel))
        similarities = dict(self.index.search(vector, range(len(self.turns))))
        groups = [ContextGroup(str(index), Content(""), similarities[index], recency=context_recency(turn.at))
                  for index, turn in enumerate(self.turns)]
        chosen = [int(group.name) for group in rank_context(groups)]

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
                                estimated_tokens=used,
                                salience={self.turns[index].record_id: similarities[index] for index in kept})


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
