"""The current situational model (CSM): unconscious content with activation.

This module ports ``CSMState``, ``CSMItem``, and ``CSMManager`` from
``private-machine-head/pm/csm/csm.py`` and ``GhostCodelets.get_snapshot``
from ``private-machine-head/pm/ghosts/ghost_codelets.py``.

The CSM holds record references with activation. The memory store owns the
records and their embeddings. The cycle state owns the CSM and saves it with
each cycle. The operations here are pure mechanics. They make no model calls
and do no I/O. The caller supplies embeddings and token counts as functions.
"""
from __future__ import annotations

import math
import random
from typing import Callable, Collection

from pydantic import BaseModel, Field

from pm.constants import (
    CSM_BASE_ACTIVATION,
    CSM_BASE_DECAY_SPREAD,
    CSM_BASE_FORGET,
    CSM_BASE_LEARN_RATE,
    CSM_BOOST,
    CSM_CAPACITY,
    CSM_DECAY,
    CSM_DEFAULT_AFFECT_TERM,
    CSM_MERGE_SIMILARITY,
    CSM_MIN_ACTIVATION,
    CSM_PRECURSOR_BONUS,
    CSM_SNAPSHOT_EXPLOIT,
    CSM_SNAPSHOT_NOISE,
)
from pm.records.workspace import ShardKey

#: Returns the embedding of a record id. A missing record or an empty
#: embedding is a caller bug, and the cosine calculation then raises.
EmbeddingOf = Callable[[int], list[float]]
#: Returns the token count of the rendered story element of a record id.
TokensOf = Callable[[int], int]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _cosine(left: list[float], right: list[float]) -> float:
    """Return the cosine similarity. A zero vector raises ZeroDivisionError."""
    dot = math.fsum(a * b for a, b in zip(left, right, strict=True))
    return dot / (math.hypot(*left) * math.hypot(*right))


class SituationalItem(BaseModel):
    """One record in the CSM.

    ``activation`` is the current activation. ``base_activation`` is the
    long-term base-level activation. It controls how fast ``activation``
    decays. ``valence`` and ``salience`` come from the percept appraisal. They
    are None for records that are not percepts. ``endorsers`` names the
    codelets and other sources that added or boosted the item. ``merged_ids``
    names records that merged into this item because their embeddings were
    similar. ``shard_hint`` names the shard that the content concerns, or None.
    """

    record_id: int = 0
    entered_tick: int = 0
    last_boost_tick: int = 0
    ticks_present: int = 0
    activation: float = 0.0
    base_activation: float = CSM_BASE_ACTIVATION
    valence: float | None = None
    salience: float | None = None
    endorsers: list[str] = Field(default_factory=list)
    merged_ids: list[int] = Field(default_factory=list)
    shard_hint: ShardKey | None = None

    def represents(self, record_ids: Collection[int]) -> bool:
        """Return True when this item or one of its merged records is in ``record_ids``."""
        return self.record_id in record_ids or any(merged in record_ids for merged in self.merged_ids)


class SituationalModel(BaseModel):
    """The CSM of one cycle: items keyed by record id.

    The cycle mutates this model in place from cycle to cycle. JSON stores
    the keys as text, and ``model_validate`` converts them back to integers.
    """

    items: dict[int, SituationalItem] = Field(default_factory=dict)

    def decay_step(self) -> None:
        """Decay all items by one cycle.

        Each item keeps ``CSM_DECAY ** e`` of its activation, with
        ``e = 1 + CSM_BASE_DECAY_SPREAD * (1 - base / CSM_BASE_ACTIVATION)``.
        At the neutral base-level activation, ``e`` is 1 and the factor is
        ``CSM_DECAY``. A lower base gives a larger ``e`` and a faster decay. A
        higher base gives a smaller ``e`` and a slower decay (LIDA base-level
        activation). The base itself drops slowly by ``CSM_BASE_FORGET``, and
        ``ticks_present`` increases by one.
        """
        for item in self.items.values():
            exponent = 1.0 + CSM_BASE_DECAY_SPREAD * (1.0 - item.base_activation / CSM_BASE_ACTIVATION)
            item.activation *= CSM_DECAY ** exponent
            item.base_activation *= CSM_BASE_FORGET
            item.ticks_present += 1

    def add_or_boost(self, item: SituationalItem, endorser: str, embedding_of: EmbeddingOf, *,
                     boost: float = CSM_BOOST) -> SituationalItem:
        """Add ``item`` or boost the existing item that holds the same content.

        The existing item is the item with the same record id, the item that
        already merged that record id, or else the item with the highest
        embedding cosine similarity of at least ``CSM_MERGE_SIMILARITY``. A
        boost adds ``boost`` to its activation (capped at 1.0), raises its
        base-level activation, sets ``last_boost_tick`` to the entered tick of
        ``item``, and records ``endorser`` once. A merge also records the
        record id of ``item`` in ``merged_ids``. Several endorsers of the same
        content therefore raise its activation.

        If no item holds the content, a copy of ``item`` enters the CSM with
        ``endorser`` in its endorsers. This method does not enforce the
        capacity. Call ``prune`` after the additions of a phase.

        Returns the CSM item that now holds the content.
        """
        target = next((existing for existing in self.items.values()
                       if existing.record_id == item.record_id or item.record_id in existing.merged_ids), None)
        if target is None:
            embedding = embedding_of(item.record_id)
            best_score = CSM_MERGE_SIMILARITY
            for existing in self.items.values():
                score = _cosine(embedding, embedding_of(existing.record_id))
                if score >= best_score:
                    target, best_score = existing, score
            if target is not None:
                target.merged_ids.append(item.record_id)
        if target is None:
            added = item.model_copy(deep=True)
            if endorser not in added.endorsers:
                added.endorsers.append(endorser)
            self.items[added.record_id] = added
            return added
        target.activation = min(1.0, target.activation + boost)
        target.base_activation += CSM_BASE_LEARN_RATE * (1.0 - target.base_activation)
        target.last_boost_tick = item.entered_tick
        if endorser not in target.endorsers:
            target.endorsers.append(endorser)
        return target

    def prune(self, protected_ids: Collection[int]) -> list[int]:
        """Remove weak items and enforce ``CSM_CAPACITY``.

        ``protected_ids`` holds the record ids that a CSM requirement
        protects. An item is protected when its record id or one of its merged
        ids is in that set. Prune first removes every unprotected item below
        ``CSM_MIN_ACTIVATION``. Then it removes unprotected items with the
        lowest activation until the CSM fits its capacity. Protection wins
        over capacity, so a CSM with many protected items can stay above it.

        Returns the removed record ids in removal order.
        """
        removed = [record_id for record_id, item in self.items.items()
                   if item.activation < CSM_MIN_ACTIVATION and not item.represents(protected_ids)]
        for record_id in removed:
            del self.items[record_id]
        excess = len(self.items) - CSM_CAPACITY
        if excess > 0:
            candidates = sorted((item for item in self.items.values() if not item.represents(protected_ids)),
                                key=lambda item: item.activation)
            for item in candidates[:excess]:
                del self.items[item.record_id]
                removed.append(item.record_id)
        return removed

    def snapshot_for(self, query_embedding: list[float], embedding_of: EmbeddingOf, tokens_of: TokensOf, *,
                     attention_bias: tuple[float, float], max_tokens: int, rng: random.Random,
                     precursor_ids: Collection[int] = ()) -> list[int]:
        """Return one codelet's view of the CSM as record ids, best first.

        Port of ``GhostCodelets.get_snapshot``. Each item gets a rating:

        1. Affect term. For a percept (``salience`` is set), it is the mean of
           a valence term and a salience term. The valence term favors
           positive or negative content by the valence focus bias. The
           salience term is ``salience ** (1 + 4 * salience_bias)``. An item
           without salience uses ``CSM_DEFAULT_AFFECT_TERM``.
        2. Relevance: cosine similarity of the record embedding and
           ``query_embedding`` (the codelet signature or the situation),
           clamped to [0, 1].
        3. Rating: relevance times affect term, plus ``CSM_PRECURSOR_BONUS``
           for an item in ``precursor_ids`` (percepts of an upstream pathway
           codelet), plus noise up to ``CSM_SNAPSHOT_NOISE * (1 -
           salience_bias)``, clamped to [0, 1].

        ``attention_bias`` is ``(valence_focus_bias, salience_bias)`` from
        ``pm.records.mental_state.compute_attention_bias``. At or above
        ``CSM_SNAPSHOT_EXPLOIT`` salience bias, the items are sorted by
        rating. Below it, the order is a weighted random order without
        replacement with weights ``rating ** (1 + 4 * salience_bias)``.

        The view takes items in that order until the next item would exceed
        ``max_tokens``. The caller orders the result by world time for a
        story. ``rng`` supplies all randomness, so a seeded generator gives a
        repeatable view. The method does not change the CSM.
        """
        valence_focus_bias, salience_bias = attention_bias
        sharp = 1.0 + 4.0 * salience_bias
        ratings: dict[int, float] = {}
        for record_id, item in self.items.items():
            affect = CSM_DEFAULT_AFFECT_TERM
            if item.salience is not None:
                valence = item.valence or 0.0
                valence_term = _clamp01((1.0 - abs(valence_focus_bias)) * 0.5 + valence_focus_bias * valence * 0.5 + 0.5)
                salience_term = math.pow(max(item.salience, 0.0), sharp)
                affect = 0.5 * valence_term + 0.5 * salience_term
            relevance = _clamp01(_cosine(query_embedding, embedding_of(record_id)))
            bonus = CSM_PRECURSOR_BONUS if record_id in precursor_ids else 0.0
            noise = rng.uniform(-CSM_SNAPSHOT_NOISE, CSM_SNAPSHOT_NOISE) * (1.0 - salience_bias)
            ratings[record_id] = _clamp01(relevance * affect + bonus + noise)

        if salience_bias >= CSM_SNAPSHOT_EXPLOIT:
            order = sorted(ratings, key=lambda record_id: ratings[record_id], reverse=True)
        else:
            weights = {record_id: math.pow(rating, sharp) for record_id, rating in ratings.items()}
            if not any(weights.values()):
                weights = {record_id: 1.0 for record_id in weights}
            # Efraimidis-Spirakis keys give a weighted order without
            # replacement, like numpy.random.choice(replace=False, p=...).
            keys = {record_id: rng.random() ** (1.0 / weight) if weight > 0.0 else 0.0
                    for record_id, weight in weights.items()}
            order = sorted(keys, key=lambda record_id: keys[record_id], reverse=True)

        selected: list[int] = []
        used = 0
        for record_id in order:
            tokens = tokens_of(record_id)
            if used + tokens > max_tokens:
                break
            selected.append(record_id)
            used += tokens
        return selected
