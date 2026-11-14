"""Choose which retained question the private attention stage is offered.

The subsystem overview is :mod:`workflows.inner_thought.host`. Several
questions can wait for the one private slot, and the attention prompt offers
one of them. This module decides which one. It makes no model call, reads no
store, and grants nothing: the attention decision still admits or refuses the
offered question.

A question comes into focus for two different reasons, and the rating keeps
them apart:

- **Source overlap.** The current context holds an original record the
  question came from.
- **Similarity.** Something in the current context means nearly the same,
  whichever record it is. Hearing something related, or a neighbouring
  thought, is enough.

The weighted sum of both is the relevance. Seeded noise is added, and the
accepted mental state decides how strictly the top rating wins. Focused
attention takes the highest rating. Wider attention samples by rating, so a
weakly cued question can still surface. The sampling follows
:meth:`pm.records.situational_model.SituationalModel.snapshot_for`, which
selects a codelet's view in the main loop.

Reading map: :func:`attention_cues` reads what the private mind holds now,
:func:`rate_retained_questions` measures each waiting question against it, and
:func:`select_retained_offer` picks one and returns the full receipt. The
caller, :func:`workflows.inner_thought.graph_execution.displaced_candidate`,
supplies the embedding vectors.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
import random
from typing import Literal

from pm.records.situational_model import cosine
from workflows.inner_thought.schemas import InnerRequest, RetainedOfferSelection, RetainedQuestionRating


#: Weight of source overlap in the relevance. The two weights sum to 1.
SOURCE_OVERLAP_WEIGHT = 0.5
#: Weight of embedding similarity in the relevance.
SIMILARITY_WEIGHT = 0.5
#: Maximum absolute rating jitter at salience bias 0. It shrinks to 0 at bias 1.
RATING_NOISE = 0.05
#: At or above this salience bias the highest rating is offered without sampling.
EXPLOIT_SALIENCE_BIAS = 0.9

CueKind = Literal["main_turn", "candidate", "recent_thought", "goal", "expectation"]


@dataclass(frozen=True, slots=True)
class AttentionCue:
    """One thing the private mind holds right now.

    ``record_id`` is None for the main turn summary when that turn has no
    frame record. Such a cue can still be similar; it cannot share a source.
    """

    record_id: int | None
    kind: CueKind
    text: str


@dataclass(frozen=True, slots=True)
class RetainedQuestion:
    """One waiting question, as the rating reads it."""

    receipt_id: int
    #: The origin's idea and its question, embedded as one text.
    text: str
    #: Original source roots of the question; internal derivations are excluded.
    source_ids: tuple[int, ...]


def attention_cues(request: InnerRequest) -> tuple[AttentionCue, ...]:
    """Read the frozen private request as cues, skipping entries without text.

    These are the records the private stages can already cite, plus the summary
    of the newest main turn. The same set feeds both signals: record identities
    for source overlap and texts for similarity.
    """
    cues: list[AttentionCue] = []
    if request.main_frame.summary.strip():
        cues.append(AttentionCue(request.main_frame.frame_id, "main_turn", request.main_frame.summary))
    for candidate in request.candidates:
        if candidate.text.strip():
            cues.append(AttentionCue(candidate.knoxel_id, "candidate", candidate.text))
    for thought in request.recent_thoughts:
        if thought.text.strip():
            cues.append(AttentionCue(thought.knoxel_id, "recent_thought", thought.text))
    for goal in request.active_goal_records:
        if goal.text.strip():
            cues.append(AttentionCue(goal.knoxel_id, "goal", goal.text))
    for expectation in request.pending_expectations:
        if expectation.text.strip():
            cues.append(AttentionCue(expectation.knoxel_id, "expectation", expectation.text))
    return tuple(cues)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def rate_retained_questions(questions: Sequence[RetainedQuestion], cues: Sequence[AttentionCue], *,
                            question_vectors: Sequence[Sequence[float]],
                            cue_vectors: Sequence[Sequence[float]], salience_bias: float,
                            rng: random.Random) -> tuple[RetainedQuestionRating, ...]:
    """Rate every waiting question against the current cues, in the given order.

    ``question_vectors`` and ``cue_vectors`` are the embeddings of
    ``questions[i].text`` and ``cues[i].text``. Similarity is the highest
    cosine to any one cue, because one related cue is enough to bring a
    question to mind. Each question draws one noise value from ``rng``, so the
    caller must pass the questions in a stable order.
    """
    if len(question_vectors) != len(questions) or len(cue_vectors) != len(cues):
        raise ValueError("Every retained question and every cue needs exactly one embedding vector.")
    cue_record_ids = {cue.record_id for cue in cues if cue.record_id is not None}
    ratings: list[RetainedQuestionRating] = []
    for question, question_vector in zip(questions, question_vectors, strict=True):
        shared_source_ids = tuple(sorted(set(question.source_ids) & cue_record_ids))
        source_overlap = len(shared_source_ids) / len(question.source_ids) if question.source_ids else 0.0

        similarity = 0.0
        most_similar_cue_id: int | None = None
        for cue, cue_vector in zip(cues, cue_vectors, strict=True):
            cue_similarity = _clamp01(cosine(list(question_vector), list(cue_vector)))
            if cue_similarity > similarity:
                similarity = cue_similarity
                most_similar_cue_id = cue.record_id

        relevance = SOURCE_OVERLAP_WEIGHT * source_overlap + SIMILARITY_WEIGHT * similarity
        noise = rng.uniform(-RATING_NOISE, RATING_NOISE) * (1.0 - salience_bias)
        ratings.append(RetainedQuestionRating(
            receipt_id=question.receipt_id, shared_source_ids=shared_source_ids, source_overlap=source_overlap,
            similarity=similarity, most_similar_cue_id=most_similar_cue_id, relevance=relevance, noise=noise,
            rating=_clamp01(relevance + noise)))
    return tuple(ratings)


def select_retained_offer(questions: Sequence[RetainedQuestion], cues: Sequence[AttentionCue], *,
                          question_vectors: Sequence[Sequence[float]], cue_vectors: Sequence[Sequence[float]],
                          salience_bias: float, seed: str) -> RetainedOfferSelection:
    """Pick the one question to offer and return why.

    ``salience_bias`` comes from
    :func:`pm.records.mental_state.compute_attention_bias`. At or above
    ``EXPLOIT_SALIENCE_BIAS`` the highest rating wins, and a newer receipt
    breaks a tie. Below it one question is sampled with weight
    ``rating ** (1 + 4 * salience_bias)``; when every rating is 0 the sample is
    uniform. ``seed`` is the opportunity's own identity, so a restart offers the
    same question for the same opportunity.
    """
    if len(questions) < 2:
        raise ValueError("A selection needs at least two waiting questions; one question is offered without rating.")
    if not 0.0 <= salience_bias <= 1.0:
        raise ValueError("The salience bias must lie in [0, 1].")
    rng = random.Random(seed)
    ordered = sorted(zip(questions, question_vectors, strict=True), key=lambda pair: pair[0].receipt_id)
    ratings = rate_retained_questions([question for question, _ in ordered], cues,
        question_vectors=[vector for _, vector in ordered], cue_vectors=cue_vectors,
        salience_bias=salience_bias, rng=rng)

    if salience_bias >= EXPLOIT_SALIENCE_BIAS:
        best = max(ratings, key=lambda item: (item.rating, item.receipt_id))
        return RetainedOfferSelection(offered_receipt_id=best.receipt_id, mode="highest_rating", seed=seed,
                                      salience_bias=salience_bias, ratings=ratings)

    sharpness = 1.0 + 4.0 * salience_bias
    weights = {item.receipt_id: math.pow(item.rating, sharpness) for item in ratings}
    if not any(weights.values()):
        weights = {receipt_id: 1.0 for receipt_id in weights}
    # Scale the largest weight to 1. The draw depends only on weight ratios, and
    # very small weights would otherwise push every sample key down to 0.0,
    # which turns the draw into a fixed tie-break.
    largest_weight = max(weights.values())
    weights = {receipt_id: weight / largest_weight for receipt_id, weight in weights.items()}
    # Efraimidis-Spirakis keys: the largest key is one weighted draw.
    sample_keys: dict[int, float] = {}
    for item in ratings:
        weight = weights[item.receipt_id]
        draw = rng.random()
        sample_keys[item.receipt_id] = draw ** (1.0 / weight) if weight > 0.0 else 0.0
    offered_receipt_id = max(sample_keys, key=lambda receipt_id: (sample_keys[receipt_id], receipt_id))
    return RetainedOfferSelection(offered_receipt_id=offered_receipt_id, mode="weighted_random", seed=seed,
                                  salience_bias=salience_bias, ratings=ratings)


__all__ = ["AttentionCue", "EXPLOIT_SALIENCE_BIAS", "RATING_NOISE", "RetainedQuestion", "SIMILARITY_WEIGHT",
           "SOURCE_OVERLAP_WEIGHT", "attention_cues", "rate_retained_questions", "select_retained_offer"]
