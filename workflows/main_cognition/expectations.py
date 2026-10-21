"""Expectations of main cognition: record what the companion expects, then check it.

Port of three functions of ``private-machine-head/pm/ghosts/ghost_lida.py``
(commit 7101933):

- ``_retrieve_active_expectations`` becomes :func:`relevant_expectations`.
  It scores each open expectation by recency and by similarity to the new
  stimuli.
- ``_appraise_stimulus_check_expectation`` becomes :func:`check` and
  :func:`outcome_of`. A story-framed call rates how directly the stimuli
  address each expectation and how well they fulfill it. Python then moves
  the fulfillment and computes the valence and fear impacts.
- ``_generate_expectations_for_action`` becomes :func:`anticipate`. After
  routing, one story-framed call writes the reaction that the companion now
  expects to each routed action. A new expectation replaces an open one
  when their embeddings are nearly equal. The old code tested
  ``similarity < threshold`` and so closed the unrelated expectations. The
  port closes the similar ones (``EXPECTATION_SAME_SIMILARITY``).

The store holds the expectations as ``ExpectationRecord`` records. The host
offers the pending ones that main cognition recorded
(``CycleRequest.open_expectations``). The private loop keeps its own
expectation lifecycle (``workflows.inner_thought.expectations``) until A19G
moves it onto this cycle. No function here writes the store. The host
stages the records and the updates at commit.
"""
from __future__ import annotations

from pm.constants import (
    EXPECTATION_ANXIETY_MULT,
    EXPECTATION_CHECK_LIMIT,
    EXPECTATION_CONTRADICTED,
    EXPECTATION_DIRECT,
    EXPECTATION_FEAR_BASE,
    EXPECTATION_FEAR_CONFIRMED,
    EXPECTATION_FEAR_DISAPPOINT,
    EXPECTATION_FEAR_RELIEF,
    EXPECTATION_MIN_CHANGE,
    EXPECTATION_MIN_RECENCY,
    EXPECTATION_MIN_RELEVANCE,
    EXPECTATION_RELEVANCE_DECAY,
    EXPECTATION_SAME_SIMILARITY,
    EXPECTATION_STRONG_VALENCE,
    EXPECTATION_SUPPORTED,
    EXPECTATION_VALENCE_MULT,
)
from pm.context_render.turns import format_content
from pm.records.situational_model import cosine
from pydantic import create_model, model_validator

from workflows.main_cognition.csm_view import EXPECTATION, Embedder
from workflows.main_cognition.cycle_work import CycleWork, provisional_tick
from workflows.main_cognition.model_json import PhasePrompts
from workflows.main_cognition.phases.model import complete
from workflows.main_cognition.schemas import (
    ExpectationAnswer,
    ExpectationCheckAnswer,
    ExpectationDraft,
    ExpectationOutcome,
    OpenExpectation,
)
from workflows.main_cognition.story_prefix import StoryPrefix

#: The provenance reference of the expectations that main cognition records.
SOURCE_REF = "main_cognition"


def recency(work: CycleWork, expectation: OpenExpectation) -> float:
    """Return ``EXPECTATION_RELEVANCE_DECAY`` to the power of the ticks since the expectation was recorded."""
    return EXPECTATION_RELEVANCE_DECAY ** max(0, provisional_tick(work) - expectation.tick_id)


def csm_expectation_ids(work: CycleWork) -> list[int]:
    """Return the ids of the recent expectations that stay open after the check of this cycle."""
    closed = {item.record_id for item in work.expectations if item.status != "pending"}
    return [item.record_id for item in recent_expectations(work) if item.record_id not in closed]


def recent_expectations(work: CycleWork) -> list[OpenExpectation]:
    """Return the offered expectations that are not too old (``EXPECTATION_MIN_RECENCY``)."""
    return [item for item in work.request.open_expectations if recency(work, item) >= EXPECTATION_MIN_RECENCY]


def salience(expectation: OpenExpectation) -> float:
    """Return the old incentive salience of an expectation: from its wanted valence and its urgency."""
    return max(0.1, min(0.9, abs(expectation.expected_valence) * 0.5 + expectation.urgency * 0.5))


def relevant_expectations(work: CycleWork, embedder: Embedder) -> list[OpenExpectation]:
    """Return the open expectations that the new stimuli can affect, most relevant first.

    Port of ``_retrieve_active_expectations``. With ``r`` the recency
    factor and ``s`` the best cosine similarity between the expectation and
    one stimulus, the relevance is
    ``(0.5 r + 0.5 s) * (0.5 + 0.25 urgency + 0.25 salience)``, clamped to
    [0, 1]. Expectations above ``EXPECTATION_MIN_RELEVANCE`` count. At most
    ``EXPECTATION_CHECK_LIMIT`` are returned. A cycle without stimuli
    checks nothing.
    """
    stimuli = [item.text for item in work.request.stimuli]
    candidates = recent_expectations(work)
    if not stimuli or not candidates:
        return []
    embedder.prefetch([*stimuli, *(item.content for item in candidates)])
    scored = []
    for item in candidates:
        similarity = max(max(0.0, min(1.0, cosine(embedder.text(text), embedder.text(item.content))))
                         for text in stimuli)
        score = (recency(work, item) * 0.5 + similarity * 0.5) * (0.5 + item.urgency * 0.25 + salience(item) * 0.25)
        score = max(0.0, min(1.0, score))
        if score > EXPECTATION_MIN_RELEVANCE:
            scored.append((score, item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored[:EXPECTATION_CHECK_LIMIT]]


def outcome_of(expectation: OpenExpectation, fulfillment_score: float, directness_score: float,
               observation_ids: list[int]) -> ExpectationOutcome:
    """Return the outcome of one check. Pure function.

    Port of the update rule of ``_appraise_stimulus_check_expectation``:

    - ``change = (fulfillment_score - previous) * directness_score``. The
      fulfillment moves toward the check result only as far as the stimulus
      addressed the expectation.
    - When ``|change|`` exceeds ``EXPECTATION_MIN_CHANGE``, the valence
      impact is ``change * wanted valence * EXPECTATION_VALENCE_MULT``. A met
      wanted outcome feels good, a met feared outcome feels bad, a failed
      wanted outcome disappoints, a failed feared outcome relieves.
    - The fear impact is the old anxiety impact: a met expectation lowers
      fear, a failed one raises it, with extra fear for a disappointment or a
      confirmed fear and less fear for relief.

    The status is ``supported`` at a fulfillment of ``EXPECTATION_SUPPORTED``
    or more, ``contradicted`` for a direct stimulus that scores at or below
    ``EXPECTATION_CONTRADICTED``, and ``pending`` otherwise.
    """
    previous = expectation.fulfillment
    change = (fulfillment_score - previous) * directness_score
    fulfillment = max(0.0, min(1.0, previous + change))
    wanted = expectation.expected_valence
    valence_impact = 0.0
    fear_impact = 0.0
    if abs(change) > EXPECTATION_MIN_CHANGE:
        valence_impact = change * wanted * EXPECTATION_VALENCE_MULT
        fear_impact = -change * EXPECTATION_FEAR_BASE * EXPECTATION_ANXIETY_MULT
        if wanted > EXPECTATION_STRONG_VALENCE and change < 0:
            fear_impact += abs(change) * EXPECTATION_FEAR_DISAPPOINT * EXPECTATION_ANXIETY_MULT
        elif wanted < -EXPECTATION_STRONG_VALENCE and change > 0:
            fear_impact += abs(change) * EXPECTATION_FEAR_CONFIRMED * EXPECTATION_ANXIETY_MULT
        elif wanted < -EXPECTATION_STRONG_VALENCE and change < 0:
            fear_impact -= abs(change) * EXPECTATION_FEAR_RELIEF * EXPECTATION_ANXIETY_MULT
    if fulfillment >= EXPECTATION_SUPPORTED:
        status = "supported"
    elif fulfillment_score <= EXPECTATION_CONTRADICTED and directness_score >= EXPECTATION_DIRECT:
        status = "contradicted"
    else:
        status = "pending"
    return ExpectationOutcome(
        record_id=expectation.record_id, revision=expectation.revision, content=expectation.content,
        previous=previous, fulfillment=round(fulfillment, 4), change=round(change, 4),
        fulfillment_score=fulfillment_score, directness_score=directness_score,
        valence_impact=round(valence_impact, 4), fear_impact=round(fear_impact, 4), status=status,
        observation_ids=list(observation_ids))


def check(context, work: CycleWork, prefix: StoryPrefix, prompts: PhasePrompts,
          expectations: list[OpenExpectation]) -> list[ExpectationOutcome]:
    """Check the relevant expectations against the new stimuli. Return one outcome for each.

    One story-framed call rates every expectation. An answer that misses an
    id, repeats one, or names an unknown one is invalid and gets a repair
    turn. Without expectations the function makes no call.
    """
    if not expectations:
        return []
    wanted = sorted(item.record_id for item in expectations)

    def exact(answer):
        given = sorted(item.expectation_id for item in answer.fulfillments)
        if given != wanted:
            raise ValueError(f"Give exactly one entry for each expectation id: {', '.join(map(str, wanted))}.")
        return answer

    schema = create_model("ExpectationCheckChecked", __base__=ExpectationCheckAnswer,
                          __validators__={"exact": model_validator(mode="after")(exact)})
    instruction = format_content(prefix.fill(prompts.check_template),
        expectations="\n".join(f"[id {item.record_id}] {item.content}" for item in expectations),
        stimuli="\n".join(f"- {item.text}" for item in work.request.stimuli),
    ).text
    answer = complete(context, work, prefix.ask(instruction), schema)
    by_id = {item.expectation_id: item for item in answer.fulfillments}
    observed = [item.knoxel_id for item in work.request.stimuli]
    return [outcome_of(item, by_id[item.record_id].fulfillment_score, by_id[item.record_id].directness_score,
                       observed) for item in expectations]


def effect(outcomes: list[ExpectationOutcome]) -> tuple[float, float]:
    """Return the summed expectation error and fear change of the outcomes.

    The expectation error is the sum of the valence impacts, clamped to
    -1..1. The appraisal stores it as ``expectation_error``. The fear change
    joins the explicit emotion deltas of the appraisal.
    """
    error = max(-1.0, min(1.0, sum(item.valence_impact for item in outcomes)))
    fear = max(-1.0, min(1.0, sum(item.fear_impact for item in outcomes)))
    return error, fear


def render_outcomes(outcomes: list[ExpectationOutcome]) -> str:
    """Render the outcomes as story lines for the appraisal prompt."""
    lines = []
    for item in outcomes:
        if item.status == "supported":
            verdict = "it came true"
        elif item.status == "contradicted":
            verdict = "the events contradict it"
        elif abs(item.change) > EXPECTATION_MIN_CHANGE:
            verdict = "it came closer" if item.change > 0 else "it moved further away"
        else:
            verdict = "nothing decided yet"
        lines.append(f"- expected: {item.content} -> {verdict}")
    return "\n".join(lines) or "(nothing in particular)"


def anticipate(context, work: CycleWork, prefix: StoryPrefix, prompts: PhasePrompts, *, decision: str,
               action: str, target: str, count: int, entity_ids: list[int],
               embedder: Embedder) -> list[ExpectationDraft]:
    """Return the reactions that the companion now expects to one routed action.

    Port of ``_generate_expectations_for_action``. ``decision`` is the
    decision text, ``action`` describes the routed action in words, and
    ``target`` names whose reaction is expected. ``count`` is the most
    expectations that the call writes. Each expectation that is
    nearly equal to an open expectation (cosine similarity of at least
    ``EXPECTATION_SAME_SIMILARITY``) supersedes it. Runs in a worker thread
    and does not change ``work``.
    """
    instruction = format_content(prefix.fill(prompts.anticipation_template),
        decision=decision, action=action, target=target, count=count).text
    answer = complete(context, work, prefix.ask(instruction), ExpectationAnswer)
    open_items = recent_expectations(work)
    embedder.prefetch([*(item.content for item in answer.expectations), *(item.content for item in open_items)])
    drafts = []
    for item in answer.expectations:
        vector = embedder.text(item.content)
        same = [other.record_id for other in open_items
                if cosine(vector, embedder.text(other.content)) >= EXPECTATION_SAME_SIMILARITY]
        drafts.append(ExpectationDraft(content=item.content, expected_valence=item.affective_valence,
                                       urgency=item.urgency, about_entity_ids=list(entity_ids),
                                       supersedes_ids=same))
    return drafts


__all__ = ["EXPECTATION", "SOURCE_REF", "anticipate", "check", "csm_expectation_ids", "effect", "outcome_of", "recency",
           "recent_expectations", "relevant_expectations", "render_outcomes", "salience"]
