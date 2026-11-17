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
offers every pending expectation of the real, in-world reality, whichever
loop recorded it (``CycleRequest.open_expectations``, B07 decision D2). The
private loop still opens expectations and may resolve one with observed
evidence (``workflows.inner_thought.expectations``); a second resolution of
the same record is refused by the head guard at commit.

Two rules keep one observation from counting twice (B07 decision D3). Only
observed stimuli are rated (``CycleRequest.observed_stimulus_ids``): a thought
that restates an event is not new evidence. And each expectation keeps the
observations it was already rated against (``counted_observation_ids``); the
check rates it only against observed stimuli not in that list, and an
expectation with none is not checked. Expectations with the same remaining
stimuli share one check call, so a normal cycle still makes one call.

No function here writes the store. The host stages the records and the
updates at commit.
"""
from __future__ import annotations
from workflows.main_cognition.context_reduction import evidence_text
from workflows.main_cognition.model_json import batch_internal_prompt

from typing import Literal, Self
from collections.abc import Sequence
from pm.context_render.budget import ContextGroup
from pm.context_render.turns import Content
from workflows.main_cognition.model_json import model_context_budget
from pm.llm.concurrency import ordered_parallel_map
from workflows.inference_settings import DECISION_WORKERS

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
    EXPECTATION_OPEN_LIMIT,
    EXPECTATION_RELEVANCE_DECAY,
    EXPECTATION_SAME_SIMILARITY,
    EXPECTATION_STRONG_VALENCE,
    EXPECTATION_SUPPORTED,
    EXPECTATION_VALENCE_MULT,
)
from pm_workflows.python_role import RoleContext

from pm.context_render.turns import format_content
from pm.memory.evidence import OBSERVED_EVIDENCE_ROLES, evidence_role
from pm.memory.snapshot import Snapshot
from pm.records.base import RecordBase, RecordScope
from pm.records.situational_model import cosine
from pydantic import model_validator

from workflows.inner_thought.expectations import PRIVATE_SOURCE_REF, pending_expectations
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
    Stimulus,
)
from workflows.main_cognition.story_prefix import StoryPrefix

#: The provenance reference of the expectations that main cognition records.
SOURCE_REF = "main_cognition"


#: The scope main cognition checks expectations in: the real world, in-world frame.
ACCEPTED_REALITY = RecordScope()


def offered_expectations(view: Snapshot) -> list[OpenExpectation]:
    """Return the newest pending expectations of the real, in-world reality, oldest first.

    Main cognition checks every such expectation, whichever loop recorded it
    (B07 decision D2). The private loop still opens expectations and may
    resolve one with observed evidence; whichever commit comes second is
    refused by the head guard, so one expectation is resolved once. An
    expectation in an imagined frame or another reality is not offered, and
    an expectation with an unknown producer raises. At most
    ``EXPECTATION_OPEN_LIMIT`` are offered.
    """
    offered: list[OpenExpectation] = []
    records = [record for record in pending_expectations(view) if record.scope == ACCEPTED_REALITY]
    for record in records[-EXPECTATION_OPEN_LIMIT:]:
        producer = record.provenance.source_ref if record.provenance is not None else None
        if producer == SOURCE_REF:
            origin: Literal["main_cognition", "private_loop"] = "main_cognition"
        elif producer == PRIVATE_SOURCE_REF:
            origin = "private_loop"
        else:
            raise ValueError(f"Expectation {record.id} has no known producer: {producer!r}.")
        offered.append(OpenExpectation(record_id=record.id, content=record.content,
            expected_valence=record.expected_valence, urgency=record.urgency, fulfillment=record.fulfillment,
            tick_id=record.tick_id, revision=view.revisions[record.id],
            about_entity_ids=list(record.about_entity_ids), origin=origin,
            counted_observation_ids=list(record.counted_observation_ids)))
    return offered


def is_observed(view: Snapshot, record: RecordBase) -> bool:
    """Return whether a stimulus record reports an observed event in the accepted reality.

    The rule is the private store owner's (``OBSERVED_EVIDENCE_ROLES``): a
    person's message, the companion's delivered speech, a world observation,
    an executed runtime event, or a recorded internal state. A thought, an
    interpretation, a reconsideration marker, or a summary restates an event
    and moves no expectation (B07 decision D3 a).
    """
    return record.scope == ACCEPTED_REALITY and evidence_role(view, record) in OBSERVED_EVIDENCE_ROLES


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


def uncounted_stimuli(work: CycleWork, expectation: OpenExpectation) -> list[Stimulus]:
    """Return the observed stimuli of this cycle not yet rated against the expectation, in stimulus order."""
    observed = set(work.request.observed_stimulus_ids)
    counted = set(expectation.counted_observation_ids)
    return [item for item in work.request.stimuli if item.knoxel_id in observed and item.knoxel_id not in counted]


def relevant_expectations(work: CycleWork, embedder: Embedder) -> list[OpenExpectation]:
    """Return the open expectations that the new stimuli can affect, most relevant first.

    Port of ``_retrieve_active_expectations``. With ``r`` the recency
    factor and ``s`` the best cosine similarity between the expectation and
    one of its uncounted observed stimuli (:func:`uncounted_stimuli`), the
    relevance is ``(0.5 r + 0.5 s) * (0.5 + 0.25 urgency + 0.25 salience)``,
    clamped to [0, 1]. Expectations above ``EXPECTATION_MIN_RELEVANCE``
    count. At most ``EXPECTATION_CHECK_LIMIT`` are returned. An expectation
    without an uncounted observed stimulus is not checked, so a cycle with
    only thoughts, or only observations already counted, checks nothing and
    makes no embedding request.
    """
    remaining: dict[int, list[Stimulus]] = {}
    for item in recent_expectations(work):
        stimuli = uncounted_stimuli(work, item)
        if stimuli:
            remaining[item.record_id] = stimuli
    candidates = [item for item in recent_expectations(work) if item.record_id in remaining]
    if not candidates:
        return []
    texts = list(dict.fromkeys(stimulus.text for stimuli in remaining.values() for stimulus in stimuli))
    vectors = embedder.prefetch([*texts, *(item.content for item in candidates)])
    scored = []
    for item in candidates:
        similarity = max(max(0.0, min(1.0, cosine(vectors[stimulus.text], vectors[item.content])))
                         for stimulus in remaining[item.record_id])
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


def check(context: RoleContext, work: CycleWork, prefix: StoryPrefix, prompts: PhasePrompts,
          expectations: list[OpenExpectation]) -> list[ExpectationOutcome]:
    """Check the relevant expectations against their uncounted observed stimuli. Return one outcome for each.

    Expectations are grouped by the stimuli they still have to be rated
    against (:func:`uncounted_stimuli`), and each group gets one story-framed
    call; in the normal case every expectation has the same stimuli and the
    cycle makes one call. Outcomes keep the order of ``expectations``.
    Without expectations the function makes no call. An expectation without
    an uncounted observed stimulus is a caller bug: :func:`relevant_expectations`
    never returns one.
    """
    groups: dict[tuple[int, ...], list[OpenExpectation]] = {}
    group_stimuli: dict[tuple[int, ...], list[Stimulus]] = {}
    for item in expectations:
        stimuli = uncounted_stimuli(work, item)
        if not stimuli:
            raise ValueError(f"Expectation {item.record_id} has no uncounted observed stimulus to be checked against.")
        key = tuple(stimulus.knoxel_id for stimulus in stimuli)
        groups.setdefault(key, []).append(item)
        group_stimuli[key] = stimuli
    outcomes: dict[int, ExpectationOutcome] = {}
    for key, members in groups.items():
        for outcome in _check_group(context, work, prefix, prompts, members, group_stimuli[key]):
            outcomes[outcome.record_id] = outcome
    return [outcomes[item.record_id] for item in expectations]


def _check_group(context: RoleContext, work: CycleWork, prefix: StoryPrefix, prompts: PhasePrompts,
                 expectations: list[OpenExpectation], stimuli: list[Stimulus]) -> list[ExpectationOutcome]:
    """Rate one group of expectations against the same stimuli in one call.

    An answer that misses an id, repeats one, or names an unknown one is
    invalid and gets a repair turn. Each outcome's ``observation_ids`` are
    exactly the rated stimuli, which the commit adds to the counted list.
    """
    by_id = {str(item.record_id): item for item in expectations}
    groups = [ContextGroup(str(item.record_id), Content(f"[id {item.record_id}] {item.content}"), salience(item),
                           source_ids=(item.record_id,), recency=float(item.tick_id)) for item in expectations]

    def instruction(chosen: Sequence[ContextGroup]) -> Content:
        return format_content(prefix.fill(prompts.check_template),
            expectations=evidence_text("\n".join(group.content.text for group in chosen), "expectation-items",
                                       [int(group.name) for group in chosen]),
            stimuli=evidence_text("\n".join(f"- {item.text}" for item in stimuli), "expectation-stimuli",
                                  [item.knoxel_id for item in stimuli]))

    batches = batch_internal_prompt(context, prefix.turns, groups, model_context_budget(context), instruction)
    work.receipt("expectation-context", f"{len(expectations)} expectations in {len(batches)} batches; none omitted")

    def rate(index: int) -> ExpectationCheckAnswer:
        batch = batches[index]
        wanted = sorted(by_id[name].record_id for name in batch.included)
        class ExpectationBatch(ExpectationCheckAnswer):
            @model_validator(mode="after")
            def exact(self) -> Self:
                """Require exactly one result per offered expectation."""
                given = sorted(item.expectation_id for item in self.fulfillments)
                if given != wanted:
                    raise ValueError(f"Give exactly one entry for each expectation id: {', '.join(map(str, wanted))}.")
                return self
        return complete(context, work, batch.turns, ExpectationBatch)

    assert context.runtime is not None
    answers = ordered_parallel_map(rate, list(range(len(batches))), workers=DECISION_WORKERS,
                                   cancel=context.runtime.stop_event)
    by_expectation = {item.expectation_id: item for answer in answers for item in answer.fulfillments}
    if set(by_expectation) != {item.record_id for item in expectations}:
        raise ValueError("Expectation batches must cover every offered expectation.")
    observed = [item.knoxel_id for item in stimuli]
    return [outcome_of(item, by_expectation[item.record_id].fulfillment_score,
                       by_expectation[item.record_id].directness_score, observed) for item in expectations]


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


def anticipate(context: RoleContext, work: CycleWork, prefix: StoryPrefix, prompts: PhasePrompts, *, decision: str,
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
    vectors = embedder.prefetch([*(item.content for item in answer.expectations), *(item.content for item in open_items)])
    drafts = []
    for item in answer.expectations:
        vector = vectors[item.content]
        same = [other.record_id for other in open_items
                if cosine(vector, vectors[other.content]) >= EXPECTATION_SAME_SIMILARITY]
        drafts.append(ExpectationDraft(content=item.content, expected_valence=item.affective_valence,
                                       urgency=item.urgency, about_entity_ids=list(entity_ids),
                                       supersedes_ids=same))
    return drafts


__all__ = ["ACCEPTED_REALITY", "EXPECTATION", "SOURCE_REF", "anticipate", "check", "is_observed",
           "offered_expectations", "csm_expectation_ids", "effect", "outcome_of", "recency",
           "recent_expectations", "relevant_expectations", "render_outcomes", "salience", "uncounted_stimuli"]
