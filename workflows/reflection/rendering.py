"""Render selected architectural evidence without exposing persisted bookkeeping."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from pydantic import BaseModel

from pm.context_render import Content
from pm.context_render.budget import ContextGroup
from pm.context_render.records import render_record
from workflows.reflection.schemas import (
    ExperimentEvidence, ExperimentPlan, ExperimentResult, ExperimentState,
    ObservationGrant, ReflectionContext, ReflectionEpisode, ReflectionSource,
)


class ExperimentSummary(BaseModel):
    """The prediction and lifecycle facts a reflection decision can inspect."""

    record_id: int
    experiment_id: int
    state: ExperimentState
    plan: ExperimentPlan
    effective_record_id: int | None
    source_ids: list[int]
    stale_source_ids: list[int]
    observation_grant: ObservationGrant
    result: ExperimentResult

    @classmethod
    def from_evidence(cls, entry: ExperimentEvidence) -> ExperimentSummary:
        """Select semantic fields. Leave revision maps and ledger history on disk."""
        return cls(record_id=entry.record_id, experiment_id=entry.experiment_id, state=entry.metadata.state, plan=entry.metadata.plan,
            effective_record_id=entry.metadata.effective_record_id, source_ids=sorted(entry.metadata.source_revisions),
            stale_source_ids=entry.stale_source_ids, observation_grant=entry.observation_grant, result=entry.metadata.result)


class ReflectionStatus(BaseModel):
    """Trigger-stage status. The reflection stage owns detailed architectural analysis."""

    review_due: bool
    attention_head_id: int | None
    newest_observation_id: int
    experiment_ids: list[int]
    omitted_experiments: int
    evidence_limits: list[str]


def status(value: ReflectionContext) -> ReflectionStatus:
    """Describe deterministic reflection availability without duplicating its record grant."""
    return ReflectionStatus(review_due=value.review_due, attention_head_id=value.attention_head_id,
        newest_observation_id=value.newest_observation_id, experiment_ids=[entry.experiment_id for entry in value.experiments],
        omitted_experiments=value.omitted_experiments, evidence_limits=value.episode_limits)


def render_status(value: ReflectionStatus) -> Content:
    lines = [f"Reflection review due: {value.review_due}; attention head: {value.attention_head_id}; "
             f"newest observation: {value.newest_observation_id}.",
             "Experiments: " + ", ".join(str(key) for key in value.experiment_ids)]
    if value.omitted_experiments:
        lines.append(f"Other experiments not supplied: {value.omitted_experiments}.")
    lines.extend("Evidence limit: " + limit for limit in value.evidence_limits)
    return Content("\n".join(lines))


def render_experiment(value: ExperimentSummary) -> Content:
    plan = value.plan
    intervention = plan.intervention
    lines = [f"Experiment {value.experiment_id}; ledger source {value.record_id}; state {value.state}; "
             f"effective source {value.effective_record_id}; result {value.result}.",
             f"Trigger episode {plan.triggering_episode_id}; {plan.diagnosis}: {plan.finding}",
             f"Scope: {plan.scope}; learning: {plan.learning}.",
             f"Prediction: {plan.predicted_effect}", f"Distinguishing observation: {plan.discriminating_observation}",
             f"Eligible observation event: {plan.observation_event}",
             f"Intervention owner: {intervention.owner}; expected source: {intervention.expected_record_id}."]
    if intervention.belief:
        lines.append("Proposed belief: " + intervention.belief)
    if intervention.focus is not None:
        focus = intervention.focus
        lines.append(f"Temporary attention: {focus.instruction}; strength {focus.strength:.3f}; "
                     f"duration {focus.seconds:g}s; promote {focus.promote_ids}; reduce {focus.reduce_ids}; "
                     f"topics {', '.join(focus.topic_hints)}. Reason: {focus.reason}")
    for hypothesis in plan.hypotheses:
        lines.append(f"Hypothesis ({hypothesis.confidence:.3f}): {hypothesis.explanation}. "
                     f"Uncertainty: {hypothesis.uncertainty}; support {hypothesis.supporting_ids}; "
                     f"opposition {hypothesis.opposing_ids}.")
    lines.append(f"Original sources: {value.source_ids}; stale sources: {value.stale_source_ids}.")
    lines.append(f"Consuming decisions: {value.observation_grant.consumer_ids}; "
                 f"eligible observations: {value.observation_grant.observation_ids}.")
    return Content("\n".join(lines))


def render_episode(value: ReflectionEpisode) -> Content:
    state = value.metadata
    lines = [f"Episode {value.episode_id}; tick {value.tick_id}; {value.kind}; {value.loop}/{value.operation}; "
             f"owner {value.owner}; revision {value.revision}.", f"Summary: {value.summary}",
             f"Recorded outcome: {value.outcome}; input sources {value.inputs}; output sources {value.outputs}."]
    for label, ids in value.links.items():
        lines.append(f"{label}: {ids}")
    for label, text in (("Trigger", state.trigger), ("Reality", state.reality), ("Expectation", state.expectation),
                        ("Execution mode", state.execution_mode), ("Inquiry outcome", state.inquiry_outcome),
                        ("Reason", state.reason), ("Information gap", state.information_gap)):
        if text:
            lines.append(f"{label}: {text}")
    if state.uncertainty is not None:
        lines.append(f"Recorded uncertainty: {state.uncertainty}")
    for alternative in state.alternatives:
        lines.append(f"Alternative ({alternative.disposition}): {alternative.text}")
    if state.progress is not None:
        progress = state.progress
        lines.append(f"Progress: {progress.kind}; meaningful difference {progress.meaningful_difference}; {progress.reason}")
    for relation in state.relations:
        lines.append(f"Relation to {relation.target_record_id}: {relation.kind}; {relation.reason}")
    lines.append(f"Structures: {state.structure_ids}; causal percepts: {state.causal_percept_ids}.")
    for run in state.codelet_runs:
        lines.append(f"Mechanism {run.codelet}; mode {run.mode}; wave {run.wave}; {run.status}; "
                     f"activation {run.score:.3f}; calls {run.calls}; reason {run.reason}.")
    return Content("\n".join(lines))


def render_source(value: ReflectionSource) -> Content:
    content = render_record(value)
    lines = [f"Belief kind: {value.belief_kind}; active: {value.active}; "
             f"supporting sources {value.supporting_ids}; opposing sources {value.opposing_ids}."]
    metadata = value.metadata
    if metadata.operation or metadata.loop or metadata.status:
        lines.append(f"Operation: {metadata.loop}/{metadata.operation}; state {metadata.status}; "
                     f"control {metadata.control_kind}; origin {metadata.origin}.")
    if metadata.reflection_versions:
        lines.append("Effective experiments: " + "; ".join(f"{key}: source {record_id}" for key, record_id in metadata.reflection_versions.items()))
    if metadata.memory_evidence:
        lines.append("Earlier selected memory sources: " + ", ".join(str(row.knoxel_id) for row in metadata.memory_evidence))
    return content + "\n" + "\n".join(lines)


def _episode_ids(episode: ReflectionEpisode) -> set[int]:
    source_ids = set(episode.inputs) | set(episode.outputs)
    for linked in episode.links.values():
        source_ids.update(linked)
    return source_ids


@dataclass(frozen=True, slots=True)
class PrivateReflectionSettings:
    maximum_experiments: int = 4
    maximum_episodes: int = 3


def private_material(value: ReflectionContext | None, settings: PrivateReflectionSettings = PrivateReflectionSettings()) -> Content:
    """Render bounded reflection evidence for one shared inquiry step.

    The reflection child selects its own groups against a token budget. A
    shared inquiry step gets one prompt and makes no separate selection call,
    so this projection is bounded by count and states what it omitted. It uses
    the same typed renderers as the reflection child, so the prompt carries no
    raw bookkeeping. The sources named by the rendered entries are included
    with their captured text, because a granted record without its text cannot
    be cited honestly.
    """
    if value is None:
        return Content("(no reflection evidence was supplied for this cycle)")
    shown_experiments = value.experiments[:settings.maximum_experiments]
    shown_episodes = value.episodes[:settings.maximum_episodes]
    if not shown_experiments and not shown_episodes:
        return Content("(no reflection evidence was supplied for this cycle)")
    parts: list[Content] = [render_status(status(value))]
    for entry in shown_experiments:
        parts.append(render_experiment(ExperimentSummary.from_evidence(entry)))
    if len(value.experiments) > len(shown_experiments):
        parts.append(Content(f"({len(value.experiments) - len(shown_experiments)} more experiments were not rendered.)"))
    for episode in shown_episodes:
        parts.append(render_episode(episode))
    if len(value.episodes) > len(shown_episodes):
        parts.append(Content(f"({len(value.episodes) - len(shown_episodes)} more episodes were not rendered.)"))
    named: set[int] = set()
    for entry in shown_experiments:
        named.update(entry.metadata.source_revisions)
        named.update(entry.observation_grant.consumer_ids)
        named.update(entry.observation_grant.observation_ids)
        if entry.metadata.effective_record_id is not None:
            named.add(entry.metadata.effective_record_id)
        if entry.metadata.plan.intervention.expected_record_id is not None:
            named.add(entry.metadata.plan.intervention.expected_record_id)
    for episode in shown_episodes:
        named.update(_episode_ids(episode))
    records = {record.knoxel_id: record for record in value.records}
    supplied = [records[key] for key in sorted(named) if key in records]
    if supplied:
        parts.extend(render_source(record) for record in supplied)
    material = parts[0]
    for part in parts[1:]:
        material = material + "\n" + part
    return material


def _sources(records: Mapping[int, ReflectionSource], source_ids: set[int]) -> Content:
    result = Content("")
    for key in sorted(source_ids):
        result += render_source(records[key]) + "\n"
    return result


def context_groups(value: ReflectionContext, salience: Mapping[int, float], *,
                   required_source_ids: frozenset[int] = frozenset(),
                   required_episode_ids: frozenset[int] = frozenset()) -> list[ContextGroup]:
    """Keep each episode and its supplied sources together during shared selection.

    Active experiment evidence is required. Optional episode and record groups
    use the workflow's existing urgency and creation order. No model scores run.
    """
    records = {record.knoxel_id: record for record in value.records}
    groups = [ContextGroup("Reflection status", render_status(status(value)), 1, required=True)]
    required_ids: set[int] = set(required_source_ids)
    required_episodes: set[int] = set(required_episode_ids)
    for entry in value.experiments:
        required_ids.add(entry.record_id)
        required_ids.update(entry.metadata.source_revisions)
        required_ids.update(entry.observation_grant.consumer_ids)
        required_ids.update(entry.observation_grant.observation_ids)
        if entry.metadata.effective_record_id is not None:
            required_ids.add(entry.metadata.effective_record_id)
        required_episodes.add(entry.metadata.plan.triggering_episode_id)
        groups.append(ContextGroup(f"Experiment {entry.record_id}", render_experiment(ExperimentSummary.from_evidence(entry)), 1, required=True))
    for episode in value.episodes:
        if episode.episode_id in required_episodes:
            required_ids.update(_episode_ids(episode) & records.keys())
            groups.append(ContextGroup(f"Required episode {episode.episode_id}", render_episode(episode), 1, required=True, episode_ids=(episode.episode_id,)))
    if required_ids:
        groups.append(ContextGroup("Required experiment sources", _sources(records, required_ids), 1,
                                   required=True, source_ids=tuple(sorted(required_ids))))
    grouped = set(required_ids)
    for episode in value.episodes:
        if episode.episode_id in required_episodes:
            continue
        ids = _episode_ids(episode) & records.keys()
        content = render_episode(episode) + "\n" + _sources(records, ids)
        priority = max((salience.get(key, 0) for key in ids), default=0) + episode.tick_id * 0.000001
        groups.append(ContextGroup(f"Episode {episode.episode_id}", content, priority, source_ids=tuple(sorted(ids)), episode_ids=(episode.episode_id,)))
        grouped.update(ids)
    for record in value.records:
        if record.knoxel_id not in grouped:
            priority = salience.get(record.knoxel_id, 0) + record.tick_id * 0.000001
            groups.append(ContextGroup(f"Source {record.knoxel_id}", render_source(record), priority,
                                       source_ids=(record.knoxel_id,)))
    return groups
