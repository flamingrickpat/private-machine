"""Render selected architectural evidence without exposing persisted bookkeeping."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel

from pm.context_render import Content, render_context
from pm.context_render.budget import ContextGroup
from pm.context_render.records import ArchitecturalEvidence
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


def _episode_ids(episode: ReflectionEpisode) -> set[int]:
    source_ids = set(episode.inputs) | set(episode.outputs)
    for linked in episode.links.values():
        source_ids.update(linked)
    return source_ids


def _sources(records: Mapping[int, ReflectionSource], source_ids: set[int]) -> Content:
    return render_context(ArchitecturalEvidence(tuple(records[key] for key in sorted(source_ids))))


def context_groups(value: ReflectionContext, salience: Mapping[int, float], *,
                   required_source_ids: frozenset[int] = frozenset(),
                   required_episode_ids: frozenset[int] = frozenset()) -> list[ContextGroup]:
    """Keep each episode and its supplied sources together during shared selection.

    Active experiment evidence is required. Optional episode and record groups
    use the workflow's existing urgency and creation order. No model scores run.
    """
    records = {record.knoxel_id: record for record in value.records}
    groups = [ContextGroup("Reflection status", render_context(status(value)), 1, required=True)]
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
        groups.append(ContextGroup(f"Experiment {entry.record_id}", render_context(ExperimentSummary.from_evidence(entry)), 1, required=True))
    for episode in value.episodes:
        if episode.episode_id in required_episodes:
            required_ids.update(_episode_ids(episode) & records.keys())
            groups.append(ContextGroup(f"Required episode {episode.episode_id}", render_context(episode, schema_path=Path(__file__).parents[2] / "pm/context_render/architectural.json"), 1, required=True, episode_ids=(episode.episode_id,)))
    if required_ids:
        groups.append(ContextGroup("Required experiment sources", _sources(records, required_ids), 1,
                                   required=True, source_ids=tuple(sorted(required_ids))))
    grouped = set(required_ids)
    for episode in value.episodes:
        if episode.episode_id in required_episodes:
            continue
        ids = _episode_ids(episode) & records.keys()
        content = render_context(episode, schema_path=Path(__file__).parents[2] / "pm/context_render/architectural.json") + "\n" + _sources(records, ids)
        priority = max((salience.get(key, 0) for key in ids), default=0) + episode.tick_id * 0.000001
        groups.append(ContextGroup(f"Episode {episode.episode_id}", content, priority, source_ids=tuple(sorted(ids)), episode_ids=(episode.episode_id,)))
        grouped.update(ids)
    for record in value.records:
        if record.knoxel_id not in grouped:
            priority = salience.get(record.knoxel_id, 0) + record.tick_id * 0.000001
            groups.append(ContextGroup(f"Source {record.knoxel_id}", render_context(record), priority,
                                       source_ids=(record.knoxel_id,)))
    return groups
