"""Select the recorded events of an episode at one accepted memory frontier.

Dialogue and world observations define most episode boundaries. A retained
finding or a private event actually used by a main decision can anchor a
private episode. Other thoughts may belong to a scene without being quoted.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from pm.constants import RECONSOLIDATION_SESSION_GAP_MINUTES
from pm.memory.cluster_membership import ClusterMembership, is_episode_cluster
from pm.memory.reconsolidation.ordering import chronological_records
from pm.memory.reconsolidation.schemas import ReconsolidationConfig
from pm.memory.snapshot import Snapshot
from pm.records.base import RecordScope
from pm.records.enums import FeatureType
from pm.records.memory_records import Feature, MemoryClusterRecord


def covered_episode_ids(clusters: Iterable[MemoryClusterRecord], membership: ClusterMembership) -> set[int]:
    covered: set[int] = set()
    for cluster in clusters:
        if cluster.id in membership.records and is_episode_cluster(cluster):
            covered.update(membership.source_ids(cluster))
    return covered


def external_event(feature: Feature, config: ReconsolidationConfig) -> bool:
    return feature.causal and feature.feature_type in config.cluster_feature_types


def eligible_external_features(features: Sequence[Feature], clusters: Sequence[MemoryClusterRecord],
                               now: datetime, config: ReconsolidationConfig, membership: ClusterMembership) -> list[Feature]:
    """Return external history known to precede each scope's still-open topic.

    A long uninterrupted conversation already has closed episodes. The final
    block from split_block can contain at most twice the minimum minus one
    anchors. Keep that tail open without running clustering in a foreground turn.
    """
    covered = covered_episode_ids(clusters, membership)
    by_scope: dict[RecordScope, list[Feature]] = {}  # pm-lint: allow PM004 -- features by reality and frame
    for feature in features:
        if feature.timestamp_world_begin <= now and external_event(feature, config) and feature.id not in covered:
            by_scope.setdefault(feature.scope, []).append(feature)
    closed: list[Feature] = []
    gap = timedelta(minutes=RECONSOLIDATION_SESSION_GAP_MINUTES)
    cutoff = now - timedelta(minutes=config.open_topic_minutes)
    for scoped in by_scope.values():
        ordered = chronological_records(scoped)
        end = len(ordered)
        if ordered[-1].timestamp_world_begin > cutoff:
            open_start = end - 1
            while open_start and ordered[open_start].timestamp_world_begin - ordered[open_start - 1].timestamp_world_begin <= gap:
                open_start -= 1
            largest_open_block = 2 * config.min_block - 1
            end = max(open_start, len(ordered) - largest_open_block)
        closed.extend(ordered[:end])
    return chronological_records(closed)


@dataclass(frozen=True, slots=True)
class EpisodeSources:
    """Full membership and the smaller set actually used to describe it."""

    members: tuple[Feature, ...]
    evidence: tuple[Feature, ...]
    participant_ids: tuple[int, ...]


class EpisodeFrontier:
    """Original records, source membership and causal salience for one pass."""

    def __init__(self, view: Snapshot, config: ReconsolidationConfig, now: datetime) -> None:
        self.config = config
        self.consequential_ids: set[int] = set()
        self.private_actions: set[int] = set()
        self.affect_scopes: dict[int, RecordScope] = {}  # pm-lint: allow PM004 -- unambiguous reality of each main tick
        for turn in view.gwt_turns.values():
            if turn.slot_id != "main":
                continue
            self.consequential_ids.update(turn.stimulus_ids)
            self.consequential_ids.update(turn.trace.causal_percept_ids)
            scopes = {view.all_knoxels[key].scope for key in turn.stimulus_ids if key in view.all_knoxels}
            if len(scopes) == 1:
                self.affect_scopes[turn.tick_id] = scopes.pop()
            for key in (turn.decision_id, turn.experience_id):
                if key is None or key not in view.all_knoxels:
                    continue
                record = view.get(key)
                self.consequential_ids.add(key)
                self.private_actions.add(key)
                if record.lineage is not None:
                    self.consequential_ids.update(record.lineage.source_event_ids)
                    self.consequential_ids.update(record.lineage.parent_record_ids)
        self.features: list[Feature] = []
        for record in view.all_knoxels.values():
            if not isinstance(record, Feature) or record.timestamp_world_begin > now or not record.content.strip():
                continue
            finding = record.metadata.get("private_inquiry_finding") is True
            if finding:
                self.consequential_ids.add(record.id)
            if record.feature_type in config.story_feature_types and (record.causal or record.id in self.consequential_ids):
                self.features.append(record)
            elif record.feature_type == FeatureType.StateUpdate and record.id in self.consequential_ids:
                self.features.append(record)
        self.features = chronological_records(self.features)

    def is_anchor(self, feature: Feature) -> bool:
        return external_event(feature, self.config) or feature.id in self.consequential_ids

    def sources(self, members: Sequence[Feature], observer_id: int | None) -> EpisodeSources:
        external: list[Feature] = []
        private: list[Feature] = []
        participants: set[int] = set()
        for feature in members:
            if external_event(feature, self.config):
                external.append(feature)
            elif feature.id in self.consequential_ids:
                private.append(feature)
            if feature.feature_type == FeatureType.Dialogue:
                participants.update(key for key in (feature.entity_id, feature.addressee_entity_id) if key > 0)
            elif feature.feature_type != FeatureType.WorldEvent and feature.entity_id > 0:
                participants.add(feature.entity_id)
            elif observer_id is not None and feature.id in self.private_actions:
                participants.add(observer_id)
        # Give the model one example of each consequential kind per main
        # decision. Keep explicit inquiry findings even when they share a tick.
        private_kinds: set[tuple[int, FeatureType]] = set()
        evidence = list(external)
        for feature in private:
            kind = (feature.tick_id, feature.feature_type)
            if feature.metadata.get("private_inquiry_finding") is True or kind not in private_kinds:
                evidence.append(feature)
                private_kinds.add(kind)
        return EpisodeSources(tuple(members), tuple(chronological_records(evidence)), tuple(sorted(participants)))
