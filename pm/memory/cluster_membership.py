"""Resolve a summary's recorded membership in one visible memory snapshot.

Episode membership answers which original features have been accounted for.
The lineage of a new summary separately identifies the evidence used to write
it. Older temporal records may describe membership through child clusters.
"""
from __future__ import annotations

from collections import ChainMap
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict

from pm.records.base import RecordBase
from pm.records.enums import ClusterType
from pm.records.memory_records import Feature, MemoryClusterRecord
from pm.memory.snapshot import Snapshot


class EpisodeMetadata(BaseModel):
    """The episode fields on a persisted memory cluster."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    memory_kind: str = ""
    participant_ids: tuple[int, ...] = ()


def is_episode_cluster(cluster: MemoryClusterRecord) -> bool:
    """Return whether the cluster represents a consolidated episode or an imported topic."""
    if cluster.cluster_type != ClusterType.Topical:
        return False
    if cluster.lineage is not None and cluster.lineage.operation == "memory_reconsolidation.topical":
        return True
    # Main cognition also publishes a small topical experience record each
    # cycle. Only a reconsolidation episode can complete those sources.
    return EpisodeMetadata.model_validate(cluster.metadata).memory_kind != "experience_summary"


def episode_participant_ids(cluster: MemoryClusterRecord) -> tuple[int, ...]:
    """Return the recorded participants. Earlier character homes may have none."""
    return EpisodeMetadata.model_validate(cluster.metadata).participant_ids


def listed_ids(value: str | None) -> tuple[int, ...]:
    """Read the comma-separated IDs on persisted legacy and current clusters."""
    return tuple(int(part) for part in (value or "").split(",") if part.strip())


def cluster_source_links(cluster: MemoryClusterRecord) -> set[int]:
    """Return the recorded members, cited originals and child summaries within this cluster's scope."""
    links = set(listed_ids(cluster.included_event_ids))
    links.update(listed_ids(cluster.included_cluster_ids))
    if cluster.lineage is not None:
        links.update(cluster.lineage.source_event_ids)
        links.update(cluster.lineage.parent_record_ids)
    links.difference_update((0, -1))
    return links


class ClusterMembership:
    """Resolve each cluster once against the same accepted, scoped record view."""

    def __init__(self, records: Mapping[int, RecordBase]) -> None:  # pm-lint: allow PM004 -- accepted records keyed by knoxel ID
        self.records = records
        self.resolved: dict[int, frozenset[int]] = {}  # pm-lint: allow PM004 -- source membership by cluster ID
        self._visiting: set[int] = set()
        self._citations: dict[int, frozenset[int]] = {}  # pm-lint: allow PM004 -- cited sources by cluster ID

    @classmethod
    def from_snapshot(cls, view: Snapshot) -> ClusterMembership:
        """Read heads and their granted historical sources without copying the record index."""
        return cls(ChainMap(view.all_knoxels, view.historical_records))

    def source_ids(self, cluster: MemoryClusterRecord) -> frozenset[int]:
        """Return certified feature membership, or empty for an unverifiable legacy record.

        Direct membership takes precedence over the child links retained for
        navigation. Some legacy parents read children crossing their calendar
        boundary. Credit only the child's sources inside the parent's span.
        """
        if cluster.id in self.resolved:
            return self.resolved[cluster.id]
        if cluster.id in self._visiting:
            raise ValueError(f"Cycle in memory cluster membership at record {cluster.id}.")
        self._visiting.add(cluster.id)
        try:
            result = self._read_members(cluster)
        finally:
            self._visiting.remove(cluster.id)
        self.resolved[cluster.id] = result
        return result

    def _read_members(self, cluster: MemoryClusterRecord) -> frozenset[int]:
        members: set[int] = set()
        if cluster.included_event_ids:
            for source_id in listed_ids(cluster.included_event_ids):
                source = self.records.get(source_id)
                if source is None or source.scope != cluster.scope:
                    return frozenset()
                if isinstance(source, Feature):
                    members.add(source_id)
        elif cluster.included_cluster_ids:
            for child_id in listed_ids(cluster.included_cluster_ids):
                child = self.records.get(child_id)
                if not isinstance(child, MemoryClusterRecord) or child.scope != cluster.scope:
                    return frozenset()
                child_sources = self.source_ids(child)
                if not child_sources:
                    return frozenset()
                for source_id in child_sources:
                    moment = self.records[source_id].timestamp_world_begin
                    if cluster.timestamp_world_begin <= moment <= cluster.timestamp_world_end:
                        members.add(source_id)
        return frozenset(members)

    def evidence_ids(self, cluster: MemoryClusterRecord) -> frozenset[int]:
        """Return the sources used to derive the recollection, independent of coverage.

        A newer summary's lineage identifies cited originals or contributing
        child summaries. A legacy cluster with only a membership list carries
        those members as its original sources.
        """
        if cluster.id in self._citations:
            return self._citations[cluster.id]
        if cluster.id in self._visiting:
            raise ValueError(f"Cycle in memory cluster evidence at record {cluster.id}.")
        if cluster.lineage is None and not cluster.included_cluster_ids:
            result = self.source_ids(cluster)  # The original inputs of a legacy leaf summary.
        else:
            self._visiting.add(cluster.id)
            try:
                cited: set[int] = set()
                if cluster.lineage is None:
                    parent_ids = listed_ids(cluster.included_cluster_ids)
                else:
                    parent_ids = tuple(cluster.lineage.parent_record_ids)
                    for source_id in cluster.lineage.source_event_ids:
                        source = self.records.get(source_id)
                        if source is not None and source.scope == cluster.scope:
                            cited.add(source_id)
                for parent_id in parent_ids:
                    parent = self.records.get(parent_id)
                    if isinstance(parent, MemoryClusterRecord) and parent.scope == cluster.scope:
                        cited.update(self.evidence_ids(parent))
                result = frozenset(cited)
            finally:
                self._visiting.remove(cluster.id)
        self._citations[cluster.id] = result
        return result
