"""Measure the external experience a character can currently sort into episodes.

Main cognition consults the assessment on its existing frozen read grant. Only
closed, unrepresented external material contributes. Story measures a different
quantity when it actually renders a reply. Its most recent observation carries
a time and a source so a later decision can distinguish the two measurements.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from math import ceil

from pydantic import BaseModel, ConfigDict, Field

from pm.constants import PROMPT_IMAGE_TOKENS, RECONSOLIDATION_SESSION_GAP_MINUTES
from pm.context_render.budget import TokenEstimator
from pm.memory.cluster_membership import ClusterMembership, is_episode_cluster
from pm.memory.reconsolidation.membership import eligible_external_features
from pm.memory.reconsolidation.schemas import ReconsolidationConfig
from pm.memory.snapshot import Snapshot
from pm.records.base import RecordScope
from pm.records.media import IMAGE_TAG
from pm.records.memory_records import Feature, MemoryClusterRecord


class StoryHistoryObservation(BaseModel):
    """Actual history coverage of a rendered public Story generation prompt."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    observed_at: datetime
    current_message_id: int | None
    covered_features: int = Field(ge=0)
    omitted_features: int = Field(ge=0)
    minimum_cover_prompt_tokens: int = Field(ge=0)
    selected_prompt_tokens: int = Field(ge=0)
    available_history_tokens: int | None = Field(default=None, ge=0)


class MemoryWorkAssessment(BaseModel):
    """External sources available for work and optional evidence from a previous reply."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    sessions_with_work: int = Field(ge=0)
    external_features: int = Field(ge=0)
    estimated_raw_text_tokens: int = Field(ge=0)
    oldest_source_id: int | None
    oldest_experience: datetime | None
    last_story: StoryHistoryObservation | None

    def decision_evidence(self) -> str:
        """Describe actionable material once as private context in the choice prompt."""
        if not self.external_features:
            return ""
        if self.oldest_experience is None:
            raise ValueError("An external memory backlog requires the oldest experience time.")
        noun = "exchange or observation" if self.external_features == 1 else "exchanges or observations"
        line = (f"{self.external_features} earlier {noun} could be filed into episodes. "
                f"The oldest dates to {self.oldest_experience:%d %b %Y}.")
        last = self.last_story
        if last is not None:
            previous_reply = last.observed_at.strftime("%H:%M on %d %b")
            if last.available_history_tokens is not None and self.estimated_raw_text_tokens > last.available_history_tokens:
                line += (f" Recalling all of that word for word would likely use more space than was available "
                         f"for history in the reply at {previous_reply}.")
            if last.omitted_features:
                line += f" Some earlier experiences did not fit in the reply at {previous_reply}."
        return line


@dataclass(frozen=True, slots=True)
class FeatureTextCost:
    revision: int
    tokens: int


class MemoryWorkReader:
    """Measure eligibility on each frozen read grant. Cache only the cost of unchanged text."""

    def __init__(self) -> None:
        self.config = ReconsolidationConfig()
        self.estimator = TokenEstimator()
        self.text_costs: dict[int, FeatureTextCost] = {}  # pm-lint: allow PM004 -- token cost by feature ID
        self.last_story: StoryHistoryObservation | None = None

    def observe_story(self, observation: StoryHistoryObservation) -> None:
        """Keep the latest measured prompt when replies finish out of order."""
        if self.last_story is None or self.last_story.observed_at <= observation.observed_at:
            self.last_story = observation

    def assess(self, view: Snapshot, scopes: set[RecordScope], now: datetime) -> MemoryWorkAssessment:
        """Read external memory from the supplied frontier without inference or embedding work."""
        selected_scopes = tuple(scopes) if scopes else (RecordScope(),)
        features: list[Feature] = []
        clusters: list[MemoryClusterRecord] = []
        for record in view.all_knoxels.values():
            if record.scope not in selected_scopes:
                continue
            if isinstance(record, Feature) and record.causal and record.feature_type in self.config.cluster_feature_types:
                features.append(record)
            elif isinstance(record, MemoryClusterRecord) and is_episode_cluster(record):
                clusters.append(record)

        membership = ClusterMembership(view.all_knoxels)
        eligible = eligible_external_features(features, clusters, now, self.config, membership)
        gap = timedelta(minutes=RECONSOLIDATION_SESSION_GAP_MINUTES)
        last_event: dict[RecordScope, datetime] = {}  # pm-lint: allow PM004 -- last eligible event time by reality and frame
        next_costs: dict[int, FeatureTextCost] = {}  # pm-lint: allow PM004 -- text cost by eligible feature ID
        sessions = 0
        tokens = 0
        for feature in eligible:
            previous = last_event.get(feature.scope)
            if previous is None or feature.timestamp_world_begin - previous > gap:
                sessions += 1
            last_event[feature.scope] = feature.timestamp_world_begin
            revision = view.revisions[feature.id]
            cost = self.text_costs.get(feature.id)
            if cost is None or cost.revision != revision:
                label = feature.override_source or feature.feature_type.value
                text = f"{label}: {feature.content}"
                image_tokens = len(IMAGE_TAG.findall(text)) * PROMPT_IMAGE_TOKENS
                estimate = (self.estimator.text_tokens(text) + image_tokens) * self.estimator.estimate_factor
                cost = FeatureTextCost(revision, ceil(estimate))
            tokens += cost.tokens
            next_costs[feature.id] = cost
        self.text_costs = next_costs
        oldest = eligible[0] if eligible else None
        last = self.last_story
        if last is not None:
            message = view.all_knoxels.get(last.current_message_id) if last.current_message_id is not None else None
            if message is None or message.scope not in selected_scopes or last.observed_at > now:
                last = None
        return MemoryWorkAssessment(
            sessions_with_work=sessions, external_features=len(eligible), estimated_raw_text_tokens=tokens,
            oldest_source_id=oldest.id if oldest else None, oldest_experience=oldest.timestamp_world_begin if oldest else None,
            last_story=last,
        )
