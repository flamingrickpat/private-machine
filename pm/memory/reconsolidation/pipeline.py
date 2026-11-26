"""Prepare episodes at one accepted frontier and publish each finished memory unit.

The pass first groups original events in bounded chronological windows. Every
finished episode carries its evidence, attribution, facts and canonical updates.
Broader summaries can cite unpublished children through the pass's private view.
World-time day closure and new evidence govern slower narrative refinement.
Offline callers checkpoint each unit. Live callers own the publication boundary.
"""
from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from heapq import heappop, heappush
from threading import Event
from uuid import uuid4

from pm import vclock
from pm.constants import RECONSOLIDATION_SESSION_GAP_MINUTES, RECONSOLIDATION_TIME_FORMAT
from pm.context_render.budget import TokenEstimator
from pm.llm.stage_model import LazyStageModel
from pm.memory.cluster_membership import ClusterMembership, is_episode_cluster
from pm.memory.embeddings import EmbeddingService, run_embedding_work
from pm.memory.media_alt import describe_image
from pm.memory.reconsolidation import RECEIPT_PREFIX, ReconsolidationProgress, reconsolidate, reconsolidation_due, unclustered_events
from pm.memory.reconsolidation.clustering import allowed_cluster_counts, contextual_embeddings, cosine_distances, optimal_labels, smooth_labels, topic_blocks
from pm.memory.reconsolidation.facts import EpisodeFacts, FiledFact
from pm.memory.reconsolidation.membership import EpisodeFrontier, EpisodeSources, covered_episode_ids, external_event
from pm.memory.reconsolidation.model_calls import ModelCalls
from pm.memory.reconsolidation.ordering import chronological_records, newest_narrative_version
from pm.memory.reconsolidation.publication import CapturedCaption, MemoryPreparation, MemoryUnitReceipt, PreparedMemoryUnit
from pm.memory.reconsolidation.rendering import (FactClassificationInput, FactExtractionInput, NarrativeEvidence,
    NarrativeRenderInput, RecordedStateInput, SceneSummaryInput, TemporalSummaryInput)
from pm.memory.reconsolidation.schemas import NarrativeDefinition, ReconsolidationConfig, ReconsolidationPrompts, ReconsolidationReport, load_prompts, naive_utc_now
from pm.memory.reconsolidation.story import ImageCatalog, count_tokens, feature_line, feature_text, render_story, render_summaries
from pm.memory.reconsolidation.temporal import CHILD_LEVEL, LEVEL_DAY, LEVEL_TOPICAL, TimeRange, closed_ranges
from pm.memory.snapshot import GhostCancelled
from pm.memory.store import MemoryStore
from pm.memory.write_batch import WriteBatch
from pm.records.base import Provenance, RecordScope
from pm.records.enums import ClusterType, SourceKind
from pm.records.lineage import RecordLineage
from pm.records.media import MEDIA_KEY, RecordImage, image_ids, metadata_media
from pm.records.memory_records import Entity, Feature, MemoryClusterRecord, Narrative
from pm.records.taxonomy import PmCategory


OPERATION_TOPICAL = "memory_reconsolidation.topical"
OPERATION_TEMPORAL = "memory_reconsolidation.temporal"
OPERATION_FACT = "memory_reconsolidation.fact"
OPERATION_NARRATIVE = "memory_reconsolidation.narrative"


@dataclass(slots=True, order=True)
class ScopeEpisodes:
    """Earliest possible next episode of a scope, before its window has been clustered."""

    earliest: datetime
    first_id: int
    episodes: Iterator[EpisodeSources] = field(compare=False)
    ready: EpisodeSources | None = field(default=None, compare=False)


class ReconsolidationPass:
    """Prepare source-bound deltas with one concrete pipeline for a live or offline Ghost."""

    def __init__(self, store: MemoryStore, *, complete: LazyStageModel, embeddings: EmbeddingService,
                 config: ReconsolidationConfig | None = None, force: bool = False, cancel: Event | None = None,
                 now: datetime | None = None, prompts: ReconsolidationPrompts | None = None,
                 progress: ReconsolidationProgress = ReconsolidationProgress()) -> None:
        self.store = store
        self.model = complete
        self.embeddings = embeddings
        self.config = config or ReconsolidationConfig()
        self.prompts = prompts or load_prompts()
        self.now = now if now is not None else vclock.utcnow()
        self.cancel = cancel if cancel is not None else Event()
        self.calls = ModelCalls(complete, cancel=self.cancel)
        self.progress = progress
        self.report = ReconsolidationReport(run_id=uuid4().hex[:12], forced=force)
        with store.lock:
            self.view = store.recovery_snapshot(capture=False)
            self.companion_name = store.ghost_config.companion_name
            self.user_name = store.ghost_config.user_name
            self.card = store.ghost_config.character_card_story
            self.states = {row.tick_id: row for row in store.states if row.tick_id in self.view.gwt_turns}
        identities = [row for row in self.view.all_knoxels.values() if isinstance(row, Entity)]
        self.observer_id = next((row.id for row in identities if self.companion_name in (row.content, *row.aliases)), None)
        self.user_id = next((row.id for row in identities if self.user_name in (row.content, *row.aliases)), None)
        self.preparation = MemoryPreparation(store, self.view, self.report.run_id, self.cancel)
        self.frontier = EpisodeFrontier(self.view, self.config, self.now)
        self.facts = EpisodeFacts(self.view, self.observer_id, self.now)
        self.system = self.prompts.system_prompt.format(companion_name=self.companion_name, user_name=self.user_name)
        self.intro = self.prompts.story_intro.format(companion_name=self.companion_name, user_name=self.user_name).strip()
        self.provenance = Provenance(source_kind=SourceKind.System, source_ref=RECEIPT_PREFIX + self.report.run_id)
        self.clusters = [row for row in self.view.all_knoxels.values() if isinstance(row, MemoryClusterRecord)]
        self.membership = ClusterMembership(self.view.all_knoxels)
        self.clusters_by_source: dict[int, dict[int, MemoryClusterRecord]] = {}  # pm-lint: allow PM004 -- cluster at each level by member ID
        for level in (LEVEL_TOPICAL, *CHILD_LEVEL.keys()):
            self.clusters_by_source[level] = {}
        for cluster in self.clusters:
            self.index_cluster(cluster)
        self.published: set[str] = set()
        self.next_unit = 0
        self.source_images: dict[str, RecordImage] = {}  # pm-lint: allow PM004 -- captured media by image ID
        self.captions: dict[str, CapturedCaption] = {}  # pm-lint: allow PM004 -- private descriptions by image ID

    def check_cancelled(self) -> None:
        if self.cancel.is_set():
            raise GhostCancelled(f"Stopped reconsolidation {self.report.run_id}.")

    def new_receipt(self, kind: str) -> MemoryUnitReceipt:
        self.next_unit += 1
        return MemoryUnitReceipt(run_id=self.report.run_id, unit=f"{kind}-{self.next_unit}")

    def index_cluster(self, cluster: MemoryClusterRecord) -> None:
        """Find a period's available children directly by their exact source IDs."""
        if cluster.cluster_type != ClusterType.Temporal and not is_episode_cluster(cluster):
            return
        level = LEVEL_TOPICAL if is_episode_cluster(cluster) else cluster.level
        if level not in self.clusters_by_source:
            self.clusters_by_source[level] = {}
        level_index = self.clusters_by_source[level]
        for key in self.membership.source_ids(cluster):
            current = level_index.get(key)
            if current is None or current.id < cluster.id:
                level_index[key] = cluster

    def seal(self, batch: WriteBatch, receipt: MemoryUnitReceipt) -> PreparedMemoryUnit:
        """Fill all pending vectors off the Ghost lock before handing the unit to publication."""
        records = batch.view().all_knoxels
        missing = [records[key] for key in batch.created_ids if not records[key].embedding
                   or records[key].embedding_model != self.embeddings.config.model]
        caption_ids: set[str] = set()
        for record in records.values():
            caption_ids.update(image_ids(record.content))
        for start in range(0, len(missing), 32):
            group = missing[start:start + 32]
            texts = [record.content or record.get_story_element() for record in group]
            keys = {key for text in texts for key in image_ids(text)}
            catalog = self.image_catalog(keys)
            vectors = run_embedding_work(self.embeddings.embed_many([catalog.alt_text(text) for text in texts], cancel=self.cancel))
            for record, vector in zip(group, vectors, strict=True):
                record.embedding = vector
                record.embedding_model = self.embeddings.config.model
        self.report.embedded_texts += len(missing)
        captions = tuple(self.captions[key] for key in sorted(caption_ids) if key in self.captions)
        return self.preparation.finish(batch, receipt, captions)

    def publish(self, unit: PreparedMemoryUnit) -> bool:
        """Accept one already prepared unit. The boundary owner decides when to save."""
        with self.store.lock:
            key = unit.batch.commit_id
            if key in self.published:
                return True
            if unit.receipt.run_id != self.report.run_id:
                raise ValueError("Cannot publish a unit prepared by another reconsolidation pass.")
            if not self.preparation.publish(unit):
                self.report.notes.append(f"Sources changed before {unit.receipt.unit} could be published.")
                return False
            self.published.add(key)
            self.report.topical_ids.extend(unit.receipt.topical_ids)
            self.report.temporal_ids.extend(unit.receipt.temporal_ids)
            self.report.fact_ids.extend(unit.receipt.fact_ids)
            self.report.semantic_ids.extend(unit.receipt.semantic_ids)
            self.report.semantic_updated_ids.extend(unit.receipt.semantic_updated_ids)
            self.report.narrative_ids.extend(unit.receipt.narrative_ids)
            self.progress.write(f"Published {unit.receipt.unit} ({len(unit.batch.created_ids)} records)")
            return True

    def image_catalog(self, keys: set[str]) -> ImageCatalog:
        """Capture model descriptions privately until a unit using the image publishes."""
        if keys and not self.source_images:
            for record in self.view.all_knoxels.values():
                if MEDIA_KEY in record.metadata:
                    for media in metadata_media(record.metadata).images:
                        if media.id not in self.source_images or media.alt_text:
                            self.source_images[media.id] = media.model_copy()
        images: list[RecordImage] = []
        for key in sorted(keys):
            media = self.source_images[key]
            if not media.alt_text:
                description = describe_image(media, self.model, cancel=self.cancel)
                media.alt_text = description
                media.alt_text_source = "model"
                self.captions[key] = CapturedCaption(key, description)
            images.append(media)
        return ImageCatalog(tuple(images))

    def capture_images(self, features: Sequence[Feature]) -> ImageCatalog:
        return self.image_catalog({key for feature in features for key in image_ids(feature.content)})

    def scene_input(self, template: str, story: str) -> SceneSummaryInput:
        return SceneSummaryInput(self.system, template, self.intro, story, self.companion_name, self.user_name)

    def state_description(self, features: Sequence[Feature], story: str) -> str:
        """Describe measured affect at cited ticks. Imported neutral states carry no affect evidence."""
        lines: list[str] = []
        for tick in sorted({row.tick_id for row in features}):
            if tick not in self.states or self.frontier.affect_scopes.get(tick) != features[0].scope:
                continue
            state = self.states[tick]
            core = state.mental_state.state_core
            emotions = [(value, name) for name, value in state.mental_state.state_emotions
                        if isinstance(value, float) and value >= .05]
            if abs(core.valence) < .02 and core.arousal < .02 and not emotions:
                continue
            emotions.sort(reverse=True)
            felt = ", ".join(f"{name} {value:.2f}" for value, name in emotions[:3])
            lines.append(f"{state.timestamp:%d %b %Y %H:%M}: valence {core.valence:+.2f}, arousal {core.arousal:.2f}, {felt}")
        if not lines:
            return ""
        input = RecordedStateInput(self.scene_input(self.prompts.mental_state, story), "\n".join(lines))
        return self.calls.describe_recorded_state(input).description

    def extract_facts(self, scene: Sequence[Feature], participants: Sequence[int]) -> list[FiledFact]:
        """Extract from original lines and bind every classification to the same source grant."""
        if not scene:
            return []
        settings = self.calls.session.renderer_settings_for(self.calls.generation)
        estimator = TokenEstimator(estimate_factor=settings.estimate_factor)
        source_budget = settings.input_ceiling // 4
        story = render_story(scene, RECONSOLIDATION_TIME_FORMAT, numbered=True, images=self.capture_images(scene))
        if estimator.text_tokens(story) > source_budget:
            if len(scene) > 1:
                middle = len(scene) // 2
                first = self.extract_facts(scene[:middle], participants)
                first.extend(self.extract_facts(scene[middle:], participants))
                return first
            story = "[1] " + self.calls.session.summarize_long_source(story, f"source {scene[0].id}", source_budget)
        intro = self.calls.session.summarize_long_source(self.intro, "character introduction", source_budget)
        source = SceneSummaryInput(self.system, self.prompts.facts_extract, intro, story, self.companion_name, self.user_name)
        extracted = self.calls.extract_facts(FactExtractionInput(source), len(scene)).facts
        if not extracted:
            return []
        subjects = self.facts.subjects(scene, participants)
        citations = [tuple(scene[line - 1] for line in fact.lines) for fact in extracted]
        claims = tuple(self.facts.claim_options(sources) for sources in citations)
        canonical = tuple(self.facts.canonical_options(fact.text, sources, subjects, participants)
                          for fact, sources in zip(extracted, citations, strict=True))
        source_times = tuple(sources[0].timestamp_world_begin for sources in citations)
        classification = FactClassificationInput(self.system, self.prompts.facts_classify, intro,
            tuple(fact.text for fact in extracted), tuple(PmCategory.all()), self.companion_name, self.user_name,
            subjects, claims, canonical, source_times)
        filings = self.calls.classify_facts(classification)
        result: list[FiledFact] = []
        for fact, filing, sources in zip(extracted, filings, citations, strict=True):
            result.extend(self.facts.attribute(fact.text, filing, sources, participants))
        return result

    def topic_blocks(self, anchors: list[Feature]) -> Iterator[list[Feature]]:
        """Cluster one bounded window at a time and leave the newest open topic alone."""
        start = 0
        gap = timedelta(minutes=RECONSOLIDATION_SESSION_GAP_MINUTES)
        while start < len(anchors):
            self.check_cancelled()
            window = anchors[start:start + self.config.chunk_events]
            sessions = [0]
            for previous, current in zip(window, window[1:]):
                sessions.append(sessions[-1] + int(current.timestamp_world_begin - previous.timestamp_world_begin > gap))
            # The original label search has no admissible cluster count for
            # small windows. Session labels are its exact result in that case.
            if len(window) >= self.config.min_events and allowed_cluster_counts(len(window)):
                images = self.capture_images(window)
                texts = [feature_text(row, images) for row in window]
                vectors = run_embedding_work(self.embeddings.embed_many(texts, cancel=self.cancel))
                self.report.embedded_texts += len(texts)
                context = contextual_embeddings(vectors, [count_tokens(text) for text in texts], self.config.context_window, sessions)
                labels = smooth_labels(optimal_labels(cosine_distances(context), fallback=sessions))
            else:
                labels = sessions
            blocks = topic_blocks(labels, self.config.min_block, sessions)
            if start + len(window) < len(anchors) and len(blocks) > 1:
                start += blocks[-1][0]
                blocks.pop()
            else:
                start += len(window)
            for block in blocks:
                rows = [window[index] for index in block]
                if rows[-1].id == anchors[-1].id and rows[-1].timestamp_world_begin > self.now - timedelta(minutes=self.config.open_topic_minutes):
                    return
                yield rows

    def episode_sources(self, features: Sequence[Feature]) -> Iterator[EpisodeSources]:
        anchors = [row for row in features if self.frontier.is_anchor(row)]
        positions = {row.id: index for index, row in enumerate(features)}
        start = 0
        for block in self.topic_blocks(anchors):
            first = positions[block[0].id]
            while start < first and block[0].timestamp_world_begin - features[start].timestamp_world_begin > timedelta(minutes=RECONSOLIDATION_SESSION_GAP_MINUTES):
                start += 1
            end = positions[block[-1].id] + 1
            while end < len(features) and not self.frontier.is_anchor(features[end]):
                if features[end].timestamp_world_begin - block[-1].timestamp_world_begin > timedelta(minutes=RECONSOLIDATION_SESSION_GAP_MINUTES):
                    break
                if features[end].timestamp_world_begin > self.now - timedelta(minutes=self.config.open_topic_minutes):
                    break
                end += 1
            yield self.frontier.sources(features[start:end], self.observer_id)
            start = end

    def prepare_episode(self, sources: EpisodeSources, previous: MemoryClusterRecord | None = None) -> PreparedMemoryUnit:
        self.check_cancelled()
        members = sources.members
        evidence = sources.evidence
        source_ids = [item.id for item in members]
        if previous is None:
            story = render_story(evidence, RECONSOLIDATION_TIME_FORMAT, images=self.capture_images(evidence))
            summary = self.calls.summarize_scene(self.scene_input(self.prompts.topic_summary, story)).summary
            description = self.state_description(evidence, story)
        else:
            summary = previous.content
            description = previous.emotion_description
        originals = [row for row in evidence if external_event(row, self.config)]
        facts = self.extract_facts(originals, sources.participant_ids)
        needed = set(source_ids) | self.facts.prerequisites(facts)
        needed.update(key for key in sources.participant_ids if key in self.facts.entities)
        needed.update(item.subject_id for item in facts if item.subject_id in self.facts.entities)
        if previous is not None:
            needed.add(previous.id)
        receipt = self.new_receipt("episode")
        batch = self.preparation.begin(receipt.unit, needed)
        family_subjects = {item.subject_id for item in facts}
        for choice in self.facts.claim_options(originals):
            family_subjects.add(int(choice.about.removeprefix("entity:")))
        self.facts.guard_subjects(batch, members[0].scope, sorted(family_subjects))
        cluster = MemoryClusterRecord(scope=members[0].scope, cluster_type=ClusterType.Topical, level=LEVEL_TOPICAL,
            content=summary, token=count_tokens(summary), emotion_description=description or None, facts_extracted=True,
            included_event_ids=",".join(map(str, source_ids)), min_tick_id=min(row.tick_id for row in members),
            max_tick_id=max(row.tick_id for row in members), min_event_id=min(source_ids), max_event_id=max(source_ids),
            timestamp_world_begin=members[0].timestamp_world_begin, timestamp_world_end=max(row.timestamp_world_end for row in members),
            provenance=self.provenance,
            lineage=RecordLineage(episode_id=batch.commit_id, operation=OPERATION_TOPICAL, source_event_ids=[item.id for item in evidence]))
        cluster.metadata["memory_kind"] = "experience_summary"
        cluster.metadata["participant_ids"] = list(sources.participant_ids)
        if previous is None:
            batch.create(cluster)
        else:
            self.replace_cluster(batch, previous, cluster)
        receipt.topical_ids.append(cluster.id)
        self.facts.stage(batch, cluster, facts, receipt, self.provenance)
        unit = self.seal(batch, receipt)
        self.clusters.append(cluster)
        self.index_cluster(cluster)
        return unit

    @staticmethod
    def replace_cluster(batch: WriteBatch, previous: MemoryClusterRecord, updated: MemoryClusterRecord) -> None:
        """Give the version a new ID while preserving the represented *event* interval."""
        begin = updated.timestamp_world_begin
        end = updated.timestamp_world_end
        updated.id = previous.id
        updated.tick_id = previous.tick_id
        batch.replace(updated)
        updated.timestamp_world_begin = begin
        updated.timestamp_world_end = end

    def temporal_units(self, features: Sequence[Feature]) -> Iterator[PreparedMemoryUnit]:
        if not features:
            return
        scope = features[0].scope
        current = {cluster.temporal_key: cluster for cluster in self.clusters
                   if cluster.scope == scope and cluster.cluster_type == ClusterType.Temporal}
        episodes = self.clusters_by_source[LEVEL_TOPICAL]
        times = [row.timestamp_world_begin for row in features]
        for span in closed_ranges(times, self.now):
            self.check_cancelled()
            low = bisect_left(times, span.start)
            high = bisect_left(times, span.end)
            inside = features[low:high]
            ids = {row.id for row in inside}
            if any(self.frontier.is_anchor(row) and row.id not in episodes for row in inside):
                continue
            existing = current.get(span.key)
            if existing is not None and self.membership.source_ids(existing) == frozenset(ids):
                parents = existing.lineage.parent_record_ids if existing.lineage else ()
                if all(key in self.view.all_knoxels for key in parents):
                    continue
            unit = self.prepare_temporal(span, inside, existing)
            current[span.key] = self.clusters[-1]
            yield unit

    def prepare_temporal(self, span: TimeRange, inside: Sequence[Feature],
                         previous: MemoryClusterRecord | None) -> PreparedMemoryUnit:
        ids = {row.id for row in inside}
        scope = inside[0].scope
        selected = self.frontier.sources(inside, self.observer_id)
        evidence = list(selected.evidence)
        # A closed period that contains only routine private thought still
        # retains one short recollection instead of disappearing from history.
        if not evidence:
            evidence = [inside[-1]]
        size = 0
        for feature in evidence:
            size += count_tokens(feature.content) + 12  # Timestamp and attribution on each original line.
            if size >= self.config.summary_token_limit:
                break
        children: list[MemoryClusterRecord] = []
        if size >= self.config.summary_token_limit:
            covered: set[int] = set()
            index = self.clusters_by_source[CHILD_LEVEL[span.level]]
            candidates = {index[key].id: index[key] for key in ids if key in index}
            for child in chronological_records(list(candidates.values())):
                if child.scope != scope or child.id not in self.view.all_knoxels:
                    continue
                membership = self.membership.source_ids(child)
                if membership and membership.issubset(ids) and membership.isdisjoint(covered):
                    children.append(child)
                    covered.update(membership)
            evidence = [row for row in evidence if row.id not in covered]
        catalog = self.capture_images(evidence)
        passages: list[str] = []
        for source in chronological_records([*children, *evidence]):
            if isinstance(source, MemoryClusterRecord):
                passages.append(render_summaries([source], RECONSOLIDATION_TIME_FORMAT))
            else:
                passages.append(feature_line(source, RECONSOLIDATION_TIME_FORMAT, catalog))
        story = "\n".join(passages)
        template = self.prompts.temporal_children if children else self.prompts.temporal_raw
        request = TemporalSummaryInput(self.scene_input(template, story), span.start.strftime(RECONSOLIDATION_TIME_FORMAT),
                                      span.end.strftime(RECONSOLIDATION_TIME_FORMAT))
        summary = self.calls.summarize_temporal(request).summary
        child_ids = [child.id for child in children]
        needed = ids | set(child_ids)
        needed.update(key for key in selected.participant_ids if key in self.facts.entities)
        episodes = self.clusters_by_source[LEVEL_TOPICAL]
        needed.update(episodes[key].id for key in ids if key in episodes)
        if previous is not None:
            needed.add(previous.id)
        receipt = self.new_receipt("period")
        batch = self.preparation.begin(receipt.unit, needed)
        cluster = MemoryClusterRecord(scope=scope, content=summary, cluster_type=ClusterType.Temporal, level=span.level,
            temporal_key=span.key, token=count_tokens(summary), included_event_ids=",".join(map(str, sorted(ids))),
            included_cluster_ids=",".join(map(str, child_ids)) or None,
            min_tick_id=min(row.tick_id for row in inside), max_tick_id=max(row.tick_id for row in inside),
            min_event_id=min(ids), max_event_id=max(ids), timestamp_world_begin=inside[0].timestamp_world_begin,
            timestamp_world_end=max(row.timestamp_world_end for row in inside), provenance=self.provenance,
            lineage=RecordLineage(episode_id=batch.commit_id, operation=OPERATION_TEMPORAL, parent_record_ids=child_ids,
                                  source_event_ids=[row.id for row in evidence]))
        cluster.metadata["memory_kind"] = "experience_summary"
        cluster.metadata["participant_ids"] = list(selected.participant_ids)
        if previous is None:
            batch.create(cluster)
        else:
            self.replace_cluster(batch, previous, cluster)
        receipt.temporal_ids.append(cluster.id)
        unit = self.seal(batch, receipt)
        self.clusters.append(cluster)
        self.index_cluster(cluster)
        return unit

    def narrative_units(self, definition: NarrativeDefinition, scope: RecordScope,
                        features: list[Feature]) -> Iterator[PreparedMemoryUnit]:
        own = definition.target == "companion"
        target_id = self.observer_id if own else self.user_id
        if target_id is None:
            return
        target, other = (self.companion_name, self.user_name) if own else (self.user_name, self.companion_name)
        versions = [row for row in self.view.all_knoxels.values() if isinstance(row, Narrative)
                    and row.scope == scope and row.entity_id == target_id and row.narrative_type == definition.type]
        latest = newest_narrative_version(versions) if versions else None
        processed: set[int] = set()
        legacy_tick = -1
        for version in versions:
            if version.lineage is not None:
                processed.update(version.lineage.source_event_ids)
            elif version.last_refined_with_tick is not None:
                legacy_tick = max(legacy_tick, version.last_refined_with_tick)
        days = self.clusters_by_source[LEVEL_DAY]
        subject = self.facts.entities[target_id]
        eligible: list[Feature] = []
        for row in features:
            if row.id not in days or row.id in processed or row.tick_id <= legacy_tick:
                continue
            if row.feature_type not in self.prompts.evidence_types[definition.type]:
                continue
            addressed = target_id in (row.entity_id, row.addressee_entity_id)
            own_experience = own and row.id in self.frontier.private_actions
            if addressed or own_experience or self.facts.mentions(subject, row.content):
                eligible.append(row)
        position = 0
        limit = (self.config.narrative_min_tokens + self.config.narrative_max_tokens) // 2
        while position < len(eligible):
            self.check_cancelled()
            buffer: list[NarrativeEvidence] = []
            cost = 0
            while position + len(buffer) < len(eligible) and cost <= limit:
                row = eligible[position + len(buffer)]
                line = feature_line(row, RECONSOLIDATION_TIME_FORMAT, self.capture_images([row]))
                buffer.append(NarrativeEvidence(row.id, line))
                cost += count_tokens(line)
            minimum = self.config.narrative_min_tokens if latest is not None else self.config.narrative_first_tokens
            if cost < minimum:
                break
            card = self.prompts.narrative_card.format(companion_name=target, character_card=self.card) if own and self.card else ""
            request = NarrativeRenderInput(self.system, self.prompts.narrative, self.intro, card, definition.label,
                target, latest.content if latest is not None else self.prompts.no_previous_narrative,
                definition.goal.format(target=target, other=other), self.companion_name, self.user_name, tuple(buffer))
            refinement = self.calls.refine_narrative(request)
            sources = eligible[position:position + len(refinement.selected_ids)]
            if not sources or tuple(row.id for row in sources) != refinement.selected_ids:
                raise ValueError("Narrative refinement lost its chronological source prefix.")
            previous_ids = [latest.id] if latest is not None else []
            receipt = self.new_receipt("narrative")
            accepted_days = {days[key].id for key in refinement.selected_ids}
            batch = self.preparation.begin(receipt.unit, [target_id, *refinement.selected_ids, *previous_ids, *accepted_days])
            latest = batch.create(Narrative(scope=scope, content=refinement.answer.narrative, narrative_type=definition.type,
                entity_id=target_id, last_refined_with_tick=max(row.tick_id for row in sources),
                timestamp_world_begin=sources[0].timestamp_world_begin, timestamp_world_end=sources[-1].timestamp_world_end,
                provenance=self.provenance, lineage=RecordLineage(episode_id=batch.commit_id, operation=OPERATION_NARRATIVE,
                    source_event_ids=list(refinement.selected_ids), parent_record_ids=previous_ids)))
            receipt.narrative_ids.append(latest.id)
            yield self.seal(batch, receipt)
            position += len(sources)

    def prepare_units(self) -> Iterator[PreparedMemoryUnit]:
        """Yield complete units progressively without adding any record to the live store."""
        by_scope: dict[RecordScope, list[Feature]] = {}  # pm-lint: allow PM004 -- chronological features by scope
        for feature in self.frontier.features:
            by_scope.setdefault(feature.scope, []).append(feature)
        covered = covered_episode_ids(self.clusters, self.membership)
        self.report.unclustered_events = sum(1 for row in self.frontier.features if row.id not in covered
                                             and external_event(row, self.config))
        if self.report.forced or self.report.unclustered_events >= self.config.trigger_events:
            streams: list[ScopeEpisodes] = []
            for features in by_scope.values():
                pending = [row for row in features if row.id not in covered]
                if pending:
                    first = pending[0]
                    heappush(streams, ScopeEpisodes(first.timestamp_world_begin, first.id, self.episode_sources(pending)))
            # Cluster other scopes only after their oldest remaining source can
            # precede the next ready episode. Initial work stays bounded.
            while streams:
                cursor = heappop(streams)
                if cursor.ready is None:
                    cursor.ready = next(cursor.episodes, None)
                    if cursor.ready is None:
                        continue
                    boundary = cursor.ready.members[0]
                else:
                    yield self.prepare_episode(cursor.ready)
                    boundary = cursor.ready.members[-1]
                    cursor.ready = None
                cursor.earliest = boundary.timestamp_world_begin
                cursor.first_id = boundary.id
                heappush(streams, cursor)
        # Imported topic records can predate fact extraction. Read only
        # verifiable, original members before replacing such a legacy head.
        for cluster in chronological_records(list(self.clusters)):
            if is_episode_cluster(cluster) and not cluster.facts_extracted:
                ids = self.membership.source_ids(cluster)
                rows = chronological_records([row for key in ids if isinstance(row := self.view.get(key), Feature)])
                if rows and all(row.timestamp_world_begin <= self.now for row in rows):
                    sources = self.frontier.sources(rows, self.observer_id)
                    yield self.prepare_episode(sources, previous=cluster)
        for features in by_scope.values():
            yield from self.temporal_units(features)
        for scope, features in by_scope.items():
            for definition in self.prompts.narratives:
                yield from self.narrative_units(definition, scope, features)
        self.report.model_calls = self.calls.calls
        self.report.finished_at = naive_utc_now()




__all__ = ["OPERATION_FACT", "OPERATION_NARRATIVE", "OPERATION_TEMPORAL", "OPERATION_TOPICAL", "RECEIPT_PREFIX",
           "PreparedMemoryUnit", "ReconsolidationPass", "ReconsolidationProgress", "is_episode_cluster",
           "reconsolidate", "reconsolidation_due", "unclustered_events"]
