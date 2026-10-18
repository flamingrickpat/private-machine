"""Memory reconsolidation over a live or imported memory store.

Port of ``DynamicMemoryConsolidator`` (``private-machine-head/pm/memory_consolidation.py``).
One run has four stages. Each stage reads the accepted store, makes its
model calls outside the commit lock, and commits its records in groups
through ``MemoryStore.commit``, so a long run keeps the work that it finished.

1. Topical clusters: the unclustered events (dialogue and world events) get
   contextual embeddings, an optimal clustering, and consecutive topic blocks.
   Each block becomes a topical ``MemoryClusterRecord`` with a summary and a
   state description.
2. Temporal clusters: every closed range (time of day, day, week, month,
   year) that holds events and has no summary yet becomes a temporal
   ``MemoryClusterRecord``. A short range reads the raw events. A long range
   reads the summaries of the next finer level.
3. Facts: every topical cluster without extracted facts gives
   ``DeclarativeFactRecord`` records with the lines that state them, and a
   canonical ``SemanticRecord`` for each new claim.
4. Narratives: each kept narrative gets a new version when enough new
   evidence exists since its last version.

Every new record names its sources: ``included_event_ids`` or
``included_cluster_ids`` on clusters, ``source_cluster_id`` and the event
range on facts, ``source_knoxel_ids`` on semantic records, and
``lineage.source_event_ids`` on all of them.

The run is incremental. Clustered events, existing temporal keys, clusters
with ``facts_extracted``, and the evidence of the newest narrative version
are skipped, so a second run without new events makes no model call.

Cancellation: ``cancel`` reaches every model and embedding call and every
commit. A cancelled run keeps the groups that it committed before.
"""
from __future__ import annotations

import asyncio
import hashlib
import re
import uuid
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from threading import Event

from pydantic import BaseModel, ConfigDict

from pm.constants import (RECONSOLIDATION_COMMIT_GROUP, RECONSOLIDATION_EMBED_WORKERS,
                          RECONSOLIDATION_SESSION_GAP_MINUTES, RECONSOLIDATION_TIME_FORMAT)
from pm.llm.completion import TextCompletion
from pm.memory.reconsolidation.clustering import (contextual_embeddings, cosine_distances, optimal_labels,
                                                  smooth_labels, topic_blocks)
from pm.memory.reconsolidation.model_calls import ModelCalls
from pm.memory.reconsolidation.schemas import (FactFiling, FactsAnswer, FilingAnswer, MentalStateAnswer,
                                               NarrativeAnswer, NarrativeDefinition, ReconsolidationConfig,
                                               ReconsolidationPrompts, ReconsolidationReport, SummaryAnswer,
                                               load_prompts)
from pm.memory.reconsolidation.story import count_tokens, feature_line, render_story, render_summaries
from pm.memory.reconsolidation.temporal import (BUILD_ORDER, CHILD_LEVEL, LEVEL_TOPICAL, TimeRange,
                                                closed_ranges)
from pm.records.base import Provenance
from pm.records.cycle_state import CycleState
from pm.records.enums import ClusterType, SourceKind
from pm.records.lineage import RecordLineage
from pm.records.memory_records import DeclarativeFactRecord, Feature, MemoryClusterRecord, Narrative, SemanticRecord
from pm.records.taxonomy import MemoryTaxonomy, MemoryVolatility, PmCategory

#: Lineage operations of the records that a run creates.
OPERATION_TOPICAL = "memory_reconsolidation.topical"
OPERATION_TEMPORAL = "memory_reconsolidation.temporal"
OPERATION_FACT = "memory_reconsolidation.fact"
OPERATION_NARRATIVE = "memory_reconsolidation.narrative"
#: The key prefix of the run receipts in ``MemoryStore.consolidations``.
RECEIPT_PREFIX = "reconsolidation:"


class RecordMetadataView(BaseModel):
    """The metadata fields that reconsolidation reads. Other keys are ignored."""

    model_config = ConfigDict(extra="ignore")

    memory_kind: str = ""
    dedup_key: str = ""


def _metadata(record) -> RecordMetadataView:
    return RecordMetadataView.model_validate(record.metadata)


def is_episode_cluster(cluster: MemoryClusterRecord) -> bool:
    """Return whether a topical cluster counts as consolidated memory.

    Main cognition writes one small topical cluster for each cycle
    (``memory_kind="experience_summary"``, no reconsolidation lineage). Such a
    cluster does not make its events clustered. Reconsolidation clusters and
    imported legacy clusters do.
    """
    if cluster.cluster_type != ClusterType.Topical:
        return False
    if cluster.lineage is not None and cluster.lineage.operation == OPERATION_TOPICAL:
        return True
    return _metadata(cluster).memory_kind != "experience_summary"


def _ids(text: str | None) -> list[int]:
    return [int(item) for item in (text or "").split(",") if item.strip()]


def unclustered_events(store, config: ReconsolidationConfig | None = None) -> list[Feature]:
    """Return the causal events that no consolidated topical cluster holds, oldest first.

    Port of ``_get_unclustered_event_knoxels``.
    """
    config = config or ReconsolidationConfig()
    with store.lock:
        features = list(store.all_features)
        clusters = [record for record in store.all_knoxels.values() if isinstance(record, MemoryClusterRecord)]
    clustered = {key for cluster in clusters if is_episode_cluster(cluster) for key in _ids(cluster.included_event_ids)}
    events = [feature for feature in features if feature.causal and feature.feature_type in config.cluster_feature_types
              and feature.id not in clustered]
    return sorted(events, key=lambda item: (item.timestamp_world_begin, item.id))


def reconsolidation_due(store, config: ReconsolidationConfig | None = None) -> bool:
    """Return whether enough events wait for a run (port of ``consolidate_memory_if_needed``)."""
    config = config or ReconsolidationConfig()
    return len(unclustered_events(store, config)) >= config.trigger_events


def _chunks(items: Sequence, size: int) -> list[list]:
    return [list(items[start:start + size]) for start in range(0, len(items), size)]


def _record_id(text: str, moment: datetime) -> str:
    """Return a stable human-facing record id like ``mem_20260522T15_001b_slug``."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:4]
    slug = "_".join(re.findall(r"[a-z0-9]+", text.casefold())[:6])
    return f"mem_{moment.strftime('%Y%m%dT%H')}_{digest}_{slug}"


class _Run:
    """The state of one run: the store, the model, the names, and the report."""

    def __init__(self, store, calls: ModelCalls, embeddings, config: ReconsolidationConfig,
                 prompts: ReconsolidationPrompts, report: ReconsolidationReport, now: datetime,
                 cancel: Event | None, log: Callable[[str], None]):
        self.store = store
        self.calls = calls
        self.embeddings = embeddings
        self.config = config
        self.prompts = prompts
        self.report = report
        self.now = now
        self.cancel = cancel
        self.log = log
        persona = store.ghost_config
        self.companion = persona.companion_name
        self.user = persona.user_name
        self.card = persona.character_card_story
        # Prompts use an unambiguous time format, not the persona format. In
        # the first real run, "Friday, 01.05.26" became "January 5" in a summary.
        self.time_format = RECONSOLIDATION_TIME_FORMAT
        with store.lock:
            self.companion_id = store.get_entity_id(self.companion)
            self.user_id = store.get_entity_id(self.user)
        self.system = self.fill(prompts.system_prompt)
        self.intro = self.fill(prompts.story_intro).strip()

    # ------------------------------------------------------------- helpers

    def fill(self, template: str, **values: object) -> str:
        """Fill a prompt template. Values are inserted as they are."""
        return template.format(companion_name=self.companion, user_name=self.user, **values)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector for each text, through the shared embedding cache."""
        async def gather():
            gate = asyncio.Semaphore(RECONSOLIDATION_EMBED_WORKERS)

            async def one(text: str) -> list[float]:
                async with gate:
                    return await self.embeddings.embed(text, cancel=self.cancel)
            return await asyncio.gather(*(one(text) for text in texts))
        if not texts:
            return []
        self.report.embedded_texts += len(texts)
        return list(asyncio.run(gather()))

    def lineage(self, operation: str, sources: Sequence[int], parents: Sequence[int] = ()) -> RecordLineage:
        return RecordLineage(episode_id=f"{RECEIPT_PREFIX}{self.report.run_id}", operation=operation,
                             source_event_ids=sorted(set(sources)), parent_record_ids=sorted(set(parents)))

    def provenance(self) -> Provenance:
        return Provenance(source_kind=SourceKind.System, source_ref=f"{RECEIPT_PREFIX}{self.report.run_id}")

    def commit(self, name: str, build: Callable[[object], None]) -> list[int]:
        """Stage records with ``build(batch)`` and commit them. Return the new ids."""
        batch = self.store.begin_batch(f"{RECEIPT_PREFIX}{self.report.run_id}:{name}", cancel=self.cancel)
        build(batch)
        receipt = self.store.commit(batch)
        return list(receipt.added_ids)

    def story_features(self) -> list[Feature]:
        """Return the causal features that summaries read, oldest first."""
        kinds = set(self.config.story_feature_types)
        with self.store.lock:
            features = [item for item in self.store.all_features if item.causal and item.feature_type in kinds]
        return sorted(features, key=lambda item: (item.timestamp_world_begin, item.id))

    def clusters(self) -> list[MemoryClusterRecord]:
        with self.store.lock:
            records = [item for item in self.store.all_knoxels.values() if isinstance(item, MemoryClusterRecord)]
        return sorted(records, key=lambda item: (item.timestamp_world_begin, item.id))

    def recorded_state(self, min_tick: int, max_tick: int) -> str:
        """Describe the accepted cycle states of a tick range in plain words.

        Imported cycles carry the neutral default state. Such a range gets
        the "no inner state was recorded" text, so the model does not read
        a neutral default as calm.
        """
        with self.store.lock:
            states: list[CycleState] = [state for state in self.store.states if min_tick <= state.tick_id <= max_tick]
        lines = []
        for state in states:
            mind = state.mental_state
            emotions = sorted(((value, name) for name, value in mind.state_emotions.model_dump().items()
                               if value >= 0.05), reverse=True)[:3]
            if abs(mind.state_core.valence) < 0.02 and mind.state_core.arousal < 0.02 and not emotions:
                continue
            felt = ", ".join(f"{name} {value:.2f}" for value, name in emotions) or "no strong emotion"
            lines.append(f"- {state.timestamp.strftime(self.time_format)}: valence {mind.state_core.valence:+.2f}, "
                         f"arousal {mind.state_core.arousal:.2f}; {felt}")
        return "\n".join(lines) if lines else self.prompts.no_recorded_state

    def describe_state(self, story: str, min_tick: int, max_tick: int) -> str:
        answer = self.calls.json(self.system, self.fill(self.prompts.mental_state, intro=self.intro, story=story,
                                 recorded=self.recorded_state(min_tick, max_tick)), MentalStateAnswer)
        return answer.description

    # ------------------------------------------------------------- stage 1

    def topical(self, events: list[Feature]) -> None:
        """Cluster the unclustered events into topic blocks and summarize each block."""
        vectors = self.embed([event.content for event in events])
        tokens = [count_tokens(event.content) for event in events]
        gap = timedelta(minutes=RECONSOLIDATION_SESSION_GAP_MINUTES)
        sessions = [0]
        for previous, event in zip(events, events[1:]):
            sessions.append(sessions[-1] + (event.timestamp_world_begin - previous.timestamp_world_begin > gap))
        blocks: list[list[Feature]] = []
        start = 0
        while start < len(events):
            end = min(len(events), start + self.config.chunk_events)
            window = slice(start, end)
            context = contextual_embeddings(vectors[window], tokens[window], self.config.context_window, sessions[window])
            labels = smooth_labels(optimal_labels(cosine_distances(context), fallback=sessions[window]))
            index_blocks = topic_blocks(labels, self.config.min_block, sessions[window])
            if end < len(events) and len(index_blocks) > 1:
                # The last block can continue in the next window. Start the
                # next window at its first event, so a topic is not cut.
                index_blocks, start = index_blocks[:-1], start + index_blocks[-1][0]
            else:
                start = end
            blocks.extend([events[window][index] for index in block] for block in index_blocks)
        if blocks and blocks[-1][-1].timestamp_world_begin > self.now - timedelta(minutes=self.config.open_topic_minutes):
            self.report.notes.append(f"The newest topic ({len(blocks[-1])} events) is still open and stays unclustered.")
            blocks.pop()
        story_features = self.story_features()
        for group_index, group in enumerate(_chunks(blocks, RECONSOLIDATION_COMMIT_GROUP)):
            plans = []
            for block in group:
                low, high = min(item.tick_id for item in block), max(item.tick_id for item in block)
                features = [item for item in story_features if low <= item.tick_id <= high]
                plans.append((block, features, low, high))

            def summarize(plan):
                _, features, low, high = plan
                story = render_story(features, self.time_format)
                summary = self.calls.json(self.system, self.fill(self.prompts.topic_summary, intro=self.intro,
                                                                 story=story), SummaryAnswer).summary
                return summary, self.describe_state(story, low, high)

            results = self.calls.map(summarize, plans)
            self.embed([description for _, description in results])

            def build(batch):
                for (block, features, low, high), (summary, description) in zip(plans, results):
                    ids = [item.id for item in features]
                    batch.create(MemoryClusterRecord(
                        content=summary, level=LEVEL_TOPICAL, cluster_type=ClusterType.Topical,
                        included_event_ids=",".join(map(str, ids)), token=count_tokens(summary),
                        timestamp_world_begin=features[0].timestamp_world_begin,
                        timestamp_world_end=max(item.timestamp_world_end for item in features),
                        min_event_id=min(ids), max_event_id=max(ids), min_tick_id=low, max_tick_id=high,
                        facts_extracted=False, emotion_description=description,
                        metadata={"memory_kind": "experience_summary"}, provenance=self.provenance(),
                        lineage=self.lineage(OPERATION_TOPICAL, ids)))
            self.report.topical_ids += self.commit(f"topical-{group_index}", build)
            self.log(f"topical clusters: {len(self.report.topical_ids)}")

    # ------------------------------------------------------------- stage 2

    def temporal(self) -> None:
        """Summarize every closed time range with events and without a summary."""
        features = self.story_features()
        existing = {item.temporal_key for item in self.clusters() if item.cluster_type == ClusterType.Temporal}
        ranges = [item for item in closed_ranges([feature.timestamp_world_begin for feature in features], self.now)
                  if item.key not in existing]
        for level in BUILD_ORDER:
            level_ranges = [item for item in ranges if item.level == level]
            for group_index, group in enumerate(_chunks(level_ranges, RECONSOLIDATION_COMMIT_GROUP)):
                clusters = self.clusters()
                plans = [self._temporal_plan(item, features, clusters) for item in group]

                def summarize(plan):
                    span, inside, children, story = plan
                    template = self.prompts.temporal_children if children else self.prompts.temporal_raw
                    summary = self.calls.json(self.system, self.fill(
                        template, intro=self.intro, story=story,
                        start=span.start.strftime(self.time_format), end=span.end.strftime(self.time_format)),
                        SummaryAnswer).summary
                    low, high = min(item.tick_id for item in inside), max(item.tick_id for item in inside)
                    return summary, self.describe_state(story, low, high)

                results = self.calls.map(summarize, plans)
                self.embed([description for _, description in results])

                def build(batch):
                    for (span, inside, children, _), (summary, description) in zip(plans, results):
                        event_ids = [item.id for item in inside]
                        child_ids = [item.id for item in children]
                        batch.create(MemoryClusterRecord(
                            content=summary, level=span.level, cluster_type=ClusterType.Temporal,
                            temporal_key=span.key, token=count_tokens(summary),
                            included_event_ids=None if children else ",".join(map(str, event_ids)),
                            included_cluster_ids=",".join(map(str, child_ids)) if children else None,
                            timestamp_world_begin=inside[0].timestamp_world_begin,
                            timestamp_world_end=max(item.timestamp_world_end for item in inside),
                            min_event_id=min(event_ids), max_event_id=max(event_ids),
                            min_tick_id=min(item.tick_id for item in inside),
                            max_tick_id=max(item.tick_id for item in inside),
                            facts_extracted=False, emotion_description=description,
                            metadata={"memory_kind": "experience_summary"}, provenance=self.provenance(),
                            lineage=self.lineage(OPERATION_TEMPORAL, [] if children else event_ids, child_ids)))
                self.report.temporal_ids += self.commit(f"temporal-{level}-{group_index}", build)
                self.log(f"temporal clusters: {len(self.report.temporal_ids)}")

    def _temporal_plan(self, span: TimeRange, features: list[Feature], clusters: list[MemoryClusterRecord]):
        """Return ``(range, events, children, story)`` for one range.

        A range whose raw story fits ``summary_token_limit`` reads the raw
        events. A longer range reads the summaries of its child level
        (:data:`~pm.memory.reconsolidation.temporal.CHILD_LEVEL`). Topical
        children overlap the range; temporal children start inside it. When
        no child exists, the range reads the raw events anyway: the old code
        raised here, and dropping evidence is worse than a long prompt.
        """
        inside = [item for item in features if span.contains(item.timestamp_world_begin)]
        story = render_story(inside, self.time_format)
        if count_tokens(story) < self.config.summary_token_limit:
            return span, inside, [], story
        level = CHILD_LEVEL[span.level]
        if level == LEVEL_TOPICAL:
            children = [item for item in clusters if is_episode_cluster(item)
                        and item.timestamp_world_begin < span.end and item.timestamp_world_end >= span.start]
        else:
            children = [item for item in clusters if item.cluster_type == ClusterType.Temporal
                        and item.level == level and span.contains(item.timestamp_world_begin)]
        if not children:
            self.report.notes.append(f"Range {span.key} is long and has no child summaries; it reads raw events.")
            return span, inside, [], story
        return span, inside, children, render_summaries(children, self.time_format)

    # ------------------------------------------------------------- stage 3

    def facts(self) -> None:
        """Extract and file the facts of every topical cluster that has none yet."""
        pending = [item for item in self.clusters() if is_episode_cluster(item) and not item.facts_extracted]
        about = [self.companion, self.user, "both", "other"]
        subjects = {self.companion: self.companion_id or 0, self.user: self.user_id or 0, "both": 0, "other": 0}
        kinds = set(self.config.story_feature_types)
        with self.store.lock:
            existing = {}
            for record in self.store.all_knoxels.values():
                if isinstance(record, SemanticRecord):
                    existing.setdefault(_metadata(record).dedup_key or record.dedup_key, record.id)
        for group_index, group in enumerate(_chunks(pending, RECONSOLIDATION_COMMIT_GROUP)):
            with self.store.lock:
                sources = [[self.store.get_knoxel_by_id(key) for key in _ids(cluster.included_event_ids)]
                           for cluster in group]
            events = [sorted((item for item in records if isinstance(item, Feature) and item.feature_type in kinds),
                             key=lambda item: (item.timestamp_world_begin, item.id)) for records in sources]

            def extract(plan):
                cluster, scene = plan
                if not scene:
                    return []
                story = render_story(scene, self.time_format, numbered=True)
                found = self.calls.json(self.system, self.fill(self.prompts.facts_extract, intro=self.intro, story=story),
                                        FactsAnswer, context={"lines": len(scene)}).facts
                if not found:
                    return []
                listing = "\n".join(f"{index}. {fact.text}" for index, fact in enumerate(found, 1))
                filings = self.calls.json(self.system, self.fill(
                    self.prompts.facts_classify, intro=self.intro, facts=listing,
                    categories=", ".join(PmCategory.all())), FilingAnswer,
                    context={"count": len(found), "about": about}).facts
                return [(fact.text, filing, [scene[line - 1] for line in fact.lines])
                        for fact, filing in zip(found, filings)]

            results = self.calls.map(extract, list(zip(group, events)))

            def build(batch):
                self._stage_facts(batch, group, results, subjects, existing)
            self.commit(f"facts-{group_index}", build)
            self.log(f"facts: {len(self.report.fact_ids)}")

    def _stage_facts(self, batch, clusters, results, subjects: dict[str, int], existing: dict[str, int]) -> None:
        """Create the facts and their canonical semantic records in ``batch``.

        A claim with a known ``(subject, essence)`` key adds the fact to the
        existing semantic record. A new claim gets a new semantic record. The
        cluster gets ``facts_extracted=True`` in the same batch, so a crash
        never extracts the same cluster twice. ``existing`` maps dedup keys to
        semantic record ids; this method adds the new keys.
        """
        view = batch.view()
        open_records: dict[int, SemanticRecord] = {}
        for cluster, items in zip(clusters, results):
            for text, filing, cited in items:
                subject = subjects[filing.about]
                fact = batch.create(self._fact(cluster, text, filing, subject, cited))
                key = MemoryTaxonomy.dedup_key(subject, filing.essence)
                if key in existing:
                    record_id = existing[key]
                    record = open_records[record_id] if record_id in open_records else view.get(record_id)
                    record.source_knoxel_ids.append(fact.id)
                    if text not in record.source_texts and text != record.content:
                        record.source_texts.append(text)
                    record.importance = max(record.importance, filing.importance)
                    open_records[record_id] = record
                    if record_id in batch.created_ids:
                        batch.revise(record)
                    else:
                        batch.replace(record)
                        if record_id not in self.report.semantic_updated_ids + self.report.semantic_ids:
                            self.report.semantic_updated_ids.append(record_id)
                else:
                    record = batch.create(self._semantic(text, filing, subject, fact.id, cited))
                    record.semantic_record_id = record.id
                    batch.revise(record)
                    existing[key] = record.id
                    open_records[record.id] = record
                    self.report.semantic_ids.append(record.id)
                fact.semantic_record_id = existing[key]
                batch.revise(fact)
                self.report.fact_ids.append(fact.id)
            done = view.get(cluster.id)
            done.facts_extracted = True
            batch.replace(done)

    def _fact(self, cluster: MemoryClusterRecord, text: str, filing: FactFiling, subject: int,
              cited: list[Feature]) -> DeclarativeFactRecord:
        ids = [item.id for item in cited]
        return DeclarativeFactRecord(
            content=text, subject_entity_id=subject, record_type=filing.record_type,
            object_kind=filing.object_kind, object_value=filing.object_value,
            epistemic_status=filing.epistemic_status, confidence=filing.confidence, topic_card=filing.topic_card,
            essence=filing.essence, source_cluster_id=cluster.id, reason=filing.reason, category=filing.category,
            importance=filing.importance, time_dependent=filing.time_dependent,
            timestamp_world_begin=min(item.timestamp_world_begin for item in cited),
            timestamp_world_end=max(item.timestamp_world_end for item in cited),
            min_event_id=min(ids), max_event_id=max(ids),
            min_tick_id=min(item.tick_id for item in cited), max_tick_id=max(item.tick_id for item in cited),
            metadata={"memory_kind": "experience_summary"}, provenance=self.provenance(),
            lineage=self.lineage(OPERATION_FACT, ids, [cluster.id]))

    def _semantic(self, text: str, filing: FactFiling, subject: int, fact_id: int,
                  cited: list[Feature]) -> SemanticRecord:
        key = MemoryTaxonomy.dedup_key(subject, filing.essence)
        label = self.companion if subject and subject == self.companion_id else (
            self.user if subject and subject == self.user_id else "")
        return SemanticRecord(
            content=text, subject_entity_id=subject, record_type=filing.record_type,
            object_kind=filing.object_kind, object_value=filing.object_value,
            epistemic_status=filing.epistemic_status, confidence=filing.confidence, topic_card=filing.topic_card,
            essence=filing.essence, record_id=_record_id(filing.essence, self.now), subject_label=label,
            importance=filing.importance,
            volatility=MemoryVolatility.Ephemeral if filing.time_dependent >= 0.7 else MemoryVolatility.Stable,
            source_knoxel_ids=[fact_id], source_method="consolidation",
            metadata={"dedup_key": key}, provenance=self.provenance(),
            lineage=self.lineage(OPERATION_FACT, [item.id for item in cited], [fact_id]))

    # ------------------------------------------------------------- stage 4

    def narratives(self) -> None:
        """Refine each kept narrative whose evidence grew enough since its newest version."""
        chains = self.calls.map(self._narrative_chain, self.prompts.narratives)
        for created in chains:
            self.report.narrative_ids += created

    def newest_narrative(self, definition: NarrativeDefinition, entity_id: int) -> Narrative | None:
        """Return the newest version, ranked like ``MainCogLoop._narrative_ids``."""
        with self.store.lock:
            versions = [item for item in self.store.all_narratives
                        if item.narrative_type == definition.type and item.entity_id == entity_id]
        if not versions:
            return None
        return max(versions, key=lambda item: (item.last_refined_with_tick or -1, item.timestamp_world_begin, item.id))

    def _narrative_chain(self, definition: NarrativeDefinition) -> list[int]:
        """Write new versions of one narrative until the new evidence is too small.

        Port of ``refine_narratives``. The evidence is the causal features of
        the narrative's evidence types. New evidence starts after the
        largest source id of the newest version (lineage), or, for a version
        without lineage, after its ``last_refined_with_tick``. Each version
        reads at most the mean of the minimum and maximum token budget.
        The first version needs only ``narrative_first_tokens``, so a small
        import still gets narratives. The old code started from placeholders.
        """
        companion = definition.target == "companion"
        entity_id = self.companion_id if companion else self.user_id
        if entity_id is None:
            return []
        target, other = (self.companion, self.user) if companion else (self.user, self.companion)
        kinds = set(self.prompts.evidence_types[definition.type])
        with self.store.lock:
            features = [item for item in self.store.all_features if item.causal and item.feature_type in kinds]
        features.sort(key=lambda item: (item.timestamp_world_begin, item.id))
        latest = self.newest_narrative(definition, entity_id)
        if latest is not None and latest.lineage is not None and latest.lineage.source_event_ids:
            cursor = max(latest.lineage.source_event_ids)
            evidence = [item for item in features if item.id > cursor]
        elif latest is not None:
            evidence = [item for item in features if item.tick_id > (latest.last_refined_with_tick or -1)]
        else:
            evidence = features
        lines = [feature_line(item, self.time_format) for item in evidence]
        costs = [count_tokens(line) for line in lines]
        created: list[int] = []
        limit = (self.config.narrative_min_tokens + self.config.narrative_max_tokens) // 2
        while evidence:
            total, size = 0, 0
            while size < len(evidence) and total <= limit:
                total += costs[size]
                size += 1
            required = self.config.narrative_first_tokens if latest is None else self.config.narrative_min_tokens
            if total < required:
                break
            buffer = evidence[:size]
            card = self.fill(self.prompts.narrative_card, character_card=self.card) if companion and self.card else ""
            text = self.calls.json(self.system, self.fill(
                self.prompts.narrative, intro=self.intro, card=card, label=definition.label, target=target,
                previous=latest.content if latest is not None else self.prompts.no_previous_narrative,
                story="\n".join(lines[:size]),
                goal=definition.goal.format(target=target, other=other)), NarrativeAnswer).narrative
            record = Narrative(
                content=text, narrative_type=definition.type, entity_id=entity_id,
                last_refined_with_tick=max(item.tick_id for item in buffer),
                timestamp_world_begin=buffer[0].timestamp_world_begin,
                timestamp_world_end=max(item.timestamp_world_end for item in buffer),
                provenance=self.provenance(),
                lineage=self.lineage(OPERATION_NARRATIVE, [item.id for item in buffer],
                                     [latest.id] if latest is not None else []))
            ids = self.commit(f"narrative-{definition.type.value}-{definition.target}-{len(created)}",
                              lambda batch: batch.create(record))
            created += ids
            with self.store.lock:
                latest = self.store.get_knoxel_by_id(ids[0])
            evidence, lines, costs = evidence[size:], lines[size:], costs[size:]
            self.log(f"narrative {definition.type.value} of {target}: version {len(created)}")
        return created

    # ------------------------------------------------------------- index

    def prepare_index(self, ids: Sequence[int]) -> None:
        """Embed the retrieval chunks of the new records into the shared cache.

        Live retrieval embeds at most 64 missing chunks per call. A prepared
        cache lets it rank every new summary, fact, and narrative at once.
        """
        from pm.memory.retrieval import index_chunks
        records = self.store.snapshot(set(ids)).serialize_knoxels()
        chunks, _ = index_chunks(records)
        self.embed([chunks[key] for key in sorted(chunks)])


def reconsolidate(store, *, complete: TextCompletion, embeddings, config: ReconsolidationConfig | None = None,
                  force: bool = False, cancel: Event | None = None, now: datetime | None = None,
                  prompts: ReconsolidationPrompts | None = None,
                  log: Callable[[str], None] | None = None) -> ReconsolidationReport:
    """Run one memory reconsolidation and return its report.

    ``complete`` is a text completion (for example ``LazyStageModel.complete``).
    ``embeddings`` has ``async embed(text, cancel=...)``, for example
    ``pm.memory.embeddings.EmbeddingService``. Without ``force``, the topical
    stage runs only when ``config.trigger_events`` events wait (the old
    trigger rule). With ``force``, it runs for ``config.min_events`` events.
    The temporal, fact, and narrative stages always check for work; they
    make no model call when nothing is new. ``now`` is naive UTC; it decides
    which time ranges are closed.

    The report is also committed as the consolidation receipt
    ``reconsolidation:<run_id>``. Model errors after the repairs raise
    ``ReconsolidationOutputError``. Provider, store, and cancellation errors
    propagate; the groups committed before stay accepted.
    """
    config = config or ReconsolidationConfig()
    prompts = prompts or load_prompts()
    calls = ModelCalls(complete, cancel=cancel)
    report = ReconsolidationReport(run_id=uuid.uuid4().hex[:12], forced=force)
    run = _Run(store, calls, embeddings, config, prompts, report, now or datetime.utcnow(), cancel,
               log or (lambda message: None))
    events = unclustered_events(store, config)
    report.unclustered_events = len(events)
    if not force and len(events) < config.trigger_events:
        report.notes.append(f"{len(events)} unclustered events are below the trigger of {config.trigger_events}.")
    elif len(events) < config.min_events:
        report.notes.append(f"{len(events)} unclustered events are below the minimum of {config.min_events}.")
    else:
        run.topical(events)
    run.temporal()
    run.facts()
    run.narratives()
    run.prepare_index(report.created_ids)
    report.model_calls = calls.calls
    report.finished_at = datetime.utcnow()

    def receipt(batch):
        batch.set_consolidation_receipt(f"{RECEIPT_PREFIX}{report.run_id}",
                                        {"run_id": report.run_id, "kind": "memory_reconsolidation",
                                         **report.model_dump(mode="json")})
    run.commit("receipt", receipt)
    return report


__all__ = ["OPERATION_FACT", "OPERATION_NARRATIVE", "OPERATION_TEMPORAL", "OPERATION_TOPICAL", "RECEIPT_PREFIX",
           "is_episode_cluster", "reconsolidate", "reconsolidation_due", "unclustered_events"]
