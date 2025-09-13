import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict
from typing import List
from typing import (
    Set,
    Tuple,
)

import numpy as np
from py_linq import Enumerable
from pydantic import BaseModel, Field
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

from pm.agents.definitions.agent_categorize_fact import FactCategorization, CategorizeFacts
from pm.agents.definitions.agent_extract_declarative_facts import ExtractDeclarativeFacts
from pm.agents.definitions.agent_summarize_topic_conversation import SummarizeTopicConversation
from pm.config_loader import user_name, companion_name
from pm.data_structures import FeatureType, narrative_feature_relevance_map, narrative_definitions, DeclarativeFactKnoxel, Feature, NarrativeTypes, Narrative, KnoxelList, KnoxelBase, MemoryClusterKnoxel, ClusterType
from pm.ghosts.base_ghost import BaseGhost
from pm.llm.llm_common import LlmPreset, CommonCompSettings
from pm.llm.llm_proxy import LlmManagerProxy
from pm.utils.cluster_utils import cluster_blocks_with_min_size
from pm.utils.time_hierarchy_utils import _floor_to_shifted_day_start, TimeRange, _iter_closed_tod_ranges, _iter_closed_day_ranges_0500, _iter_closed_week_ranges_0500, _iter_closed_month_ranges_0500, _iter_closed_year_ranges_0500
from pm.utils.token_utils import get_token_count

logger = logging.getLogger(__name__)


class MemoryConsolidationConfig(BaseModel):
    clustered_feature_types: List[FeatureType] = Field(default_factory=lambda: [
        FeatureType.Dialogue, FeatureType.WorldEvent
    ])
    included_feature_types: List[FeatureType] = []

    topical_clustering_context_window: int = 3
    min_events_for_processing: int = 3  # Min events for a topical cluster to be processed
    min_event_memory_consolidation_threshold: int = 60

    narrative_rolling_tokens_min: int = 3000
    narrative_rolling_tokens_nax: int = 4500
    narrative_rolling_overlap: float = 0.2
    hierchical_summary_token_limit: int = 4000
    min_split_up_cluster_size: int = 8


class DynamicMemoryConsolidator:
    def __init__(self, llm: LlmManagerProxy, ghost: BaseGhost, config: MemoryConsolidationConfig):
        self.ghost = ghost
        self.config = config
        self.llm = llm
        self.narrative_feature_map = narrative_feature_relevance_map
        self.narrative_definitions = narrative_definitions  # Assuming it's imported/available

    def consolidate_memory_if_needed(self):
        logger.info("Checking if memory consolidation is needed...")
        unclustered_events = self._get_unclustered_event_knoxels()

        if unclustered_events:
            num_unclustered = len(unclustered_events)
            logger.info(f"Unclustered event knoxels: {num_unclustered}. Threshold: {self.config.min_event_memory_consolidation_threshold}")

            if num_unclustered >= self.config.min_event_memory_consolidation_threshold:
                logger.critical("Threshold met. Starting memory consolidation...")
                self._perform_consolidation(unclustered_events)
                logger.critical("Memory consolidation finished.")
            else:
                logger.info("Threshold not met. Skipping consolidation.")
        else:
            logger.info("No unclustered event knoxels found.")

        self.refine_narratives()

    def refine_narratives(self):
        """
        Iterates through defined narratives and refines them if enough relevant new features exist.
        """
        current_tick_id = self.ghost._get_current_tick_id()
        logger.info(f"Starting narrative refinement check (Current Tick: {current_tick_id}).")

        # Get all current Feature knoxels, sorted by tick_id then id
        all_features = sorted(
            [k for k in self.ghost.all_knoxels.values() if isinstance(k, Feature)],
            key=lambda f: f.timestamp_world_begin)

        # Group existing narratives by type and target for easy lookup of the latest one
        latest_narratives: Dict[Tuple[NarrativeTypes, str], Narrative] = {}
        for knoxel in self.ghost.all_knoxels.values():
            if isinstance(knoxel, Narrative):
                key = (knoxel.narrative_type, knoxel.target_name)
                if key not in latest_narratives or knoxel.id > latest_narratives[key].id:
                    latest_narratives[key] = knoxel

        for definition in self.narrative_definitions:
            narrative_type: NarrativeTypes = definition["type"]
            target_name: str = definition["target"]
            prompt_goal: str = definition["prompt"]
            narrative_key = (narrative_type, target_name)

            logger.debug(f"Checking narrative: {narrative_type.name} for {target_name}")

            previous_narrative = latest_narratives.get(narrative_key)
            previous_content = previous_narrative.content if previous_narrative else "No previous narrative."
            last_refined_tick = previous_narrative.last_refined_with_tick if previous_narrative and previous_narrative.last_refined_with_tick is not None else -1
            relevant_feature_types = self.narrative_feature_map[narrative_type]

            # Filter recent features based on type and tick range
            features_since_last_refinement = [
                f for f in all_features
                if f.feature_type in relevant_feature_types and f.causal and f.tick_id > last_refined_tick
            ]
            features_since_last_refinement.sort(key=lambda x: x.timestamp_world_begin)

            logger.debug(f"Found {len(features_since_last_refinement)} relevant features since tick {last_refined_tick}")

            # Check if enough new features exist to warrant refinement
            while True:
                buffer = KnoxelList()
                i = 0
                for i, f in enumerate(features_since_last_refinement):
                    buffer.add(f)
                    if buffer.get_token_count() > int((self.config.narrative_rolling_tokens_min + self.config.narrative_rolling_tokens_nax) / 2):
                        break

                # remove from candidates
                features_since_last_refinement = features_since_last_refinement[i:]
                # no enough?
                if buffer.get_token_count() < self.config.narrative_rolling_tokens_min:
                    break

                logger.info(f"Refining narrative '{narrative_type.name}' for '{target_name}' based on {len(buffer._list)} new features.")
                feature_context = buffer.get_story(self.ghost)

                # --- Construct the Refinement Prompt ---
                sysprompt = f"""You are a helpful assistant for memory consolidation in a chatbot with the cognitive architecture based on LIDA. You specialize in creating 'self-narratives'.
You get a long history of events that happened to a chatbot, and you must create a new self-narrative that focuses on a specific aspect: {prompt_goal} """
                user_prompt = f"""
You are refining the psychological/behavioral narrative profile for '{target_name}'.
Narrative Type Goal: "{prompt_goal}"

Current Narrative for '{target_name}' ({narrative_type.name}):
---
{previous_content}
---

New Evidence:
---
{feature_context}
---

Task: Analyze the 'New Evidence'. Based *only* on this new information, provide an *updated and improved* version of the narrative for '{target_name}' regarding '{narrative_type.name}'. Integrate insights, modify conclusions subtly, add new observations, or keep the original narrative section if no new relevant information contradicts or enhances it. Ensure the output is the complete, updated narrative text, fulfilling the Narrative Type Goal.

Output *only* the complete, updated narrative text below. Use no more than 512 tokens!
"""

                assistant_seed = f"""Okay, here is a self-narrative for {target_name} for the {narrative_type.name} category with less than 512 tokens:

{target_name} is """

                prompt = [
                    ("system", sysprompt),
                    ("user", user_prompt),
                    ("assistant", assistant_seed)
                ]

                updated_content_raw = f"{target_name} is " + self.llm.completion_text(LlmPreset.Default, prompt, CommonCompSettings(max_tokens=2048, temperature=0.6))

                # --- Process LLM Response ---
                if updated_content_raw:
                    # Clean up potential LLM artifacts like "Updated Narrative:" prefixes
                    updated_content = updated_content_raw.strip()
                    if updated_content.startswith("Updated Narrative:"):
                        updated_content = updated_content[len("Updated Narrative:"):].strip()

                    logger.info(f"Narrative '{narrative_type.name}' for '{target_name}' successfully updated.")
                    # Create the NEW narrative knoxel
                    new_narrative = Narrative(
                        narrative_type=narrative_type,
                        target_name=target_name,
                        content=updated_content,
                        last_refined_with_tick=max(buffer._list.select(lambda x: x.tick_id)) - 1,  # buffer._list[0].tick_id,  # Mark refinement tick # use latest tick - 1 for now?
                        # tick_id, id, timestamp_creation set by add_knoxel
                    )
                    # Add to ghost state (automatically archives the old one by latest ID/tick)
                    # Generate embedding for the new narrative content
                    self.ghost.add_knoxel(new_narrative, generate_embedding=True)
                else:
                    logger.warning(f"LLM call failed for narrative refinement: {narrative_type.name} for {target_name}.")
            # else: logger.debug(f"Not enough new features to refine narrative {narrative_type.name} for {target_name}.") # Too verbose maybe

        logger.info("Narrative refinement check complete.")

    def _get_unclustered_event_knoxels(self) -> List[KnoxelBase]:
        """Fetches event-like knoxels not yet part of a Topical cluster."""
        clustered_event_ids: Set[int] = set()
        topical_clusters = [
            k for k in self.ghost.all_knoxels.values()
            if isinstance(k, MemoryClusterKnoxel) and k.cluster_type == ClusterType.Topical
        ]

        for cluster in topical_clusters:
            if cluster.included_event_ids:
                try:
                    ids = [int(eid) for eid in cluster.included_event_ids.split(',') if eid]
                    clustered_event_ids.update(ids)
                except ValueError:
                    logger.warning(f"Could not parse included_event_ids for cluster {cluster.id}: '{cluster.included_event_ids}'")

        unclustered_knoxels = []
        candidate_knoxels = Enumerable(self.ghost.all_features) \
            .where(lambda x: x.causal) \
            .where(lambda x: x.feature_type in self.config.clustered_feature_types) \
            .order_by(lambda x: x.timestamp_world_begin) \
            .to_list()

        for knoxel in candidate_knoxels:  # Process in ID order
            if knoxel.id not in clustered_event_ids:
                unclustered_knoxels.append(knoxel)

        logger.debug(f"Found {len(unclustered_knoxels)} unclustered event knoxels with embeddings.")
        return unclustered_knoxels

    def _perform_consolidation(self, events_to_cluster: List[KnoxelBase]):
        if len(events_to_cluster) < self.config.min_events_for_processing:
            logger.warning(f"Not enough new events ({len(events_to_cluster)}) for consolidation. Min: {self.config.min_events_for_processing}")
            return

        def safelen(lst):
            return 1 if len(lst) == 0 else len(lst)

        events_to_cluster = events_to_cluster[:100]

        # 1. Topical Clustering
        t = time.time()
        new_topical_clusters = self._perform_topical_clustering(events_to_cluster)
        logger.critical(f"Finished _perform_topical_clustering. Items: {len(new_topical_clusters)}, Duration: {int(time.time() - t)}, Avg. Time: {int(time.time() - t) / safelen(new_topical_clusters)}")
        if not new_topical_clusters:
            logger.info("No new topical clusters formed.")
            return

        # 2. temporal clustering
        t = time.time()
        new_temporal_cluster = self._generate_missing_closed_temporal_ranges()
        logger.critical(f"Finished _generate_missing_closed_temporal_ranges. Items: {len(new_temporal_cluster)}, Duration: {int(time.time() - t)}, Avg. Time: {int(time.time() - t) / safelen(new_temporal_cluster)}")
        all_new = new_topical_clusters + new_temporal_cluster

        # 3. extract data
        t = time.time()
        all_new_facts = []
        for clust in all_new:
            all_new_facts.extend(self._extract_declarative_memory(clust))
        logger.critical(f"Finished _extract_declarative_memory. Items: {len(all_new_facts)}, Duration: {int(time.time() - t)}, Avg. Time: {int(time.time() - t) / safelen(all_new_facts)}")

        # 4. categorize
        t = time.time()
        all_new_declarative_facts = []
        for fact_pair in all_new_facts:
            _id, raw_fact = fact_pair
            fact = self._categorize_fact(_id, raw_fact)
            all_new_declarative_facts.append(fact)
        logger.critical(f"Finished _categorize_fact. Items: {len(all_new_declarative_facts)}, Duration: {int(time.time() - t)}, Avg. Time: {int(time.time() - t) / safelen(all_new_declarative_facts)}")

        # 5. add to graph
        t = time.time()
        for fact_cated in all_new_declarative_facts:
            pass

        logger.info(f"Finished processing {len(all_new)} new clusters.")

    def _perform_topical_clustering(self, event_knoxels: List[KnoxelBase]) -> List[MemoryClusterKnoxel]:
        """Clusters event knoxels contextually and saves them as new Topical Clusters."""
        logger.info(f"Performing topical clustering on {len(event_knoxels)} knoxels...")
        new_cluster_knoxels: List[MemoryClusterKnoxel] = []
        if len(event_knoxels) < 2: return new_cluster_knoxels

        # Compute contextual embeddings
        contextual_embeddings = self._compute_contextual_embeddings(event_knoxels, self.config.topical_clustering_context_window)
        if contextual_embeddings.shape[0] != len(event_knoxels):
            logger.error("Mismatch between knoxel count and contextual embeddings count. Aborting topical clustering.")
            return new_cluster_knoxels

        # Get optimal cluster labels
        conceptual_cluster = self._get_optimal_clusters(event_knoxels, contextual_embeddings)
        if not conceptual_cluster:
            logger.warning("Optimal clustering did not produce results.")
            return new_cluster_knoxels

        """
        # Group knoxels by assigned cluster label
        clusters_data: Dict[int, List[KnoxelBase]] = defaultdict(list)
        for knoxel in event_knoxels:
            label = conceptual_cluster.get(knoxel.id)
            if label is not None:
                clusters_data[label].append(knoxel)
            # else: logger.warning(f"Knoxel {knoxel.id} was not assigned a cluster label.") # Should not happen if cluster_desc is valid

        logger.info(f"Identified {len(clusters_data)} potential topical groups.")

        # Determine the order of cluster creation based on the first knoxel ID in each group
        cluster_creation_order = sorted(clusters_data.keys(), key=lambda k: min(knox.id for knox in clusters_data[k]))

        added_event_ids = set()  # Track which event knoxels get linked
        """

        clustered_ids_blocks = cluster_blocks_with_min_size(conceptual_cluster, self.config.min_split_up_cluster_size)
        clustered_knoxel_blocks = [[self.ghost.get_knoxel_by_id(x) for x in y] for y in clustered_ids_blocks]

        for block in clustered_knoxel_blocks:
            cluster_knoxels = sorted(block, key=lambda k: k.id)

            # get all causal features from that tick range
            min_tick_id = min(k.tick_id for k in cluster_knoxels)
            max_tick_id = max(k.tick_id for k in cluster_knoxels)

            features = Enumerable(self.ghost.all_features) \
                .where(lambda x: x.causal) \
                .where(lambda x: x.tick_id >= min_tick_id and x.tick_id <= max_tick_id) \
                .where(lambda x: (x.feature_type in self.config.included_feature_types) if len(self.config.included_feature_types) > 0 else True) \
                .order_by(lambda x: x.timestamp_world_begin) \
                .to_list()
            if len(features) == 0:
                continue
            kl = KnoxelList(features)
            data = kl.get_story(ghost=self.ghost)

            # get topic summary
            inp = {
                "context": f"This is from a story about the user {user_name} and the next-gen, sentient AI companion {companion_name}.",
                "content": data
            }
            res = SummarizeTopicConversation.execute(inp, self.llm, None)
            content = res["summary"]

            min_feature_id = features[0].id
            max_feature_id = features[-1].id
            token = get_token_count(content)
            emb = self.llm.get_embedding(content)
            timestamp_begin = min(k.timestamp_world_begin for k in features)
            timestamp_end = max(k.timestamp_world_end for k in features)
            event_ids_str = ",".join(str(k.id) for k in features)

            # Create the new MemoryClusterKnoxel (Topical)
            topical_cluster = MemoryClusterKnoxel(
                level=100,
                cluster_type=ClusterType.Topical,
                content=content,
                included_event_ids=event_ids_str,
                token=token,
                embedding=emb,
                timestamp_world_begin=timestamp_begin,
                timestamp_world_end=timestamp_end,
                min_event_id=min_feature_id,
                max_event_id=max_feature_id,
                min_tick_id=min_tick_id,
                max_tick_id=max_tick_id,
                facts_extracted=False,
            )

            self.ghost.add_knoxel(topical_cluster)
            new_cluster_knoxels.append(topical_cluster)

        logger.info(f"Created {len(new_cluster_knoxels)} new topical clusters.")
        return new_cluster_knoxels

    def _generate_missing_closed_temporal_ranges(self) -> None:
        now = datetime.now()
        min_ts = datetime(year=2020, month=1, day=1)
        new_temp_knoxels = []

        # 2) Build CLOSED logical ranges in order: TOD -> Day -> Week -> Month -> Year
        time_ranges: List[TimeRange] = []

        # Per-day TOD: iterate every logical day from min_ts to today (include closed segments of current logical day)
        day_cursor = _floor_to_shifted_day_start(min_ts)
        last_full_day_start = _floor_to_shifted_day_start(now)  # start of current logical day (@05:00)
        while day_cursor <= last_full_day_start:
            time_ranges.extend(_iter_closed_tod_ranges(self, day_cursor, now))
            day_cursor += timedelta(days=1)

        # Full closed logical days
        time_ranges.extend(_iter_closed_day_ranges_0500(self, min_ts, now))
        # Full closed logical weeks
        time_ranges.extend(_iter_closed_week_ranges_0500(self, min_ts, now))
        # Full closed logical months
        time_ranges.extend(_iter_closed_month_ranges_0500(self, min_ts, now))
        # Full closed logical years
        time_ranges.extend(_iter_closed_year_ranges_0500(self, min_ts, now))

        # 3) Emit only missing summaries (same as before)
        created_count = 0
        skipped_existing = 0
        for level, start, end, key, label in time_ranges:
            if end > now:  # only closed ranges
                continue

            exists = False
            for mem in self.ghost.all_episodic_memories:
                if mem.cluster_type == ClusterType.Temporal and mem.temporal_key == key:
                    exists = True
                    break

            if exists:
                skipped_existing += 1
                continue

            features = Enumerable(self.ghost.all_features) \
                .where(lambda x: x.causal) \
                .where(lambda x: x.timestamp_world_begin <= end and x.timestamp_world_end >= start) \
                .where(lambda x: (x.feature_type in self.config.included_feature_types) if len(self.config.included_feature_types) > 0 else True) \
                .order_by(lambda x: x.timestamp_world_begin) \
                .to_list()

            if len(features) == 0:
                continue

            kl = KnoxelList(features)
            data = kl.get_story(ghost=self.ghost)

            # if raw features have less than 4k tokens, summarize from raw data
            if get_token_count(data) < self.config.hierchical_summary_token_limit:
                # get topic summary
                inp = {
                    "context": f"This is from a story about the user {user_name} and the AI companion {companion_name}. They communicate via text messages.",
                    "content": data
                }
                res = SummarizeTopicConversation.execute(inp, self.llm, None)
                content = res["summary"]

                min_tick_id = features[0].tick_id
                max_tick_id = features[-1].tick_id
                min_feature_id = features[0].id
                max_feature_id = features[-1].id
                token = get_token_count(content)
                emb = self.llm.get_embedding(content)
                timestamp_begin = min(k.timestamp_world_begin for k in features)
                timestamp_end = max(k.timestamp_world_end for k in features)
                event_ids_str = ",".join(str(k.id) for k in features)

                # Create the new MemoryClusterKnoxel (Topical)
                topical_cluster = MemoryClusterKnoxel(
                    level=level,
                    cluster_type=ClusterType.Temporal,
                    temporal_key=key,
                    content=content,
                    included_event_ids=event_ids_str,
                    token=token,
                    embedding=emb,
                    timestamp_world_begin=timestamp_begin,
                    timestamp_world_end=timestamp_end,
                    min_event_id=min_feature_id,
                    max_event_id=max_feature_id,
                    min_tick_id=min_tick_id,
                    max_tick_id=max_tick_id,
                    facts_extracted=False,
                )
                self.ghost.add_knoxel(topical_cluster)
                new_temp_knoxels.append(topical_cluster)
            else:
                # otherwise get summaries and summarize those
                # first try all time of days inside current range, if thats too big try all days, if thats too big try all weeks
                # only consider shorter levels that fall inside current level
                # if everything is too big, choose least bad option
                # llm shortens prompt itself, can't ever fail

                # Helper: collect existing temporal children at a specific level within [start, end)
                def _collect_temporal_children(level_filter: int):
                    # Use your episodic memories store (already used above)
                    blocks = []
                    for mem in self.ghost.all_episodic_memories:
                        if isinstance(mem, MemoryClusterKnoxel) and mem.level == level_filter:
                            mb = getattr(mem, "timestamp_world_begin", None)
                            me = getattr(mem, "timestamp_world_end", None)
                            if me > start and mb < end:
                                blocks.append(mem)
                    blocks.sort(key=lambda m: m.timestamp_world_begin)
                    return blocks

                candidate_levels = [l for l in [100, 6, 5, 4, 3, 2] if l > level]

                best_blocks = None
                best_level_used = None
                best_tokens = None
                fallback_blocks = None
                fallback_level = None
                fallback_tokens = None

                for child_level in candidate_levels:
                    blocks = _collect_temporal_children(child_level)
                    if not blocks:
                        continue

                    kl = KnoxelList(blocks)
                    t = get_token_count(kl.get_story(self.ghost))

                    if best_tokens is None or t < best_tokens:
                        best_blocks = blocks
                        best_level_used = child_level
                        best_tokens = t
                        if self.config.hierchical_summary_token_limit:
                            break

                # Decide which set we’ll use
                if best_blocks is not None:
                    chosen_blocks = best_blocks
                    chosen_level = best_level_used
                else:
                    raise Exception(f"Can't summarize between {start} - {end}")

                # Build the input text for the hierarchical summarizer
                if chosen_blocks:
                    kl = KnoxelList(chosen_blocks)
                    combined_text = kl.get_story(self.ghost)
                else:
                    # No lower-level blocks found; use the raw story (let LLM compress)
                    combined_text = data  # from the raw features above

                # Context tailored for hierarchical summarization
                hier_context = (
                    f"This is from a story about the user {user_name} and the next-gen, sentient AI companion {companion_name}."
                    f"This events happened between {start.isoformat()} to {end.isoformat()}."
                )

                # Run the summarizer on the combined lower-level content
                inp_hier = {
                    "context": hier_context,
                    "content": combined_text,
                }
                res_hier = SummarizeTopicConversation.execute(inp_hier, self.llm, None)
                content_hier = res_hier["summary"]

                # Aggregate metadata from chosen lower-level blocks (if any), else from raw features
                if chosen_blocks:
                    ts_begin = chosen_blocks[0].timestamp_world_begin
                    ts_end = chosen_blocks[-1].timestamp_world_end
                    min_tick_id = chosen_blocks[0].min_tick_id
                    max_tick_id = chosen_blocks[-1].max_tick_id
                    min_feature_id = chosen_blocks[0].min_feature_id
                    max_feature_id = chosen_blocks[-1].max_feature_id
                    event_ids_str = ",".join(str(k.event_ids_str) for k in chosen_blocks)
                else:
                    ts_begin = min(k.timestamp_world_begin for k in features)
                    ts_end = max(k.timestamp_world_end for k in features)
                    min_tick_id = features[0].tick_id
                    max_tick_id = features[-1].tick_id
                    min_feature_id = features[0].id
                    max_feature_id = features[-1].id
                    event_ids_str = ",".join(str(k.id) for k in features)

                token_hier = get_token_count(content_hier)
                emb_hier = self.llm.get_embedding(content_hier)

                topical_cluster = MemoryClusterKnoxel(
                    level=level,
                    cluster_type=ClusterType.Temporal,
                    temporal_key=key,
                    content=content_hier,
                    included_event_ids=event_ids_str,
                    token=token_hier,
                    embedding=emb_hier,
                    timestamp_world_begin=ts_begin,
                    timestamp_world_end=ts_end,
                    min_event_id=min_feature_id,
                    max_event_id=max_feature_id,
                    min_tick_id=min_tick_id,
                    max_tick_id=max_tick_id,
                    facts_extracted=False,
                )
                self.ghost.add_knoxel(topical_cluster)
                new_temp_knoxels.append(topical_cluster)
                created_count += 1

        print(f"[Temporal] Done. Created (stubs printed): {created_count}, skipped existing: {skipped_existing}")
        return new_temp_knoxels

    def _extract_declarative_memory(self, cluster: MemoryClusterKnoxel) -> List[Tuple[int, str]]:
        res = []
        inp_hier = {
            "context": f"This is from a story about the user {user_name} and the next-gen, sentient AI companion {companion_name}.",
            "content": cluster.content,
        }
        res_hier = ExtractDeclarativeFacts.execute(inp_hier, self.llm, None)
        facts: str = res_hier["facts"]

        for fact in facts.split("\n"):
            res.append((cluster.id, fact.lstrip("-").strip()))

        return res

    def _categorize_fact(self, source_cluster_id: int, raw_fact: str) -> DeclarativeFactKnoxel:
        inp_hier = {
            "fact": raw_fact
        }
        res = CategorizeFacts.execute(inp_hier, self.llm, None)
        fc: FactCategorization = res["category"]

        matchingVals = [x for x in self.ghost.all_episodic_memories if x.id == source_cluster_id]
        if len(matchingVals) == 1:
            mem: MemoryClusterKnoxel = matchingVals[0]
            f = DeclarativeFactKnoxel(
                timestamp_world_begin=mem.timestamp_world_begin,
                timestamp_world_end=mem.timestamp_world_end,
                min_event_id=mem.min_event_id,
                max_event_id=mem.max_event_id,
                min_tick_id=mem.min_tick_id,
                max_tick_id=mem.max_tick_id,
                content=raw_fact,
                source_cluster_id=source_cluster_id,
                reason=fc.reason,
                category=fc.category,
                importance=fc.importance,
                time_dependent=fc.time_dependent,
            )
            self.ghost.add_knoxel(f)
            return f
        else:
            raise Exception("non-existant memory")

    def _compute_contextual_embeddings(self, event_knoxels: List[KnoxelBase], context_window: int) -> np.ndarray:
        if not event_knoxels: return np.array([])
        # Assume knoxels have 'embedding' and calculate 'token' if not present
        # TODO: Ensure knoxels passed here *have* embeddings. Pre-filter or generate?
        embeddings = []
        tokens = []
        valid_knoxels = []
        for k in event_knoxels:
            if k.embedding:
                embeddings.append(k.embedding)
                token_count = get_token_count(k.content)
                tokens.append(max(1, token_count))
                valid_knoxels.append(k)
            else:
                logger.warning(f"Skipping knoxel {k.id} in contextual embedding calculation: missing embedding.")

        if not valid_knoxels: return np.array([])

        embeddings = np.array(embeddings)
        tokens = np.array(tokens)
        contextual_embeddings = []

        for i in range(len(valid_knoxels)):
            # Using simple token count as weight proxy - adjust if needed
            current_token_weight = np.log(1 + tokens[i])
            current_temporal_weight = 1  # Base weight for the item itself
            weight = current_temporal_weight * current_token_weight

            combined_embedding = embeddings[i] * weight
            weight_sum = weight

            start_idx = max(0, i - context_window)
            end_idx = min(len(valid_knoxels), i + context_window + 1)

            for j in range(start_idx, end_idx):
                if i == j: continue

                context_token_weight = np.log(1 + tokens[j])
                # Use index difference as simple temporal distance proxy
                context_temporal_weight = 1 / (1 + abs(i - j))
                context_weight = context_temporal_weight * context_token_weight

                combined_embedding += context_weight * embeddings[j]
                weight_sum += context_weight

            if weight_sum > 1e-6:  # Avoid division by zero
                combined_embedding /= weight_sum
            else:  # Should not happen if valid_knoxels is not empty
                combined_embedding = embeddings[i]  # Fallback to original

            contextual_embeddings.append(combined_embedding)

        return np.array(contextual_embeddings)

    def _get_optimal_clusters(self, event_knoxels: List[KnoxelBase], contextual_embeddings: np.ndarray) -> Dict[int, int]:
        num_events = len(event_knoxels)
        if num_events < 2 or contextual_embeddings.shape[0] != num_events:
            logger.warning(f"Insufficient data for clustering. Events: {num_events}, Embeddings shape: {contextual_embeddings.shape}")
            return {}

        try:
            # Calculate cosine distance matrix from contextual embeddings
            cosine_sim_matrix = cosine_similarity(contextual_embeddings)
            # Ensure distances are non-negative and handle potential floating point issues
            cosine_dist_matrix = np.maximum(0, 1 - cosine_sim_matrix)
            np.fill_diagonal(cosine_dist_matrix, 0)  # Ensure zero distance to self

            max_metric = -1.1  # Initialize below silhouette's range [-1, 1]
            best_threshold = None
            best_labels = None

            # Iterate through potential distance thresholds (adjust range/step as needed)
            # Start coarser, get finer near potential optimum? Or use n_clusters search?
            # Let's stick to distance threshold for now.
            for threshold in np.linspace(1.0, 0.1, 91):  # Example: from 1.0 down to 0.1
                if threshold <= 0: continue  # Skip zero or negative threshold

                agg_clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=threshold,
                    metric='precomputed',
                    linkage='average',  # Average linkage is often good for text topics
                    compute_full_tree=True
                )
                labels = agg_clustering.fit_predict(cosine_dist_matrix)
                n_clusters_found = len(np.unique(labels))

                # Calculate silhouette score only if we have a valid number of clusters
                if max(2, int(num_events / 20)) < n_clusters_found < min(num_events, int(num_events / 8)):
                    try:
                        # Use cosine *distance* for silhouette score
                        metric = silhouette_score(cosine_dist_matrix, labels, metric='precomputed')
                        if metric > max_metric:
                            max_metric = metric
                            best_threshold = threshold
                            best_labels = labels
                            logger.debug(f"New best silhouette score: {metric:.4f} at threshold {threshold:.3f} ({n_clusters_found} clusters)")
                    except ValueError as e:
                        logger.warning(f"Silhouette score error at threshold {threshold:.3f} ({n_clusters_found} clusters): {e}")
                        metric = -1.1  # Invalid score
                else:
                    # logger.debug(f"Skipping silhouette score: Threshold {threshold:.3f} -> {n_clusters_found} clusters")
                    metric = -1.1

            # If no suitable clustering found, return empty
            if best_labels is None:
                logger.warning("Could not find an optimal clustering threshold.")
                # Fallback: assign all to one cluster? or return empty? Let's return empty.
                return {}
                # Fallback 2: Assign all to cluster 0
                # best_labels = np.zeros(num_events, dtype=int)

            logger.info(f"Optimal clustering found: Threshold {best_threshold:.3f}, Score {max_metric:.4f}, Clusters {len(np.unique(best_labels))}")

            # Optional: Smooth labels (reduce single-event clusters between same-label clusters)
            final_labels = best_labels.copy()
            for i in range(1, num_events - 1):
                if final_labels[i] != final_labels[i - 1] and final_labels[i] != final_labels[i + 1] and final_labels[i - 1] == final_labels[i + 1]:
                    final_labels[i] = final_labels[i - 1]  # Assign to surrounding cluster label

            res = {}
            for i, knoxel in enumerate(event_knoxels):
                res[knoxel.id] = int(final_labels[i])  # Ensure integer label

            return res
        except Exception as e:
            logger.error(f"Error during optimal cluster calculation: {e}", exc_info=True)
            return {}

    # --- Temporal Key Generation Functions (Keep these) ---
    def get_temporal_key(self, timestamp: datetime, level: int) -> str:
        # ... (implementation from previous response) ...
        if level == 6:  # Time Of Day (combining with day for uniqueness)
            hour = timestamp.hour
            tod = "NIGHT"  # Default
            if 5 <= hour < 12:
                tod = "MORNING"
            elif 12 <= hour < 14:
                tod = "NOON"
            elif 14 <= hour < 18:
                tod = "AFTERNOON"
            elif 18 <= hour < 22:
                tod = "EVENING"
            return f"{timestamp.strftime('%Y-%m-%d')}-{tod}"
        elif level == 5:
            return timestamp.strftime('%Y-%m-%d')  # Day
        elif level == 4:
            return timestamp.strftime('%Y-W%W')  # Week
        elif level == 3:
            return timestamp.strftime('%Y-%m')  # Month
        elif level == 2:  # Season
            season = (timestamp.month % 12 + 3) // 3
            return f"{timestamp.year}-S{season}"
        elif level == 1:
            return str(timestamp.year)  # Year
        else:
            raise ValueError(f"Invalid temporal level: {level}")
