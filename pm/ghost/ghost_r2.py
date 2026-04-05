import copy
import logging
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Dict
from typing import List
from typing import (
    Optional,
)
from typing import TypeVar

import numpy as np
from py_linq import Enumerable
from pydantic import BaseModel
from scipy.spatial.distance import cosine as cosine_distance  # Use scipy for cosine distance

from pm.agents.definitions.agent_select_codelets import AgentSelectCodelets
from pm.ghost.ghost_r1 import *
from pm.ghost.ghost_state import GhostState
from pm.model.csm import CSMManager, CSMState, CSMItem
from pm.model.knoxel_core import KnoxelBase
from pm.model.knoxel_enums import StimulusType, ClusterType
from pm.model.knoxel_list import KnoxelList
from pm.model.mental_state_vectors import compute_attention_bias, _clamp01, AppraisalGeneral, AppraisalSocial, compute_state_delta, _features_to_history, _collect_axis_bounds, _init_ms_from_vec
from pm.model.runtime_models import ShellCCQUpdate
from pm.subsystems.codelet.codelet import CodeletExecutor, CodeletContext, CodeletRegistry
from pm.subsystems.codelet.codelet_percepts import CodeletPercept
from pm.subsystems.codelet.pathways_detailed import get_codelet_pathways, CodeletActivation
from pm.subsystems.memory.graph_memory import CognitiveMemoryManager
from pm.subsystems.memory.common_memory import MemoryInterface
from pm.utils.emb_utils import cosine_sim
from pm.utils.pydantic_utils import basemodel_to_text, pydandic_model_to_dict_jsonable, group_by_int
from pm.utils.system_utils import generate_start_message
from pm.utils.token_utils import get_token_count

logger = logging.getLogger(__name__)

@dataclass
class ContextContainerPam:
    ids: List[int]
    story: str
    story_new: str
    embedding: List[float]

class GhostR2(GhostR1):
    def __init__(self, llm, config):
        super().__init__(llm, config)
        self.stimulus_feature_map: Dict[int, Feature] = {}
        self.stimulus_triage: StimulusTriage = StimulusTriage.Moderate
        self.meta_insights = []

        self.narrative_definitions = narrative_definitions
        self.input_knoxels = []
        self.primary_stimulus: Stimulus = None

    def add_stimulus(self, stimulus: Stimulus):
        """
        Add singular stimulus to cycle execution.
        Convert the stimulus to a non-causal feature and remember the IDs.

        This logic exists b
        :param stimulus:
        :return:
        """
        self.stimulus_feature_map.clear()
        if stimulus is not None:
            self.add_knoxel(stimulus)
            self.primary_stimulus = stimulus

            if stimulus.stimulus_type == StimulusType.UserMessage:
                name = self.ghost_config.user_name
                ftype = FeatureType.Dialogue
            elif stimulus.stimulus_type == StimulusType.CompanionMessage:
                name = self.ghost_config.companion_name
                ftype = FeatureType.Dialogue
            else:
                raise ValueError()

            story_feature = Feature(content=stimulus.content, source=name, feature_type=ftype, interlocus=1, causal=False)

            self.add_knoxel(story_feature)
            self.input_knoxels.append(story_feature)
            self.stimulus_feature_map[stimulus.id] = story_feature

    def postprocess_stimuli(self):
        stimuli = [self.get_knoxel_by_id(x) for x in self.stimulus_feature_map.keys()]
        for stims_tick in group_by_int(stimuli, lambda x: x.based_on_tick):
            based_on_tick = stims_tick[0].based_on_tick + 1
            inserted_begin = False

            for tmp in group_by_int(stims_tick, lambda x: x.async_sub_tick):
                tick_min_time = min([x.timestamp_creation for x in [y for y in self.all_knoxels.values() if y.tick_id == based_on_tick]])
                tick_max_time = max([x.timestamp_creation for x in [y for y in self.all_knoxels.values() if y.tick_id == based_on_tick]])

                stims_subtick = sorted(tmp, key=lambda x: x.async_tick_source_order)
                for stimulus in stims_subtick:
                    self.add_knoxel(stimulus)
                    self.primary_stimulus = stimulus

                    new_dt = datetime.now
                    if stimulus.async_tick_insert_begin:
                        if inserted_begin:
                            raise Exception("Can't add multiple in the beginning")
                        new_dt = tick_min_time - timedelta(seconds=1)
                        inserted_begin = True
                    else:
                        new_dt = tick_max_time + timedelta(seconds=1)
                        tick_max_time = new_dt

                    if stimulus.stimulus_type == StimulusType.UserMessage:
                        name = self.ghost_config.user_name
                        ftype = FeatureType.Dialogue
                    elif stimulus.stimulus_type == StimulusType.CompanionMessage:
                        name = self.ghost_config.companion_name
                        ftype = FeatureType.Dialogue
                    else:
                        raise ValueError()

                    story_feature = self.stimulus_feature_map[stimulus.id]
                    story_feature.timestamp_creation = new_dt
                    story_feature.timestamp_world_begin = new_dt
                    story_feature.timestamp_world_end = new_dt
                    story_feature.causal = True

    def create_context_pam(self, allowed: List[int], max_tokens: int) -> ContextContainerPam:
        ids = []
        kl = KnoxelList([k for k in self.all_features if k.causal and k.id in allowed]).order_by(lambda x: x.timestamp_world_begin)
        story = kl.get_story(max_tokens=max_tokens, target_knoxel_ids=ids)

        kl = KnoxelList(self.input_knoxels).order_by(lambda x: x.timestamp_world_begin)
        story_new = kl.get_story(max_tokens=max_tokens)

        res = ContextContainerPam(ids=ids, story=story, story_new=story_new, embedding=self.llm.get_embedding(story + story_new))
        return res

    def create_initial_csm(self, allowed: List[int], embedding: List[float], max_tokens: int) -> ContextContainerPam:
        csm_state = CSMState()

        relevant = MemoryInterface.sample_knoxels_embedding([k for k in self.all_knoxels.values() if k.id in allowed], embedding, 128)
        kl = KnoxelList(relevant).order_by(lambda x: x.timestamp_world_begin)
        res = []
        kl.get_story(max_tokens, res)

        for kid in res:
            csm_state.csm_item_states[kid] = CSMItem(knoxel_id=kid, first_tick=self.current_tick_id, last_tick=self.current_tick_id)

        return csm_state

    def update_csm_items(self, manager: CSMManager, search_embedding: List[float], target_codelet: CodeletExecutor, ms: FullMentalState, precursor_dict: Dict[str, List[int]], max_tokens: int) -> KnoxelList:
        if len(manager.state.csm_item_states) == 0:
            raise Exception("No knowledge for codelets!")

        valence_focus_bias, salience_bias = compute_attention_bias(ms)

        ratings = {}
        for _id, item in manager.state.csm_item_states.items():
            knoxel: KnoxelBase = self.get_knoxel_by_id(_id)

            # default activation baseline
            activation = 0.5

            if isinstance(knoxel, Feature) and knoxel.feature_type in [FeatureType.CodeletPercept, FeatureType.CodeletOutput]:
                f_val = item.affective_valence
                f_sal = item.incentive_salience

                # --- 1. Valence modulation --------------------------------------
                # If valence_focus_bias > 0, favor positive f_val; if < 0, favor negative.
                valence_term = (1.0 - abs(valence_focus_bias)) * 0.5 \
                               + valence_focus_bias * f_val * 0.5 \
                               + 0.5  # center around 0.5
                # clamp to [0..1]
                valence_term = _clamp01(valence_term)

                # --- 2. Salience sharpening -------------------------------------
                # Apply power-law sharpening: higher salience_bias → more top-focused
                sharp = 1.0 + 4.0 * salience_bias  # [1..5] exponent scale
                salience_term = math.pow(max(f_sal, 0.0), sharp)

                activation = 0.5 * valence_term + 0.5 * salience_term

            # --- 3. Codelet relevance --------------------------------------------
            dist = cosine_sim(search_embedding, knoxel.embedding) or 0.0  # 0..1 similarity
            goal_relevance = _clamp01(dist)

            # --- 4. Precursor reinforcement -------------------------------------
            precursor_boost = 0.0
            if target_codelet and target_codelet.signature.name in precursor_dict.keys() and _id in precursor_dict[target_codelet.signature.name]:
                precursor_boost = 1.0

            # --- 5. Combine all factors -----------------------------------------
            # Base metric = semantic alignment × attention activation + precursor reward
            metric = (goal_relevance * activation) + 0.3 * precursor_boost

            # optional small random jitter depending on salience_bias (more randomness when low)
            noise = random.uniform(-0.05, 0.05) * (1.0 - salience_bias)
            metric = _clamp01(metric + noise)

            ratings[_id] = metric

        ids = list(ratings.keys())
        scores = np.array([ratings[_id] for _id in ids])

        # -------- 1. Sort or sample depending on salience_bias --------
        if salience_bias >= 0.9:
            # deterministic top-down selection (exploit)
            sampled_ids = [i for _, i in sorted(zip(scores, ids), reverse=True)]
        else:
            # probabilistic selection (explore/exploit mix)
            # sharpen the distribution depending on salience_bias
            sharp = 1.0 + 4.0 * salience_bias  # [1..5]
            probs = np.power(np.clip(scores, 0, 1e6), sharp)
            if probs.sum() == 0:
                probs = np.ones_like(probs)
            probs /= probs.sum()

            # choose all percepts once, in weighted random order
            sampled_ids = list(np.random.choice(ids, size=len(ids), replace=False, p=probs))

        # add to list
        token_budget = 0
        kl = KnoxelList()

        for sampled_id in sampled_ids:
            k = self.get_knoxel_by_id(sampled_id)
            kl.add(k)
            token_budget += get_token_count(k.get_story_element())
            if token_budget > max_tokens:
                break

        kl = kl.order_by(lambda x: x.timestamp_world_begin)
        return kl

    def cognitive_cycle(self, buffer_input: List[Stimulus]) -> ShellCCQUpdate:
        # Start new tick and create state
        self.current_tick_id += 1
        self.input_knoxels.clear()
        self.states.append(GhostState(tick_id=self._get_current_tick_id()))

        # Add stimuli features to buffer
        conv_partner = 0
        for stim in buffer_input:
            conv_partner = self.get_entity_id(stim.source)
            self.add_stimulus(stim)

        # get current mental state
        rtm = 24 * 60
        start_latent_ms = self._get_current_mental_state(reference_timeframe_minutes=rtm, conversation_partner_entity_id=conv_partner)

        # === BEGIN PAM ===
        # Get last few messages with new stimulus for building base prompt
        context_pam = self.create_context_pam(allowed=[k.id for k in self.all_features], max_tokens=2048)
        # === END PAM ===

        # build initial new csm from previous and decay
        if self.previous_state and self.previous_state.csm_state and len(self.previous_state.csm_state.csm_item_states) > 0:
            csm_state = self.previous_state.csm_state.copy(deep=True)
        else:
            csm_state = self.create_initial_csm(allowed=[k.id for k in self.all_knoxels.values() if k.id not in context_pam.ids], embedding=context_pam.embedding, max_tokens=1024)

        csm_manager = CSMManager(self, state=csm_state)
        csm_manager.decay_step()

        context_codelets = CodeletContext(
            tick_id=self.current_tick_id,
            stimulus=self.primary_stimulus,
            context_embedding=context_pam.embedding,
            llm=self.llm,
            mental_state=start_latent_ms,
            story=context_pam.story,
            story_new=context_pam.story_new,
            csm_snapshot=""
        )

        codelet_registry = CodeletRegistry.init_from_simple_codelets(context_codelets)
        if self.previous_state and len(self.previous_state.codelet_state.states) > 0:
            codelet_registry.apply_codelet_state(self.previous_state.codelet_state)
            codelet_registry.decay_step()

            cands = codelet_registry.pick_candidates(context_codelets)
            codelet_cands = [x[0] for x in cands[:20]]

            inp_hier = {
                "full_prompt": context_pam.story + context_pam.story_new,
                "codelets": codelet_cands,
            }
            res_hier = AgentSelectCodelets.execute(inp_hier, self.llm, None)
            codelet_ratings: BaseModel = res_hier["codelets"]

            for k, v in codelet_ratings.dict():
                codelet_registry.items[k].runtime.activation += v

        # gather codelets for data aquisition
        # run codelet, add to csm, boost paths if possible, sample next codelet, call with extended csm
        # to prevent csm overflow we need to tag each codelet output with ms vector, and salience and valence
        precursor_dict = {}
        cnt = 0
        while True:
            codelet_candidates = codelet_registry.pick_candidates(context_codelets)
            codelet_executor, activation = codelet_candidates[0]
            codelet_executor.agent_manager.ghost = self

            codelet_container = None
            first_pass = None
            while codelet_container is None and first_pass is None:
                #snap = self.update_csm_items(csm_manager, search_embedding=self.llm.get_embedding(codelet_executor.signature.create_embedding_string()), target_codelet=codelet_executor, ms=start_latent_ms, precursor_dict=precursor_dict, max_tokens=512)
                #context_codelets.csm_snapshot = snap.get_story()
                res = codelet_executor.run(context_codelets)

                first_pass = res["first_pass"]
                codelet_container = res["output_feature"]

            # create percepts and sum the mental states
            latent_list = []
            delta_list = []
            sum_salience = 0
            sum_valence = 0
            feature_ids = []
            for field_name, model_field in codelet_container.__class__.model_fields.items():
                sub: CodeletPercept = getattr(codelet_container, field_name)
                content = basemodel_to_text(sub)
                print(codelet_candidates[0][0].signature.create_embedding_string())
                print()
                print(sub)
                for i in range(10):
                    print()

                sum_salience += sub.salience
                sum_valence += sub.valence

                temp_ms = start_latent_ms.copy(deep=True)
                if "appraisal_general" in sub.dict().keys():
                    appraisal_general: AppraisalGeneral = sub.appraisal_general
                    appraisal_social: AppraisalSocial = sub.appraisal_social
                    temp_ms.appraisal_general = appraisal_general
                    temp_ms.appraisal_social = appraisal_social
                latent, delta = compute_state_delta(temp_ms, 1)

                latent_list.append(latent.to_list())
                delta_list.append(delta.to_list())

                f = Feature(
                    source=field_name,
                    content=content,
                    feature_type=FeatureType.CodeletPercept,
                    interlocus=-2,
                    causal=False,
                    metadata=pydandic_model_to_dict_jsonable(sub),
                    mental_state_appraisal=latent.to_list(),
                    mental_state_delta=delta.to_list(),
                )
                self.add_knoxel(f)
                feature_ids.append(f.id)
                csm_manager.add_or_boost(CSMItem(knoxel_id=f.id, first_tick=self.current_tick_id, last_tick=-1, affective_valence=sum_valence, incentive_salience=sum_salience))

            # create codelet feature
            if latent_list:
                latent_stack = np.vstack(latent_list)  # shape: (n, VectorModelReservedSize)
                delta_stack = np.vstack(delta_list)

                combined_latent = np.mean(latent_stack, axis=0)
                combined_delta = np.sum(delta_stack, axis=0)
            else:
                combined_latent = np.zeros(VectorModelReservedSize)
                combined_delta = np.zeros(VectorModelReservedSize)

            content = f"## {codelet_executor.signature.name}\n{first_pass}\n"
            f = Feature(
                content=first_pass,
                source=codelet_executor.signature.name,
                feature_type=FeatureType.CodeletOutput,
                interlocus=-1,
                causal=False,
                metadata=pydandic_model_to_dict_jsonable(codelet_container),
                mental_state_appraisal=combined_latent.tolist(),
                mental_state_delta=combined_delta.tolist()
            )
            feature_ids.append(f.id)
            self.add_knoxel(f)
            csm_manager.add_or_boost(CSMItem(knoxel_id=f.id, first_tick=self.current_tick_id, last_tick=-1, affective_valence=sum_valence, incentive_salience=sum_salience))

            # apply pathway factors
            codelet_pathways = get_codelet_pathways()
            for concept, pathways in codelet_pathways.items():
                for pathway in pathways:
                    found = False
                    for i, step in enumerate(pathway):
                        if isinstance(step, CodeletActivation):
                            if found and i + 1 < len(pathway):
                                next_codelet = pathway[i + 1]
                                codelet_registry.boost_codelet(next_codelet, factor=step.strength)
                                if next_codelet.name not in precursor_dict:
                                    precursor_dict[next_codelet.name] = []
                                precursor_dict[next_codelet.name] += feature_ids
                        else:
                            if step.name == codelet_executor.signature.name:
                                found = True

            if cnt == 4:
                break
            cnt += 1

        exit(1)

        # get most salient codelt features and make them causal
        workspace_content = self.update_csm_items(csm_manager, search_embedding=context_pam.embedding, target_codelet=None, ms=start_latent_ms, precursor_dict=precursor_dict, max_tokens=1024)
        for k in workspace_content.to_list():
            k.causal = True

        ccq = self.get_ccq(context_pam.embedding, workspace_content, max_total_tokens=self.get_max_tokens_reply())

        self.postprocess_stimuli()
        self.current_state.csm_state = csm_manager.state
        self.current_state.ccq_state = {x.id: 1 for x in ccq.knoxels.to_list()}
        self.current_state.latent_mental_state = self._compute_mental_state()

        return ccq

    def _compute_mental_state(self, reference_timeframe_minutes: int = 60 * 24, half_life_factor: float = 1):
        conversation_partner_entity_id = 2
        if conversation_partner_entity_id is not None:
            feats = [f for f in self.all_features if (f is None or self.get_entity_id(f.source) == conversation_partner_entity_id) and (f.causal)]
        else:
            feats = [f for f in self.all_features if (f.causal)]
        feats = KnoxelList(feats).order_by(lambda x: x.timestamp_world_begin).to_list()

        history = _features_to_history(feats)
        vec_len = VectorModelReservedSize
        axis_bounds = _collect_axis_bounds()

        half_life_s = max(1.0, reference_timeframe_minutes * 60.0 * half_life_factor)

        normalized_vec = ema_baselined_normalize(
            history=history,
            vec_len=vec_len,
            half_life_s=half_life_s,
            axis_bounds=axis_bounds,
            start_level=None,  # or pass last persisted level if you persist between runs
        )

        return _init_ms_from_vec(normalized_vec)

    def get_ccq(
            self,
            query_embedding: List[float],
            conscious_workspace_content: KnoxelList,
            max_total_tokens: int = 1500,  # Adjusted default
            recent_budget_ratio: float = 0.50,  # Give more to recent direct features
            topical_budget_ratio: float = 0.25,
            temporal_budget_ratio: float = 0.25
    ) -> ShellCCQUpdate:
        """
        Creates an optimized story prompt by balancing recent causal features,
        relevant topical cluster events, and relevant temporal summaries.
        """
        # --- Token Budget Allocation ---
        # Ensure ratios sum to 1, adjust if not (though here they do

        workspace_size = get_token_count(conscious_workspace_content.get_story())
        max_total_tokens -= workspace_size

        selected_knoxels_for_story: Dict[int, KnoxelBase] = {}  # Use dict to ensure unique knoxels by ID
        for f in conscious_workspace_content._list:
            selected_knoxels_for_story[f.id] = f

        total_ratio = recent_budget_ratio + topical_budget_ratio + temporal_budget_ratio
        if not math.isclose(total_ratio, 1.0):
            logging.warning(f"Budget ratios do not sum to 1.0 (sum: {total_ratio}). Normalizing.")
            recent_budget_ratio /= total_ratio
            topical_budget_ratio /= total_ratio
            temporal_budget_ratio /= total_ratio

        recent_token_budget = int(max_total_tokens * recent_budget_ratio)
        topical_token_budget = int(max_total_tokens * topical_budget_ratio)
        temporal_token_budget = int(max_total_tokens * temporal_budget_ratio)

        # --- 1. Select Most Recent Causal Features ---
        logging.debug(f"Recent causal feature budget: {recent_token_budget} tokens.")

        dialouge = Enumerable(self.all_features) \
            .where(lambda x: x.causal) \
            .where(lambda x: x.tick_id != self.current_tick_id) \
            .order_by(lambda x: x.timestamp_world_begin) \
            .to_list()

        current_tick = Enumerable(self.all_features) \
            .where(lambda x: x.causal) \
            .where(lambda x: x.tick_id == self.current_tick_id) \
            .order_by(lambda x: x.timestamp_world_begin) \
            .to_list()

        recent_causal_features = dialouge + current_tick
        current_recent_tokens = 0
        for feature in recent_causal_features[::-1]:
            tokens = get_token_count(feature)
            if current_recent_tokens + tokens <= recent_token_budget:
                if feature.id not in selected_knoxels_for_story:
                    selected_knoxels_for_story[feature.id] = feature
                    current_recent_tokens += tokens
            else:
                break
        logging.info(f"Selected {len(selected_knoxels_for_story)} recent causal features, using {current_recent_tokens} tokens.")

        # --- 2. Select Relevant Topical Cluster Events ---
        logging.debug(f"Topical cluster event budget: {topical_token_budget} tokens.")
        all_topical_clusters = [
            k for k in self.all_episodic_memories  # Assuming all_episodic_memories holds MemoryClusterKnoxels
            if k.cluster_type == ClusterType.Topical and k.embedding and k.included_event_ids
        ]

        if all_topical_clusters:
            # Rank topical clusters by relevance
            ranked_topical_clusters = sorted(
                all_topical_clusters,
                key=lambda c: cosine_distance(np.array(query_embedding), np.array(c.embedding))
            )  # Low distance = high relevance

            current_topical_tokens = 0
            for cluster in ranked_topical_clusters:
                if current_topical_tokens >= topical_token_budget:
                    break

                event_ids_in_cluster = [int(eid) for eid in cluster.included_event_ids.split(',') if eid]
                events_to_add_from_cluster: List[KnoxelBase] = []
                tokens_for_this_cluster_events = 0

                # Get events from this cluster, preferring those not already selected
                # and sort them chronologically within the cluster
                cluster_event_knoxels = sorted(
                    [self.get_knoxel_by_id(eid) for eid in event_ids_in_cluster if self.get_knoxel_by_id(eid)],
                    key=lambda e: (e.tick_id, e.id)
                )

                for event in cluster_event_knoxels:
                    if event.id not in selected_knoxels_for_story:
                        tokens = get_token_count(event)
                        if current_topical_tokens + tokens_for_this_cluster_events + tokens <= topical_token_budget:
                            events_to_add_from_cluster.append(event)
                            tokens_for_this_cluster_events += tokens
                        else:  # Not enough budget for this specific event from cluster
                            break

                # Add the collected events from this cluster
                for event in events_to_add_from_cluster:
                    selected_knoxels_for_story[event.id] = event
                current_topical_tokens += tokens_for_this_cluster_events
            logging.info(f"Added events from topical clusters, using {current_topical_tokens} tokens.")

        # --- 3. Select Relevant Temporal Summaries ---
        logging.debug(f"Temporal summary budget: {temporal_token_budget} tokens.")
        all_temporal_summaries = [
            k for k in self.all_episodic_memories
            if k.cluster_type == ClusterType.Temporal and k.embedding and k.content
        ]

        if all_temporal_summaries:
            # Rank temporal summaries
            ranked_temporal_summaries = sorted(
                all_temporal_summaries,
                key=lambda s: cosine_distance(np.array(query_embedding), np.array(s.embedding)) if s.embedding else 1.0
            )

            current_temporal_tokens = 0
            for summary in ranked_temporal_summaries:
                if summary.id not in selected_knoxels_for_story:  # Don't add if somehow already there
                    tokens = get_token_count(summary)
                    if current_temporal_tokens + tokens <= temporal_token_budget:
                        selected_knoxels_for_story[summary.id] = summary
                        current_temporal_tokens += tokens
                    else:
                        break
            logging.info(f"Selected temporal summaries, using {current_temporal_tokens} tokens.")

        # --- Combine, Sort, and Generate Story ---
        final_knoxels_for_story = sorted(
            selected_knoxels_for_story.values(),
            key=lambda k: k.timestamp_world_begin  # Chronological by creation
        )

        logger.info("Logging current story elements:")
        for k in final_knoxels_for_story:
            logger.info(k.get_story_element().replace("\n", "\\n"))

        story_list = KnoxelList(final_knoxels_for_story)
        final_story_str = story_list.get_story(max_tokens=max_total_tokens)  # Pass the original max_total_tokens

        res = ShellCCQUpdate(last_causal_id=self.max_knoxel_id, current_tick=self.current_tick_id, knoxels=story_list, as_story=final_story_str)
        return res

    def get_max_tokens_reply(self):
        return self.llm.get_max_tokens(LlmPreset.Default) - 512
