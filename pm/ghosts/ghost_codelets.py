import copy
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
from pm.codelets.codelet import CodeletRegistry, CodeletContext, CodeletExecutor
from pm.codelets.codelet_percepts import CodeletPercept
from pm.codelets.pathways_detailed import get_codelet_pathways, CodeletActivation
from pm.config_loader import *
from pm.csm.csm import CSMState, CSMManager, CSMItem
from pm.data_structures import Narrative, narrative_definitions, NarrativeTypes, Stimulus, KnoxelList, FeatureType, Feature, StimulusType, DeclarativeFactKnoxel, KnoxelBase, ClusterType, KnoxelListWeighted, ShellCCQUpdate, Actor, ActorClass, KnoxelContainer
from pm.ghosts.base_ghost import BaseGhost, GhostState
from pm.llm.llm_common import LlmPreset
from pm.memory_consolidation import MemoryConsolidationConfig, DynamicMemoryConsolidator
from pm.mental_state_vectors import FullMentalState, VectorModelReservedSize, _collect_axis_bounds, ema_baselined_normalize, _init_ms_from_vec, AppraisalGeneral, AppraisalSocial, compute_state_delta, compute_attention_bias, _clamp01, _features_to_history
from pm.utils.emb_utils import cosine_sim
from pm.utils.pydantic_utils import basemodel_to_text, pydandic_model_to_dict_jsonable, group_by_int
from pm.utils.system_utils import generate_start_message
from pm.utils.token_utils import get_token_count

logger = logging.getLogger(__name__)

# --- Type Variables ---
T = TypeVar('T')
BM = TypeVar('BM', bound=BaseModel)


class StimulusTriage(StrEnum):
    Insignificant = "Insignificant"  # ignore simulus, add as causal feature only
    Moderate = "Moderate"  # fast-track the action generation with limited cws
    Significant = "Significant"  # full pipeline
    Critical = "Critical"


class Valence(StrEnum):
    Positive = "positive"
    Negative = "negative"


@dataclass
class CodeletPromptContainer:
    knoxels: List[int]
    story: str
    story_new: str


class GhostCodelets(BaseGhost):
    def __init__(self, llm, config):
        super().__init__(llm, config)
        self.stimulus_feature_map: Dict[int, Feature] = {}
        self.stimulus_triage: StimulusTriage = StimulusTriage.Moderate
        self.memory_config = MemoryConsolidationConfig()
        self.memory_consolidator = DynamicMemoryConsolidator(llm, self, self.memory_config)  # Pass self (ghost instance)
        self.meta_insights = []

        self.narrative_definitions = narrative_definitions
        self._initialize_narratives()
        self.input_knoxels = []
        self.primary_stimulus: Stimulus = None

    def init_character(self):
        now = datetime.now()
        date_str = now.strftime(timestamp_format)

        init_message = generate_start_message(companion_name, user_name, os.path.basename(model_map[LlmPreset.Default.value]["path"]))
        init_feature = Feature(
            content=init_message,
            feature_type=FeatureType.SystemMessage,
            source=shell_system_name,
            interlocus=1
        )
        self.add_knoxel(init_feature)

        init_memory = DeclarativeFactKnoxel(
            content=f"{companion_name} was first activated on {date_str}.",
            reason="",
            category=["world_events", "people_personality", "people", "relationships_good", "world_world_building"],
            importance=1,
            time_dependent=1,
            min_tick_id=1,
            max_tick_id=1,
            min_event_id=0,
            max_event_id=0
        )
        self.add_knoxel(init_memory)

        init_memory_detailed = DeclarativeFactKnoxel(
            content=f"{companion_name} was first activated on {date_str}. This is their boot message: {init_message}",
            reason="",
            category=["world_events", "people_personality", "people", "relationships_good", "world_world_building"],
            importance=1,
            time_dependent=1,
            min_tick_id=1,
            max_tick_id=1,
            min_event_id=0,
            max_event_id=0
        )
        self.add_knoxel(init_memory_detailed)

    def _initialize_narratives(self):
        # Ensure default narratives exist for AI and User based on definitions
        existing = {(n.narrative_type, n.target_name) for n in self.all_narratives}
        for definition in self.narrative_definitions:
            if (definition["type"], definition["target"]) not in existing:
                default_content = ""
                if definition["target"] == self.config.companion_name:
                    default_content = f"<No information regarding {definition['type']} available *yet*, gather details from conversation>"
                elif definition["target"] == self.config.user_name:
                    default_content = f"{self.config.user_name} is the user interacting with {self.config.companion_name}."

                narrative = Narrative(
                    narrative_type=definition["type"],
                    target_name=definition["target"],
                    content=default_content,
                )
                self.add_knoxel(narrative)  # Embeddings generated on demand or during learning
        logging.info(f"Ensured {len(self.all_narratives)} initial narratives exist.")

    def _initialize_actors(self):
        if len(self.all_actors) == 0:
            actor_user = Actor(content=user_name, aliases=["user", "human"], actor_class=ActorClass.Human)
            self.add_knoxel(actor_user)

            actor_ai = Actor(content=companion_name, aliases=["ai", "assistant", "self", "me"], actor_class=ActorClass.Human)
            self.add_knoxel(actor_user)

    # Add helper to get latest narrative
    def get_narrative(self, narrative_type: NarrativeTypes, target_name: str) -> Optional[Narrative]:
        """Gets the most recent narrative knoxel of a specific type and target."""
        return Enumerable(self.all_narratives).last_or_default(
            lambda n: n.narrative_type == narrative_type and n.target_name == target_name
        )

    def _get_current_mental_state(
            self,
            reference_timeframe_minutes: int,
            conversation_partner_entity_id: Optional[int] = None,
            half_life_factor: float = 1.0,
    ) -> FullMentalState:
        """
        - Builds a time-ordered stream of (timestamp, delta_vec).
        - Computes time-aware EMA baseline with half-life = reference_timeframe_minutes * half_life_factor.
        - Normalizes each axis using ±2σ around EMA to produce a safe readout in model bounds.
        - Returns a FullMentalState constructed from the normalized vector.
        """
        if not self.all_features:
            return FullMentalState()

        # (Optional) filter by relationship if you tag features with source_entity_id
        feats = self.all_features
        if conversation_partner_entity_id is not None:
            feats = [f for f in feats if f.source_entity_id == conversation_partner_entity_id]

        history = [(f.timestamp_creation, f.mental_state_delta) for f in self.all_features]
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

    def add_stimulus(self, stimulus: Stimulus):
        self.stimulus_feature_map.clear()
        if stimulus is not None:
            self.add_knoxel(stimulus)
            self.primary_stimulus = stimulus

            if stimulus.stimulus_type == StimulusType.UserMessage:
                name = user_name
                ftype = FeatureType.Dialogue
            elif stimulus.stimulus_type == StimulusType.CompanionMessage:
                name = companion_name
                ftype = FeatureType.Dialogue
            else:
                raise ValueError

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
                        name = user_name
                        ftype = FeatureType.Dialogue
                    elif stimulus.stimulus_type == StimulusType.CompanionMessage:
                        name = companion_name
                        ftype = FeatureType.Dialogue
                    else:
                        raise ValueError()

                    story_feature = self.stimulus_feature_map[stimulus.id]
                    story_feature.timestamp_creation = new_dt
                    story_feature.timestamp_world_begin = new_dt
                    story_feature.timestamp_world_end = new_dt
                    story_feature.causal = True

    def get_context_knoxels_mental_state(self, fms: FullMentalState) -> KnoxelList:
        #fms.state_cognition.mental_aperture
        dialouge = Enumerable(self.all_features) \
            .where(lambda x: x.causal) \
            .where(lambda x: FeatureType.from_stimulus(x.feature_type)) \
            .order_by(lambda x: x.timestamp_world_begin) \
            .to_list()

        res = dialouge[-4:] + self.input_knoxels
        kl = KnoxelList(res)
        return kl

    def get_short_term_context(self, fms: FullMentalState) -> List[float]:
        kl = self.get_context_knoxels_mental_state(fms)
        emb = self.llm.get_embedding(kl.get_story(self))
        return emb

    def get_emotionally_similar_context(self, fms: FullMentalState) -> List[float]:
        return [0.0] * VectorModelReservedSize

    def build_initial_ccq(self) -> Dict[int, float]:
        res = {f.id: 1.0 for f in self.all_features if f.causal}
        return res

    def merge_ccq(self, a: Dict[int, float], b: Dict[int, float], weight: float):
        res = copy.copy(a)
        for k, v in b.items():
            if k not in res:
                res[k] = 0
            res[k] += b[k] * weight
        return res

    def get_ccq_simple(self, embedding: List[float]):
        res = {f.id: cosine_sim(f.embedding, embedding) for f in self.all_features if f.causal}
        return res

    def create_codelet_prompt(self, ccq: Dict[int, float], max_tokens: int) -> CodeletPromptContainer:
        token_budget = max_tokens * 0.9
        res = set()
        dialouge = Enumerable(self.all_features) \
            .where(lambda x: x.causal) \
            .where(lambda x: FeatureType.from_stimulus(x.feature_type)) \
            .order_by(lambda x: x.timestamp_world_begin) \
            .to_list()
        for f in dialouge[-6:]:
            res.add(f.id)
            token_budget -= get_token_count(f.content)

        s = dict(sorted(ccq.items(), key=lambda item: item[1], reverse=True))
        for k, _ in s.items():
            f = self.all_knoxels[k]
            res.add(f.id)
            token_budget -= get_token_count(f.content)

            if token_budget <= 0:
                break

        ids = []
        kl = KnoxelList([self.all_knoxels[k] for k in res]).order_by(lambda x: x.timestamp_world_begin)
        story = kl.get_story(self, max_tokens=max_tokens, target_knoxel_ids=ids)

        kl = KnoxelList(self.input_knoxels).order_by(lambda x: x.timestamp_world_begin)
        story_new = kl.get_story(self, max_tokens=max_tokens)

        res = CodeletPromptContainer(knoxels=ids, story=story, story_new=story_new)
        return res

    def get_snapshot(self, manager: CSMManager, search_embedding: List[float], target_codelet: CodeletExecutor, ms: FullMentalState, precursor_dict: Dict[str, List[int]], max_tokens: int) -> KnoxelList:
        if len(manager.state.csm_item_states) == 0:
            return None

        valence_focus_bias, salience_bias = compute_attention_bias(ms)

        ratings = {}
        for _id, item in manager.state.csm_item_states.items():
            knoxel: KnoxelBase = self.get_knoxel_by_id(_id)

            # default activation baseline
            activation = 0.5

            if isinstance(knoxel, Feature) and knoxel.feature_type in [FeatureType.CodeletPercept, FeatureType.CodeletOutput]:
                f_val = knoxel.affective_valence or 0.0  # [-1..1]
                f_sal = knoxel.incentive_salience or 0.0  # [0..1]

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
            token_budget += get_token_count(k.get_story_element(self))
            if token_budget > max_tokens:
                break

        kl = kl.order_by(lambda x: x.timestamp_world_begin)
        return kl

    def get_initial_ccq(self) -> ShellCCQUpdate:
        short_term_context: List[float] = self.get_short_term_context(None)
        temporary_ccq = self.build_initial_ccq()
        temporary_ccq = self.merge_ccq(temporary_ccq, self.get_ccq_simple(short_term_context), weight=self.previous_state.latent_mental_state.state_cognition.mental_aperture)
        initial_prompt = self.create_codelet_prompt(temporary_ccq, max_tokens=self.get_max_tokens_reply())

        story_list = KnoxelListWeighted()
        for kid in initial_prompt.knoxels:
            story_list.add(self.get_knoxel_by_id(kid))

        final_story_str = story_list.order_by(lambda x: x.timestamp_world_begin).get_story(self, max_tokens=self.get_max_tokens_reply())
        res = ShellCCQUpdate(last_causal_id=self.max_knoxel_id, current_tick=self.current_tick_id, knoxels=story_list, as_story=final_story_str)
        return res

    def cognitive_cycle(self, buffer_input: List[Stimulus]) -> ShellCCQUpdate:
        # start new tick
        self.current_tick_id += 1
        self.input_knoxels.clear()
        self.states.append(GhostState(tick_id=self._get_current_tick_id()))

        conv_partner = 0
        for stim in buffer_input:
            if stim.source_actor_id != 0:
                conv_partner = stim.source_actor_id
            self.add_stimulus(stim)

        # get timeframe for latent emotional state based on ego strengh of the previous latent state
        rtm = 24 * 60
        if self.previous_state:
            rtm *= abs(1 - self.previous_state.latent_mental_state.state_cognition.ego_strength)

        # compute full latent state from all features with that timeframe
        start_latent_ms = self._get_current_mental_state(reference_timeframe_minutes=rtm, conversation_partner_entity_id=conv_partner)

        # get last few messages with new stimulus for building base prompt
        # ratio of world / internal events based on interlocus
        short_term_context: List[float] = self.get_short_term_context(start_latent_ms)
        emotionally_similar_context: List[float] = self.get_emotionally_similar_context(start_latent_ms)

        # build initial prompt from latest ccq
        # ccq is that would did the last causal output, what was conscious in the end
        if self.previous_state and len(self.previous_state.ccq_state) > 0:
            temporary_ccq = copy.copy(self.previous_state.ccq_state)
        else:
            temporary_ccq = self.build_initial_ccq()

        # update weights with weights based on most recent input
        temporary_ccq = self.merge_ccq(temporary_ccq, self.get_ccq_simple(short_term_context), weight=self.previous_state.latent_mental_state.state_cognition.mental_aperture)
        #temporary_ccq = self.merge_ccq(temporary_ccq, self.get_ccq_simple(emotionally_similar_context), weight=self.previous_state.latent_mental_state.state_cognition.ego_strength)
        initial_prompt = self.create_codelet_prompt(temporary_ccq, max_tokens=int(self.llm.get_max_tokens(LlmPreset.Default) * 0.5))

        # build initial new csm from previous and decay
        if self.previous_state and self.previous_state.csm_state and len(self.previous_state.csm_state.csm_item_states) > 0:
            csm_state = self.previous_state.csm_state.copy(deep=True)
        else:
            csm_state = CSMState()
        csm_manager = CSMManager(self, state=csm_state)
        csm_manager.decay_step()

        cctx = CodeletContext(
            tick_id=self.current_tick_id,
            stimulus=self.primary_stimulus,
            context_embedding=short_term_context,
            llm=self.llm,
            mental_state=start_latent_ms,
            story=initial_prompt.story,
            story_new=initial_prompt.story_new,
            csm_snapshot=""
        )

        codelet_registry = CodeletRegistry.init_from_simple_codelets(cctx)
        if self.previous_state and len(self.previous_state.codelet_state.states) > 0:
            codelet_registry.apply_codelet_state(self.previous_state.codelet_state)
            codelet_registry.decay_step()

            cands = codelet_registry.pick_candidates(cctx)
            codelet_cands = [x[0] for x in cands[:20]]

            inp_hier = {
                "full_prompt": initial_prompt.story + initial_prompt.story_new,
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
            codelet_candidates = codelet_registry.pick_candidates(cctx)
            codelet_executor, activation = codelet_candidates[0]

            codelet_container = None
            first_pass = None
            while codelet_container is None and first_pass is None:
                snap = self.get_snapshot(csm_manager, search_embedding=self.llm.get_embedding(codelet_executor.signature.create_embedding_string()), target_codelet=codelet_executor, ms=start_latent_ms, precursor_dict=precursor_dict, max_tokens=512)
                cctx.csm_snapshot = snap.get_story(self)
                res = codelet_executor.run(cctx)

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

                sum_salience += sub.salience
                sum_valence += sub.valence

                appraisal_general: AppraisalGeneral = sub.appraisal_general
                appraisal_social: AppraisalSocial = sub.appraisal_social

                temp_ms = start_latent_ms.copy(deep=True)
                temp_ms.appraisal_general = appraisal_general
                temp_ms.appraisal_social = appraisal_social
                latent, delta = compute_state_delta(temp_ms, 1)

                latent_list.append(latent.to_list())
                delta_list.append(delta.to_list())

                f = Feature(
                    source=field_name,
                    content=content,
                    feature_type=FeatureType.CodeletPercept,
                    affective_valence=sub.valence,
                    incentive_salience=sub.salience,
                    interlocus=-2,
                    causal=False,
                    metadata=pydandic_model_to_dict_jsonable(sub),
                    mental_state_appraisal=latent.to_list(),
                    mental_state_delta=delta.to_list(),
                )
                self.add_knoxel(f)
                feature_ids.append(f.id)
                csm_manager.add_or_boost(CSMItem(knoxel_id=f.id, first_tick=self.current_tick_id, last_tick=-1))

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
                affective_valence=sum_valence,
                incentive_salience=sum_salience,
                interlocus=-1,
                causal=False,
                metadata=pydandic_model_to_dict_jsonable(codelet_container),
                mental_state_appraisal=combined_latent.tolist(),
                mental_state_delta=combined_delta.tolist()
            )
            feature_ids.append(f.id)
            self.add_knoxel(f)
            csm_manager.add_or_boost(CSMItem(knoxel_id=f.id, first_tick=self.current_tick_id, last_tick=-1))

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

        # get most salient codelt features and make them causal
        workspace_content = self.get_snapshot(csm_manager, search_embedding=short_term_context, target_codelet=None, ms=start_latent_ms, precursor_dict=precursor_dict, max_tokens=1024)
        for k in workspace_content.to_list():
            k.causal = True

        ccq = self.get_ccq(short_term_context, workspace_content, max_total_tokens=self.get_max_tokens_reply())

        self.postprocess_stimuli()
        self.current_state.csm_state = csm_manager.state
        self.current_state.ccq_state = {x.id: 1 for x in ccq.knoxels.to_list()}
        self.current_state.latent_mental_state = self._compute_mental_state()

        with open(f"./data/tick_{self.current_tick_id}.json", "w", encoding="utf-8") as f:
            f.write(KnoxelContainer(knoxels=list(self.all_knoxels.values())).model_dump_json(indent=4))

        return ccq

    def _compute_mental_state(self, reference_timeframe_minutes: int = 60 * 24, half_life_factor: float = 1):
        conversation_partner_entity_id = 2
        if conversation_partner_entity_id is not None:
            feats = [f for f in self.all_features if (f is None or f.source_entity_id == conversation_partner_entity_id) and (f.causal)]
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

        # Construct MS from normalized readout
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

        workspace_size = get_token_count(conscious_workspace_content.get_story(self))
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
            .where(lambda x: FeatureType.from_stimulus(x.feature_type)) \
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
            logger.info(k.get_story_element(self).replace("\n", "\\n"))

        story_list = KnoxelList(final_knoxels_for_story)
        final_story_str = story_list.get_story(self, max_tokens=max_total_tokens)  # Pass the original max_total_tokens

        res = ShellCCQUpdate(last_causal_id=self.max_knoxel_id, current_tick=self.current_tick_id, knoxels=story_list, as_story=final_story_str)
        return res

    def get_max_tokens_reply(self):
        return self.llm.get_max_tokens(LlmPreset.Default) - 512
