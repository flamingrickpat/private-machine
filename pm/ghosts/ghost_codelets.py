import copy
import json
import math
import random
import time
from datetime import datetime
from enum import StrEnum
from pprint import pprint
from typing import Dict, Any
from typing import List
from typing import Type
from typing import TypeVar
from typing import (
    Union,
    Optional,
    Tuple,
)

import numpy as np
from py_linq import Enumerable
from pydantic import BaseModel, Field, create_model
from pydantic import ValidationError
from scipy.spatial.distance import cosine as cosine_distance  # Use scipy for cosine distance
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from pm.agents.definitions.agent_select_codelets import AgentSelectCodelets
from pm.character_card import default_agent_sysprompt, character_card_assistant
from pm.codelets.codelet import CodeletRegistry, CodeletFamily, CodeletContext, CodeletState
from pm.config_loader import *
from pm.csm.csm import CSMState, CSMBuffer, CSMItem
from pm.data_structures import ActionType, Narrative, narrative_definitions, NarrativeTypes, Stimulus, Action, KnoxelList, FeatureType, Feature, StimulusType, Intention, CoversTicksEventsKnoxel, MemoryClusterKnoxel, DeclarativeFactKnoxel, ACTION_DESCRIPTIONS, KnoxelBase, ClusterType, MLevel, KnoxelListWeighted
from pm.dialog import DialogActPool
from pm.ghosts.base_ghost import BaseGhost, GhostState, GhostConfig
from pm.ghosts.ghost_lida import EngagementIdea
from pm.llm.llm_common import LlmPreset, CommonCompSettings
from pm.memory_consolidation import MemoryConsolidationConfig, DynamicMemoryConsolidator
from pm.mental_state_vectors import FullMentalState, VectorModelReservedSize, _collect_axis_bounds, ema_baselined_normalize, _init_ms_from_vec
from pm.mental_states import NeedsAxesModel, CognitiveEventTriggers, EmotionalAxesModel, _describe_emotion_valence_anxiety, _verbalize_emotional_state
from pm.thoughts import TreeOfThought
from pm.utils.emb_utils import cosine_pair
from pm.utils.profile_utils import profile
from pm.utils.pydantic_utils import basemodel_to_text, pydandic_model_to_dict_jsonable
from pm.utils.system_utils import generate_start_message
from pm.utils.token_utils import get_token_count
from pm.utils.utterance_extractor import UtteranceExtractor
from pm.codelets.data_acquisition_codelet import DataAcquisitionCodelet

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

class PromptContainer(BaseModel):
    knoxels: List[int]
    story: str
    story_new: str


class GhostCodelets(BaseGhost):
    def __init__(self, llm, config):
        super().__init__(llm, config)
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

        history = [(f.timestamp_creation, f.state_delta_vector) for f in self.all_features]
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
        if stimulus is not None:
            self.add_knoxel(stimulus)
            self.primary_stimulus = stimulus

            if stimulus.stimulus_type == StimulusType.UserMessage:
                story_feature = Feature(content=stimulus.content, source=user_name, feature_type=FeatureType.Dialogue, interlocus=1, causal=False)
            elif stimulus.stimulus_type in [StimulusType.LowNeedTrigger, StimulusType.WakeUp, StimulusType.TimeOfDayChange, StimulusType.SystemMessage, StimulusType.UserInactivity]:
                story_feature = Feature(content=stimulus.content, source=stimulus.source, feature_type=FeatureType.SystemMessage, interlocus=-1, causal=False)
            elif stimulus.stimulus_type in [StimulusType.EngagementOpportunity]:
                idea_json = stimulus.content
                idea = EngagementIdea.model_validate_json(idea_json)
                idea_str = f"{idea.thought_process} Action Plan: {idea.suggested_action} {idea.action_content}"
                story_feature = Feature(content=idea_str, source=companion_name, feature_type=FeatureType.ExternalThought, interlocus=-1, causal=False)
            else:
                raise Exception()

            self.add_knoxel(story_feature)
            self.input_knoxels.append(story_feature)

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

    def build_initial_ccq(self) -> KnoxelListWeighted:
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
        res = {f.id: cosine_pair(f.embedding, embedding) for f in self.all_features if f.causal}
        return res

    def create_codelet_prompt(self, ccq: Dict[int, float], max_tokens: int) -> PromptContainer:
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

        kl = KnoxelList([self.all_knoxels[k] for k in res]).order_by(lambda x: x.timestamp_world_begin)
        story = kl.get_story(self, max_tokens=max_tokens)

        kl = KnoxelList(self.input_knoxels).order_by(lambda x: x.timestamp_world_begin)
        ids = []
        story_new = kl.get_story(self, max_tokens=max_tokens, target_knoxel_ids=ids)

        res = PromptContainer(knoxels=ids, story=story, story_new=story_new)
        return res


    def cognitive_cycle(self, buffer_input: List[Stimulus]):
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
        if self.previous_state and len(self.previous_state.ccq) > 0:
            temporary_ccq = copy.copy(self.previous_state.ccq)
        else:
            temporary_ccq = self.build_initial_ccq()

        # update weights with weights based on most recent input
        temporary_ccq = self.merge_ccq(temporary_ccq, self.get_ccq_simple(short_term_context), weight=self.previous_state.latent_mental_state.state_cognition.mental_aperture)
        #temporary_ccq = self.merge_ccq(temporary_ccq, self.get_ccq_simple(emotionally_similar_context), weight=self.previous_state.latent_mental_state.state_cognition.ego_strength)
        initial_prompt = self.create_codelet_prompt(temporary_ccq, max_tokens=int(self.llm.get_max_tokens(LlmPreset.Default) * 0.5))

        # build initial new csm from previous and decay
        if self.previous_state and self.previous_state.current_situational_model and len(self.previous_state.current_situational_model.items) > 0:
            csm_state = self.previous_state.current_situational_model.copy(deep=True)
        else:
            csm_state = CSMState()
        csm = CSMBuffer(self, state=csm_state)
        csm.decay_step()

        cctx = CodeletContext(
            tick_id=self.current_tick_id,
            stimulus=self.primary_stimulus,
            context_embedding=short_term_context,
            llm=self.llm,
            mental_state=start_latent_ms,
            story=initial_prompt.story,
            story_new=initial_prompt.story_new,
            csm_snapshot=csm.get_snapshot()
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
                "content": "I have made a list with possible mental functions. I want to psychologically simulate what {companion_name} would say based on her psyche."
                           "But the list is long. You now act as a selector for possible mental functions. Here is the JSON schema with all the descriptions, and names. "
                           "Please give each codelet a rating between 0 and 1, for how likely it is I should invoke it for this conversation.",
            }
            res_hier = AgentSelectCodelets.execute(inp_hier, self.llm, None)
            codelet_ratings: BaseModel = res_hier["codelets"]

            for k, v in codelet_ratings.dict():
                codelet_registry.items[k].runtime.activation += v

        # gather codelets for data aquisition
        # run codelet, add to csm, boost paths if possible, sample next codelet, call with extended csm
        # to prevent csm overflow we need to tag each codelet output with ms vector, and salience and valence
        cnt = 0
        while True:
            codelet_candidates = codelet_registry.pick_candidates(cctx)
            codelet, activation = codelet_candidates[0]

            cctx.csm_snapshot=csm.get_snapshot()
            res = codelet.run(cctx)

            first_pass = res["first_pass"]
            codelet_container = res["output_feature"]

            content = f"## {codelet.signature.name}\n{first_pass}\n"
            f = Feature(
                content=first_pass,
                source=codelet.signature.name,
                feature_type=FeatureType.CodeletOutput,
                affective_valence=0,
                interlocus=-1,
                causal=False,
                metadata=pydandic_model_to_dict_jsonable(codelet_container)
            )

            test = pydandic_model_to_dict_jsonable(codelet_container)
            test2 = codelet_container.__class__.model_validate_json(json.dumps(test))

            self.add_knoxel(f)
            csm.add_or_boost(CSMItem(knoxel_id=f.id, first_tick=self.current_tick_id, last_tick=-1))

            #from rich import print as rprint
            #rprint(codelet_container)

            #for field_name, model_field in codelet_container.__class__.model_fields.items():
            #    sub = getattr(codelet_container, field_name)
            #    content = f"## {sub.__class__.__name__}\n{basemodel_to_text(sub)}\n"
            #    f = Feature(
            #        content=content,
            #        feature_type=FeatureType.Narrative,
            #        affective_valence=0,
            #        interlocus=-1,
            #        causal=False,
            #        metadata=pydandic_model_to_dict_jsonable(sub)
            #    )
            #    self.add_knoxel(f)
            #    csm.add_or_boost(CSMItem(knoxel_id=f.id, first_tick=self.current_tick_id, last_tick=-1))

            if cnt == 8:
                break
            cnt += 1

        # simulate attention
        exit(1)

        # 1. Initialize State (Handles decay, internal stimulus gen, carries over intentions/expectations)
        self.initialize_tick_state(stimulus)
        if not self.current_state:
            raise Exception("Empty state!")

        # build initial state
        # 1. build latent state from all features
        # 2. rebuild csm from csm buffer of last state
        # 2a. list of dicts [{id: 333, cur_c
        # 3. get ccq of last state
        # 3a. create master prompt
        """
        you are a story writer. you can write stories and extract information without making something up.
        you are an expert in immersive storytelling with vibrant characters.
        user
        <last ccq>
        do you get all that?
        ass
        yes i do, what is your task?
        """


        pass