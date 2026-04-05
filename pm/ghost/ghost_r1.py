import copy
import math
import os
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

from pm.ghost.ghost_base import BaseGhost
from pm.model.knoxel_common import Narrative, Stimulus, Entity, DeclarativeFactKnoxel
from pm.model.knoxel_enums import StimulusTriage, NarrativeTypes, EntityClass, FeatureType
from pm.model.knoxel_feature import Feature
from pm.model.mental_state_vectors import FullMentalState, VectorModelReservedSize, _collect_axis_bounds, ema_baselined_normalize, _init_ms_from_vec
from pm.model.meta_narrative import narrative_definitions
from pm.subsystems.memory.graph_memory import CognitiveMemoryManager
from pm.subsystems.memory.memory_consolidation import MemoryConsolidationConfig, DynamicMemoryConsolidator
from pm.system.llm.llm_common import LlmPreset
from pm.utils.datetime_utils import get_causal_datetime
from pm.utils.system_utils import generate_start_message


class GhostR1(BaseGhost):
    """
    Ghost layer with character initialization and context building.
    """
    def __init__(self, llm, config):
        super().__init__(llm, config)
        self.memory_config = MemoryConsolidationConfig()
        self.memory_consolidator = DynamicMemoryConsolidator(llm, self, self.memory_config)
        self.memory_manager: CognitiveMemoryManager = CognitiveMemoryManager(llm, self)

    def initialize_basic_knoxels(self):
        self.init_actors()
        self.init_narratives()
        self.init_character()

    def init_character(self):
        if len(self.all_features) > 0:
            return

        now = get_causal_datetime()
        date_str = now.strftime(self.system_config.timestamp_format)
        comp_name = self.system_config.companion_name
        usr_name = self.system_config.user_name

        init_message = generate_start_message(comp_name, usr_name, os.path.basename(self.system_config.model_map[LlmPreset.Default.value]["path"]))
        init_feature = Feature(
            content=init_message,
            feature_type=FeatureType.SystemMessage,
            source=self.system_config.shell_system_name,
            interlocus=1
        )
        self.add_knoxel(init_feature)

        init_memory = DeclarativeFactKnoxel(
            content=f"{comp_name} was first activated on {date_str}.",
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
            content=f"{comp_name} was first activated on {date_str}. This is their boot message: {init_message}",
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

    def init_narratives(self):
        if len(self.all_narratives) > 0:
            return
        existing = {(n.narrative_type, n.target_name) for n in self.all_narratives}
        for definition in narrative_definitions:
            if (definition["type"], definition["target"]) not in existing:
                default_content = ""
                if definition["target"] == "{companion_name}":
                    default_content = f"<No information regarding {definition['type']} available *yet*, gather details from conversation>"
                elif definition["target"] == "{user_name}":
                    default_content = f"{self.system_config.user_name} is the user interacting with {self.system_config.companion_name}."

                narrative = Narrative(
                    narrative_type=definition["type"],
                    target_name=definition["target"].format(companion_name=self.system_config.companion_name, user_name=self.system_config.user_name),
                    content=default_content,
                )
                self.add_knoxel(narrative)

    def init_actors(self):
        if len(self.all_entities) > 0:
            return
        actor_user = Entity(content=self.system_config.user_name, aliases=["user", "human"], entity_class=EntityClass.Human)
        self.add_knoxel(actor_user)

        actor_ai = Entity(content=self.system_config.companion_name, aliases=["ai", "assistant", "ghost", "me"], entity_class=EntityClass.AI)
        self.add_knoxel(actor_ai)

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

        feats = self.all_features
        if conversation_partner_entity_id is not None:
            conv_partner = self.get_entity_name(conversation_partner_entity_id)
            feats = [f for f in feats if f.source in (None, conv_partner)]

        history = [(f.timestamp_creation, f.mental_state_delta) for f in feats]
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
