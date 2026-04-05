from datetime import datetime
from typing import Optional, List, Dict

from pydantic import BaseModel, Field, PrivateAttr

from pm.ghost.ghost_config import GhostConfig
from pm.ghost.ghost_state import GhostState
from pm.model.knoxel_common import Stimulus, Intention, Narrative, Action, MemoryClusterKnoxel, DeclarativeFactKnoxel, CauseEffectKnoxel, Entity
from pm.model.knoxel_core import KnoxelHaver, KnoxelBase
from pm.model.knoxel_feature import Feature
from pm.model.knoxel_graph import ConceptNode, GraphNode, GraphEdge
from pm.system.llm.llm_proxy import LlmManagerProxy
from pm.system.load_config import PmConfig
from pm.utils.datetime_utils import get_causal_datetime


class BaseGhost(KnoxelHaver):
    """
    Base ghost layer that has configs and knoxels accessors.
    """
    def __init__(self, llm: LlmManagerProxy, config: GhostConfig):
        super().__init__()
        self.system_config: PmConfig | None = PrivateAttr(default=None)
        self.ghost_config: GhostConfig = config

        self.llm: LlmManagerProxy = llm
        self.current_db_path: str = ""

        self.current_tick_id: int = 0
        self.current_knoxel_id: int = 0
        self.states: List[GhostState] = []

        self.all_features: List[Feature] = []
        self.all_stimuli: List[Stimulus] = []
        self.all_intentions: List[Intention] = []  # Includes internal goals AND external expectations
        self.all_narratives: List[Narrative] = []
        self.all_actions: List[Action] = []
        self.all_episodic_memories: List[MemoryClusterKnoxel] = []
        self.all_declarative_facts: List[DeclarativeFactKnoxel] = []
        self.all_cause_effects: List[CauseEffectKnoxel] = []
        self.all_entities: List[Entity] = []

        self.all_concepts: List[ConceptNode] = []
        self.all_graph_nodes: List[GraphNode] = []
        self.all_graph_edges: List[GraphEdge] = []

        self.simulated_reply: Optional[str] = None

    @property
    def current_state(self) -> GhostState:
        if len(self.states) > 0:
            return self.states[-1]
        return None

    @property
    def previous_state(self) -> GhostState:
        if len(self.states) > 1:
            return self.states[-2]
        return None

    def _get_state_at_tick(self, tick_id: int) -> GhostState | None:
        if tick_id == self.current_tick_id:
            return self.current_state

        for s in self.states:
            if s.tick_id == tick_id:
                return s
        return None

    def _get_next_id(self) -> int:
        self.current_knoxel_id += 1
        return self.current_knoxel_id

    def _get_current_tick_id(self) -> int:
        return self.current_tick_id

    def build_body_and_situation_addendum(self) -> str:
        current_causal_datetime = get_causal_datetime()
        dt_text = current_causal_datetime.strftime("%Y-%m-%d %H:%M:%S UTC")
        return (
            "Body and Situation\n"
            f"Current causal datetime: {dt_text}\n"
            "This is an in-character temporal anchor for reasoning."
        )

    def get_all_knoxels(self) -> List[KnoxelBase]:
        return list(self.all_knoxels.values())

    def get_simple_character_story_block(self) -> str:
        return self.ghost_config.universal_character_card

    def get_companion_name(self) -> str:
        return self.ghost_config.companion_name

    def get_user_name(self) -> str:
        return self.ghost_config.user_name

    def add_knoxel(self, knoxel: KnoxelBase, generate_embedding: bool = True):
        if knoxel.id == -1: knoxel.id = self._get_next_id()
        if knoxel.tick_id == -1: knoxel.tick_id = self._get_current_tick_id()
        knoxel._owner = self

        self.all_knoxels[knoxel.id] = knoxel

        # Add to specific lists
        if isinstance(knoxel, Feature):
            self.all_features.append(knoxel)
        elif isinstance(knoxel, Stimulus):
            self.all_stimuli.append(knoxel)
        elif isinstance(knoxel, Intention):
            self.all_intentions.append(knoxel)
        elif isinstance(knoxel, Narrative):
            self.all_narratives.append(knoxel)
        elif isinstance(knoxel, Action):
            self.all_actions.append(knoxel)
        elif isinstance(knoxel, MemoryClusterKnoxel):
            self.all_episodic_memories.append(knoxel)
        elif isinstance(knoxel, DeclarativeFactKnoxel):
            self.all_declarative_facts.append(knoxel)
        elif isinstance(knoxel, CauseEffectKnoxel):
            self.all_cause_effects.append(knoxel)
        elif isinstance(knoxel, ConceptNode):
            self.all_concepts.append(knoxel)
        elif isinstance(knoxel, GraphNode):
            self.all_graph_nodes.append(knoxel)
        elif isinstance(knoxel, GraphEdge):
            self.all_graph_edges.append(knoxel)
        elif isinstance(knoxel, Entity):
            self.all_entities.append(knoxel)

        # Generate embedding if requested and not present
        if generate_embedding and not knoxel.embedding and knoxel.content:
            knoxel.embedding = self.llm.get_embedding(knoxel.content)

    def get_knoxel_by_id(self, knoxel_id: int) -> Optional[KnoxelBase]:
        return self.all_knoxels.get(knoxel_id, None)

    def get_entity_id(self, name: str) -> int:
        for ent in self.all_entities:
            if name == ent.content:
                return ent.id

    def get_entity_name(self, entity_id: int) -> str:
        for ent in self.all_entities:
            if entity_id == ent.id:
                return ent.content

    def _reset_internal_state(self):
        """Clears all knoxels, states, and resets IDs."""
        self.current_tick_id = 0
        self.current_knoxel_id = 0
        self.all_knoxels: Dict[int, KnoxelBase] = {}
        self.all_features = []
        self.all_stimuli = []
        self.all_intentions = []
        self.all_narratives = []
        self.all_actions = []
        self.all_episodic_memories = []
        self.all_declarative_facts = []
        self.all_cause_effects = []
        self.all_concepts: List[ConceptNode] = []
        self.all_graph_nodes: List[GraphNode] = []
        self.all_graph_edges: List[GraphEdge] = []
        self.states = []
        #self.current_state = None
        self.simulated_reply = None

    def _rebuild_specific_lists(self):
        for k in self.all_knoxels.values():
            k._owner = self

        """Helper to re-populate specific lists from all_knoxels after loading."""
        self.all_features = [k for k in self.sorted_knoxels_causal if isinstance(k, Feature)]
        self.all_stimuli = [k for k in self.sorted_knoxels_causal if isinstance(k, Stimulus)]
        self.all_intentions = [k for k in self.sorted_knoxels_causal if isinstance(k, Intention)]
        self.all_narratives = [k for k in self.sorted_knoxels_causal if isinstance(k, Narrative)]
        self.all_actions = [k for k in self.sorted_knoxels_causal if isinstance(k, Action)]
        self.all_episodic_memories = [k for k in self.sorted_knoxels_causal if isinstance(k, MemoryClusterKnoxel)]
        self.all_declarative_facts = [k for k in self.sorted_knoxels_causal if isinstance(k, DeclarativeFactKnoxel)]
        self.all_cause_effects = [k for k in self.sorted_knoxels_causal if isinstance(k, CauseEffectKnoxel)]
        self.all_concepts: List[ConceptNode] = [k for k in self.sorted_knoxels_causal if isinstance(k, ConceptNode)]
        self.all_graph_nodes: List[GraphNode] = [k for k in self.sorted_knoxels_causal if isinstance(k, GraphNode)]
        self.all_graph_edges: List[GraphEdge] = [k for k in self.sorted_knoxels_causal if isinstance(k, GraphEdge)]

if __name__ == '__main__':
    gs = GhostState(tick_id=1)
