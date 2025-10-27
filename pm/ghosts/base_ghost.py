from datetime import datetime
from typing import Optional, List, Dict

from pydantic import BaseModel, Field

from pm.codelets.codelet import CodeletState
from pm.data_structures import KnoxelBase, Feature, Stimulus, Intention, Narrative, Action, MemoryClusterKnoxel, DeclarativeFactKnoxel, KnoxelHaver, ConceptNode, GraphNode, GraphEdge, Actor
from pm.csm.csm import CSMState
from pm.llm.llm_proxy import LlmManagerProxy
from pm.mental_state_vectors import FullMentalState, VectorModelReservedSize, create_empty_ms_vector, create_empty_state
from pm.config_loader import companion_name, user_name, character_card_story


# --- Configuration ---
class GhostConfig(BaseModel):
    companion_name: str = companion_name
    user_name: str = user_name
    universal_character_card: str = character_card_story
    default_decay_factor: float = 0.95
    retrieval_limit_episodic: int = 8
    retrieval_limit_facts: int = 16  # Reduced slightly
    retrieval_limit_features_context: int = 16  # Reduced slightly
    retrieval_limit_features_causal: int = 256  # Reduced slightly
    context_events_similarity_max_tokens: int = 512
    short_term_intent_count: int = 3
    # --- Expectation Config ---
    retrieval_limit_expectations: int = 5  # How many active expectations to check against stimulus
    expectation_relevance_decay: float = 0.8  # Decay factor per tick for expectation relevance (recency)
    expectation_generation_count: int = 2  # How many expectations to try generating per action
    # --- Action Selection Config ---
    min_simulations_per_reply: int = 1
    mid_simulations_per_reply: int = 1
    max_simulations_per_reply: int = 1  # Keep low initially for performance
    importance_threshold_more_sims: float = 0.5  # Stimulus valence abs() or max urgency > this triggers more sims
    force_assistant: bool = False
    remember_per_category_limit: int = 4
    coalition_aux_rating_factor: float = 0.03
    cognition_delta_factor: float = 0.33
    complex_coalition_rating: bool = False


class GhostState(BaseModel):
    tick_id: int
    previous_tick_id: int = -1
    timestamp: datetime = Field(default_factory=datetime.now)
    rating: int = Field(default=0)

    latent_mental_state: FullMentalState = Field(default_factory=create_empty_state)
    csm_state: CSMState = Field(default_factory=CSMState)
    codelet_state: CodeletState = Field(default_factory=CodeletState)
    ccq_state: Dict[int, float] = Field(default_factory=dict)

    #primary_stimulus: Optional[Stimulus] = None
    #attention_candidates: KnoxelList = KnoxelList()
    #attention_focus: KnoxelList = KnoxelList()
    #conscious_workspace: KnoxelList = KnoxelList()
#
    #coalitions_hard: Dict[int, List[KnoxelBase]] = {}
    #coalitions_balanced: Dict[int, List[KnoxelBase]] = {}
    #subjective_experience: Optional[Feature] = None
    #subjective_experience_tool: Optional[Feature] = None
#
    ## Action Deliberation & Simulation Results
    #action_simulations: List[Dict[str, Any]] = []  # Store results of MC simulations [{sim_id, type, content, rating, predicted_state, user_reaction}, ...]
    #selected_action_details: Optional[Dict[str, Any]] = None  # Details of the chosen simulation/action before execution
    #selected_action_knoxel: Optional[Action] = None  # The final Action knoxel generated (set in execute)
#
    ## user rating 0 neutral, -1 bad, 1 good
    #rating: int = Field(default=0)
#
    ## State snapshots
    #state_emotions: EmotionalAxesModel = EmotionalAxesModel()
    #state_needs: NeedsAxesModel = NeedsAxesModel()
    #state_cognition: CognitionAxesModel = CognitionAxesModel()


class BaseGhost(KnoxelHaver):
    def __init__(self, llm: LlmManagerProxy, config: GhostConfig):
        super().__init__()
        self.llm = llm
        self.config = config
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
        self.all_actors: List[Actor] = []

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

    def add_knoxel(self, knoxel: KnoxelBase, generate_embedding: bool = True):
        if knoxel.id == -1: knoxel.id = self._get_next_id()
        if knoxel.tick_id == -1: knoxel.tick_id = self._get_current_tick_id()

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
        elif isinstance(knoxel, ConceptNode):
            self.all_concepts.append(knoxel)
        elif isinstance(knoxel, GraphNode):
            self.all_graph_nodes.append(knoxel)
        elif isinstance(knoxel, GraphEdge):
            self.all_graph_edges.append(knoxel)
        elif isinstance(knoxel, Actor):
            self.all_actors.append(knoxel)

        # Generate embedding if requested and not present
        if generate_embedding and not knoxel.embedding and knoxel.content:
            knoxel.embedding = self.llm.get_embedding(knoxel.content)

    def get_knoxel_by_id(self, knoxel_id: int) -> Optional[KnoxelBase]:
        return self.all_knoxels.get(knoxel_id, None)

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
        self.all_concepts: List[ConceptNode] = []
        self.all_graph_nodes: List[GraphNode] = []
        self.all_graph_edges: List[GraphEdge] = []
        self.states = []
        #self.current_state = None
        self.simulated_reply = None

    def _rebuild_specific_lists(self):
        """Helper to re-populate specific lists from all_knoxels after loading."""
        self.all_features = [k for k in self.sorted_knoxels if isinstance(k, Feature)]
        self.all_stimuli = [k for k in self.sorted_knoxels if isinstance(k, Stimulus)]
        self.all_intentions = [k for k in self.sorted_knoxels if isinstance(k, Intention)]
        self.all_narratives = [k for k in self.sorted_knoxels if isinstance(k, Narrative)]
        self.all_actions = [k for k in self.sorted_knoxels if isinstance(k, Action)]
        self.all_episodic_memories = [k for k in self.sorted_knoxels if isinstance(k, MemoryClusterKnoxel)]
        self.all_declarative_facts = [k for k in self.sorted_knoxels if isinstance(k, DeclarativeFactKnoxel)]
        self.all_concepts: List[ConceptNode] = [k for k in self.sorted_knoxels if isinstance(k, ConceptNode)]
        self.all_graph_nodes: List[GraphNode] = [k for k in self.sorted_knoxels if isinstance(k, GraphNode)]
        self.all_graph_edges: List[GraphEdge] = [k for k in self.sorted_knoxels if isinstance(k, GraphEdge)]

if __name__ == '__main__':
    gs = GhostState(tick_id=1)