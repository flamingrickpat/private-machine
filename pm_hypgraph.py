import networkx as nx
from datetime import datetime
import json  # For pretty printing node data


# --- The Universal Atom of Thought ---
# While we don't need a rigid Pydantic model for networkx attributes,
# this serves as the conceptual reference for what every node represents.
class UniversalNodeReference:
    """
    This is a conceptual guide, not a class we instantiate.
    Every node in our NetworkX graph will have these attributes.
    """
    hri: str  # Human-Readable Identifier, the primary key.
    concept_type_hri: str  # Link to its type in the Lexical-Conceptual Ontology.
    description: str  # Natural language description of the node.
    # Optional attributes
    # embedding: List[float]
    # grounding_anchor_hri: str


class CognitiveHypergraph:
    """
    Manages the unified hypergraph, representing the AI's entire knowledge,
    memory, and reasoning structure.
    """

    def __init__(self, companion_name: str):
        """
        Initializes the hypergraph, building the foundational ontologies
        and the AI's core self-concept.

        Args:
            companion_name: The name of the AI agent who "owns" this brain.
        """
        print("Initializing the Cognitive Hypergraph...")
        self.graph = nx.MultiDiGraph()
        self.companion_name = companion_name

        # The orchestration of creation
        self._initialize_lexical_ontology()
        self._initialize_structural_memory_ontology()
        self._initialize_thematic_ontology()
        self._initialize_temporal_ontology()
        self._initialize_causal_logical_ontology()
        self._initialize_ego()

        print(f"Hypergraph initialized for companion '{self.companion_name}'.")
        print(
            f"Foundational structure contains {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")

    def _add_node(self, hri: str, concept_type_hri: str, description: str, **kwargs):
        """
        The fundamental operation to add a new node (an 'atom') to the graph.
        """
        if self.graph.has_node(hri):
            return  # Avoid overwriting existing nodes during init

        self.graph.add_node(
            hri,
            hri=hri,
            concept_type_hri=concept_type_hri,
            description=description,
            created_at=datetime.utcnow(),
            **kwargs
        )

    def _add_hyper_edge(self, source_hri: str, target_hri: str, relation_type_hri: str):
        """
        Creates a directed relationship between two nodes. This is the simplest
        form of link, used to build the ontological structures.
        """
        self.graph.add_edge(source_hri, target_hri, key=relation_type_hri, relation_type_hri=relation_type_hri)

    def _initialize_lexical_ontology(self):
        """
        ONTOLOGY 1: The Dictionary. Defines the vocabulary of concepts and relations.
        """
        print("Building Lexical-Conceptual Ontology...")
        # Core Root Nodes
        self._add_node("Concept:Thing", "Concept:OntologyRoot",
                       "The ultimate root of all concepts. Everything is a 'Thing'.")
        self._add_node("RelationType:RelatedTo", "Concept:OntologyRoot", "The ultimate root of all relations.")

        # Build out a sample concept hierarchy
        concepts = {
            "Concept:Agent": ("Concept:Thing", "An entity that can act."),
            "Concept:AI_Companion": ("Concept:Agent", "A personalized AI agent."),
            "Concept:Person": ("Concept:Agent", "A human being."),
            "Concept:Object": ("Concept:Thing", "An inanimate object."),
            "Concept:Place": ("Concept:Thing", "A location."),
            "Concept:Abstract": ("Concept:Thing", "A concept that has no physical referent."),
            "Concept:Emotion": ("Concept:Abstract", "A mental state associated with feelings."),
            "Concept:Fact": ("Concept:Abstract", "A statement representing a piece of knowledge."),
            "Concept:Event": ("Concept:Abstract", "Something that happens at a particular time and place."),
        }
        for hri, (parent_hri, desc) in concepts.items():
            self._add_node(hri, "Concept:Lexical", desc)
            self._add_hyper_edge(hri, parent_hri, "RelationType:IsA")

        # Build out a sample relation hierarchy
        relations = {
            "RelationType:IsA": ("RelationType:RelatedTo", "Expresses inheritance or class membership."),
            "RelationType:HasProperty": ("RelationType:RelatedTo", "Links an entity to one of its attributes."),
            "RelationType:InterpersonalFeeling": ("RelationType:HasProperty",
                                                  "A feeling one agent has towards another."),
            "RelationType:Likes": ("RelationType:InterpersonalFeeling", "A positive interpersonal feeling."),
            "RelationType:HappenedAt": ("RelationType:RelatedTo", "Links an event to a time or place."),
        }
        for hri, (parent_hri, desc) in relations.items():
            self._add_node(hri, "Concept:Lexical", desc)
            self._add_hyper_edge(hri, parent_hri, "RelationType:IsA")

    def _initialize_structural_memory_ontology(self):
        """
        ONTOLOGY 2: The Library Map. Defines the AI's own mental architecture.
        """
        print("Building Structural-Memory Ontology...")
        self._add_node("MemorySystem:Core", "Concept:MemorySystem", "The root of the AI's memory systems.")

        systems = {
            "MemorySystem:LongTermMemory": ("MemorySystem:Core", "The persistent store of knowledge."),
            "MemorySystem:Semantic": ("MemorySystem:LongTermMemory",
                                      "Memory for facts, concepts, and general knowledge."),
            "MemorySystem:Episodic": ("MemorySystem:LongTermMemory", "Memory for specific events and experiences."),
            "MemorySystem:Autobiographical": ("MemorySystem:Episodic", "Memory for events related to the self."),
        }
        for hri, (parent_hri, desc) in systems.items():
            self._add_node(hri, "Concept:MemorySystem", desc)
            self._add_hyper_edge(hri, parent_hri, "RelationType:IsSubsystemOf")

    def _initialize_thematic_ontology(self):
        """
        ONTOLOGY 3: The Card Catalog. Defines the hierarchy of topics.
        """
        print("Building Thematic-Semantic Ontology...")
        self._add_node("Theme:World", "Concept:Theme", "The root of all thematic topics about the world.")

        themes = {
            "Theme:User": ("Theme:World", "Topics related to the primary user."),
            "Theme:User_Preferences": ("Theme:User", "The user's likes, dislikes, and preferences."),
            "Theme:AI": ("Theme:World", "Topics related to artificial intelligence."),
            "Theme:AI_Self": ("Theme:AI", "Topics related to the AI companion itself."),
        }
        for hri, (parent_hri, desc) in themes.items():
            self._add_node(hri, "Concept:Theme", desc)
            self._add_hyper_edge(hri, parent_hri, "RelationType:HasSubtopic")

    def _initialize_temporal_ontology(self):
        """
        ONTOLOGY 4: The Universal Clock. Defines concepts of time.
        """
        print("Building Temporal Ontology...")
        self._add_node("TimeConcept:Time", "Concept:Abstract", "The root concept for all temporal notions.")

        time_concepts = {
            "TimeType:WorldTime": ("TimeConcept:Time", "Objective, real-world time."),
            "TimeType:LearningTime": ("TimeConcept:Time", "The subjective time at which a fact was learned."),
            "TimeType:ValidityTime": ("TimeConcept:Time", "The time interval during which a fact is true."),
            "TimeConcept:Eternity": ("TimeConcept:Time", "A state of timelessness or infinite duration."),
        }
        for hri, (parent_hri, desc) in time_concepts.items():
            self._add_node(hri, "Concept:Temporal", desc)
            self._add_hyper_edge(hri, parent_hri, "RelationType:IsA")

    def _initialize_causal_logical_ontology(self):
        """
        ONTOLOGY 5: The Engine of Reason. Defines concepts for logic and causality.
        """
        print("Building Causal-Logical Ontology...")
        self._add_node("LogicConcept:Logic", "Concept:Abstract", "The root for all logical and causal concepts.")

        logic_concepts = {
            "CausalRelation:Causes": ("LogicConcept:Logic", "Indicates that one event or state brings about another."),
            "CausalRelation:Enables": ("LogicConcept:Logic",
                                       "Indicates a state or event is a prerequisite for another."),
            "LogicalOperator:IfThen": ("LogicConcept:Logic", "Represents a conditional rule."),
        }
        for hri, (parent_hri, desc) in logic_concepts.items():
            self._add_node(hri, "Concept:Logical", desc)
            self._add_hyper_edge(hri, parent_hri, "RelationType:IsA")

    def _initialize_ego(self):
        """
        Creates the central "I" node for the AI companion.
        """
        print(f"Creating Ego Node for '{self.companion_name}'...")
        ego_hri = f"AI:{self.companion_name}"

        self._add_node(
            ego_hri,
            "Concept:AI_Companion",
            f"The self-concept node for me, {self.companion_name}."
        )
        # Link the ego to its self-identity in the thematic ontology
        self._add_hyper_edge(ego_hri, "Theme:AI_Self", "RelationType:IsThemedBy")

    def save_graph(self, path: str = "cognitive_hypergraph.graphml"):
        """
        Saves the current graph to a file.
        GraphML is a good, human-readable choice.
        """
        print(f"Saving graph to {path}...")
        # Convert datetime objects to ISO format strings for serialization
        for node, data in self.graph.nodes(data=True):
            if 'created_at' in data and isinstance(data['created_at'], datetime):
                data['created_at'] = data['created_at'].isoformat()

        nx.write_graphml(self.graph, path)
        print("Save complete.")

    def display_node(self, hri: str):
        """Utility to pretty-print a node's data."""
        if self.graph.has_node(hri):
            data = self.graph.nodes[hri]
            print(f"--- Node Data: {hri} ---")
            print(json.dumps(data, indent=2, default=str))

            print("\n--- Incoming Edges ---")
            for source, _, key, data in self.graph.in_edges(hri, keys=True, data=True):
                print(f"  <-[{data['relation_type_hri']}]- ({source})")

            print("\n--- Outgoing Edges ---")
            for _, target, key, data in self.graph.out_edges(hri, keys=True, data=True):
                print(f"  - [{data['relation_type_hri']}]-> ({target})")

        else:
            print(f"Node '{hri}' not found.")


# --- Example Usage ---
if __name__ == "__main__":
    # 1. Initialize the "brain" for an AI named "Companion"
    ai_mind = CognitiveHypergraph(companion_name="Companion")

    # 2. Let's inspect a few key nodes to see the structure
    print("\n\n--- Inspecting Initialized Structure ---")
    ai_mind.display_node("AI:Companion")
    print("\n" + "=" * 40 + "\n")
    ai_mind.display_node("RelationType:Likes")

    # 3. Save the foundational graph to a file for later use
    ai_mind.save_graph("companion_brain_foundation.graphml")