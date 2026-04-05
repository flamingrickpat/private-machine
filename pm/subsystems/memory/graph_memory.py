# --- IMPORTS ---

import logging
from collections import defaultdict
from datetime import datetime
from enum import StrEnum
from typing import Dict, List, Optional, Tuple, Union, Set

import networkx as nx
import numpy as np
import regex as re
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
from scipy.spatial.distance import cosine  # For similarity calculation

from pm.agents.definitions.agent_graph_memory import (
    CheckGraphMemoryContradiction,
    ClassifyGraphMemoryEntity,
    DefineGraphMemoryConcept,
    ExtractGraphMemoryEntities,
    ExtractGraphMemoryRelationships,
    GraphMemoryConceptDefinition,
    GraphMemoryContradictionCheck,
    GraphMemoryEntityClassification,
    GraphMemoryExtractedEntities,
    GraphMemoryExtractedEntity,
    GraphMemoryExtractedRelationship,
    GraphMemoryExtractedRelationships,
    GraphMemoryRelevantItems,
    GraphMemoryRephrasedQuestions,
    PruneGraphMemoryContext,
    RephraseGraphMemoryQuestion,
    SynthesizeGraphMemoryAnswer,
)
from pm.consts import TIMESTAMP_FORMAT
from pm.ghost.ghost_base import BaseGhost
from pm.model.knoxel_common import MemoryClusterKnoxel
from pm.model.knoxel_enums import ClusterType
from pm.model.knoxel_graph import GraphNode, GraphEdge, ConceptNode
from pm.system.llm.llm_common import LlmPreset
from pm.system.llm.llm_proxy import LlmManagerProxy
from pm.utils.datetime_utils import get_causal_datetime
from pm.utils.string_utils import are_similar

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)


class StructuralEdgeLabel(StrEnum):
    INSTANCE_OF = "INSTANCE_OF"
    IS_A = "IS_A"
    EXPERIENCED_BY = "EXPERIENCED_BY"


SEMANTIC_AXES_DEFINITION = {
    "animacy": ("living being, animal, person", "inanimate object, rock, tool"),
    "social_interaction": ("conversation, relationship, team", "solitude, isolation, individual work"),
    "concreteness": ("physical object, location", "abstract idea, concept, plan"),
    "temporality": ("event, moment in time, history", "enduring state, permanent characteristic"),
    "emotional_tone": ("joy, success, happiness", "sadness, failure, conflict"),
    "agency": ("action, decision, intention", "observation, passive event, occurrence")
}

# Default embedding dimension (example, should match your LLM's output)
EMBEDDING_DIM = 768  # Replace with actual dimension of your embeddings

# Pre-compiled set of common English stop words for the lightweight keyword extractor.
STOP_WORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at',
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', "can't", 'cannot',
    'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few',
    'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll",
    "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll",
    "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', "let's", 'me', 'more', 'most',
    "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our',
    'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should',
    "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves',
    'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those',
    'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're",
    "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who',
    "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're",
    "you've", 'your', 'yours', 'yourself', 'yourselves', 'tell', 'give', 'ask', 'name', 'list', 'describe', 'can', 'could'
}

class MemoryQueryParameters(BaseModel):
    question: str = Field(default="")
    semantic_top_k: int = 24
    semantic_similarity_threshold: float = 0.4
    keyword_top_k: int = 40
    keyword_min_term_matches: int = 1
    keyword_min_score: float = 0.05
    hybrid_top_k: int = 48
    hybrid_semantic_weight: float = 1.0
    hybrid_keyword_weight: float = 0.9
    min_candidate_count: int = 8
    max_retry_rounds: int = 3
    max_query_expansions: int = 3
    prune_trigger_count: int = 16
    max_context_items: int = 20
    llm_temperature: float = 0.3
    llm_max_tokens: int = 2048
    max_agent_calls: int = 8
    direct_answer_weight: float = 1.0
    regularities_weight: float = 1.0
    evidence_weight: float = 1.0
    unknowns_weight: float = 1.0
    target_word_count: int = 512


class MemoryQueryResult(BaseModel):
    content: str
    source_ids: List[int]


class CognitiveMemoryManager:
    _instances = {}

    def __init__(self, llm_manager: LlmManagerProxy, ghost: BaseGhost):
        self.llm_preset = LlmPreset.Default
        self.ghost = ghost
        self.llm_manager: LlmManagerProxy = llm_manager
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()  # Combined graph
        logger.info("CognitiveMemoryManager initialized successfully.")
        self._rebuild_networkx_graph()

        self.dirty: bool = True
        self._concept_cache: Dict[int, ConceptNode] = {}
        self._node_cache: Dict[int, GraphNode] = {}
        self._edge_cache: Dict[int, GraphEdge] = {}

        self._bootstrap_core_concepts()
        self.__class__._instances[self.__class__.__name__] = self

    @classmethod
    def get_latest_instance(cls):
        return cls._instances[cls.__name__]

    @property
    def concepts(self) -> Dict[int, ConceptNode]:
        return {x.id: x for x in self.ghost.all_concepts}

    @property
    def nodes(self) -> Dict[int, GraphNode]:
        return {x.id: x for x in self.ghost.all_graph_nodes}

    @property
    def edges(self) -> Dict[int, GraphEdge]:
        return {x.id: x for x in self.ghost.all_graph_edges}

    @property
    def memorys(self) -> Dict[int, MemoryClusterKnoxel]:
        return {x.id: x for x in self.ghost.all_episodic_memories}

    def _get_or_create_node_for_relationship(self, node_name: str, resolved_nodes_map: Dict[str, GraphNode],
                                             current_memory_id: int) -> Optional[GraphNode]:
        """
        Checks if a node exists in the resolved map; if not, creates it.
        This handles cases where relationship extraction finds an entity that entity extraction missed.
        """
        # 1. Check if the node is already resolved from the initial entity pass.
        if node_name in resolved_nodes_map:
            return resolved_nodes_map[node_name]

        # 2. If not, this entity was likely part of a relationship but not a primary "named entity".
        # We must now create it to complete the edge.
        logger.info(f"Implicitly resolving node for '{node_name}' mentioned in a relationship.")

        # Use the existing robust _resolve_entity_and_concept flow.
        # We create a temporary extracted entity to pass to it.
        # We can guess a generic label like "Concept" or let the classification figure it out.
        implicit_entity_data = GraphMemoryExtractedEntity(name=node_name, label="Entity")
        newly_resolved_node = self._resolve_entity_and_concept(implicit_entity_data, current_memory_id)

        if newly_resolved_node:
            # Add it to the map so we don't create it again in this integration cycle.
            resolved_nodes_map[node_name] = newly_resolved_node

        return newly_resolved_node

    def _get_or_create_concept_on_the_fly(self, concept_name: str) -> Optional[ConceptNode]:
        """
        Finds a concept by name. If it doesn't exist, it uses the LLM
        to define it and place it in the concept hierarchy.
        """
        # 1. First, try to find the concept (case-insensitive search).
        for concept in self.concepts.values():
            if concept.content.lower() == concept_name.lower():
                logger.debug(f"Found existing concept: '{concept.content}'")
                return concept

        # 2. If not found, create it using an LLM to define it and find its parent.
        logger.info(f"Concept '{concept_name}' not found. Attempting to create it on-the-fly.")

        existing_concept_names = sorted([c.content for c in self.concepts.values()])
        res = DefineGraphMemoryConcept.execute(
            {
                "concept_name": concept_name,
                "existing_concepts": existing_concept_names,
            },
            self.llm_manager,
            None,
        )
        definition_result = res.get("concept_definition")

        if not isinstance(definition_result, GraphMemoryConceptDefinition):
            logger.error(f"Graph-memory concept definition agent failed to return a valid concept definition for '{concept_name}'.")
            return None

        # Find the parent concept's id
        parent_concept_id = None
        parent_name = definition_result.parent_concept_name
        for id_val, concept_obj in self.concepts.items():
            if concept_obj.content.lower() == parent_name.lower():
                parent_concept_id = id_val
                break

        if not parent_concept_id:
            logger.warning(
                f"LLM suggested parent '{parent_name}' for '{concept_name}', but it was not found. Defaulting to 'Entity' as parent.")
            # Fallback to the root "Entity" concept
            for id_val, concept_obj in self.concepts.items():
                if concept_obj.content == "Entity":
                    parent_concept_id = id_val
                    break

        # Create the new concept
        new_concept = ConceptNode(
            content=concept_name,  # Use the original casing
            description=definition_result.description,
            parent_concept_id=parent_concept_id
        )
        self.ghost.add_knoxel(new_concept)
        self.graph.add_node(new_concept.id, data=new_concept, type="ConceptNode", name=new_concept.content)
        if new_concept.parent_concept_id:
            self.graph.add_edge(new_concept.id, new_concept.parent_concept_id, label=StructuralEdgeLabel.IS_A)

        logger.info(f"Successfully created new concept '{new_concept.content}' with parent '{parent_name}'.")
        return new_concept

    def _rebuild_networkx_graph(self):
        """Reconstructs the in-memory NetworkX graph from loaded Pydantic models."""
        self.graph.clear()
        logger.debug("Rebuilding NetworkX graph...")

        # Add ConceptNodes
        for concept_id, concept in self.concepts.items():
            self.graph.add_node(concept_id, data=concept, type="ConceptNode", name=concept.content)

        # Add GraphNodes (including EgoNode)
        for node_id, node in self.nodes.items():
            self.graph.add_node(node_id, data=node, type="GraphNode", name=node.content)

        # Add MemoryClusterKnoxels
        for memory_id, memory in self.memorys.items():
            self.graph.add_node(memory_id, data=memory, type="MemoryClusterKnoxel", summary=memory.content)

        # Add structural edges for Concepts (IS_A)
        for concept_id, concept in self.concepts.items():
            if concept.parent_concept_id and concept.parent_concept_id in self.concepts:
                self.graph.add_edge(concept_id, concept.parent_concept_id,
                                    label=StructuralEdgeLabel.IS_A)  # type: ignore

        # Add structural edges for GraphNodes (INSTANCE_OF)
        for node_id, node in self.nodes.items():
            if node.concept_id and node.concept_id in self.concepts:
                self.graph.add_edge(node_id, node.concept_id, label=StructuralEdgeLabel.INSTANCE_OF)  # type: ignore

        # Add factual GraphEdges
        for edge_id, edge in self.edges.items():
            if edge.source_id in self.graph and edge.target_id in self.graph:
                self.graph.add_edge(edge.source_id, edge.target_id, key=edge_id, label=edge.label, data=edge,
                                    type="GraphEdge")
            else:
                logger.warning(f"Skipping edge {edge_id} due to missing source/target node in graph during rebuild.")
        logger.debug(
            f"NetworkX graph rebuilt with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")

    def _bootstrap_core_concepts(self):
        logger.info("Bootstrapping core concepts...")
        core_concepts_definitions = [
            {"name": "Entity", "description": "A distinct and independent existent thing.", "parent": None},
            {"name": "Abstract Concept",
             "description": "An idea or notion not having a physical or concrete existence.", "parent": "Entity"},
            {"name": "Physical Object", "description": "A tangible item that can be seen and touched.",
             "parent": "Entity"},
            {"name": "Living Being", "description": "An organism that has life.", "parent": "Physical Object"},
            {"name": "Person", "description": "A human being.", "parent": "Living Being"},
            {"name": "Animal", "description": "A living organism that feeds on organic matter.",
             "parent": "Living Being"},
            {"name": "Plant",
             "description": "A living organism of the kind exemplified by trees, shrubs, herbs, grasses, ferns, and mosses.",
             "parent": "Living Being"},
            {"name": "Organization", "description": "An organized group of people with a particular purpose.",
             "parent": "Abstract Concept"},  # Or could be Entity directly
            {"name": "Company", "description": "A commercial business.", "parent": "Organization"},
            {"name": "Location", "description": "A particular place or position.", "parent": "Physical Object"},
            # Or Abstract Concept for general locations
            {"name": "Event", "description": "A thing that happens, especially one of importance.",
             "parent": "Abstract Concept"},
            {"name": "Action", "description": "The fact or process of doing something.", "parent": "Abstract Concept"},
            {"name": "Information", "description": "Facts provided or learned about something or someone.",
             "parent": "Abstract Concept"},
            {"name": "Emotion",
             "description": "A strong feeling deriving from one's circumstances, mood, or relationships with others.",
             "parent": "Abstract Concept"},
            {"name": "Property", "description": "An attribute, quality, or characteristic of something.",
             "parent": "Abstract Concept"},
            {"name": "Time", "description": "The indefinite continued progress of existence and events.",
             "parent": "Abstract Concept"},
            {"name": "Occupation", "description": "A job or profession.", "parent": "Abstract Concept"},
            {"name": "User", "description": "The human user interacting with the companion.", "parent": "Person"},
            {"name": "Companion", "description": "The AI companion as an agentive social entity.", "parent": "Entity"},
            {"name": "Agent", "description": "A non-human agent such as an assistant, bot, or internal agent.", "parent": "Entity"},
            {"name": "Group", "description": "A social collection of multiple people or agents.", "parent": "Organization"},
            {"name": "Role", "description": "A social or operational role held by a person or agent.", "parent": "Abstract Concept"},
            {"name": "Relationship", "description": "A durable interpersonal connection between parties.", "parent": "Abstract Concept"},
            {"name": "RelationshipState", "description": "The current quality or condition of a relationship.", "parent": "Relationship"},
            {"name": "Mood", "description": "A broad affective state that persists over some time.", "parent": "Emotion"},
            {"name": "Need", "description": "An explicit need, requirement, or ongoing constraint.", "parent": "Abstract Concept"},
            {"name": "Value", "description": "A principle or value that guides preference or choice.", "parent": "Abstract Concept"},
            {"name": "Preference", "description": "A favored style, choice, or option.", "parent": "Abstract Concept"},
            {"name": "Boundary", "description": "A limit, rule, or interpersonal constraint that should be respected.", "parent": "Abstract Concept"},
            {"name": "Goal", "description": "A desired future state or target outcome.", "parent": "Abstract Concept"},
            {"name": "Intention", "description": "A near-term intended course of action.", "parent": "Goal"},
            {"name": "Motivation", "description": "A driving reason or incentive shaping behavior.", "parent": "Abstract Concept"},
            {"name": "Habit", "description": "A recurring behavior pattern.", "parent": "Action"},
            {"name": "Trait", "description": "A relatively stable personal characteristic.", "parent": "Property"},
            {"name": "CommunicationStyle", "description": "A recurring interpersonal style of expression or tone.", "parent": "Property"},
            {"name": "InteractionPattern", "description": "A repeated interpersonal or conversational pattern.", "parent": "Abstract Concept"},
            {"name": "Capability", "description": "An ability the agent or a person can exercise.", "parent": "Property"},
            {"name": "Limitation", "description": "A constraint on what the agent or a person can do.", "parent": "Property"},
            {"name": "MetacognitiveAbility", "description": "An explicit ability to monitor or manage one's own internal processes.", "parent": "Capability"},
            {"name": "CognitiveProcess", "description": "An internal process such as reasoning, reflection, or planning.", "parent": "Abstract Concept"},
            {"name": "Tool", "description": "An internal or external tool that can be used to perform work.", "parent": "Physical Object"},
            {"name": "ToolCall", "description": "A specific use of a tool to perform an operation.", "parent": "Action"},
            {"name": "Codelet", "description": "A bounded internal agent or process used for cognitive work.", "parent": "Agent"},
            {"name": "MemorySubsystem", "description": "A named memory-related internal subsystem or capability.", "parent": "CognitiveProcess"},
            {"name": "Project", "description": "A coherent work effort or long-running undertaking.", "parent": "Abstract Concept"},
            {"name": "Activity", "description": "A concrete activity or practice.", "parent": "Action"},
            {"name": "Task", "description": "A specific unit of work to be carried out.", "parent": "Activity"},
            {"name": "Topic", "description": "A subject of discussion, reflection, or focus.", "parent": "Information"},
            {"name": "Technology", "description": "A technical system, tool, or framework.", "parent": "Information"},
        ]

        # Create concepts if they don't exist by name
        name_to_id_map = {c.content: id_val for id_val, c in self.concepts.items()}
        made_changes = False

        for concept_def in core_concepts_definitions:
            if concept_def["name"] not in name_to_id_map:
                logger.debug(f"Creating core concept: {concept_def['name']}")
                parent_id = name_to_id_map.get(concept_def["parent"]) if concept_def["parent"] else 0
                new_concept = ConceptNode(
                    content=concept_def["name"],
                    description=concept_def["description"],
                    parent_concept_id=parent_id
                )
                self.ghost.add_knoxel(new_concept)
                self.graph.add_node(new_concept.id, data=new_concept, type="ConceptNode", name=new_concept.content)
                name_to_id_map[new_concept.content] = new_concept.id  # Update map for subsequent parent lookups
                made_changes = True

        # Link concepts with IS_A edges after all are potentially created
        for concept_id, concept in self.concepts.items():
            if concept.parent_concept_id and concept.parent_concept_id in self.graph:
                if not self.graph.has_edge(concept_id, concept.parent_concept_id):
                    self.graph.add_edge(concept_id, concept.parent_concept_id, label=StructuralEdgeLabel.IS_A)  # type: ignore
                    made_changes = True

    # --- Knowledge Integration (The Write Path) ---
    def integrate_text(self, text_from_interaction, ai_internal_state_text):
        logger.info(f"Integrating new experience. Interaction (start): '{text_from_interaction[:100]}...'")
        # 1. Create the MemoryEpisode node
        memory = self._create_memory_memory(text_from_interaction, ai_internal_state_text)
        if not memory:
            logger.error("Failed to create memory memory. Aborting integration.")
            return None
        return self.integrate_experience(memory)

    def integrate_experience(self, memory: MemoryClusterKnoxel) -> Optional[MemoryClusterKnoxel]:
        text_from_interaction = memory.get_story_element()
        if memory.emotion_description:
            text_from_interaction += memory.emotion_description

        # 2. Extract entities from the interaction text.
        extracted_entities_model = self._extract_entities_from_text(text_from_interaction)
        if not extracted_entities_model:
            logger.warning("No entities extracted from interaction.")
            return memory

        # 3. For each entity, find or create its GraphNode, ensuring it's linked to a concept and the current memory.
        resolved_nodes_map: Dict[str, GraphNode] = {}  # name -> GraphNode
        for extracted_entity_obj in extracted_entities_model.entities:
            node = self._resolve_entity_and_concept(extracted_entity_obj, memory.id)
            if node:
                resolved_nodes_map[node.content] = node
            else:
                logger.warning(f"Could not resolve or create node for entity: {extracted_entity_obj.name}")

        # 4. Extract relationships between the identified entities.
        if resolved_nodes_map:  # Only extract relationships if we have nodes
            extracted_relationships_model = self._extract_relationships_from_text(text_from_interaction,
                                                                                  list(resolved_nodes_map.values()))
            if extracted_relationships_model:
                # 5. For each relationship, create a GraphEdge, check for contradictions, and add it.
                for rel_data in extracted_relationships_model.relationships:
                    self._resolve_and_add_edge(rel_data, resolved_nodes_map, memory.id)
            else:
                logger.info("No relationships extracted from interaction.")
        else:
            logger.info("No resolved nodes, skipping relationship extraction.")
        return memory

    def _create_memory_memory(self, text: str, state_text: str) -> Optional[MemoryClusterKnoxel]:
        # Create the new MemoryClusterKnoxel (Topical)
        memory = MemoryClusterKnoxel(
            level=100,
            cluster_type=ClusterType.Topical,
            content=text,
            included_event_ids="",
            facts_extracted=False,
            emotion_description=state_text
        )
        self.ghost.add_knoxel(memory)
        return memory

    def _extract_entities_from_text(self, text: str) -> Optional[GraphMemoryExtractedEntities]:
        inp = {"content": text}
        res = ExtractGraphMemoryEntities.execute(inp, self.llm_manager, None)
        return res["entities"]

    def _find_existing_node(self, name: str, label: Optional[str] = None, threshold: float = 0.85) -> Optional[GraphNode]:
        """Tries to find an existing node by name and semantic similarity."""
        logger.debug(f"Attempting to find existing node for name: '{name}', label: '{label}'")
        # Direct name match (case-insensitive for robustness)
        for node in self.nodes.values():
            if are_similar(node.content, name):
                for lbl in node.labels:
                    if are_similar(lbl, label):
                        return node

        query_embedding_text = f"{name} {label}" if label else name
        query_embedding = np.array(self.llm_manager.get_embedding(query_embedding_text))

        best_match: Optional[GraphNode] = None
        highest_similarity = -1.0

        for node in self.nodes.values():
            if node.embedding and len(node.embedding) == EMBEDDING_DIM:
                node_embedding_np = np.array(node.embedding)
                # Cosine similarity is 1 - cosine_distance
                similarity = 1 - cosine(query_embedding, node_embedding_np)
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = node

        if best_match and highest_similarity >= threshold:
            logger.info(f"Found potential existing node by semantic similarity: {best_match.id} ({best_match.content}) with score {highest_similarity:.2f}")
            return best_match
        else:
            logger.debug(f"No sufficiently similar node found for '{name}' (highest score: {highest_similarity:.2f}, threshold: {threshold}).")
            return None

    def _resolve_entity_and_concept(self, entity_data: GraphMemoryExtractedEntity, current_memory_id: int) -> Optional[GraphNode]:
        logger.debug(f"Resolving entity: {entity_data.name} (Label: {entity_data.label})")

        # 1. Attempt to find an existing node
        existing_node = self._find_existing_node(entity_data.name, entity_data.label)

        # 2. If a strong candidate exists, use it.
        if existing_node:
            logger.info(f"Using existing node {existing_node.id} for '{entity_data.name}'.")
            return existing_node

        # 3. If no existing node, create a new one
        logger.info(f"No definitive existing node found for '{entity_data.name}'. Creating new node.")
        new_node = GraphNode(
            content=entity_data.name,
            name=entity_data.name,
            labels=[entity_data.label] if entity_data.label else ["Unknown"],
            source_memory_id=current_memory_id,
            attributes={"original_extracted_label": entity_data.label},
            concept_id=0)

        self.ghost.add_knoxel(new_node)
        self.graph.add_node(new_node.id, data=new_node, type="GraphNode", name=new_node.content)
        logger.info(f"Created new GraphNode {new_node.id} for '{new_node.content}'.")

        # 4. Classify the new node against the Concept Graph and link it
        self._classify_and_link_entity_to_concept(new_node)

        return new_node

    def _classify_and_link_entity_to_concept(self, node_to_classify: GraphNode):
        logger.debug(f"Classifying node '{node_to_classify.content}' ({node_to_classify.id}) against Concept Graph.")
        if not self.concepts:
            logger.warning("No concepts available for classification.")
            return

        concept_names = [c.content for c in self.concepts.values()]
        # Simple heuristic: if a label matches a concept name, use that.
        for label in node_to_classify.labels:
            matching_concepts = [c for c_id, c in self.concepts.items() if c.content.lower() == label.lower()]
            if matching_concepts:
                chosen_concept = matching_concepts[0]
                logger.info(f"Directly classified '{node_to_classify.content}' as '{chosen_concept.content}' based on label.")
                node_to_classify.concept_id = chosen_concept.id
                self.graph.add_edge(node_to_classify.id, chosen_concept.id,
                                    label=StructuralEdgeLabel.INSTANCE_OF)  # type: ignore
                # Update node in self.nodes to persist concept_id
                self.nodes[node_to_classify.id] = node_to_classify
                return

        # If no direct match, use LLM for classification
        res = ClassifyGraphMemoryEntity.execute(
            {
                "entity_name": node_to_classify.content,
                "labels": node_to_classify.labels,
                "available_concepts": concept_names,
            },
            self.llm_manager,
            None,
        )
        classification_result = res.get("classification")
        if isinstance(classification_result, GraphMemoryEntityClassification) and classification_result.most_likely_concept_name:
            target_concept_name = classification_result.most_likely_concept_name
            target_concept = None
            for concept in self.concepts.values():
                if concept.content.lower() == target_concept_name.lower():
                    target_concept = concept
                    break

            if target_concept is None and classification_result.should_create_new_concept:
                target_concept = self._get_or_create_concept_on_the_fly(target_concept_name)
            elif target_concept is None:
                fallback_concept = next(
                    (concept for concept in self.concepts.values() if concept.content.lower() == "entity"),
                    None,
                )
                target_concept = fallback_concept

            if target_concept:
                node_to_classify.concept_id = target_concept.id
                self.graph.add_edge(node_to_classify.id, target_concept.id,
                                    label=StructuralEdgeLabel.INSTANCE_OF)  # type: ignore
                self.nodes[node_to_classify.id] = node_to_classify  # Persist change
                logger.info(
                    f"LLM classified '{node_to_classify.content}' as '{target_concept.content}'. Reasoning: {classification_result.reasoning}")
            else:
                logger.warning(
                    f"LLM suggested concept '{target_concept_name}' for '{node_to_classify.content}', but this concept was not found in the Concept Graph.")
        else:
            logger.warning(f"Could not classify entity '{node_to_classify.content}' using LLM or result was invalid.")

    def _extract_relationships_from_text(self, text: str, resolved_nodes: List[GraphNode]) -> Optional[
        GraphMemoryExtractedRelationships]:
        """
        [REPLACEMENT] Extracts relationships with a more flexible prompt that allows discovering new entities.
        """
        logger.debug("Extracting relationships from text with flexible prompt...")
        if not resolved_nodes:
            logger.info("No resolved nodes provided, cannot extract relationships between them.")
            return None

        entity_names = [node.content for node in resolved_nodes]
        res = ExtractGraphMemoryRelationships.execute(
            {
                "content": text,
                "known_entities": entity_names,
            },
            self.llm_manager,
            None,
        )
        extracted_data = res.get("relationships")
        if isinstance(extracted_data, GraphMemoryExtractedRelationships):
            if extracted_data.relationships:
                logger.info(f"Extracted {len(extracted_data.relationships)} potential relationships.")
                return extracted_data
            else:
                logger.info("No relationships extracted by LLM or result was invalid.")
                return None

    def _resolve_and_add_edge(self, rel_data: GraphMemoryExtractedRelationship, resolved_nodes_map: Dict[str, GraphNode],
                              current_memory_id: int):
        logger.debug(
            f"Resolving and adding edge for relationship: {rel_data.source_name} -({rel_data.label})-> {rel_data.target_name}")

        source_node = self._get_or_create_node_for_relationship(
            rel_data.source_name, resolved_nodes_map, current_memory_id
        )
        target_node = self._get_or_create_node_for_relationship(
            rel_data.target_name, resolved_nodes_map, current_memory_id
        )

        if not source_node or not target_node:
            logger.warning(
                f"Could not find source ('{rel_data.source_name}') or target ('{rel_data.target_name}') node for relationship. Skipping.")
            return

        # Contradiction/Duplicate Check
        # Find existing edges between these two nodes with a similar label
        existing_relevant_edges: List[GraphEdge] = []
        for u, v, key, data in self.graph.out_edges(source_node.id, data=True, keys=True):
            if v == target_node.id and isinstance(data.get('data'), GraphEdge):
                edge_obj: GraphEdge = data['data']
                # Simple label match for now; could use semantic similarity for labels too
                if edge_obj.label.lower() == rel_data.label.lower() and not edge_obj.invalid_at:
                    existing_relevant_edges.append(edge_obj)

        if existing_relevant_edges:
            existing_facts_texts = [f"{edge.fact_text} [edge_id={edge.id}]" for edge in existing_relevant_edges]
            res = CheckGraphMemoryContradiction.execute(
                {
                    "new_fact": rel_data.fact,
                    "existing_facts": existing_facts_texts,
                },
                self.llm_manager,
                None,
            )
            check_result = res.get("check")
            if isinstance(check_result, GraphMemoryContradictionCheck):
                if check_result.is_duplicate:
                    logger.info(f"Fact '{rel_data.fact}' is a duplicate of an existing fact. Skipping addition.")
                    # Potentially update timestamp or certainty of existing fact if needed.
                    return
                if check_result.is_contradictory:
                    logger.warning(f"Fact '{rel_data.fact}' is contradictory. Reasoning: {check_result.reasoning}")
                    # Invalidate the old edge(s)
                    # This mock assumes only one conflicting edge for simplicity
                    conflicting_edge_id_to_invalidate = check_result.conflicting_edge_id
                    if not conflicting_edge_id_to_invalidate and existing_relevant_edges:  # Fallback if LLM didn't specify
                        conflicting_edge_id_to_invalidate = existing_relevant_edges[0].id

                    if conflicting_edge_id_to_invalidate and conflicting_edge_id_to_invalidate in self.edges:
                        old_edge = self.edges[conflicting_edge_id_to_invalidate]
                        old_edge.invalid_at = get_causal_datetime()
                        self.edges[old_edge.id] = old_edge  # Update in-memory store
                        # Update in graph (NetworkX stores dicts, so update data attribute)
                        if self.graph.has_edge(old_edge.source_id, old_edge.target_id, key=old_edge.id):
                            self.graph.edges[old_edge.source_id, old_edge.target_id, old_edge.id][
                                'data'] = old_edge
                        logger.info(f"Invalidated old edge {old_edge.id} due to contradiction.")
                    else:
                        logger.warning(
                            f"Contradiction detected, but could not identify or find specific edge '{conflicting_edge_id_to_invalidate}' to invalidate.")
            else:
                logger.warning("Contradiction check agent failed to return a valid contradiction result.")

        # Create and add the new edge
        new_edge = GraphEdge(
            content=rel_data.label,
            source_id=source_node.id,
            target_id=target_node.id,
            label=rel_data.label,
            fact_text=rel_data.fact,
            source_memory_id=current_memory_id)

        source_name = self.nodes.get(new_edge.source_id).content
        target_name = self.nodes.get(new_edge.target_id).content

        self.ghost.add_knoxel(new_edge)
        self.graph.add_edge(source_node.id, target_node.id, key=new_edge.id, label=new_edge.label, data=new_edge, type="GraphEdge")
        logger.info(f"Added new GraphEdge {new_edge.id}: '{new_edge.fact_text}'.")

    def _extract_keywords(self, text: str) -> Set[str]:
        """
        [REPLACEMENT - NO SPACY]
        Extracts keywords using rule-based heuristics:
        1. Finds all-caps words (e.g., "AWS", "HIPAA").
        2. Finds capitalized words (potential proper nouns).
        3. Takes all other words and filters them against a stop-word list.
        """
        keywords = set()

        # 1. Heuristic for acronyms and important capitalized terms (case-sensitive)
        # Find all-caps words of 2 or more letters
        acronyms = re.findall(r'\b[A-Z]{2,}\b', text)
        for acronym in acronyms:
            keywords.add(acronym.lower())

        # Find capitalized words (potential proper nouns)
        capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', text)
        for cap_word in capitalized_words:
            # Avoid adding words that are only capitalized because they start a sentence.
            # This is a simple check; a more complex one isn't needed for keyword soup.
            if cap_word.lower() not in STOP_WORDS:
                keywords.add(cap_word.lower())

        # 2. General keyword extraction (case-insensitive)
        # Remove punctuation and normalize
        normalized_text = re.sub(r'[^\w\s]', '', text.lower())
        tokens = normalized_text.split()

        for token in tokens:
            if token.strip() and token not in STOP_WORDS and not token.isdigit():
                keywords.add(token)

        logger.debug(f"Extracted keywords (spaCy-free): {keywords}")
        return keywords

    def _item_search_text(self, item: Union[GraphNode, GraphEdge, MemoryClusterKnoxel]) -> str:
        if isinstance(item, GraphNode):
            return (
                f"{item.content}\n"
                f"labels: {', '.join(item.labels)}\n"
                f"emotion_state: {item.get_emotional_state_at_time()}\n"
                f"emotion_appraisal: {item.get_emotional_appraisal()}\n"
                f"emotion_delta: {item.get_emotional_delta()}"
            )
        if isinstance(item, GraphEdge):
            source_name = self.nodes[item.source_id].content
            target_name = self.nodes[item.target_id].content
            return (
                f"{item.fact_text}\n"
                f"relation: {source_name} {item.label} {target_name}\n"
                f"emotion_state: {item.get_emotional_state_at_time()}\n"
                f"emotion_appraisal: {item.get_emotional_appraisal()}\n"
                f"emotion_delta: {item.get_emotional_delta()}"
            )
        return (
            f"{item.content}\n"
            f"emotion_state: {item.get_emotional_state_at_time()}\n"
            f"emotion_appraisal: {item.get_emotional_appraisal()}\n"
            f"emotion_delta: {item.get_emotional_delta()}"
        )

    def _tokenize_for_lexical_search(self, text: str) -> List[str]:
        normalized_text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = normalized_text.split()
        return [token for token in tokens if token and token not in STOP_WORDS and not token.isdigit()]

    def _search_by_keywords(
        self,
        question: str,
        searchable_items: List[Union[GraphNode, GraphEdge, MemoryClusterKnoxel]],
        top_k: int,
        min_term_matches: int,
        min_score: float,
    ) -> List[Tuple[float, Union[GraphNode, GraphEdge, MemoryClusterKnoxel]]]:
        query_tokens = self._tokenize_for_lexical_search(question)
        if not query_tokens:
            return []

        corpus_tokens = [self._tokenize_for_lexical_search(self._item_search_text(item)) for item in searchable_items]
        bm25 = BM25Okapi(corpus_tokens)
        bm25_scores = bm25.get_scores(query_tokens)
        query_token_set = set(query_tokens)

        scored_items: List[Tuple[float, Union[GraphNode, GraphEdge, MemoryClusterKnoxel]]] = []
        for idx, item in enumerate(searchable_items):
            overlap = len(query_token_set.intersection(set(corpus_tokens[idx])))
            score = float(bm25_scores[idx]) + (0.1 * overlap)
            if overlap >= min_term_matches and score >= min_score:
                scored_items.append((score, item))

        scored_items.sort(key=lambda x: x[0], reverse=True)
        logger.info(f"Keyword search returned {len(scored_items)} matches above score {min_score}.")
        return scored_items[:top_k]

    def _rephrase_question(self, question: str, variant_count: int, llm_temperature: float, llm_max_tokens: int) -> List[str]:
        """Uses LLM to generate alternative phrasings of a question."""
        logger.info(f"Rephrasing question: '{question}'")
        _ = llm_temperature
        _ = llm_max_tokens
        res = RephraseGraphMemoryQuestion.execute(
            {
                "question": question,
                "variant_count": variant_count,
            },
            self.llm_manager,
            None,
        )
        result = res.get("rephrased")
        if isinstance(result, GraphMemoryRephrasedQuestions) and result.questions:
            variants = result.questions[:variant_count]
            logger.info(f"Generated rephrasings: {variants}")
            return variants
        return []

    def _prune_and_rerank_context(self, question: str,
                                  candidate_items: List[Union[GraphNode, GraphEdge, MemoryClusterKnoxel]],
                                  max_context_items: int,
                                  llm_temperature: float,
                                  llm_max_tokens: int) -> List[
        Union[GraphNode, GraphEdge, MemoryClusterKnoxel]]:
        """Uses an LLM to prune a list of candidate items down to the most relevant ones."""
        if not candidate_items:
            return []
        logger.info(f"Pruning {len(candidate_items)} candidate items for relevance to question: '{question}'")

        # Create a mapping from a unique identifier (id) to the item
        item_map = {item.id: item for item in candidate_items[:max_context_items]}

        context_block = []
        for id_val, item in item_map.items():
            text = ""
            if isinstance(item, GraphNode):
                text = f"Entity: '{item.content}' (Type: {', '.join(item.labels)})"
            elif isinstance(item, GraphEdge):
                text = f"Fact: '{item.fact_text}'"
            elif isinstance(item, MemoryClusterKnoxel):
                text = f"Memory Summary: '{item.content}'"
            context_block.append(f"Identifier: {id_val}\nContent: {text}\n---")

        _ = llm_temperature
        _ = llm_max_tokens
        nl = '\n'
        res = PruneGraphMemoryContext.execute(
            {
                "question": question,
                "context_block": nl.join(context_block),
            },
            self.llm_manager,
            None,
        )
        result = res.get("relevant")
        if isinstance(result, GraphMemoryRelevantItems):
            pruned_ids = set(result.relevant_item_identifiers)
            pruned_items = [item_map[id_val] for id_val in pruned_ids if id_val in item_map]
            logger.info(f"Pruned context from {len(candidate_items)} to {len(pruned_items)} items.")
            return pruned_items

        # Fallback: if LLM pruning fails, return the original candidates to not lose all context
        logger.warning("LLM pruning failed. Returning all candidate items.")
        return candidate_items

    def _find_relevant_items(self, question: str, top_k: int = 10, similarity_threshold: float = 0.4) -> List[
        Tuple[float, Union[GraphNode, GraphEdge, MemoryClusterKnoxel]]]:
        """
        [NEW - Previously Missing]
        Performs a pure semantic search across all memory items (nodes, edges, memorys).
        It retrieves items whose embeddings are most similar to the question's embedding.
        """
        logger.debug(f"Finding relevant items via semantic search for: '{question}'")

        question_embedding = np.array(self.llm_manager.get_embedding(question))
        scored_items: List[Tuple[float, Union[GraphNode, GraphEdge, MemoryClusterKnoxel]]] = []

        # Combine all searchable items into one list
        all_items = list(self.nodes.values()) + list(self.edges.values()) + list(self.memorys.values())

        for item in all_items:
            # Skip edges that have been invalidated
            if isinstance(item, GraphEdge) and item.invalid_at:
                continue

            if item.embedding and len(item.embedding) == EMBEDDING_DIM:
                item_embedding_np = np.array(item.embedding)

                # Cosine similarity is 1 - cosine_distance
                similarity = 1 - cosine(question_embedding, item_embedding_np)

                if similarity >= similarity_threshold:
                    scored_items.append((similarity, item))

        # Sort by similarity (highest first) and take the top k results
        scored_items.sort(key=lambda x: x[0], reverse=True)
        logger.info(
            f"Found {len(scored_items)} semantically relevant items above threshold {similarity_threshold}. Returning top {top_k}.")
        return scored_items[:top_k]

    def _fuse_hybrid_results(
        self,
        semantic_results: List[Tuple[float, Union[GraphNode, GraphEdge, MemoryClusterKnoxel]]],
        keyword_results: List[Tuple[float, Union[GraphNode, GraphEdge, MemoryClusterKnoxel]]],
        top_k: int,
        semantic_weight: float,
        keyword_weight: float,
    ) -> List[Union[GraphNode, GraphEdge, MemoryClusterKnoxel]]:
        k_rrf = 60.0
        fused_scores: Dict[int, float] = {}
        item_map: Dict[int, Union[GraphNode, GraphEdge, MemoryClusterKnoxel]] = {}

        for rank, (_, item) in enumerate(semantic_results):
            fused_scores[item.id] = fused_scores.get(item.id, 0.0) + (semantic_weight / (k_rrf + rank + 1))
            item_map[item.id] = item

        for rank, (_, item) in enumerate(keyword_results):
            fused_scores[item.id] = fused_scores.get(item.id, 0.0) + (keyword_weight / (k_rrf + rank + 1))
            item_map[item.id] = item

        ranked_ids = sorted(fused_scores.keys(), key=lambda item_id: fused_scores[item_id], reverse=True)
        return [item_map[item_id] for item_id in ranked_ids[:top_k]]

    def _rerank_candidates_dummy(
        self,
        question: str,
        candidates: List[Union[GraphNode, GraphEdge, MemoryClusterKnoxel]],
        params: MemoryQueryParameters,
    ) -> List[Union[GraphNode, GraphEdge, MemoryClusterKnoxel]]:
        """
        Placeholder reranker hook.
        This method intentionally performs no reranking yet.
        """
        _ = question
        _ = params
        return candidates

    def _trace_items_to_memorys(self, items: List[Union[GraphNode, GraphEdge, MemoryClusterKnoxel]]) -> Dict[int, List[Union[GraphNode, GraphEdge]]]:
        """
        [NEW - Previously Missing]
        Takes a list of relevant items (nodes, edges, or memorys) and groups them by the
        MemoryEpisode in which they were learned. This is a key step for building a
        narrative context for recall.

        If a MemoryClusterKnoxel itself is in the list of items, its id is added as a key
        to the map, ensuring its summary is included in the final narrative context even if no
        specific facts from that memory were individually returned by the pruner.

        Args:
            items: A list of relevant GraphNode, GraphEdge, or MemoryClusterKnoxel objects.

        Returns:
            A dictionary where keys are memory ids and values are lists of the
            facts (GraphNodes or GraphEdges) learned in that memory.
        """
        logger.debug(f"Tracing {len(items)} items to their source memorys.")
        memorys_map: Dict[int, List[Union[GraphNode, GraphEdge]]] = defaultdict(list)

        for item in items:
            if isinstance(item, MemoryClusterKnoxel):
                # If an memory itself is deemed relevant, ensure its id is a key in the map.
                # This guarantees it will be included in the narrative construction step.
                # We don't append the memory to the list of facts; its presence as a key is enough.
                _ = memorys_map[item.id]

            elif isinstance(item, (GraphNode, GraphEdge)):
                # For a specific fact (node or edge), find its source memory.
                if item.source_memory_id in self.memorys:
                    memorys_map[item.source_memory_id].append(item)
                else:
                    logger.warning(f"Fact id '{item.id}' points to missing source_memory_id '{item.source_memory_id}'.")

        logger.info(f"Relevant items were traced to {len(memorys_map)} unique memorys.")
        return memorys_map

    def answer_question(self, question: str, query: MemoryQueryParameters | None = None) -> MemoryQueryResult:
        """
        Answers a question using configurable multi-step retrieval.
        """
        if query is None:
            query = MemoryQueryParameters()

        logger.info(f"Answering question with multi-step retrieval: '{question}'")

        all_items: List[Union[GraphNode, GraphEdge, MemoryClusterKnoxel]] = (
            list(self.nodes.values()) + list(self.edges.values()) + list(self.memorys.values())
        )
        remaining_agent_calls = query.max_agent_calls

        semantic_results = self._find_relevant_items(
            question,
            top_k=query.semantic_top_k,
            similarity_threshold=query.semantic_similarity_threshold,
        )
        keyword_results = self._search_by_keywords(
            question,
            searchable_items=all_items,
            top_k=query.keyword_top_k,
            min_term_matches=query.keyword_min_term_matches,
            min_score=query.keyword_min_score,
        )
        fused_items = self._fuse_hybrid_results(
            semantic_results,
            keyword_results,
            top_k=query.hybrid_top_k,
            semantic_weight=query.hybrid_semantic_weight,
            keyword_weight=query.hybrid_keyword_weight,
        )

        candidate_items_map: Dict[int, Union[GraphNode, GraphEdge, MemoryClusterKnoxel]] = {
            item.id: item for item in fused_items
        }

        retry_round = 0
        while len(candidate_items_map) < query.min_candidate_count and retry_round < query.max_retry_rounds and remaining_agent_calls > 0:
            retry_round += 1
            remaining_agent_calls -= 1
            logger.info(f"Retrieval retry round {retry_round}/{query.max_retry_rounds}: running query expansion.")

            rephrased_qs = self._rephrase_question(
                question,
                variant_count=query.max_query_expansions,
                llm_temperature=query.llm_temperature,
                llm_max_tokens=query.llm_max_tokens,
            )
            for q_variant in rephrased_qs:
                variant_semantic = self._find_relevant_items(
                    q_variant,
                    top_k=max(2, query.semantic_top_k // 2),
                    similarity_threshold=query.semantic_similarity_threshold,
                )
                variant_keyword = self._search_by_keywords(
                    q_variant,
                    searchable_items=all_items,
                    top_k=max(4, query.keyword_top_k // 2),
                    min_term_matches=query.keyword_min_term_matches,
                    min_score=query.keyword_min_score,
                )
                fused_variant = self._fuse_hybrid_results(
                    variant_semantic,
                    variant_keyword,
                    top_k=query.hybrid_top_k,
                    semantic_weight=query.hybrid_semantic_weight,
                    keyword_weight=query.hybrid_keyword_weight,
                )
                for item in fused_variant:
                    candidate_items_map[item.id] = item

        candidate_items = list(candidate_items_map.values())[:query.hybrid_top_k]
        if not candidate_items:
            return MemoryQueryResult(
                content="Unknown: this information is not present in memory yet.",
                source_ids=[],
            )

        pruned_items = candidate_items[:query.max_context_items]
        if len(candidate_items) > query.prune_trigger_count and remaining_agent_calls > 0:
            remaining_agent_calls -= 1
            pruned_items = self._prune_and_rerank_context(
                question,
                candidate_items,
                max_context_items=query.max_context_items,
                llm_temperature=query.llm_temperature,
                llm_max_tokens=query.llm_max_tokens,
            )

        if not pruned_items:
            return MemoryQueryResult(
                content="Unknown: relevant evidence was retrieved, but none directly supports an answer.",
                source_ids=[],
            )

        pruned_items = self._rerank_candidates_dummy(question, pruned_items, query)

        memorys_map = self._trace_items_to_memorys(pruned_items)
        if not memorys_map:
            return MemoryQueryResult(
                content="Unknown: retrieved items are not linked to a valid source memory.",
                source_ids=[],
            )

        narrative_context = self._construct_narrative_context(memorys_map, TIMESTAMP_FORMAT)
        if not narrative_context:
            return MemoryQueryResult(
                content="Unknown: evidence exists, but no narrative context could be constructed.",
                source_ids=sorted({item.id for item in pruned_items}),
            )

        if remaining_agent_calls <= 0:
            return MemoryQueryResult(
                content="Unknown: agent call budget was exhausted before final synthesis.",
                source_ids=sorted({item.id for item in pruned_items}),
            )

        final_answer = self._synthesize_final_answer(
            question,
            narrative_context,
            llm_temperature=query.llm_temperature,
            llm_max_tokens=query.llm_max_tokens,
            direct_answer_weight=query.direct_answer_weight,
            regularities_weight=query.regularities_weight,
            evidence_weight=query.evidence_weight,
            unknowns_weight=query.unknowns_weight,
            target_word_count=query.target_word_count,
        )
        logger.info(f"Generated answer: '{final_answer[:100]}...'")
        return MemoryQueryResult(content=final_answer, source_ids=sorted({item.id for item in pruned_items}))

    def _find_relevant_facts(self, question: str, top_k: int = 5, similarity_threshold: float = 0.7) -> List[
        Tuple[float, Union[GraphNode, GraphEdge]]]:
        logger.debug(f"Finding relevant facts for question: '{question}'")
        if not EMBEDDING_DIM:
            logger.warning("EMBEDDING_DIM not set, cannot perform semantic search.")
            return []
        question_embedding = np.array(self.llm_manager.get_embedding(question))
        scored_items: List[Tuple[float, Union[GraphNode, GraphEdge]]] = []

        # Search GraphNodes
        for node in self.nodes.values():
            if node.embedding and len(node.embedding) == EMBEDDING_DIM:
                node_embedding_np = np.array(node.embedding)
                similarity = 1 - cosine(question_embedding, node_embedding_np)
                if similarity >= similarity_threshold:
                    scored_items.append((similarity, node))

        # Search GraphEdges (only those not invalidated)
        for edge in self.edges.values():
            if not edge.invalid_at and edge.embedding and len(edge.embedding) == EMBEDDING_DIM:
                edge_embedding_np = np.array(edge.embedding)
                similarity = 1 - cosine(question_embedding, edge_embedding_np)
                if similarity >= similarity_threshold:
                    scored_items.append((similarity, edge))

        # Sort by similarity (highest first) and take top_k
        scored_items.sort(key=lambda x: x[0], reverse=True)
        logger.info(f"Found {len(scored_items)} relevant facts above threshold. Returning top {top_k}.")
        return scored_items[:top_k]

    def _trace_facts_to_memorys(self, facts: List[Union[GraphNode, GraphEdge]]) -> Dict[
        int, List[Union[GraphNode, GraphEdge]]]:
        logger.debug(f"Tracing {len(facts)} facts to their source memorys.")
        memorys_map: Dict[int, List[Union[GraphNode, GraphEdge]]] = defaultdict(list)
        for fact in facts:
            if fact.source_memory_id in self.memorys:
                memorys_map[fact.source_memory_id].append(fact)
            else:
                logger.warning(f"Fact '{fact.id}' has no valid source_memory_id or memory not found.")
        logger.info(f"Facts traced to {len(memorys_map)} unique memorys.")
        return memorys_map

    def _construct_narrative_context(self, memorys_map: Dict[int, List[Union[GraphNode, GraphEdge]]],
                                     date_format: str) -> str:
        logger.debug("Constructing narrative context from memorys map.")
        context_parts = []
        # Sort memorys by timestamp, most recent first, for a more natural recall flow
        sorted_memory_ids = sorted(
            memorys_map.keys(),
            key=lambda id_val: self.memorys[id_val].timestamp_world_begin,
            reverse=True
        )

        for memory_id in sorted_memory_ids:
            memory: MemoryClusterKnoxel = self.memorys.get(memory_id)
            facts_in_memory = memorys_map[memory_id]

            if not memory:
                logger.warning(f"Episode {memory_id} not found while constructing narrative. Skipping.")
                continue

            date_str = memory.timestamp_world_begin.strftime(date_format)
            topic = memory.content
            #emotion = memory.internal_state.emotional_label
            #cognitive_process = memory.internal_state.cognitive_process  # e.g. "learning new information"

            learned_items_str_parts = []
            for fact in facts_in_memory:
                if isinstance(fact, GraphEdge):
                    source_name = self.nodes.get(fact.source_id).content
                    target_name = self.nodes.get(fact.target_id).content
                    learned_items_str_parts.append(f"that {fact.fact_text} (connecting {source_name} to {target_name})")
                elif isinstance(fact, GraphNode):
                    learned_items_str_parts.append(f"about '{fact.content}' (which is a {', '.join(fact.labels)})")

            if not learned_items_str_parts:
                learned_summary = "something I was reflecting on"
            else:
                learned_summary = ", and ".join(learned_items_str_parts)

            narrative = (
                f"On {date_str}, there is a memory related to '{topic}'. "
                #f"During that time, I was primarily {cognitive_process} and I recall feeling {emotion}. "
                f"{learned_summary}."
            )
            context_parts.append(narrative)

        if not context_parts:
            logger.info("No narrative parts constructed.")
            return ""

        full_narrative = "\n\n".join(context_parts)
        logger.info(
            f"Narrative context constructed (length: {len(full_narrative)} chars). Preview: {full_narrative[:200]}...")
        return full_narrative

    def _synthesize_final_answer(
        self,
        question: str,
        narrative_context: str,
        llm_temperature: float,
        llm_max_tokens: int,
        direct_answer_weight: float,
        regularities_weight: float,
        evidence_weight: float,
        unknowns_weight: float,
        target_word_count: int,
    ) -> str:
        logger.debug("Synthesizing final answer using LLM with narrative context.")
        _ = llm_temperature
        _ = llm_max_tokens
        res = SynthesizeGraphMemoryAnswer.execute(
            {
                "question": question,
                "narrative_context": narrative_context,
                "body_and_situation": self.ghost.build_body_and_situation_addendum(),
                "direct_answer_weight": direct_answer_weight,
                "regularities_weight": regularities_weight,
                "evidence_weight": evidence_weight,
                "unknowns_weight": unknowns_weight,
                "target_word_count": target_word_count,
            },
            self.llm_manager,
            None,
        )
        answer = res.get("answer_body", "").strip()
        return "1) Direct Answer\n" + answer
