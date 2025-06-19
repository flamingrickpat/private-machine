import json
import logging
import sqlite3
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from pydantic import BaseModel, Field, field_validator
from scipy.spatial.distance import cosine

# Assuming pm_lida.py is in the same directory or accessible in the python path
# We'll import the necessary components from your existing file.
from pm_lida import LlmPreset, LlmManagerProxy, CommonCompSettings

logger = logging.getLogger(__name__)


# --- Pydantic Models for Graph Elements ---

class GraphNode(BaseModel):
    """Represents a single entity (a node) in the knowledge graph."""
    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(description="The primary name or identifier for the entity.")
    labels: List[str] = Field(default_factory=list,
                              description="A list of labels for categorization (e.g., ['Person', 'Employee']).")
    summary: str = Field(default="",
                         description="An LLM-generated summary of the node and its most important relationships.")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    attributes: Dict[str, Any] = Field(default_factory=dict,
                                       description="A dictionary for storing additional structured data about the node.")

    # In-memory only, not persisted directly in the node's JSON
    embedding: List[float] = Field(default_factory=list, exclude=True)

    def __hash__(self):
        return hash(self.uuid)

    def __eq__(self, other):
        if not isinstance(other, GraphNode):
            return NotImplemented
        return self.uuid == other.uuid


class GraphEdge(BaseModel):
    """Represents a single relationship (an edge) between two nodes in the knowledge graph."""
    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_uuid: str = Field(description="The UUID of the source node.")
    target_uuid: str = Field(description="The UUID of the target node.")
    label: str = Field(description="The type of relationship (e.g., 'WORKS_AT', 'IS_FRIENDS_WITH').")
    fact_text: str = Field(description="The natural language sentence describing the fact of this relationship.")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    valid_at: Optional[datetime] = Field(default=None, description="The timestamp when the fact became true.")
    invalid_at: Optional[datetime] = Field(default=None, description="The timestamp when the fact ceased to be true.")
    attributes: Dict[str, Any] = Field(default_factory=dict,
                                       description="A dictionary for storing additional structured data about the edge.")

    # In-memory only, not persisted directly
    embedding: List[float] = Field(default_factory=list, exclude=True)

    @field_validator('valid_at', 'invalid_at', 'created_at', mode='before')
    def ensure_datetime(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v)
        return v

    def __hash__(self):
        return hash(self.uuid)

    def __eq__(self, other):
        if not isinstance(other, GraphEdge):
            return NotImplemented
        return self.uuid == other.uuid


# --- Pydantic Models for LLM Operations ---

class ExtractedEntity(BaseModel):
    name: str = Field(description="The name of the extracted entity.")
    label: str = Field(description="A suitable label for the entity (e.g., 'Person', 'Company', 'Concept').")


class ExtractedEntities(BaseModel):
    entities: List[ExtractedEntity]


class ExtractedRelationship(BaseModel):
    source_name: str = Field(description="The name of the source entity.")
    target_name: str = Field(description="The name of the target entity.")
    label: str = Field(
        description="A concise, descriptive label for the relationship in SCREAMING_SNAKE_CASE (e.g., 'WORKS_FOR', 'IS_LOCATED_IN').")
    fact: str = Field(description="The natural language sentence from the text that states the relationship.")


class ExtractedRelationships(BaseModel):
    relationships: List[ExtractedRelationship]


class Confirmation(BaseModel):
    is_same: bool = Field(
        description="True if the two items refer to the same real-world entity or concept, False otherwise.")
    reasoning: str = Field(description="A brief explanation for the decision.")


class ContradictionCheck(BaseModel):
    is_contradictory: bool = Field(description="True if the new fact contradicts the existing fact.")
    is_duplicate: bool = Field(description="True if the new fact is a duplicate of the existing fact.")
    reasoning: str = Field(description="Brief explanation for the decisions.")


# --- The Main Graph Memory Manager Class ---

class GraphMemoryManager:
    """Manages a knowledge graph using NetworkX and persists it to SQLite."""

    def __init__(self, llm_manager: LlmManagerProxy, db_path: str):
        self.llm_manager = llm_manager
        self.db_path = db_path
        self.graph = nx.MultiDiGraph()
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, GraphEdge] = {}
        self.load_graph()

    # --- Persistence ---
    def save_graph(self):
        """Saves the current in-memory graph to the SQLite database."""
        logger.info(f"Saving knowledge graph to {self.db_path}...")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Node Table
            cursor.execute("DROP TABLE IF EXISTS graph_nodes")
            cursor.execute("CREATE TABLE graph_nodes (uuid TEXT PRIMARY KEY, data TEXT)")
            node_data = [
                (node.uuid, node.model_dump_json()) for node in self.nodes.values()
            ]
            if node_data:
                cursor.executemany("INSERT OR REPLACE INTO graph_nodes (uuid, data) VALUES (?, ?)", node_data)

            # Edge Table
            cursor.execute("DROP TABLE IF EXISTS graph_edges")
            cursor.execute("CREATE TABLE graph_edges (uuid TEXT PRIMARY KEY, data TEXT)")
            edge_data = [
                (edge.uuid, edge.model_dump_json()) for edge in self.edges.values()
            ]
            if edge_data:
                cursor.executemany("INSERT OR REPLACE INTO graph_edges (uuid, data) VALUES (?, ?)", edge_data)
            conn.commit()
        logger.info(f"Graph saved. {len(self.nodes)} nodes, {len(self.edges)} edges.")

    def load_graph(self):
        """Loads the graph from the SQLite database into memory."""
        logger.info(f"Loading knowledge graph from {self.db_path}...")
        self.graph = nx.MultiDiGraph()
        self.nodes = {}
        self.edges = {}

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Load Nodes
                cursor.execute("SELECT data FROM graph_nodes")
                for row in cursor.fetchall():
                    node = GraphNode.model_validate_json(row[0])
                    self.nodes[node.uuid] = node
                    self.graph.add_node(node.uuid, **node.model_dump())

                # Load Edges
                cursor.execute("SELECT data FROM graph_edges")
                for row in cursor.fetchall():
                    edge = GraphEdge.model_validate_json(row[0])
                    # Only add edge if both nodes exist (data integrity)
                    if edge.source_uuid in self.nodes and edge.target_uuid in self.nodes:
                        self.edges[edge.uuid] = edge
                        self.graph.add_edge(
                            edge.source_uuid,
                            edge.target_uuid,
                            key=edge.uuid,
                            **edge.model_dump()
                        )
        except sqlite3.OperationalError:
            logger.warning("Graph database tables not found. Starting with an empty graph.")
            # This is expected on the first run, so we just continue.

        # Generate embeddings for loaded items if they are missing
        self._ensure_embeddings()
        logger.info(f"Graph loaded. {len(self.nodes)} nodes, {len(self.edges)} edges.")

    def _ensure_embeddings(self):
        """Ensures all loaded nodes and edges have embeddings."""
        for node in self.nodes.values():
            if not node.embedding and node.name:
                node.embedding = self.llm_manager.get_embedding(node.name)
        for edge in self.edges.values():
            if not edge.embedding and edge.fact_text:
                edge.embedding = self.llm_manager.get_embedding(edge.fact_text)

    # --- Knowledge Integration ---
    def integrate_text(self, text: str, group_id: str):
        """
        Main entry point to extract and integrate knowledge from a block of text.
        This process is incremental and designed to be called during idle time.
        """
        logger.info("Starting knowledge integration from text.")

        # 1. Extract potential entities from the text
        extracted_entities = self._extract_entities(text)
        if not extracted_entities:
            logger.info("No entities extracted from text.")
            return

        # 2. Resolve entities (deduplicate against existing nodes)
        resolved_nodes: Dict[str, GraphNode] = {}  # name -> Node
        for entity in extracted_entities:
            node = self._resolve_entity(entity.name, entity.label, group_id)
            resolved_nodes[node.name] = node

        logger.info(f"Resolved {len(resolved_nodes)} unique entities.")

        # 3. Extract potential relationships between the resolved entities
        extracted_relationships = self._extract_relationships(text, list(resolved_nodes.values()))
        if not extracted_relationships:
            logger.info("No relationships extracted from text.")
            return

        # 4. Resolve and add relationships to the graph
        for rel in extracted_relationships:
            source_node = resolved_nodes.get(rel.source_name)
            target_node = resolved_nodes.get(rel.target_name)
            if source_node and target_node:
                self._resolve_and_add_edge(source_node, target_node, rel.label, rel.fact)

        logger.info("Finished knowledge integration cycle.")
        self.save_graph()

    def _extract_entities(self, text: str) -> List[ExtractedEntity]:
        """Uses LLM to extract entities and their labels from text."""
        prompt = [
            ("system",
             "You are an expert at identifying and classifying named entities in a given text. Extract all people, organizations, locations, and key concepts."),
            ("user", f"Extract all entities from the following text:\n\n---\n{text}\n---")
        ]
        settings = CommonCompSettings(temperature=0.0, max_tokens=1024)
        _, calls = self.llm_manager.completion_tool(LlmPreset.Default, prompt, settings, tools=[ExtractedEntities])

        if calls and isinstance(calls[0], ExtractedEntities):
            return calls[0].entities
        return []

    def _resolve_entity(self, name: str, label: str, group_id: str) -> GraphNode:
        """Finds or creates a node, ensuring no duplicates."""
        # Simple exact match first
        for node in self.nodes.values():
            if node.name.lower() == name.lower():
                logger.debug(f"Resolved entity '{name}' to existing node '{node.name}' by exact match.")
                return node

        # Semantic search for potential duplicates
        if not self.nodes:  # First node ever
            return self._create_new_node(name, label, group_id)

        name_embedding = self.llm_manager.get_embedding(name)

        # Find top 3 most similar nodes
        candidates = sorted(
            self.nodes.values(),
            key=lambda node: cosine(name_embedding, node.embedding) if node.embedding else 1.0
        )[:3]

        for candidate in candidates:
            similarity = 1 - cosine(name_embedding, candidate.embedding)
            if similarity > 0.90:  # High threshold for consideration
                prompt = [
                    ("system",
                     "You are an expert in entity resolution. Determine if two names refer to the same real-world entity."),
                    ("user",
                     f"Does the name '{name}' refer to the same entity as '{candidate.name}'? Consider that they might be aliases, misspellings, or full vs. partial names.")
                ]
                settings = CommonCompSettings(temperature=0.0)
                _, calls = self.llm_manager.completion_tool(LlmPreset.Default, prompt, settings, tools=[Confirmation])

                if calls and calls[0].is_same:
                    logger.info(f"Resolved '{name}' to existing node '{candidate.name}' via LLM confirmation.")
                    return candidate

        # If no duplicates found, create a new node
        return self._create_new_node(name, label, group_id)

    def _create_new_node(self, name: str, label: str, group_id: str) -> GraphNode:
        logger.info(f"Creating new graph node for entity: '{name}'")
        new_node = GraphNode(name=name, labels=[label], summary=f"An entity named '{name}' of type '{label}'.")
        new_node.embedding = self.llm_manager.get_embedding(new_node.name)

        self.nodes[new_node.uuid] = new_node
        self.graph.add_node(new_node.uuid, **new_node.model_dump())
        return new_node

    def _extract_relationships(self, text: str, nodes_in_text: List[GraphNode]) -> List[ExtractedRelationship]:
        """Uses LLM to extract relationships between a given set of resolved nodes."""
        if len(nodes_in_text) < 2:
            return []

        node_context = "\n".join(
            [f"- {node.name} (type: {node.labels[0] if node.labels else 'Unknown'})" for node in nodes_in_text])

        prompt = [
            ("system",
             "You are an expert at extracting factual relationships between entities from text. You create a knowledge graph edge for each fact."),
            ("user",
             f"Given the following text and the list of entities present in it, extract all relationships between these entities. Express each relationship as a directed edge from a source entity to a target entity.\n\nText:\n---\n{text}\n---\n\nEntities Present:\n{node_context}")
        ]
        settings = CommonCompSettings(temperature=0.0, max_tokens=2048)
        _, calls = self.llm_manager.completion_tool(LlmPreset.Default, prompt, settings, tools=[ExtractedRelationships])

        if calls and isinstance(calls[0], ExtractedRelationships):
            return calls[0].relationships
        return []

    def _resolve_and_add_edge(self, source: GraphNode, target: GraphNode, label: str, fact_text: str):
        """Checks for duplicate/contradictory edges before adding a new one."""
        new_edge_embedding = self.llm_manager.get_embedding(fact_text)

        # Get existing edges between these two nodes
        existing_edges = []
        if self.graph.has_edge(source.uuid, target.uuid):
            for edge_uuid in self.graph[source.uuid][target.uuid]:
                existing_edges.append(self.edges[edge_uuid])

        for old_edge in existing_edges:
            # Skip if embeddings are missing
            if not old_edge.embedding:
                continue

            similarity = 1 - cosine(new_edge_embedding, old_edge.embedding)
            if similarity > 0.95:  # High threshold for considering a check
                prompt = [
                    ("system", "You are an expert in identifying duplicate and contradictory information."),
                    ("user",
                     f"Analyze the following two facts.\nFact 1: '{old_edge.fact_text}'\nFact 2: '{fact_text}'\n\nIs Fact 2 a duplicate of Fact 1? Does Fact 2 contradict Fact 1?")
                ]
                settings = CommonCompSettings(temperature=0.0)
                _, calls = self.llm_manager.completion_tool(LlmPreset.Default, prompt, settings,
                                                            tools=[ContradictionCheck])

                if calls and calls[0].is_duplicate:
                    logger.info(f"New fact is a duplicate of existing edge {old_edge.uuid}. Ignoring.")
                    return  # It's a duplicate, do nothing

                if calls and calls[0].is_contradictory:
                    logger.info(f"New fact contradicts existing edge {old_edge.uuid}. Invalidating old edge.")
                    old_edge.invalid_at = datetime.now(timezone.utc)
                    # No break, continue checking other edges in case of multiple contradictions

        # If we get here, it's not a duplicate. Add the new edge.
        logger.info(f"Adding new edge: '{source.name}' -> '{label}' -> '{target.name}'")
        new_edge = GraphEdge(
            source_uuid=source.uuid,
            target_uuid=target.uuid,
            label=label,
            fact_text=fact_text,
            valid_at=datetime.now(timezone.utc)
        )
        new_edge.embedding = new_edge_embedding

        self.edges[new_edge.uuid] = new_edge
        self.graph.add_edge(source.uuid, target.uuid, key=new_edge.uuid, **new_edge.model_dump())

    # --- Retrieval ---

    def answer_question(self, question: str, group_id: str) -> str:
        """
        Answers a question by querying the knowledge graph and synthesizing the findings.
        This is the primary retrieval method for the architecture.
        """
        logger.info(f"Answering question: '{question}'")

        # 1. Find relevant entry points into the graph
        question_embedding = self.llm_manager.get_embedding(question)

        # Filter nodes by group_id before sorting
        relevant_nodes = [n for n in self.nodes.values()]

        if not relevant_nodes:
            return "I don't have enough information in my memory to answer that yet."

        # Sort by similarity to the question
        relevant_nodes.sort(key=lambda n: cosine(question_embedding, n.embedding) if n.embedding else 1.0)

        entry_points = relevant_nodes[:5]  # Top 5 nodes as starting points
        entry_point_uuids = [n.uuid for n in entry_points]

        if not entry_point_uuids:
            return "I couldn't find any relevant concepts in my memory to answer that."

        # 2. Build a relevant subgraph
        # Get the 2-hop neighborhood around the entry points
        subgraph_nodes = set(entry_point_uuids)
        for uuid in entry_point_uuids:
            # Add neighbors (successors and predecessors)
            subgraph_nodes.update(self.graph.successors(uuid))
            subgraph_nodes.update(self.graph.predecessors(uuid))

        subgraph = self.graph.subgraph(subgraph_nodes)

        if subgraph.number_of_edges() == 0:
            # If no edges, just use info from the entry point nodes
            context = "Based on my memory, here's what I know about the most relevant concepts:\n"
            for node_uuid in entry_point_uuids:
                node = self.nodes[node_uuid]
                context += f"- Entity: {node.name} (Type: {', '.join(node.labels)})\n  Summary: {node.summary}\n"
        else:
            # 3. Serialize the subgraph to text for the LLM
            context_parts = []
            for u, v, data in subgraph.edges(data=True):
                source_node = self.nodes[u]
                target_node = self.nodes[v]
                context_parts.append(
                    f"Fact: '{source_node.name} {data['label']} {target_node.name}'. Evidence: '{data['fact_text']}'"
                )
            context = "Based on my memory, here are the relevant facts I found:\n" + "\n".join(context_parts)

        # 4. Ask the LLM to synthesize an answer
        prompt = [
            ("system",
             "You are an AI assistant answering questions based *only* on the provided context from your memory. If the context doesn't contain the answer, say so."),
            ("user", f"Context from Memory:\n---\n{context}\n---\n\nQuestion: {question}")
        ]
        settings = CommonCompSettings(temperature=0.1)
        answer = self.llm_manager.completion_text(LlmPreset.Default, prompt, settings)

        return answer or "I found some relevant information but couldn't form a conclusive answer."


if __name__ == '__main__':
    # This is a test block to demonstrate the functionality of the GraphMemoryManager.
    # It requires a running LlmManagerProxy instance.

    # Setup basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # In a real application, you would get this from your main thread.
    # For this test, we'll assume a mock or real one is available.
    # You would replace this with the actual proxy from pm_lida.

    # We need to start the llama_worker thread from pm_lida for the proxy to work.
    from pm_lida import llama_worker
    import threading

    worker_thread = threading.Thread(target=llama_worker, daemon=True)
    worker_thread.start()

    # Use the aux_llm proxy for memory tasks to give it lower priority
    from pm_lida import aux_llm as llm_manager

    # 1. Initialize the manager
    DB_FILE = "./test_knowledge_graph.db"
    graph_manager = GraphMemoryManager(llm_manager=llm_manager, db_path=DB_FILE)

    # 2. Integrate some text
    conversation_text = """
    User: Hey, I was thinking about starting a new project with my friend Sarah. She works at a company called Innovate Inc.
    AI: That's great! What kind of project?
    User: We're building a mobile app. By the way, Sarah mentioned she's been working at Innovate Inc. since 2022. Before that, she was at a startup called 'QuickSolve'.
    AI: Interesting. It sounds like she has good experience.
    User: Yeah, she does. Oh, and QuickSolve was acquired by Innovate Inc. last year. Total coincidence.
    """
    graph_manager.integrate_text(conversation_text, group_id="test_conversation_1")

    # 3. Ask some questions to test retrieval
    print("\n--- Answering Questions ---")

    question1 = "Where does Sarah work?"
    answer1 = graph_manager.answer_question(question1, group_id="test_conversation_1")
    print(f"Q: {question1}\nA: {answer1}\n")

    question2 = "What is the relationship between Innovate Inc. and QuickSolve?"
    answer2 = graph_manager.answer_question(question2, group_id="test_conversation_1")
    print(f"Q: {question2}\nA: {answer2}\n")

    # 4. Add new, contradictory information
    contradictory_text = """
    User: Quick update on Sarah - she just told me she left Innovate Inc. today to start her own company!
    """
    graph_manager.integrate_text(contradictory_text, group_id="test_conversation_1")

    # 5. Ask the same question again to see if the memory evolved
    print("\n--- Answering Questions After Update ---")
    answer3 = graph_manager.answer_question(question1, group_id="test_conversation_1")
    print(f"Q: {question1}\nA: {answer3}\n")

    # 6. Test loading from DB
    print("\n--- Testing Persistence ---")
    del graph_manager
    print("Deleted in-memory graph. Reloading from database...")
    reloaded_manager = GraphMemoryManager(llm_manager=llm_manager, db_path=DB_FILE)

    answer4 = reloaded_manager.answer_question(question1, group_id="test_conversation_1")
    print(f"Q (after reload): {question1}\nA: {answer4}\n")

    print("Test complete.")