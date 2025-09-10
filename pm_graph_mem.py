# --- IMPORTS ---

import json
import logging
import sqlite3
import threading
import uuid
import os  # For test cleanup
from collections import defaultdict
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Dict, List, Optional, Tuple, Literal, Type, Union, Set
import regex as re

import networkx as nx
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator
from scipy.spatial.distance import cosine  # For similarity calculation

from pm.llm.llm_common import CommonCompSettings, LlmPreset
from pm.llm.llm_proxy import llama_worker

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONSTANTS AND PREDEFINED VALUES ---

EGO_NODE_UUID = "00000000-0000-0000-0000-000000000001"
EGO_NODE_NAME = "Self"


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


# --- PLACEHOLDER LLM MANAGER INTERFACE (pm_lida.py) ---
# This section simulates what might be in pm_lida.py for clarity

class SmallLlmManagerProxy:
    """
    A placeholder for the actual LLM interaction manager.
    This class would handle API calls, potentially caching, rate limiting, etc.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "default_model"):
        self.api_key = api_key
        self.model_name = model_name
        logger.info(f"SmallLlmManagerProxy initialized for model: {model_name}")
        from pm_lida import main_llm, task_queue, model_map
        worker_thread = threading.Thread(target=llama_worker, args=(model_map, task_queue), daemon=True)
        worker_thread.start()

        self.llm_manager = main_llm
        # In a real scenario, initialize API clients here

    def get_embedding(self, text: str, model_id: Optional[str] = None) -> List[float]:
        """Generates an embedding for the given text."""
        logger.debug(f"Requesting embedding for text (first 50 chars): '{text[:50]}...'")
        return self.llm_manager.get_embedding(text)

    def call_tool(self, system_prompt: str, user_prompt: str, tool_model: Type[BaseModel],
                  settings: Optional[CommonCompSettings] = None) -> BaseModel:
        """
        Simulates an LLM call that is expected to return a JSON object
        conforming to the `tool_model` Pydantic schema.
        """
        logger.debug(f"Calling tool {tool_model.__name__} with user prompt (first 50 chars): '{user_prompt[:50]}...'")
        _, calls = self.llm_manager.completion_tool(LlmPreset.Default,
                                                    [("system", system_prompt), ("user", user_prompt)],
                                                    tools=[tool_model])
        return calls[0]

    def get_completion(self, system_prompt: str, user_prompt: str,
                       settings: Optional[CommonCompSettings] = None) -> str:
        """Simulates a general text completion LLM call."""
        logger.debug(
            f"Requesting completion with system prompt and user prompt (first 50 chars): '{user_prompt[:50]}...'")
        return self.llm_manager.completion_text(LlmPreset.Default, [("system", system_prompt), ("user", user_prompt)])


# --- PYDANTIC MODELS FOR GRAPH ELEMENTS (The Three Layers) ---

class InternalState(BaseModel):
    emotional_valence: float = Field(description="A score from -1.0 (highly negative) to 1.0 (highly positive).", default=0)
    emotional_label: str = Field(
        description="A concise, descriptive word for the primary emotion (e.g., 'curiosity', 'satisfaction', 'confusion').", default="")
    cognitive_process: str = Field(
        description="The dominant cognitive process (e.g., 'storing_new_fact', 'detecting_contradiction', 'forming_goal').", default="")
    certainty: float = Field(default=1.0,
                             description="The AI's confidence in the information processed, from 0.0 to 1.0.")
    salience_focus: List[str] = Field(default_factory=list,
                                      description="List of entity names or concepts that were the main focus of attention.")


class MemoryEpisodeNode(BaseModel):
    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    ego_uuid: str = Field(default=EGO_NODE_UUID, description="Link to the AI's central EgoNode.")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    topic_summary: str = Field(description="An LLM-generated, brief summary of what the episode was about.")
    internal_state: InternalState = Field(description="The structured internal state of the AI during this episode.", default_factory=InternalState)
    embedding: List[float] = Field(default_factory=list)  # Persisted as BLOB

    @model_validator(mode='before')
    @classmethod
    def ensure_utc_timestamp(cls, data: Any) -> Any:
        if isinstance(data, dict):
            ts = data.get('timestamp')
            if isinstance(ts, str):
                data['timestamp'] = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            if isinstance(ts, datetime) and ts.tzinfo is None:
                data['timestamp'] = ts.replace(tzinfo=timezone.utc)
        return data


class GraphNode(BaseModel):
    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    labels: List[str]  # e.g., ["Person"], ["Location"], ["Organization", "Client"]
    source_episode_uuid: str  # Episode where this node was primarily defined or last significantly updated
    concept_uuid: Optional[str] = None  # Link to a ConceptNode
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    attributes: Dict[str, Any] = Field(default_factory=dict)  # For additional, non-relational data
    embedding: List[float] = Field(default_factory=list)  # Persisted as BLOB

    @model_validator(mode='before')
    @classmethod
    def ensure_utc_created_at(cls, data: Any) -> Any:
        if isinstance(data, dict):
            ts = data.get('created_at')
            if isinstance(ts, str):
                data['created_at'] = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            if isinstance(ts, datetime) and ts.tzinfo is None:
                data['created_at'] = ts.replace(tzinfo=timezone.utc)
        return data


class GraphEdge(BaseModel):
    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_uuid: str  # UUID of the source GraphNode
    target_uuid: str  # UUID of the target GraphNode
    label: str  # e.g., "works_for", "knows", "located_in"
    fact_text: str  # Natural language representation of the fact, e.g., "Alex works for Acme Corp"
    source_episode_uuid: str  # Episode where this fact was learned
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    valid_at: Optional[datetime] = Field(default=None,
                                         description="Timestamp when this fact became valid (if applicable).")
    invalid_at: Optional[datetime] = Field(default=None,
                                           description="Timestamp when this fact became invalid (due to contradiction/update).")
    attributes: Dict[str, Any] = Field(default_factory=dict)
    embedding: List[float] = Field(default_factory=list)  # Persisted as BLOB

    @model_validator(mode='before')
    @classmethod
    def ensure_utc_timestamps(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for key in ['created_at', 'valid_at', 'invalid_at']:
                ts = data.get(key)
                if ts:
                    if isinstance(ts, str):
                        data[key] = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    if isinstance(ts, datetime) and ts.tzinfo is None:
                        data[key] = ts.replace(tzinfo=timezone.utc)
        return data


class ConceptNode(BaseModel):
    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(description="The name of the concept (e.g., 'Person', 'Company', 'Action').")
    description: str = Field(description="A brief explanation of what this concept represents.")
    parent_concept_uuid: Optional[str] = Field(default=None,
                                               description="UUID of the parent concept (for IS_A hierarchy).")
    embedding: List[float] = Field(default_factory=list)  # Persisted as BLOB


# --- PYDANTIC MODELS FOR LLM TOOL-CALLING OPERATIONS ---
class AnalyzedInternalState(BaseModel):
    state: InternalState


class EpisodeTopicSummary(BaseModel):
    summary: str


class ExtractedEntity(BaseModel):
    name: str
    label: str  # Suggested primary label, e.g., 'Person', 'Project', 'Location'


class ExtractedEntities(BaseModel):
    entities: List[ExtractedEntity]


class ExtractedRelationship(BaseModel):
    source_name: str
    target_name: str
    label: str  # e.g., "works_at", "is_located_in"
    fact: str  # The sentence or phrase representing the fact


class ExtractedRelationships(BaseModel):
    relationships: List[ExtractedRelationship]


class EntityClassifier(BaseModel):
    entity_name: str
    most_likely_concept_name: str
    reasoning: str


class Confirmation(BaseModel):
    is_same: bool
    reasoning: str


class ContradictionCheck(BaseModel):
    is_contradictory: bool
    is_duplicate: bool  # If the new fact is essentially identical to an existing one
    reasoning: str
    conflicting_edge_uuid: Optional[str] = None  # UUID of the edge that is contradicted


class NewConceptDefinition(BaseModel):
    """Defines a new concept and places it within the existing hierarchy."""
    description: str = Field(description="A brief, clear explanation of what this new concept represents.")
    parent_concept_name: str = Field(
        description="The name of the single most appropriate existing concept that is the parent of this new one.")

class RephrasedQuestions(BaseModel):
    """A list of alternative ways to ask a question to improve search recall."""
    questions: List[str] = Field(description="A list of 3 diverse rephrasings of the original question.")

class RelevantItems(BaseModel):
    """A list of items deemed relevant to a user's question from a larger context."""
    relevant_item_identifiers: List[str] = Field(description="A list of UUIDs of the items that directly help answer the question.")


class CognitiveMemoryManager:
    def __init__(self, llm_manager: SmallLlmManagerProxy, db_path: str = "cognitive_memory.db"):
        logger.info(f"Initializing CognitiveMemoryManager with DB path: {db_path}")
        self.llm_manager: SmallLlmManagerProxy = llm_manager
        self.db_path: str = db_path
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()  # Combined graph

        # In-memory stores for quick access; DB is source of truth
        self.ego_node_model: Optional[GraphNode] = None
        self.episodes: Dict[str, MemoryEpisodeNode] = {}
        self.concepts: Dict[str, ConceptNode] = {}
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, GraphEdge] = {}

        self.semantic_axes: Dict[str, np.ndarray] = {}

        self.db_conn = sqlite3.connect(self.db_path)
        self.db_conn.row_factory = sqlite3.Row
        self._create_tables_if_not_exist()

        self.load_from_db()
        self._initialize_ego_node()
        self._bootstrap_core_concepts()
        self._compute_semantic_axes()
        # No longer need _ensure_all_embeddings if we always save them
        # self._ensure_all_embeddings() # This can be removed if saving is reliable
        self._save_to_db()
        logger.info("CognitiveMemoryManager initialized successfully.")

    def _map_pydantic_type_to_sql(self, pydantic_type: Any) -> str:
        """Maps a Pydantic/Python type to a SQLite data type."""
        # Handle Optional[T] by getting the inner type
        if hasattr(pydantic_type, '__origin__') and pydantic_type.__origin__ is Union:
            # Get the first type argument that is not NoneType
            args = [arg for arg in pydantic_type.__args__ if arg is not type(None)]
            if args:
                pydantic_type = args[0]

        if pydantic_type is str:
            return "TEXT"
        if pydantic_type is int:
            return "INTEGER"
        if pydantic_type is float:
            return "REAL"
        if pydantic_type is bool:
            return "INTEGER"  # Stored as 0 or 1
        if pydantic_type is datetime:
            return "TEXT"  # Stored as ISO 8601 string

        # Complex types (that are not embeddings) are serialized to JSON
        if pydantic_type is list or pydantic_type is dict or issubclass(pydantic_type, BaseModel):
            return "TEXT"

        return "TEXT"

    def _prepare_data_for_sql(self, model_instance: BaseModel) -> Tuple:
        """Serializes a Pydantic model instance into a tuple for SQL insertion."""
        values = []
        # The calling context (_save_to_db) ensures we only iterate over fields to be persisted.
        for field_name in [name for name, f in model_instance.__class__.model_fields.items() if not f.exclude]:
            value = getattr(model_instance, field_name)

            if field_name == 'embedding' and value:
                np_array = np.array(value, dtype=np.float32)
                values.append(np_array.tobytes())
            elif value is None:
                values.append(None)
            elif isinstance(value, (list, dict)) or issubclass(type(value), BaseModel):
                values.append(json.dumps(value, default=str))
            elif isinstance(value, datetime):
                values.append(value.isoformat())
            elif isinstance(value, bool):
                values.append(1 if value else 0)
            else:
                values.append(value)
        return tuple(values)

    def _create_tables_if_not_exist(self):
        """Dynamically creates SQLite tables based on Pydantic model schemas."""
        table_map = {
            "episodes": MemoryEpisodeNode,
            "concepts": ConceptNode,
            "graph_nodes": GraphNode,
            "graph_edges": GraphEdge,
        }

        with self.db_conn:
            cursor = self.db_conn.cursor()
            for table_name, model_class in table_map.items():
                columns = []
                for field_name, field_info in model_class.model_fields.items():
                    if field_info.exclude:
                        continue

                    if field_name == 'embedding':
                        sql_type = "BLOB"
                    else:
                        sql_type = self._map_pydantic_type_to_sql(field_info.annotation)

                    column_def = f"{field_name} {sql_type}"
                    if field_name == 'uuid':
                        column_def += " PRIMARY KEY"
                    columns.append(column_def)

                create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns)})"
                cursor.execute(create_table_sql)
            logger.info("Database tables ensured dynamically from Pydantic models (with BLOB for embeddings).")

    # --- Persistence ---
    def _save_to_db(self):
        """Saves the current in-memory state to the SQLite database, storing embeddings as BLOBs."""
        logger.debug("Saving cognitive memory state to database...")
        persistence_map = {
            "episodes": (MemoryEpisodeNode, self.episodes.values()),
            "concepts": (ConceptNode, self.concepts.values()),
            "graph_nodes": (GraphNode, self.nodes.values()),
            "graph_edges": (GraphEdge, self.edges.values()),
        }

        with self.db_conn:
            cursor = self.db_conn.cursor()
            for table_name, (model_class, data_iterable) in persistence_map.items():
                data_list = list(data_iterable)
                if not data_list:
                    continue

                columns = [name for name, field in model_class.model_fields.items() if not field.exclude]
                data_to_insert = [self._prepare_data_for_sql(item) for item in data_list]
                placeholders = ", ".join(["?"] * len(columns))
                column_names = ", ".join(columns)

                sql = f"INSERT OR REPLACE INTO {table_name} ({column_names}) VALUES ({placeholders})"
                cursor.executemany(sql, data_to_insert)
                logger.info(f"Saved {cursor.rowcount} items to '{table_name}' table.")

    def load_from_db(self):
        """
        [REPLACEMENT V3]
        Loads state from the database, correctly deserializing complex types.
        Includes a special-case "dirty parser" for the 'internal_state' field
        to handle its non-JSON string format.
        """
        logger.info("Loading cognitive memory state from database...")
        from typing import get_origin, get_args, Union  # For dynamic type inspection
        import ast  # For safely evaluating list/dict strings

        persistence_map = {
            "episodes": (MemoryEpisodeNode, self.episodes),
            "concepts": (ConceptNode, self.concepts),
            "graph_nodes": (GraphNode, self.nodes),
            "graph_edges": (GraphEdge, self.edges),
        }

        # Clear in-memory stores before loading
        for _, store in persistence_map.values():
            store.clear()

        cursor = self.db_conn.cursor()
        for table_name, (model_class, store) in persistence_map.items():
            try:
                cursor.execute(f"SELECT * FROM {table_name}")
                rows = cursor.fetchall()
                for row in rows:
                    row_data = dict(row)

                    # --- SPECIAL CASE: DIRTY PARSER FOR 'internal_state' ---
                    if 'internal_state' in row_data and isinstance(row_data['internal_state'], str):
                        state_str = row_data['internal_state'].strip('"')
                        # Check if it looks like the non-JSON format
                        if '=' in state_str and '{' not in state_str:
                            logger.debug(f"Applying dirty parser for internal_state: {state_str[:60]}...")
                            parsed_state = {}
                            # Regex to find key='value' or key=number or key=[list]
                            pattern = re.compile(r"(\w+)=('([^']*)'|\[.*?\]|[\d\.]+)")
                            matches = pattern.findall(state_str)

                            for key, full_value, str_value in matches:
                                value_to_process = str_value if str_value else full_value
                                try:
                                    # Use ast.literal_eval for safe evaluation of lists, numbers, etc.
                                    parsed_state[key] = ast.literal_eval(value_to_process)
                                except (ValueError, SyntaxError):
                                    # Fallback for simple strings that literal_eval can't handle
                                    parsed_state[key] = value_to_process
                            row_data['internal_state'] = parsed_state
                    # --- END OF DIRTY PARSER ---

                    # --- DYNAMIC DESERIALIZATION FOR OTHER JSON FIELDS ---
                    for field_name, field_info in model_class.model_fields.items():
                        # Skip fields we've already handled or don't need to process
                        if field_name == 'internal_state' or field_name not in row_data or not isinstance(
                                row_data[field_name], str):
                            continue

                        main_type = field_info.annotation
                        if get_origin(main_type) is Union:
                            args = [arg for arg in get_args(main_type) if arg is not type(None)]
                            if args: main_type = args[0]

                        origin = get_origin(main_type)
                        is_json_field = (
                                origin in (list, dict) or
                                (isinstance(main_type, type) and issubclass(main_type, BaseModel))
                        )

                        if is_json_field:
                            try:
                                row_data[field_name] = json.loads(row_data[field_name])
                            except (json.JSONDecodeError, TypeError):
                                logger.warning(
                                    f"Could not JSON-decode field '{field_name}' in table '{table_name}'. "
                                    f"Value (first 50 chars): '{row_data[field_name][:50]}...'. Leaving as string."
                                )
                    # --- END OF DESERIALIZATION LOGIC ---

                    # Handle the embedding BLOB
                    if 'embedding' in row_data and row_data['embedding'] is not None:
                        embedding_blob = row_data['embedding']
                        np_array = np.frombuffer(embedding_blob, dtype=np.float32)
                        row_data['embedding'] = np_array.tolist()

                    instance = model_class.model_validate(row_data)
                    store[instance.uuid] = instance

                logger.info(f"Loaded {len(rows)} items from '{table_name}'.")

            except sqlite3.OperationalError as e:
                logger.warning(f"Could not load from table '{table_name}': {e}. It might be new.")

        self._rebuild_networkx_graph()

    def _get_or_create_node_for_relationship(self, node_name: str, resolved_nodes_map: Dict[str, GraphNode],
                                             current_episode_uuid: str) -> Optional[GraphNode]:
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
        # We create a temporary ExtractedEntity to pass to it.
        # We can guess a generic label like "Concept" or let the classification figure it out.
        implicit_entity_data = ExtractedEntity(name=node_name, label="Entity")
        newly_resolved_node = self._resolve_entity_and_concept(implicit_entity_data, current_episode_uuid)

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
            if concept.name.lower() == concept_name.lower():
                logger.debug(f"Found existing concept: '{concept.name}'")
                return concept

        # 2. If not found, create it using an LLM to define it and find its parent.
        logger.info(f"Concept '{concept_name}' not found. Attempting to create it on-the-fly.")

        existing_concept_names = sorted([c.name for c in self.concepts.values()])
        concept_list_str = ", ".join(existing_concept_names)

        system_prompt = "You are an ontology expert. Your task is to define a new concept and correctly place it within an existing conceptual hierarchy."
        user_prompt = (
            f"A new concept, '{concept_name}', needs to be added to our knowledge graph. "
            f"Please provide a brief, one-sentence description for it. "
            f"Then, select the single most logical parent for '{concept_name}' from the following list of existing concepts:\n"
            f"---_LIST_OF_CONCEPTS---\n{concept_list_str}\n---_END_OF_LIST_---"
        )

        try:
            definition_result = self.llm_manager.call_tool(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                tool_model=NewConceptDefinition
            )

            if not isinstance(definition_result, NewConceptDefinition):
                logger.error(f"LLM failed to return a valid NewConceptDefinition for '{concept_name}'.")
                return None

            # Find the parent concept's UUID
            parent_concept_uuid = None
            parent_name = definition_result.parent_concept_name
            for uuid_val, concept_obj in self.concepts.items():
                if concept_obj.name.lower() == parent_name.lower():
                    parent_concept_uuid = uuid_val
                    break

            if not parent_concept_uuid:
                logger.warning(
                    f"LLM suggested parent '{parent_name}' for '{concept_name}', but it was not found. Defaulting to 'Entity' as parent.")
                # Fallback to the root "Entity" concept
                for uuid_val, concept_obj in self.concepts.items():
                    if concept_obj.name == "Entity":
                        parent_concept_uuid = uuid_val
                        break

            # Create the new concept
            new_concept = ConceptNode(
                name=concept_name,  # Use the original casing
                description=definition_result.description,
                parent_concept_uuid=parent_concept_uuid
            )
            new_concept.embedding = self.llm_manager.get_embedding(
                f"Concept: {new_concept.name}. Description: {new_concept.description}")

            # Add it to memory and the graph
            self.concepts[new_concept.uuid] = new_concept
            self.graph.add_node(new_concept.uuid, data=new_concept, type="ConceptNode", name=new_concept.name)
            if new_concept.parent_concept_uuid:
                self.graph.add_edge(new_concept.uuid, new_concept.parent_concept_uuid, label=StructuralEdgeLabel.IS_A)

            logger.info(f"Successfully created new concept '{new_concept.name}' with parent '{parent_name}'.")
            return new_concept

        except Exception as e:
            logger.error(f"Error creating concept '{concept_name}' on the fly: {e}", exc_info=True)
            return None

    def _rebuild_networkx_graph(self):
        """Reconstructs the in-memory NetworkX graph from loaded Pydantic models."""
        self.graph.clear()
        logger.debug("Rebuilding NetworkX graph...")

        # Add ConceptNodes
        for concept_uuid, concept in self.concepts.items():
            self.graph.add_node(concept_uuid, data=concept, type="ConceptNode", name=concept.name)

        # Add GraphNodes (including EgoNode)
        for node_uuid, node in self.nodes.items():
            self.graph.add_node(node_uuid, data=node, type="GraphNode", name=node.name)

        # Add MemoryEpisodeNodes
        for episode_uuid, episode in self.episodes.items():
            self.graph.add_node(episode_uuid, data=episode, type="MemoryEpisodeNode", summary=episode.topic_summary)

        # Add structural edges for Concepts (IS_A)
        for concept_uuid, concept in self.concepts.items():
            if concept.parent_concept_uuid and concept.parent_concept_uuid in self.concepts:
                self.graph.add_edge(concept_uuid, concept.parent_concept_uuid,
                                    label=StructuralEdgeLabel.IS_A)  # type: ignore

        # Add structural edges for GraphNodes (INSTANCE_OF)
        for node_uuid, node in self.nodes.items():
            if node.concept_uuid and node.concept_uuid in self.concepts:
                self.graph.add_edge(node_uuid, node.concept_uuid, label=StructuralEdgeLabel.INSTANCE_OF)  # type: ignore

        # Add structural edges for Episodes (EXPERIENCED_BY)
        for episode_uuid in self.episodes:
            if EGO_NODE_UUID in self.graph:  # Ensure ego node exists
                self.graph.add_edge(episode_uuid, EGO_NODE_UUID,
                                    label=StructuralEdgeLabel.EXPERIENCED_BY)  # type: ignore
            else:
                logger.warning(
                    f"EgoNode {EGO_NODE_UUID} not found in graph while rebuilding EXPERIENCED_BY for episode {episode_uuid}")

        # Add factual GraphEdges
        for edge_uuid, edge in self.edges.items():
            if edge.source_uuid in self.graph and edge.target_uuid in self.graph:
                self.graph.add_edge(edge.source_uuid, edge.target_uuid, key=edge_uuid, label=edge.label, data=edge,
                                    type="GraphEdge")
            else:
                logger.warning(f"Skipping edge {edge_uuid} due to missing source/target node in graph during rebuild.")
        logger.debug(
            f"NetworkX graph rebuilt with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")

    def _ensure_all_embeddings(self):
        logger.info("Ensuring all items have embeddings...")
        items_embedded = 0

        for episode in self.episodes.values():
            if not episode.embedding:
                text_to_embed = f"{episode.topic_summary} Emotional state: {episode.internal_state.emotional_label}. Cognitive process: {episode.internal_state.cognitive_process}."
                episode.embedding = self.llm_manager.get_embedding(text_to_embed)
                items_embedded += 1

        for concept in self.concepts.values():
            if not concept.embedding:
                text_to_embed = f"Concept: {concept.name}. Description: {concept.description}"
                concept.embedding = self.llm_manager.get_embedding(text_to_embed)
                items_embedded += 1

        for node in self.nodes.values():
            if not node.embedding:
                text_to_embed = f"Entity: {node.name}. Labels: {', '.join(node.labels)}. Attributes: {json.dumps(node.attributes)}"
                node.embedding = self.llm_manager.get_embedding(text_to_embed)
                items_embedded += 1

        for edge in self.edges.values():
            if not edge.embedding:
                # To get context for an edge, might need source/target node names
                source_node_name = self.nodes.get(edge.source_uuid, GraphNode(name="Unknown Source", labels=[],
                                                                              source_episode_uuid="N/A")).name
                target_node_name = self.nodes.get(edge.target_uuid, GraphNode(name="Unknown Target", labels=[],
                                                                              source_episode_uuid="N/A")).name
                text_to_embed = f"Fact: {edge.fact_text}. Relates: {source_node_name} {edge.label} {target_node_name}."
                edge.embedding = self.llm_manager.get_embedding(text_to_embed)
                items_embedded += 1
        if items_embedded > 0:
            logger.info(f"Generated embeddings for {items_embedded} items.")
            self._save_to_db()  # Persist new embeddings
        else:
            logger.info("All items already have embeddings.")

    # --- Bootstrap and Setup ---
    def _initialize_ego_node(self):
        logger.debug("Initializing EgoNode...")
        if EGO_NODE_UUID not in self.nodes:
            logger.info(f"EgoNode with UUID {EGO_NODE_UUID} not found. Creating new EgoNode.")
            # Create a "birth" episode for the EgoNode itself.
            birth_internal_state = InternalState(
                emotional_valence=0.5,
                emotional_label="awareness",
                cognitive_process="self_initialization",
                certainty=1.0,
                salience_focus=[EGO_NODE_NAME]
            )
            birth_episode = MemoryEpisodeNode(
                topic_summary="System initialization and self-awareness.",
                internal_state=birth_internal_state,
                # uuid will be auto-generated
            )
            # Add birth episode to memory
            self.episodes[birth_episode.uuid] = birth_episode
            if not birth_episode.embedding:  # Generate embedding if not mocked
                birth_episode.embedding = self.llm_manager.get_embedding(
                    f"{birth_episode.topic_summary} {birth_internal_state.emotional_label}")
            self.graph.add_node(birth_episode.uuid, data=birth_episode, type="MemoryEpisodeNode")

            ego = GraphNode(
                uuid=EGO_NODE_UUID,
                name=EGO_NODE_NAME,
                labels=["AI", "Self", "CognitiveAgent"],
                source_episode_uuid=birth_episode.uuid,  # Link to its "birth"
                attributes={"description": "The core identity of this AI agent."}
            )
            if not ego.embedding:  # Generate embedding if not mocked
                ego.embedding = self.llm_manager.get_embedding(f"{ego.name} {', '.join(ego.labels)}")

            self.nodes[EGO_NODE_UUID] = ego
            self.ego_node_model = ego
            self.graph.add_node(EGO_NODE_UUID, data=ego, type="GraphNode", name=ego.name)
            # Link birth episode to ego
            self.graph.add_edge(birth_episode.uuid, EGO_NODE_UUID,
                                label=StructuralEdgeLabel.EXPERIENCED_BY)  # type: ignore
            logger.info(f"EgoNode '{EGO_NODE_NAME}' created and linked to birth episode '{birth_episode.uuid}'.")
            self._save_to_db()  # Save the new EgoNode and its birth episode
        else:
            self.ego_node_model = self.nodes[EGO_NODE_UUID]
            logger.info(f"EgoNode '{self.ego_node_model.name}' loaded from database.")
        # Ensure it's in the graph if loaded from DB but graph was cleared
        if EGO_NODE_UUID not in self.graph:
            self.graph.add_node(EGO_NODE_UUID, data=self.ego_node_model, type="GraphNode",
                                name=self.ego_node_model.name)

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
        ]

        # Create concepts if they don't exist by name
        name_to_uuid_map = {c.name: uuid_val for uuid_val, c in self.concepts.items()}
        made_changes = False

        for concept_def in core_concepts_definitions:
            if concept_def["name"] not in name_to_uuid_map:
                logger.debug(f"Creating core concept: {concept_def['name']}")
                parent_uuid = name_to_uuid_map.get(concept_def["parent"]) if concept_def["parent"] else None
                new_concept = ConceptNode(
                    name=concept_def["name"],
                    description=concept_def["description"],
                    parent_concept_uuid=parent_uuid
                )
                if not new_concept.embedding:  # Generate embedding
                    new_concept.embedding = self.llm_manager.get_embedding(
                        f"Concept: {new_concept.name}. {new_concept.description}")

                self.concepts[new_concept.uuid] = new_concept
                self.graph.add_node(new_concept.uuid, data=new_concept, type="ConceptNode", name=new_concept.name)
                name_to_uuid_map[new_concept.name] = new_concept.uuid  # Update map for subsequent parent lookups
                made_changes = True

        # Link concepts with IS_A edges after all are potentially created
        for concept_uuid, concept in self.concepts.items():
            if concept.parent_concept_uuid and concept.parent_concept_uuid in self.graph:
                if not self.graph.has_edge(concept_uuid, concept.parent_concept_uuid):
                    self.graph.add_edge(concept_uuid, concept.parent_concept_uuid,
                                        label=StructuralEdgeLabel.IS_A)  # type: ignore
                    made_changes = True  # IS_A edges are part of graph structure

        if made_changes:
            logger.info(f"Core concepts bootstrapped. Total concepts: {len(self.concepts)}.")
            self._save_to_db()
        else:
            logger.info("Core concepts already up-to-date.")

    def _compute_semantic_axes(self):
        logger.info("Computing semantic axes...")
        updated_axes = False
        for axis_name, (pos_term, neg_term) in SEMANTIC_AXES_DEFINITION.items():
            if axis_name not in self.semantic_axes:  # Only compute if not already done (or allow re-compute flag)
                try:
                    pos_embedding = np.array(self.llm_manager.get_embedding(pos_term))
                    neg_embedding = np.array(self.llm_manager.get_embedding(neg_term))
                    axis_vector = pos_embedding - neg_embedding
                    norm = np.linalg.norm(axis_vector)
                    if norm > 0:
                        self.semantic_axes[axis_name] = axis_vector / norm
                        logger.debug(f"Computed semantic axis: {axis_name}")
                        updated_axes = True
                    else:
                        logger.warning(f"Could not compute semantic axis {axis_name}: zero norm vector.")
                except Exception as e:
                    logger.error(f"Error computing semantic axis {axis_name}: {e}")
        if updated_axes:
            logger.info(f"Semantic axes computed for: {list(self.semantic_axes.keys())}")
        else:
            logger.info("Semantic axes already computed or no new axes defined.")

    # --- Knowledge Integration (The Write Path) ---
    def integrate_experience(self, text_from_interaction: str, ai_internal_state_text: str) -> Optional[
        MemoryEpisodeNode]:
        logger.info(f"Integrating new experience. Interaction (start): '{text_from_interaction[:100]}...'")
        try:
            # 1. Create the MemoryEpisode node
            episode = self._create_memory_episode(text_from_interaction, ai_internal_state_text)
            if not episode:
                logger.error("Failed to create memory episode. Aborting integration.")
                return None

            # 2. Extract entities from the interaction text.
            extracted_entities_model = self._extract_entities_from_text(text_from_interaction)
            if not extracted_entities_model:
                logger.warning("No entities extracted from interaction.")
                # Still save the episode, as it captures the AI's internal state
                self._save_to_db()
                return episode

            # 3. For each entity, find or create its GraphNode, ensuring it's linked to a concept and the current episode.
            resolved_nodes_map: Dict[str, GraphNode] = {}  # name -> GraphNode
            for extracted_entity_obj in extracted_entities_model.entities:
                node = self._resolve_entity_and_concept(extracted_entity_obj, episode.uuid)
                if node:
                    resolved_nodes_map[node.name] = node
                else:
                    logger.warning(f"Could not resolve or create node for entity: {extracted_entity_obj.name}")

            # 4. Extract relationships between the identified entities.
            if resolved_nodes_map:  # Only extract relationships if we have nodes
                extracted_relationships_model = self._extract_relationships_from_text(text_from_interaction,
                                                                                      list(resolved_nodes_map.values()))
                if extracted_relationships_model:
                    # 5. For each relationship, create a GraphEdge, check for contradictions, and add it.
                    for rel_data in extracted_relationships_model.relationships:
                        self._resolve_and_add_edge(rel_data, resolved_nodes_map, episode.uuid)
                else:
                    logger.info("No relationships extracted from interaction.")
            else:
                logger.info("No resolved nodes, skipping relationship extraction.")

            # 6. Save the updated graph to the database.
            self._save_to_db()
            logger.info(f"Experience integration completed for episode {episode.uuid}.")
            return episode

        except Exception as e:
            logger.error(f"Error during experience integration: {e}", exc_info=True)
            return None

    def _create_memory_episode(self, text: str, state_text: str) -> Optional[MemoryEpisodeNode]:
        logger.debug("Creating memory episode...")
        try:
            # Use LLM with EpisodeTopicSummary tool
            summary_prompt = f"Summarize this interaction concisely: {text}"
            summary_tool_result = self.llm_manager.call_tool(
                system_prompt="You are a summarization assistant.",
                user_prompt=summary_prompt,
                tool_model=EpisodeTopicSummary
            )
            topic_summary = summary_tool_result.summary if isinstance(summary_tool_result,
                                                                      EpisodeTopicSummary) else "Summary unavailable"

            # Use LLM with AnalyzedInternalState tool
            state_prompt = f"Analyze this internal state description: {state_text}"
            internal_state_tool_result = self.llm_manager.call_tool(
                system_prompt="You are an AI internal state analyzer.",
                user_prompt=state_prompt,
                tool_model=AnalyzedInternalState
            )
            internal_state = internal_state_tool_result.state if isinstance(internal_state_tool_result,
                                                                            AnalyzedInternalState) else InternalState(
                emotional_valence=0, emotional_label="unknown", cognitive_process="unknown")

            episode = MemoryEpisodeNode(
                topic_summary=topic_summary,
                internal_state=internal_state
            )

            # Generate embedding for the episode
            embedding_text = f"Summary: {episode.topic_summary}. Emotion: {episode.internal_state.emotional_label}. Process: {episode.internal_state.cognitive_process}."
            episode.embedding = self.llm_manager.get_embedding(embedding_text)

            # Add to in-memory store and graph
            self.episodes[episode.uuid] = episode
            self.graph.add_node(episode.uuid, data=episode, type="MemoryEpisodeNode", summary=episode.topic_summary)

            # Add 'EXPERIENCED_BY' edge from the episode to the EgoNode
            if EGO_NODE_UUID in self.graph:
                self.graph.add_edge(episode.uuid, EGO_NODE_UUID,
                                    label=StructuralEdgeLabel.EXPERIENCED_BY)  # type: ignore
            else:
                logger.error(f"EgoNode {EGO_NODE_UUID} not found in graph. Cannot link episode {episode.uuid}.")
                # This should ideally not happen if _initialize_ego_node works correctly.

            logger.info(f"MemoryEpisodeNode {episode.uuid} created. Summary: {topic_summary}")
            return episode

        except Exception as e:
            logger.error(f"Failed to create memory episode: {e}", exc_info=True)
            return None

    def _extract_entities_from_text(self, text: str) -> Optional[ExtractedEntities]:
        """
        [REPLACEMENT] Extracts entities with a more detailed and guiding prompt.
        """
        logger.debug("Extracting entities from text with enhanced prompt...")
        try:
            # The prompt is now more descriptive, giving the LLM better examples of what to look for.
            prompt = (
                "Extract all named entities from the following text. "
                "Entities can be people (Alex), organizations (FinSecure, AWS), products or services (React, Kafka, S3), "
                "projects (Project Aurora), specific concepts (HIPAA compliance, RBAC, G1GC), dates, or other proper nouns. "
                "For each entity, provide its name and a general label (e.g., Person, Organization, Technology, Project, Concept, Date).\n\n"
                f"Text: {text}"
            )
            system_prompt = "You are an expert entity extraction system. Return entities in the specified JSON format, capturing technical terms and concepts as entities."
            extracted_data = self.llm_manager.call_tool(
                system_prompt=system_prompt,
                user_prompt=prompt,
                tool_model=ExtractedEntities
            )
            if isinstance(extracted_data, ExtractedEntities) and extracted_data.entities:
                logger.info(f"Extracted {len(extracted_data.entities)} entities.")
                return extracted_data
            else:
                logger.info("No entities extracted by LLM or result was not ExtractedEntities type.")
                return None
        except Exception as e:
            logger.error(f"Error during entity extraction: {e}", exc_info=True)
            return None


    def _find_existing_node(self, name: str, label: Optional[str] = None, threshold: float = 0.85) -> Optional[
        GraphNode]:
        """Tries to find an existing node by name and semantic similarity."""
        logger.debug(f"Attempting to find existing node for name: '{name}', label: '{label}'")
        # Direct name match (case-insensitive for robustness)
        for node in self.nodes.values():
            if node.name.lower() == name.lower():
                # If label is provided, check if it's compatible (e.g., new label is in existing or vice-versa)
                if label and label.lower() not in [l.lower() for l in node.labels]:
                    logger.debug(
                        f"Node '{name}' found by name, but label mismatch: existing '{node.labels}', new '{label}'. Considering it a potential match for LLM confirmation.")
                else:
                    logger.info(f"Found existing node by exact name match: {node.uuid} ({node.name})")
                    return node

        # Semantic similarity search
        if not EMBEDDING_DIM:  # Cannot do semantic search if embeddings are not configured
            logger.warning("EMBEDDING_DIM not set, skipping semantic search for node resolution.")
            return None

        try:
            query_embedding_text = f"{name} {label}" if label else name
            query_embedding = np.array(self.llm_manager.get_embedding(query_embedding_text))
        except Exception as e:
            logger.error(f"Could not generate embedding for query '{query_embedding_text}': {e}")
            return None

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
            logger.info(
                f"Found potential existing node by semantic similarity: {best_match.uuid} ({best_match.name}) with score {highest_similarity:.2f}")
            return best_match
        else:
            logger.debug(
                f"No sufficiently similar node found for '{name}' (highest score: {highest_similarity:.2f}, threshold: {threshold}).")
            return None

    def _resolve_entity_and_concept(self, entity_data: ExtractedEntity, current_episode_uuid: str) -> Optional[
        GraphNode]:
        logger.debug(f"Resolving entity: {entity_data.name} (Label: {entity_data.label})")

        # 1. Attempt to find an existing node
        existing_node = self._find_existing_node(entity_data.name, entity_data.label)

        # 2. If a strong candidate exists, use it.
        if existing_node:
            logger.info(f"Using existing node {existing_node.uuid} for '{entity_data.name}'.")
            return existing_node

        # 3. If no existing node, create a new one
        logger.info(f"No definitive existing node found for '{entity_data.name}'. Creating new node.")
        new_node = GraphNode(
            name=entity_data.name,
            labels=[entity_data.label] if entity_data.label else ["Unknown"],
            source_episode_uuid=current_episode_uuid,
            attributes={"original_extracted_label": entity_data.label}  # Store the initially extracted label
        )
        new_node.embedding = self.llm_manager.get_embedding(
            f"Entity: {new_node.name}. Labels: {', '.join(new_node.labels)}")

        self.nodes[new_node.uuid] = new_node
        self.graph.add_node(new_node.uuid, data=new_node, type="GraphNode", name=new_node.name)
        logger.info(f"Created new GraphNode {new_node.uuid} for '{new_node.name}'.")

        # 4. Classify the new node against the Concept Graph and link it
        self._classify_and_link_entity_to_concept(new_node)

        return new_node

    def _classify_and_link_entity_to_concept(self, node_to_classify: GraphNode):
        logger.debug(f"Classifying node '{node_to_classify.name}' ({node_to_classify.uuid}) against Concept Graph.")
        if not self.concepts:
            logger.warning("No concepts available for classification.")
            return

        concept_names = [c.name for c in self.concepts.values()]
        # Simple heuristic: if a label matches a concept name, use that.
        for label in node_to_classify.labels:
            matching_concepts = [c for c_uuid, c in self.concepts.items() if c.name.lower() == label.lower()]
            if matching_concepts:
                chosen_concept = matching_concepts[0]
                logger.info(f"Directly classified '{node_to_classify.name}' as '{chosen_concept.name}' based on label.")
                node_to_classify.concept_uuid = chosen_concept.uuid
                self.graph.add_edge(node_to_classify.uuid, chosen_concept.uuid,
                                    label=StructuralEdgeLabel.INSTANCE_OF)  # type: ignore
                # Update node in self.nodes to persist concept_uuid
                self.nodes[node_to_classify.uuid] = node_to_classify
                return

        # If no direct match, use LLM for classification
        prompt = f"Classify the entity '{node_to_classify.name}' (described by labels: {node_to_classify.labels}) into one of the following existing concepts. Provide the name of the most appropriate concept. If none seem right, you can suggest a new, more specific concept name. Available concepts: {', '.join(concept_names)}."
        system_prompt = "You are an expert entity classifier. Choose the best concept from the provided list, or propose a new one if necessary."

        try:
            classification_result = self.llm_manager.call_tool(
                system_prompt=system_prompt,
                user_prompt=prompt,
                tool_model=EntityClassifier
            )

            if isinstance(classification_result, EntityClassifier) and classification_result.most_likely_concept_name:
                target_concept_name = classification_result.most_likely_concept_name

                target_concept = self._get_or_create_concept_on_the_fly(target_concept_name)

                if target_concept:
                    node_to_classify.concept_uuid = target_concept.uuid
                    self.graph.add_edge(node_to_classify.uuid, target_concept.uuid,
                                        label=StructuralEdgeLabel.INSTANCE_OF)  # type: ignore
                    self.nodes[node_to_classify.uuid] = node_to_classify  # Persist change
                    logger.info(
                        f"LLM classified '{node_to_classify.name}' as '{target_concept.name}'. Reasoning: {classification_result.reasoning}")
                else:
                    logger.warning(
                        f"LLM suggested concept '{target_concept_name}' for '{node_to_classify.name}', but this concept was not found in the Concept Graph.")
            else:
                logger.warning(f"Could not classify entity '{node_to_classify.name}' using LLM or result was invalid.")

        except Exception as e:
            logger.error(f"Error during LLM entity classification for '{node_to_classify.name}': {e}", exc_info=True)


    def _extract_relationships_from_text(self, text: str, resolved_nodes: List[GraphNode]) -> Optional[
        ExtractedRelationships]:
        """
        [REPLACEMENT] Extracts relationships with a more flexible prompt that allows discovering new entities.
        """
        logger.debug("Extracting relationships from text with flexible prompt...")
        if not resolved_nodes:
            logger.info("No resolved nodes provided, cannot extract relationships between them.")
            return None

        entity_names = [node.name for node in resolved_nodes]
        # The prompt is relaxed to allow finding relationships involving entities that might have been missed in the first pass.
        prompt = (
            f"From the following text, extract relationships between entities. "
            f"Prioritize relationships involving these known entities: {', '.join(entity_names)}. "
            "However, if you find a clear relationship involving an entity NOT on this list, include it as well. "
            "For each relationship, identify the source entity name, target entity name, a descriptive verb phrase for the label "
            "(e.g., 'works_for', 'is_located_in', 'replaced_by', 'uses_service'), and the specific fact sentence.\n\n"
            f"Text: {text}"
        )
        system_prompt = "You are an expert relationship extraction system. Use the provided entity names but also be open to new ones. Return relationships in the specified JSON format."

        try:
            extracted_data = self.llm_manager.call_tool(
                system_prompt=system_prompt,
                user_prompt=prompt,
                tool_model=ExtractedRelationships
            )
            if isinstance(extracted_data, ExtractedRelationships) and extracted_data.relationships:
                logger.info(f"Extracted {len(extracted_data.relationships)} potential relationships.")
                return extracted_data
            else:
                logger.info("No relationships extracted by LLM or result was invalid.")
                return None
        except Exception as e:
            logger.error(f"Error during relationship extraction: {e}", exc_info=True)
            return None

    def _resolve_and_add_edge(self, rel_data: ExtractedRelationship, resolved_nodes_map: Dict[str, GraphNode],
                              current_episode_uuid: str):
        logger.debug(
            f"Resolving and adding edge for relationship: {rel_data.source_name} -({rel_data.label})-> {rel_data.target_name}")

        source_node = self._get_or_create_node_for_relationship(
            rel_data.source_name, resolved_nodes_map, current_episode_uuid
        )
        target_node = self._get_or_create_node_for_relationship(
            rel_data.target_name, resolved_nodes_map, current_episode_uuid
        )

        if not source_node or not target_node:
            logger.warning(
                f"Could not find source ('{rel_data.source_name}') or target ('{rel_data.target_name}') node for relationship. Skipping.")
            return

        # Contradiction/Duplicate Check
        # Find existing edges between these two nodes with a similar label
        existing_relevant_edges: List[GraphEdge] = []
        for u, v, key, data in self.graph.out_edges(source_node.uuid, data=True, keys=True):
            if v == target_node.uuid and isinstance(data.get('data'), GraphEdge):
                edge_obj: GraphEdge = data['data']
                # Simple label match for now; could use semantic similarity for labels too
                if edge_obj.label.lower() == rel_data.label.lower() and not edge_obj.invalid_at:
                    existing_relevant_edges.append(edge_obj)

        if existing_relevant_edges:
            existing_facts_texts = [edge.fact_text for edge in existing_relevant_edges]
            prompt = f"A new fact is proposed: '{rel_data.fact}' (meaning {rel_data.source_name} {rel_data.label} {rel_data.target_name}). Existing related facts are: {'; '.join(existing_facts_texts)}. Is the new fact contradictory to any existing fact? Is it a duplicate? Provide reasoning."
            system_prompt = "You are a contradiction and duplication detection expert."
            try:
                check_result = self.llm_manager.call_tool(
                    system_prompt=system_prompt,
                    user_prompt=prompt,
                    tool_model=ContradictionCheck
                )
                if isinstance(check_result, ContradictionCheck):
                    if check_result.is_duplicate:
                        logger.info(f"Fact '{rel_data.fact}' is a duplicate of an existing fact. Skipping addition.")
                        # Potentially update timestamp or certainty of existing fact if needed.
                        return
                    if check_result.is_contradictory:
                        logger.warning(f"Fact '{rel_data.fact}' is contradictory. Reasoning: {check_result.reasoning}")
                        # Invalidate the old edge(s)
                        # This mock assumes only one conflicting edge for simplicity
                        conflicting_edge_uuid_to_invalidate = check_result.conflicting_edge_uuid
                        if not conflicting_edge_uuid_to_invalidate and existing_relevant_edges:  # Fallback if LLM didn't specify
                            conflicting_edge_uuid_to_invalidate = existing_relevant_edges[0].uuid

                        if conflicting_edge_uuid_to_invalidate and conflicting_edge_uuid_to_invalidate in self.edges:
                            old_edge = self.edges[conflicting_edge_uuid_to_invalidate]
                            old_edge.invalid_at = datetime.now(timezone.utc)
                            self.edges[old_edge.uuid] = old_edge  # Update in-memory store
                            # Update in graph (NetworkX stores dicts, so update data attribute)
                            if self.graph.has_edge(old_edge.source_uuid, old_edge.target_uuid, key=old_edge.uuid):
                                self.graph.edges[old_edge.source_uuid, old_edge.target_uuid, old_edge.uuid][
                                    'data'] = old_edge
                            logger.info(f"Invalidated old edge {old_edge.uuid} due to contradiction.")
                        else:
                            logger.warning(
                                f"Contradiction detected, but could not identify or find specific edge '{conflicting_edge_uuid_to_invalidate}' to invalidate.")
                else:
                    logger.warning("Contradiction check LLM call failed to return valid ContradictionCheck object.")
            except Exception as e:
                logger.error(f"Error during contradiction check for fact '{rel_data.fact}': {e}", exc_info=True)

        # Create and add the new edge
        new_edge = GraphEdge(
            source_uuid=source_node.uuid,
            target_uuid=target_node.uuid,
            label=rel_data.label,
            fact_text=rel_data.fact,
            source_episode_uuid=current_episode_uuid
        )
        source_name = self.nodes.get(new_edge.source_uuid,
                                     GraphNode(name="Unknown", labels=[], source_episode_uuid="N/A")).name
        target_name = self.nodes.get(new_edge.target_uuid,
                                     GraphNode(name="Unknown", labels=[], source_episode_uuid="N/A")).name
        new_edge.embedding = self.llm_manager.get_embedding(
            f"Fact: {new_edge.fact_text}. Connects: {source_name} to {target_name}.")

        self.edges[new_edge.uuid] = new_edge
        self.graph.add_edge(source_node.uuid, target_node.uuid, key=new_edge.uuid, label=new_edge.label, data=new_edge,
                            type="GraphEdge")
        logger.info(f"Added new GraphEdge {new_edge.uuid}: '{new_edge.fact_text}'.")

    def _extract_keywords_spacy(self, text: str) -> Set[str]:
        """Uses spaCy to extract named entities, nouns, and proper nouns as keywords."""
        doc = self.nlp(text)
        keywords = set()
        # Add named entities
        for ent in doc.ents:
            keywords.add(ent.text.lower())
        # Add nouns and proper nouns
        for token in doc:
            if token.pos_ in ('NOUN', 'PROPN'):
                keywords.add(token.lemma_.lower())
        logger.debug(f"Extracted keywords: {keywords}")
        return keywords

    def _extract_keywords(self, text: str) -> Set[str]:
        """
        [REPLACEMENT - NO SPACY]
        Extracts keywords using rule-based heuristics:
        1. Finds all-caps words (e.g., "AWS", "HIPAA").
        2. Finds capitalized words (potential proper nouns).
        3. Takes all other words and filters them against a stop-word list.
        """
        if not hasattr(self, 'STOP_WORDS'):  # Lazily define to avoid import-time issues if moved
            self.STOP_WORDS = STOP_WORDS

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
            if cap_word.lower() not in self.STOP_WORDS:
                keywords.add(cap_word.lower())

        # 2. General keyword extraction (case-insensitive)
        # Remove punctuation and normalize
        normalized_text = re.sub(r'[^\w\s]', '', text.lower())
        tokens = normalized_text.split()

        for token in tokens:
            if token.strip() and token not in self.STOP_WORDS and not token.isdigit():
                keywords.add(token)

        logger.debug(f"Extracted keywords (spaCy-free): {keywords}")
        return keywords

    def _search_by_keywords(self, keywords: Set[str]) -> List[Union[GraphNode, GraphEdge, MemoryEpisodeNode]]:
        """Performs a case-insensitive keyword search across nodes, edges, and episodes."""
        found_items = []
        for node in self.nodes.values():
            if any(re.search(r'\b' + re.escape(kw) + r'\b', node.name, re.IGNORECASE) for kw in keywords):
                found_items.append(node)
        for edge in self.edges.values():
            if not edge.invalid_at and any(
                    re.search(r'\b' + re.escape(kw) + r'\b', edge.fact_text, re.IGNORECASE) for kw in keywords):
                found_items.append(edge)
        for episode in self.episodes.values():
            if any(re.search(r'\b' + re.escape(kw) + r'\b', episode.topic_summary, re.IGNORECASE) for kw in keywords):
                found_items.append(episode)
        logger.info(f"Found {len(found_items)} items via keyword search.")
        return found_items

    def _rephrase_question(self, question: str) -> List[str]:
        """Uses LLM to generate alternative phrasings of a question."""
        logger.info(f"Rephrasing question: '{question}'")
        system_prompt = "You are a search query expansion expert. Your task is to rephrase a user's question in 3 different ways to maximize the chance of finding relevant information in a semantic search system. The rephrasings should be diverse."
        user_prompt = f"Original question: '{question}'"
        try:
            result = self.llm_manager.call_tool(system_prompt, user_prompt, RephrasedQuestions)
            if isinstance(result, RephrasedQuestions) and result.questions:
                logger.info(f"Generated rephrasings: {result.questions}")
                return result.questions
        except Exception as e:
            logger.error(f"Failed to rephrase question: {e}", exc_info=True)
        return []

    def _prune_and_rerank_context(self, question: str,
                                  candidate_items: List[Union[GraphNode, GraphEdge, MemoryEpisodeNode]]) -> List[
        Union[GraphNode, GraphEdge, MemoryEpisodeNode]]:
        """Uses an LLM to prune a list of candidate items down to the most relevant ones."""
        if not candidate_items:
            return []
        logger.info(f"Pruning {len(candidate_items)} candidate items for relevance to question: '{question}'")

        # Create a mapping from a unique identifier (UUID) to the item
        item_map = {item.uuid: item for item in candidate_items}

        context_block = []
        for uuid_val, item in item_map.items():
            text = ""
            if isinstance(item, GraphNode):
                text = f"Entity: '{item.name}' (Type: {', '.join(item.labels)})"
            elif isinstance(item, GraphEdge):
                text = f"Fact: '{item.fact_text}'"
            elif isinstance(item, MemoryEpisodeNode):
                text = f"Memory Summary: '{item.topic_summary}'"
            context_block.append(f"Identifier: {uuid_val}\nContent: {text}\n---")

        nl = '\n'
        system_prompt = "You are a relevance ranking assistant. Your job is to analyze a list of retrieved facts and determine which ones are truly relevant to answering the user's original question. Return only the identifiers of the relevant items."
        user_prompt = (
            f"Original Question: \"{question}\"\n\n"
            "Review the following items and identify which ones are directly relevant to answering the question. "
            "Return a list of their identifiers.\n\n"
            f"{nl.join(context_block)}"
        )

        try:
            result = self.llm_manager.call_tool(system_prompt, user_prompt, RelevantItems)
            if isinstance(result, RelevantItems):
                pruned_uuids = set(result.relevant_item_identifiers)
                pruned_items = [item_map[uuid_val] for uuid_val in pruned_uuids if uuid_val in item_map]
                logger.info(f"Pruned context from {len(candidate_items)} to {len(pruned_items)} items.")
                return pruned_items
        except Exception as e:
            logger.error(f"LLM-based pruning failed: {e}", exc_info=True)

        # Fallback: if LLM pruning fails, return the original candidates to not lose all context
        logger.warning("LLM pruning failed. Returning all candidate items.")
        return candidate_items

    # --- Main Retrieval Orchestrator ---
    def _find_relevant_items(self, question: str, top_k: int = 10, similarity_threshold: float = 0.7) -> List[
        Tuple[float, Union[GraphNode, GraphEdge, MemoryEpisodeNode]]]:
        """
        [NEW - Previously Missing]
        Performs a pure semantic search across all memory items (nodes, edges, episodes).
        It retrieves items whose embeddings are most similar to the question's embedding.
        """
        logger.debug(f"Finding relevant items via semantic search for: '{question}'")

        # Check if embeddings are usable
        if not EMBEDDING_DIM:
            logger.warning("EMBEDDING_DIM not set, cannot perform semantic search.")
            return []

        try:
            question_embedding = np.array(self.llm_manager.get_embedding(question))
        except Exception as e:
            logger.error(f"Could not generate embedding for question '{question}': {e}")
            return []

        scored_items: List[Tuple[float, Union[GraphNode, GraphEdge, MemoryEpisodeNode]]] = []

        # Combine all searchable items into one list
        all_items = list(self.nodes.values()) + list(self.edges.values()) + list(self.episodes.values())

        for item in all_items:
            # Skip edges that have been invalidated
            if isinstance(item, GraphEdge) and item.invalid_at:
                continue

            if hasattr(item, 'embedding') and item.embedding and len(item.embedding) == EMBEDDING_DIM:
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

    def _trace_items_to_episodes(self, items: List[Union[GraphNode, GraphEdge, MemoryEpisodeNode]]) -> Dict[str, List[Union[GraphNode, GraphEdge]]]:
        """
        [NEW - Previously Missing]
        Takes a list of relevant items (nodes, edges, or episodes) and groups them by the
        MemoryEpisode in which they were learned. This is a key step for building a
        narrative context for recall.

        If a MemoryEpisodeNode itself is in the list of items, its UUID is added as a key
        to the map, ensuring its summary is included in the final narrative context even if no
        specific facts from that episode were individually returned by the pruner.

        Args:
            items: A list of relevant GraphNode, GraphEdge, or MemoryEpisodeNode objects.

        Returns:
            A dictionary where keys are episode UUIDs and values are lists of the
            facts (GraphNodes or GraphEdges) learned in that episode.
        """
        logger.debug(f"Tracing {len(items)} items to their source episodes.")
        episodes_map = defaultdict(list)

        for item in items:
            if isinstance(item, MemoryEpisodeNode):
                # If an episode itself is deemed relevant, ensure its UUID is a key in the map.
                # This guarantees it will be included in the narrative construction step.
                # We don't append the episode to the list of facts; its presence as a key is enough.
                _ = episodes_map[item.uuid]

            elif isinstance(item, (GraphNode, GraphEdge)):
                # For a specific fact (node or edge), find its source episode.
                if hasattr(item, 'source_episode_uuid') and item.source_episode_uuid:
                    if item.source_episode_uuid in self.episodes:
                        episodes_map[item.source_episode_uuid].append(item)
                    else:
                        # This is a data integrity check.
                        logger.warning(
                            f"Fact '{getattr(item, 'name', getattr(item, 'fact_text', item.uuid))}' has source episode UUID "
                            f"'{item.source_episode_uuid}' which was not found in the episode list."
                        )
                else:
                    logger.warning(
                        f"Fact '{getattr(item, 'name', getattr(item, 'fact_text', item.uuid))}' is missing a valid source_episode_uuid."
                    )

        logger.info(f"Relevant items were traced to {len(episodes_map)} unique episodes.")
        return episodes_map

    def answer_question(self, question: str, top_k_facts: int = 10, episode_date_format: str = "%B %d, %Y") -> str:
        """
        [REPLACEMENT] Answers a question using a multi-step, robust retrieval process.
        """
        logger.info(f"Answering question with multi-step retrieval: '{question}'")
        if question is not None:
            # 1. Hybrid Search: Combine vector and keyword search for a broad initial context.
            vector_items_with_scores = self._find_relevant_items(question, top_k=top_k_facts)
            vector_items = [item for score, item in vector_items_with_scores]

            # This line is the only change needed here. It calls our new spaCy-free function.
            keywords = self._extract_keywords(question)
            keyword_items = self._search_by_keywords(keywords)

            # Combine and deduplicate
            candidate_items_map = {item.uuid: item for item in vector_items + keyword_items}

            # 2. Rephrase and Retry if necessary
            if not candidate_items_map:
                logger.info("Initial search yielded no results. Rephrasing question...")
                rephrased_qs = self._rephrase_question(question)
                for q_variant in rephrased_qs:
                    vector_results = self._find_relevant_items(q_variant, top_k=top_k_facts // 2)
                    for _, item in vector_results:
                        candidate_items_map[item.uuid] = item  # Add to map
                    keyword_results = self._search_by_keywords(self._extract_keywords(q_variant))
                    for item in keyword_results:
                        candidate_items_map[item.uuid] = item

            candidate_items = list(candidate_items_map.values())

            if not candidate_items:
                logger.info("No relevant items found even after rephrasing.")
                return "I don't have enough information in my personal memory to answer that question."

            # 3. Prune and Re-rank the context using an LLM for relevance.
            pruned_items = self._prune_and_rerank_context(question, candidate_items)

            if not pruned_items:
                logger.info("After pruning, no items were deemed relevant.")
                return "I found some related topics, but nothing that directly answers your question."

            # 4. Construct Narrative from the high-quality, pruned context.
            episodes_map = self._trace_items_to_episodes(pruned_items)
            if not episodes_map:
                logger.info("Could not trace pruned items to any specific episodes.")
                return "I found some related information, but I can't recall the specific experiences where I learned it."

            narrative_context = self._construct_narrative_context(episodes_map, episode_date_format)
            if not narrative_context:
                logger.info("Failed to construct narrative context from episodes.")
                return "I found some relevant memories, but I'm having trouble putting them into words right now."

            # 5. Synthesize Final Answer using the rich, relevant narrative.
            final_answer = self._synthesize_final_answer(question, narrative_context)
            logger.info(f"Generated answer: '{final_answer[:100]}...'")
            return final_answer

    def _find_relevant_facts(self, question: str, top_k: int = 5, similarity_threshold: float = 0.7) -> List[
        Tuple[float, Union[GraphNode, GraphEdge]]]:
        logger.debug(f"Finding relevant facts for question: '{question}'")
        if not EMBEDDING_DIM:
            logger.warning("EMBEDDING_DIM not set, cannot perform semantic search.")
            return []
        try:
            question_embedding = np.array(self.llm_manager.get_embedding(question))
        except Exception as e:
            logger.error(f"Could not generate embedding for question '{question}': {e}")
            return []

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

    def _trace_facts_to_episodes(self, facts: List[Union[GraphNode, GraphEdge]]) -> Dict[
        str, List[Union[GraphNode, GraphEdge]]]:
        logger.debug(f"Tracing {len(facts)} facts to their source episodes.")
        episodes_map = defaultdict(list)
        for fact in facts:
            if hasattr(fact,
                       'source_episode_uuid') and fact.source_episode_uuid and fact.source_episode_uuid in self.episodes:
                episodes_map[fact.source_episode_uuid].append(fact)
            else:
                logger.warning(
                    f"Fact '{getattr(fact, 'name', getattr(fact, 'fact_text', fact.uuid))}' has no valid source_episode_uuid or episode not found.")
        logger.info(f"Facts traced to {len(episodes_map)} unique episodes.")
        return episodes_map

    def _construct_narrative_context(self, episodes_map: Dict[str, List[Union[GraphNode, GraphEdge]]],
                                     date_format: str) -> str:
        logger.debug("Constructing narrative context from episodes map.")
        context_parts = []
        # Sort episodes by timestamp, most recent first, for a more natural recall flow
        sorted_episode_uuids = sorted(
            episodes_map.keys(),
            key=lambda uuid_val: self.episodes[uuid_val].timestamp,
            reverse=True
        )

        for episode_uuid in sorted_episode_uuids:
            episode = self.episodes.get(episode_uuid)
            facts_in_episode = episodes_map[episode_uuid]

            if not episode:
                logger.warning(f"Episode {episode_uuid} not found while constructing narrative. Skipping.")
                continue

            try:
                date_str = episode.timestamp.strftime(date_format)
            except ValueError:  # Handle potential strftime errors with bad format
                date_str = episode.timestamp.isoformat()

            topic = episode.topic_summary
            emotion = episode.internal_state.emotional_label
            cognitive_process = episode.internal_state.cognitive_process  # e.g. "learning new information"

            learned_items_str_parts = []
            for fact in facts_in_episode:
                if isinstance(fact, GraphEdge):
                    source_name = self.nodes.get(fact.source_uuid,
                                                 GraphNode(name="Something", labels=[], source_episode_uuid="N/A")).name
                    target_name = self.nodes.get(fact.target_uuid, GraphNode(name="something else", labels=[],
                                                                             source_episode_uuid="N/A")).name
                    learned_items_str_parts.append(f"that {fact.fact_text} (connecting {source_name} to {target_name})")
                elif isinstance(fact, GraphNode):
                    learned_items_str_parts.append(f"about '{fact.name}' (which is a {', '.join(fact.labels)})")

            if not learned_items_str_parts:
                learned_summary = "something I was reflecting on"
            else:
                learned_summary = ", and ".join(learned_items_str_parts)

            narrative = (
                f"On {date_str}, I remember an experience related to '{topic}'. "
                f"During that time, I was primarily {cognitive_process} and I recall feeling {emotion}. "
                f"Specifically, I learned {learned_summary}."
            )
            context_parts.append(narrative)

        if not context_parts:
            logger.info("No narrative parts constructed.")
            return ""

        full_narrative = "\n\n".join(context_parts)
        logger.info(
            f"Narrative context constructed (length: {len(full_narrative)} chars). Preview: {full_narrative[:200]}...")
        return full_narrative

    def _synthesize_final_answer(self, question: str, narrative_context: str) -> str:
        logger.debug("Synthesizing final answer using LLM with narrative context.")
        system_prompt = (
            "You are an AI assistant with a personal memory. "
            "Answer the user's question based *only* on the following first-person memories you have. "
            "If the memories don't directly answer the question, say you can't recall the specific detail from those experiences. "
            "Speak naturally in the first person (e.g., 'I remember', 'I learned')."
        )
        user_prompt = (
            f"Here are your relevant personal memories:\n"
            f"---\n"
            f"{narrative_context}\n"
            f"---\n\n"
            f"Based on these memories, please answer this question: {question}"
        )

        try:
            answer = self.llm_manager.get_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
            return answer
        except Exception as e:
            logger.error(f"Error during final answer synthesis with LLM: {e}", exc_info=True)
            return "I had trouble formulating a response based on my memories."

    def close_db(self):
        """Closes the database connection."""
        if self.db_conn:
            self.db_conn.close()
            logger.info("Database connection closed.")

    def integrate_text(self, text, group_id):
        self.integrate_experience(text, text)


# --- NEW TEST SUITE ---
def run_cognitive_test_scenario():
    logger.info("--- Starting Cognitive Test Scenario ---")
    DB_FILE = "test_cognitive_memory.db"
    # Cleanup previous test DB if it exists
    #if os.path.exists(DB_FILE):
    #    os.remove(DB_FILE)
    #    logger.info(f"Removed old test database: {DB_FILE}")

    # 1. Setup: Initialize logging, LLM manager, and the CognitiveMemoryManager.
    # For testing, we use the mock SmallLlmManagerProxy defined earlier.
    mock_llm_manager = SmallLlmManagerProxy(model_name="mock_test_model")
    manager = CognitiveMemoryManager(llm_manager=mock_llm_manager, db_path=DB_FILE)
    graph_manager = manager

    # Assert initial state
    assert manager.ego_node_model is not None, "EgoNode should be initialized."
    assert EGO_NODE_UUID in manager.nodes, "EgoNode UUID should be in manager.nodes"
    assert EGO_NODE_UUID in manager.graph, "EgoNode should be in the graph."
    assert len(manager.concepts) > 5, "Core concepts should be bootstrapped." # Check for a few concepts
    assert "animacy" in manager.semantic_axes, "Semantic axes should be computed."
    initial_episode_count = len(manager.episodes) # Should be 1 (birth episode)
    assert initial_episode_count >= 1, "Should have at least one initial (birth) episode."

    if not os.path.exists(DB_FILE):
    # ───── Cluster 1: Greetings and Session Setup ─────
        text_cluster_1 = """
        AI: Hello! I'm Chatty, a conversational AI built to assist you with any questions or tasks you have today. It's nice to meet you.
        User: Hi Chatty! Nice to meet you too. I'm Alex, a software developer working on a new project. How's your performance today?
        AI: I'm running at peak efficiency. Thank you for asking, Alex. I'm ready to help you brainstorm, troubleshoot code, plan events, or just chat.
        User: Wonderful, I could use some brainstorming help. But first, I'd like to share a bit about my current project scope.
        AI: Absolutely, please go ahead and describe your project. I'm listening.
        User: Before we dive into specifics, could you tell me what capabilities you have? For instance, can you recall our past conversations or suggestions?
        AI: I can integrate information you provide, remember details within our session, and use that to answer follow-up questions, diagnose issues, and offer personalized recommendations.
        User: That's impressive. Do you also have knowledge of various programming languages and frameworks?
        AI: Yes, I have familiarity with common languages like Python, JavaScript, Java, C++, and frameworks such as React, Django, and TensorFlow.
        User: Perfect. Let's start with an overview of my project timeline and milestones.
        AI: Sure thing. I'll keep track of our discussion and recall any details when you need them.
        """
        graph_manager.integrate_text(text_cluster_1, group_id="story_test")

        # ───── Cluster 2: Personal and Professional Background ─────
        text_cluster_2 = """
        User: So, Chatty, here's some background: I graduated with a degree in computer science five years ago. Since then, I've worked mostly in backend systems.
        AI: Understood. Do you have specific industries you've been involved with, such as finance, healthcare, or e-commerce?
        User: Yes, I spent two years at FinSecure, building microservices for secure transactions. After that, I joined HealthNet to develop APIs for patient data management.
        AI: I see. That gives me context on your experience with security and compliance. How has that shaped your current project requirements?
        User: My new undertaking, Project Aurora, requires both robust security measures and high throughput, combining my previous experiences.
        AI: Excellent. Are there particular compliance standards you need to adhere to, like HIPAA or PCI DSS?
        User: Definitely HIPAA compliance is crucial, along with encryption at rest and in transit, and detailed audit logging.
        AI: Got it. Any preferences for technology stacks or cloud providers given your history?
        User: I've primarily used AWS with services like Lambda, API Gateway, DynamoDB, and S3, but I'm open to alternatives if they offer better cost efficiency.
        AI: That helps paint a clear picture of your background and priorities. Ready to proceed to feature brainstorming.
        """
        graph_manager.integrate_text(text_cluster_2, group_id="story_test")

        # ───── Cluster 3: Project Scope and Data Flow ─────
        text_cluster_3 = """
        User: Let's dive into Project Aurora. The goal is to build a real-time analytics dashboard for healthcare metrics.
        AI: What specific metrics are you tracking, and what sources feed into the dashboard?
        User: We're aggregating patient vital stats, appointment rates, and medication adherence from several hospital systems via secure APIs.
        AI: For real-time analytics, do you plan to use streaming platforms like Kafka or AWS Kinesis?
        User: I'm considering Kafka for its mature ecosystem, but Kinesis could simplify integration with AWS Lambda functions.
        AI: Both are viable. Kafka offers greater flexibility, whereas Kinesis integrates seamlessly with other AWS services. Would you like a detailed pros and cons list?
        User: Yes, please list the pros and cons of using Kafka versus Kinesis for our use case.
        AI: Sure. Kafka pros: high throughput, fault tolerance, rich ecosystem; cons: self-managed infrastructure, operational overhead. Kinesis pros: fully managed, auto-scaling; cons: vendor lock-in, cost unpredictability.
        User: That analysis is helpful. Let's consider cost modeling next, including data retention and scaling costs.
        AI: Understood. I'll keep track of cost comparison metrics as we proceed.
        """
        graph_manager.integrate_text(text_cluster_3, group_id="story_test")

        # ───── Cluster 4: Feature Brainstorming ─────
        text_cluster_4 = """
        User: For the dashboard features, I want user authentication, role-based access, customizable views, and alerting.
        AI: Do you need multi-tenant isolation for different hospitals or departments?
        User: Yes, each hospital should only see its own data, but admins can oversee multiple tenants.
        AI: Role-based access can map to hospital IDs and department IDs. Would you like recommendations on implementing RBAC?
        User: Absolutely, outline a simple RBAC model for our microservices architecture.
        AI: We can define roles like admin, clinician, and auditor. Each role maps to permissions such as readMetrics, configureAlerts, and manageUsers.
        User: Should permissions be stored in a central identity service or within each microservice?
        AI: Best practice is to have a centralized identity provider like AWS Cognito or Auth0. Microservices validate JWT tokens issued by the provider.
        User: That makes sense. Let's discuss alert thresholds and notification channels next.
        AI: Sure. What metrics require alerting, and what severity levels are you considering?
        """
        graph_manager.integrate_text(text_cluster_4, group_id="story_test")

        # ───── Cluster 5: Troubleshooting Performance ─────
        text_cluster_5 = """
        User: I'm encountering a latency spike when ingesting data from our Kafka cluster into the aggregation service.
        AI: What is the current throughput, and how many partitions are you using for the topic?
        User: We're pushing fifteen thousand events per second with four partitions. Latency jumps above two seconds intermittently.
        AI: Two seconds is high. Have you monitored consumer lag or GC pauses in your Java application?
        User: The consumer lag rarely exceeds a few hundred messages. GC logs show occasional full GC lasting close to 500 milliseconds.
        AI: Full GC of 500 milliseconds every few seconds could cause spikes. Consider tuning the JVM flags or switching to G1GC.
        User: I’m using CMS by default. Should I configure G1GC with specific heap region sizes?
        AI: Yes. Try `-XX:+UseG1GC` with a `MaxGCPauseMillis` target of `200ms` and adjust `-XX:MaxGCPauseMillis=200` accordingly.
        User: I'll test those settings and increase partitions to six to distribute load more evenly.
        AI: Good plan. Monitor after changes and let me know if latency improves or if new issues arise.
        """
        graph_manager.integrate_text(text_cluster_5, group_id="story_test")

        # ───── Cluster 6: Memory and Preferences Recall ─────
        text_cluster_6 = """
        User: Earlier, I mentioned my AWS preference. Can you recall what AWS services I commonly use?
        AI: You specified AWS Lambda, API Gateway, DynamoDB, S3, and CloudWatch for monitoring.
        User: Correct. Also, I use CloudWatch logs streamed via Kinesis Firehose into S3, then query with Athena.
        AI: Right. Do you plan to automate Athena queries for regular reporting?
        User: Yes, I’ll schedule them daily and send summaries via email.
        AI: And for compliance, you mentioned using AWS Config and strict IAM roles?
        User: Exactly. IAM roles for least privilege and AWS Config rules for automated audits.
        AI: Noted. I'll remember all your AWS preferences, data pipelines, and compliance settings.
        """
        graph_manager.integrate_text(text_cluster_6, group_id="story_test")

        # ───── Cluster 7: Contradictory Update ─────
        text_cluster_7 = """
        User: Actually, we decided to switch from AWS to Google Cloud Platform for cost and performance benefits.
        AI: Understood. Which GCP services will you use instead of AWS Lambda and API Gateway?
        User: We'll use Cloud Functions for serverless, Cloud Run for container endpoints, and Cloud Pub/Sub for messaging.
        AI: And BigQuery in place of Athena, plus Cloud Storage instead of S3?
        User: Yes, exactly. BigQuery for analytics, Cloud Storage for object data, and Stackdriver for monitoring.
        AI: That’s a significant shift. Shall I remove AWS configurations from memory and overwrite with GCP choices?
        User: Yes, overwrite the AWS choices and disregard prior cost estimates.
        AI: Done. I'll remember Cloud Functions, Cloud Run, Pub/Sub, BigQuery, Cloud Storage, and Stackdriver.
        User: Perfect. Moving forward, use GCP service names and pricing for any cost analysis.
        """
        graph_manager.integrate_text(text_cluster_7, group_id="story_test")

        # ───── Cluster 8: UI/UX Preferences ─────
        text_cluster_8 = """
        User: Let's talk about user interface preferences for the dashboard.
        AI: Do you prefer a web-based single-page application or a desktop client?
        User: A web SPA is better for accessibility. I like React for the frontend framework.
        AI: Any CSS framework preferences, like Material UI or Bootstrap?
        User: I prefer Material UI for its design consistency and built-in theming.
        AI: And charting libraries for real-time graphs? D3.js, Chart.js, or Recharts?
        User: Recharts aligns well with React’s component model and provides the interactivity we need.
        AI: Should dark mode be supported by default, with a toggle for users?
        User: Yes, default to system preference and allow manual override for dark/light themes.
        """
        graph_manager.integrate_text(text_cluster_8, group_id="story_test")

        # ───── Cluster 9: Ethical and Philosophical Reflection ─────
        text_cluster_9 = """
        User: I’d like your thoughts on the ethics of AI assistance in healthcare applications.
        AI: Ethical AI in healthcare involves fairness, transparency, patient privacy, and accountability.
        User: How can we ensure fairness when algorithms influence treatment recommendations?
        AI: Use diverse training data representative of patient demographics and continuously audit model outcomes.
        User: For transparency, should we implement explainable AI techniques?
        AI: Yes, techniques like LIME or SHAP can provide feature-level explanations for model predictions.
        User: How do we protect patient privacy while using their data to train models?
        AI: Implement strong de-identification, encryption at rest and in transit, and strict access controls with audit logging.
        User: That covers key aspects. Let’s document these ethical guidelines within our project spec.
        """
        graph_manager.integrate_text(text_cluster_9, group_id="story_test")

        # ───── Cluster 10: Timeline and Milestones ─────
        text_cluster_10 = """
        User: Thanks for all the help, Chatty. What are the next steps summary?
        AI: Summarizing: GCP stack chosen, UI with React/Material UI/Recharts, ethical guidelines outlined, monitoring strategies set.
        User: Exactly. Now, can you outline the milestones for the next eight weeks?
        AI: Week 1: Provision GCP environment, configure IAM and basic networking. Week 2: Set up Pub/Sub topics and Cloud Functions. Week 3: Design BigQuery schemas and ETL pipelines. Week 4: Scaffold React frontend with Material UI components. Week 5: Integrate frontend with backend via Cloud Run endpoints. Week 6: Develop alerting and notification system using Cloud Monitoring. Week 7: Conduct performance/load testing and optimize. Week 8: Finalize documentation, ethics review, and plan rollout.
        User: Perfect timeline. Please store these milestones and remind me when we reach each week’s start.
        AI: Milestones stored. I’ll notify you at the start of each week about your next tasks.
        User: Great. Looking forward to our progress together.
        """
        graph_manager.integrate_text(text_cluster_10, group_id="story_test")

        # ───── Cluster 11: Scheduling and Reminders ─────
        text_cluster_11 = """
        User: Can you remind me about each milestone on Mondays at 9 AM?
        AI: Certainly. I will schedule reminders for Week 1 through Week 8 milestones at 9 AM each Monday.
        User: Also, send me a weekly summary every Friday afternoon.
        AI: Understood. A weekly summary every Friday at 3 PM will be set up.
        User: And if any system monitoring alerts trigger, notify me immediately via email.
        AI: I'll configure immediate notifications for critical alerts alongside your weekly reminders.
        User: One more: remind me to review the ethical guidelines document at the end of this month.
        AI: Reminder scheduled for the last day of this month to review the ethics document.
        User: Thank you. That covers all my scheduling needs for now.
        """
        graph_manager.integrate_text(text_cluster_11, group_id="story_test")

        # ───── Cluster 12: Reflection and Farewell ─────
        text_cluster_12 = """
        AI: It’s been great helping you shape Project Aurora, Alex. Do you have any final questions before we wrap up?
        User: Just one: how would you describe our collaboration so far?
        AI: Our collaboration has been structured, efficient, and adaptive. You provided clear goals, and I responded with detailed analyses, troubleshooting advice, and scheduling support.
        User: I appreciate your thoroughness. I feel confident moving forward.
        AI: I'm glad to hear that. Remember, you can revisit any of these details anytime, and I'll recall them accurately.
        User: Thanks, Chatty. Talk to you tomorrow when the Week 1 milestone reminder arrives.
        AI: I’ll be here and ready. Have a productive day, Alex!
        User: You too. Goodbye for now.
        AI: Goodbye, Alex. Looking forward to our next session.
        """
        graph_manager.integrate_text(text_cluster_12, group_id="story_test")

        print("Extended dialogue story test integration complete.")
    else:
        graph_manager.load_from_db()

    # Assuming `manager` is your GraphMemoryManager instance
    qa_expected = [
        # Cluster 1
        ("What name does the AI introduce itself with?", "Chatty"),
        ("How does the user introduce themself?", "Alex"),
        ("Which capabilities does the AI list when the user asks?", "integrate"),
        ("How does the AI describe its ability to recall past conversation details?", "remember"),
        ("What programming domains does the AI claim familiarity with?", "Python"),

        # Cluster 2
        ("How many years ago did the user graduate?", "five"),
        ("Which two previous employers does the user mention?", "FinSecure"),
        ("What compliance standard is crucial for the user’s new project?", "HIPAA"),
        ("When asked about cloud preferences, which provider does the user say they’ve primarily used?", "AWS"),
        ("Which services on AWS does the user list as their go-to?", "Lambda"),

        # Cluster 3
        ("What is the name of the user’s project?", "Aurora"),
        ("Which two streaming platforms does the AI compare for real-time analytics?", "Kafka"),
        ("What pros and cons does the AI give for Kafka?", "throughput"),
        ("What pros and cons does the AI give for Kinesis?", "managed"),
        ("What cost-modeling factors does the AI say it will track next?", "retention"),

        # Cluster 4
        ("Name three key dashboard features the user requests.", "authentication"),
        ("What roles are defined in the simple RBAC model?", "admin"),
        ("Which identity provider options does the AI recommend?", "Cognito"),
        ("What notification channels are discussed?", "alerting"),
        ("How does the AI plan to remember the user’s RBAC preferences?", "remember"),

        # Cluster 5
        ("What throughput and partition count does the user report?", "partitions"),
        ("Which garbage collector is the user using by default?", "CMS"),
        ("What JVM flag does the AI suggest to switch to G1GC?", "-XX:+UseG1GC"),
        ("What `MaxGCPauseMillis` target does the AI recommend?", "200ms"),
        ("After the tuning suggestions, what two actions does the user plan to take?", "increase"),

        # Cluster 6
        ("Which AWS service for monitoring does the AI recall?", "CloudWatch"),
        ("How does the user stream CloudWatch logs into S3?", "Kinesis Firehose"),
        ("What tool does the user plan to use to query S3 data?", "Athena"),
        ("What compliance tools does the user mention alongside IAM roles?", "Config"),
        ("What schedule does the user set for Athena queries?", "daily"),

        # Cluster 7
        ("To which cloud provider does the user switch?", "Google"),
        ("Which GCP service replaces AWS Lambda?", "Cloud Functions"),
        ("What does the AI ask before overwriting memory?", "remove"),
        ("What new GCP monitoring service does the user choose?", "Stackdriver"),
        ("After overwriting, what six GCP services does the AI now remember?", "Pub/Sub"),

        # Cluster 8
        ("Which frontend framework does the user prefer?", "React"),
        ("Which component library does the AI note for theming?", "Material UI"),
        ("What charting library does the user select?", "Recharts"),
        ("How should the theme default be determined?", "system"),
        ("How will the AI recall the dark-mode preference?", "default"),

        # Cluster 9
        ("Name two explainable AI techniques the AI mentions.", "LIME"),
        ("Which four ethical principles does the AI list?", "transparency"),
        ("What method does the AI recommend for de-identification?", "de-identification"),
        ("Where does the AI say it will document these guidelines?", "project spec"),

        # Cluster 10
        ("What is the Week 3 milestone?", "schemas"),
        ("Which week is dedicated to performance/load testing?", "load"),
        ("What tasks are planned for Week 5?", "frontend"),
        ("How many total weeks are in the timeline?", "8"),
        ("Which milestone does the AI promise to remind at the start of each week?", "tasks"),

        # Cluster 11
        ("On which day and time should the weekly summary go out?", "Friday"),
        ("When should milestone reminders fire each week?", "Monday"),
        ("What triggers immediate notifications?", "critical"),
        ("When is the ethics-document review scheduled?", "last"),
        ("How many distinct reminder schedules did the user request?", "reminder"),

        # Cluster 12
        ("How does the AI describe the collaboration?", "structured"),
        ("When will Alex next hear from Chatty?", "tomorrow"),
        ("What phrase does the AI use to express readiness for the next session?", "ready"),
        ("What final well-wish does the AI give Alex?", "productive"),
    ]

    for question, expected in qa_expected:
        answer_str = manager.answer_question(question)
        print(f"Q: {question}")
        print(f"A: {answer_str}")
        if expected.lower() in answer_str.lower():
            print(f"✅ Found '{expected}' in answer_str.\n")
        else:
            print(f"❌ '{expected}' not found in answer_str.\n")


if __name__ == '__main__':
    run_cognitive_test_scenario()