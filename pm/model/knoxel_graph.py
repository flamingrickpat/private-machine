from datetime import datetime
import logging
from typing import Dict, Any
from typing import List
from typing import (
    Optional,
)

from pydantic import Field

from pm.model.knoxel_core import KnoxelBase

logger = logging.getLogger(__name__)


class GraphNode(KnoxelBase):
    """
    Graph-memory entity node derived from consolidated memories and fact extraction.

    Represents a durable concept instance (person/place/object/topic) with typed labels,
    normalized name, and optional attributes. ``source_memory_id`` ties the node to the
    memory episode that most strongly defined or updated it.
    """
    name: str
    labels: List[str]  # e.g., ["Person"], ["Location"], ["Organization", "Client"]
    source_memory_id: int  # Episode where this node was primarily defined or last significantly updated
    concept_id: int
    attributes: Dict[str, Any] = Field(default_factory=dict)  # For additional, non-relational data


class GraphEdge(KnoxelBase):
    """
    Graph-memory relation between two ``GraphNode`` entries.

    Stores symbolic relation type (``label``), natural-language statement
    (``fact_text``), and temporal validity window for facts that may change over time.
    Used for relation retrieval, contradiction handling, and evidence tracing.
    """
    source_id: int  # UUID of the source GraphNode
    target_id: int  # UUID of the target GraphNode
    label: str  # e.g., "works_for", "knows", "located_in"
    fact_text: str  # Natural language representation of the fact, e.g., "Alex works for Acme Corp"
    source_memory_id: int  # Episode where this fact was learned
    valid_at: Optional[datetime] = Field(default=None, description="Timestamp when this fact became valid (if applicable).")
    invalid_at: Optional[datetime] = Field(default=None, description="Timestamp when this fact became invalid (due to contradiction/update).")
    attributes: Dict[str, Any] = Field(default_factory=dict)


class ConceptNode(KnoxelBase):
    """
    Concept hierarchy node for abstraction and taxonomy alignment.

    Allows graph memory to map entities to higher-level concepts through
    ``parent_concept_id`` and optional ``entity_id`` linkage, supporting generalized
    retrieval beyond literal entity names.
    """
    description: str
    entity_id: Optional[int] = Field(default=None)
    parent_concept_id: int = Field(default=None, description="UUID of the parent concept (for IS_A hierarchy).")
