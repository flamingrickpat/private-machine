"""Graph-memory records: entity nodes, relation edges, and concept nodes.

owns: the three record families that memory reconsolidation writes when it
connects consolidated memories into a graph. They hold ids, labels, and short
text; embeddings and search live in the memory store. See
``pm.records.base`` for the package overview.

``GraphEdge`` links two ``GraphNode`` records and carries the natural-language
claim plus its validity window, so a changed fact keeps its history instead of
overwriting the old one. ``ConceptNode`` is the abstraction layer above the
entity nodes. ``attributes`` stays a free-form JSON bag on purpose: it holds
labels that a future schema may promote to a field.
"""
from datetime import datetime
import logging
from typing import Any

from pydantic import Field

from pm.records.base import RecordBase
from pm.records.memory_records import SemanticallyClassifiedRecord

logger = logging.getLogger(__name__)


class GraphNode(RecordBase):
    """
    Graph-memory entity node derived from consolidated memories and fact extraction.

    Represents a durable concept instance (person/place/object/topic) with typed labels,
    normalized name, and optional attributes. ``source_memory_id`` ties the node to the
    memory episode that most strongly defined or updated it.
    """
    name: str
    labels: list[str]  # e.g., ["Person"], ["Location"], ["Organization", "Client"]
    source_memory_id: int  # Episode where this node was primarily defined or last significantly updated
    concept_id: int
    attributes: dict[str, Any] = Field(default_factory=dict)  # For additional, non-relational data


class GraphEdge(SemanticallyClassifiedRecord):
    """
    Graph-memory relation between two ``GraphNode`` entries.

    Stores symbolic relation type (``label``), natural-language statement
    (``fact_text``), and temporal validity window for facts that may change over time.
    Used for relation retrieval, contradiction handling, and evidence tracing.

    The semantic classification layer (subject/record type/epistemic/essence) marks
    the edge's fact_text as a claim and links it to its canonical SemanticRecord.
    """
    source_id: int  # Id of the source GraphNode
    target_id: int  # Id of the target GraphNode
    label: str  # e.g., "works_for", "knows", "located_in"
    fact_text: str  # Natural language representation of the fact, e.g., "Alex works for Acme Corp"
    source_memory_id: int  # Episode where this fact was learned
    valid_at: datetime | None = Field(default=None, description="Timestamp when this fact became valid (if applicable).")
    invalid_at: datetime | None = Field(default=None, description="Timestamp when this fact became invalid (due to contradiction/update).")
    attributes: dict[str, Any] = Field(default_factory=dict)


class ConceptNode(RecordBase):
    """
    Concept hierarchy node for abstraction and taxonomy alignment.

    Allows graph memory to map entities to higher-level concepts through
    ``parent_concept_id`` and optional ``entity_id`` linkage, supporting generalized
    retrieval beyond literal entity names.
    """
    description: str
    entity_id: int | None = Field(default=None)
    parent_concept_id: int | None = Field(default=None, description="Id of the parent concept (for IS_A hierarchy).")
