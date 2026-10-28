"""Check typed memory references in new proposals, not free-form metadata.

Legacy seed graphs remain importable. New invalid references fail the whole
batch. Numeric tick IDs and string external IDs are not memory references.
"""
from pydantic import BaseModel


REFERENCE_FIELDS = frozenset({
    "supersedes_id",
    "reality_id", "entity_id", "generated_expectation_ids", "intention_ids",
    "min_event_id", "max_event_id", "subject_entity_id",
    "semantic_record_id", "source_cluster_id", "superseded_by_ids", "source_knoxel_ids",
    "supersedes_record_ids", "contradicted_by_record_ids", "supports_record_ids", "related_record_ids",
    "precursor_ids", "owner_ids", "about_entity_ids", "supporting_knoxel_ids", "contradicting_knoxel_ids",
    "superseded_by_id", "target_belief_id", "causal_knoxel_ids", "target_knoxel_ids",
    "focus_knoxel_ids", "winning_attention_ids", "active_goal_ids", "example_knoxel_ids",
    "related_term_ids", "term_id", "bound_knoxel_ids", "source_event_ids", "source_memory_id",
    "concept_id", "source_id", "target_id", "parent_concept_id", "internal_knoxel_ids",
    "structure_ids", "predecessor_id", "produced_structure_ids", "primary_structure_id",
    "control_knoxel_id", "predecessor_output_ids", "last_driver_ids", "driver_ids",
    "knoxel_ids",
    "parent_record_ids",
})


def referenced_ids(record: BaseModel) -> set[int]:
    """Return the memory record IDs that ``record`` names.

    Only declared reference fields count. ``0`` and ``-1`` are sentinels for
    "no record" and are not references, and nested models are searched so a
    lineage or embedded structure cannot hide a dangling link.
    """
    result: set[int] = set()
    for name, value in record:
        if name not in type(record).model_fields:
            continue
        if name in REFERENCE_FIELDS:
            values = value if isinstance(value, list) else [value]
            result.update(item for item in values if item not in (None, 0, -1))
        elif name in {"included_event_ids", "included_cluster_ids"} and value:
            result.update(int(item.strip()) for item in value.split(",") if item.strip())
        elif isinstance(value, BaseModel):
            result.update(referenced_ids(value))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, BaseModel):
                    result.update(referenced_ids(item))
    return result


def check_references(record: BaseModel, allowed_ids: set[int]) -> None:
    """Fail the whole batch when an accepted reference names an unavailable record."""
    missing = referenced_ids(record) - allowed_ids
    if missing:
        raise ValueError(f"Memory references are missing or outside the batch view: {sorted(missing)}")
