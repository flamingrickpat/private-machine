"""Read ordinary selection receipts from their canonical memory records.

See :mod:`workflows.interaction.host` for acceptance and recovery ownership.
"""
from __future__ import annotations

from pm.memory.snapshot import Snapshot
from pm.records.memory_records import Feature
from pydantic import BaseModel, ConfigDict, Field
from workflows.interaction.schemas import InteractionMetadata, InteractionReceipt


class MainSelectionMetadata(BaseModel):
    """Local selections reserved to interaction by a completed main appraisal."""

    model_config = ConfigDict(extra="ignore")
    interaction_source_ids: list[int] = Field(default_factory=list[int])


def interaction_owned_sources(view: Snapshot) -> set[int]:
    """Read input ownership from accepted main decision records."""
    sources: set[int] = set()
    for turn in view.gwt_turns.values():
        if turn.slot_id == "main" and turn.decision_id is not None:
            metadata = MainSelectionMetadata.model_validate(view.get(turn.decision_id).metadata)
            sources.update(metadata.interaction_source_ids)
    return sources


def interaction_receipts(view: Snapshot) -> dict[int, InteractionReceipt]:
    """Map each selected source ID to its latest accepted ordinary receipt."""
    receipts: dict[int, InteractionReceipt] = {}
    for record in view.all_knoxels.values():
        if isinstance(record, Feature) and "interaction_receipt" in record.metadata:
            receipt = InteractionMetadata.model_validate(record.metadata).interaction_receipt
            for source_id in receipt.source_ids:
                receipts[source_id] = receipt
    return receipts
