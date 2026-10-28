"""Record envelopes of the legacy YAML/JSON formats, read and written once.

An imported persona seed is free-form YAML, and a JSON export is a flat
mapping of persona fields plus a list of tagged record payloads.
:class:`RecordPayload` turns one tagged mapping into the typed record that its
``__type__`` names, so nothing downstream sees an untyped record. Runtime
state does not use this module: an accepted store is saved as a typed pickle
checkpoint (:mod:`pm.memory.checkpoint`). Load order and ownership live in
:mod:`pm.memory.store`; the package overview is :mod:`pm.memory`.

Invalid input fails here and never becomes partial memory: an unknown type tag
raises ``KeyError``, and a payload that contradicts its record model raises a
pydantic validation error. The base YAML loader keeps every scalar as text, so
``_normalize_base_loader_nulls`` restores the ``null`` strings it produces.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import JsonValue, RootModel, SerializeAsAny, model_validator

from pm.records.cycle_state import CycleStateModel
from pm.records.base import persisted_type_name, RecordBase
from pm.records.memory_records import Action, ActionSchemaRecord, CauseEffectRecord, CoversTicksEventsRecord, DeclarativeFactRecord, Entity, Feature, MemoryClusterRecord, Narrative, RealityRecord, SemanticRecord, Stimulus
from pm.records.codelet_records import CodeletPercept, Structure
from pm.records.codelet_records import ModuleOutput
from pm.records.world_records import BodyEpisodeRecord, ExecutableSkillRecord, WorldObjectRecord, WorldPlaceRecord
from pm.records.graph_records import ConceptNode, GraphEdge, GraphNode
from pm.records.qualia_records import QualeRecord
from pm.records.cognitive_records import (AttentionFrameRecord, AttentionRecord, BeliefRecord,
                                           BeliefShiftRecord, ExpectationRecord, Intention,
                                           OntologyBindingRecord, OntologyTermRecord)

KNOXEL_TYPE_REGISTRY: dict[str, type[RecordBase]] = {
    persisted_type_name(cls): cls
    for cls in [
        Action,
        ActionSchemaRecord,
        AttentionFrameRecord,
        AttentionRecord,
        BeliefRecord,
        BeliefShiftRecord,
        BodyEpisodeRecord,
        CauseEffectRecord,
        CodeletPercept,
        ConceptNode,
        CoversTicksEventsRecord,
        DeclarativeFactRecord,
        Entity,
        ExecutableSkillRecord,
        ExpectationRecord,
        Feature,
        GraphEdge,
        GraphNode,
        Intention,
        MemoryClusterRecord,
        ModuleOutput,
        Narrative,
        OntologyBindingRecord,
        OntologyTermRecord,
        QualeRecord,
        RealityRecord,
        SemanticRecord,
        Stimulus,
        Structure,
        WorldObjectRecord,
        WorldPlaceRecord,
    ]
}


class RecordPayload(RootModel[SerializeAsAny[RecordBase]]):
    """Validate a record only at a legacy file or external tool boundary."""

    @model_validator(mode="before")
    @classmethod
    def read_legacy(cls, raw: Any) -> RecordBase:
        """Keep typed records intact. Parse legacy tagged mappings once.

        ``__type__``/``data`` is the current envelope; the older ``type`` plus
        inline-fields form is still read so an untouched seed file imports.
        """
        if isinstance(raw, RecordBase):
            return raw
        if "__type__" in raw:
            return KNOXEL_TYPE_REGISTRY[raw["__type__"]].model_validate(raw["data"])
        return KNOXEL_TYPE_REGISTRY[raw["type"]].model_validate({key: value for key, value in raw.items() if key != "type"})


def ghost_state_to_dict(state: CycleStateModel) -> dict[str, JsonValue]:
    """Render one JSON export of persona fields plus tagged record payloads."""
    data = state.model_dump(mode="json", exclude={"knoxels"}, serialize_as_any=True)
    data["knoxels"] = [knoxel.as_payload() for knoxel in state.knoxels]
    return data


def ghost_state_from_dict(data: Mapping[str, Any]) -> CycleStateModel:
    """Parse one legacy state mapping into typed records and cycle state.

    The mapping is external input, so this is the one place that accepts a
    general JSON shape. Record envelopes are validated individually, and a
    duplicate record ID raises from ``CycleStateModel.rebuild_knoxel_index``.
    """
    if not isinstance(data, dict):
        raise ValueError("Ghost state YAML root must be a mapping")
    prepared = dict(data)
    prepared["knoxels"] = [RecordPayload.model_validate(item).root for item in prepared.get("knoxels", [])]
    state = CycleStateModel.model_validate(prepared)
    state.rebuild_knoxel_index()
    return state


def _normalize_base_loader_nulls(value: Any) -> Any:
    """Restore real nulls that the base YAML loader rendered as the text ``null``."""
    if value == "null":
        return None
    if isinstance(value, list):
        return [_normalize_base_loader_nulls(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_base_loader_nulls(item) for key, item in value.items()}
    return value


def load_ghost_state_from_yaml(path: str | Path) -> CycleStateModel:
    """Read and parse one seed file. The file itself is never rewritten."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.load(handle, Loader=yaml.CBaseLoader) or {}
    data = _normalize_base_loader_nulls(data)
    return ghost_state_from_dict(data)
