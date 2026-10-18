from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Type

import yaml

from pm.utils import yaml_util

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

KNOXEL_TYPE_REGISTRY: Dict[str, Type[RecordBase]] = {
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


class _GhostStateYamlDumper(yaml.SafeDumper):
    pass


def _is_plain_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _represent_list(dumper: _GhostStateYamlDumper, data: list[Any]) -> yaml.nodes.SequenceNode:
    flow_style = True if data and all(_is_plain_number(item) for item in data) else None
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow_style)


_GhostStateYamlDumper.add_representer(list, _represent_list)


def _serialize_knoxel(knoxel: RecordBase) -> Dict[str, Any]:
    data = knoxel.model_dump(mode="json", serialize_as_any=True)
    # Keep legacy records unchanged until an owner adds actual episode links.
    if data.get("lineage") is None:
        data.pop("lineage", None)
    return {
        "__type__": persisted_type_name(type(knoxel)),
        "data": data,
    }


def _deserialize_knoxel(raw: Any) -> RecordBase:
    if not isinstance(raw, dict):
        raise ValueError("Knoxel YAML entry must be a mapping")

    if "__type__" in raw:
        type_name = str(raw["__type__"])
        data = raw.get("data", {})
    else:
        type_name = str(raw.get("type", "KnoxelBase"))
        data = {key: value for key, value in raw.items() if key != "type"}

    cls = KNOXEL_TYPE_REGISTRY.get(type_name)
    if cls is None:
        raise ValueError(f"Unsupported knoxel type in YAML: {type_name}")
    return cls.model_validate(data)


def ghost_state_to_dict(state: CycleStateModel) -> Dict[str, Any]:
    data = state.model_dump(mode="json", exclude={"knoxels"}, serialize_as_any=True)
    data["knoxels"] = [_serialize_knoxel(knoxel) for knoxel in state.knoxels]
    return data


def ghost_state_from_dict(data: Dict[str, Any]) -> CycleStateModel:
    if not isinstance(data, dict):
        raise ValueError("Ghost state YAML root must be a mapping")
    prepared = dict(data)
    prepared["knoxels"] = [_deserialize_knoxel(item) for item in prepared.get("knoxels", [])]
    state = CycleStateModel.model_validate(prepared)
    state.rebuild_knoxel_index()
    return state


def dump_ghost_state_to_yaml(state: CycleStateModel, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.dump(
            ghost_state_to_dict(state),
            handle,
            Dumper=_GhostStateYamlDumper,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
            width=2**31 - 1,
        )


def _normalize_base_loader_nulls(value: Any) -> Any:
    if value == "null":
        return None
    if isinstance(value, list):
        return [_normalize_base_loader_nulls(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_base_loader_nulls(item) for key, item in value.items()}
    return value


def load_ghost_state_from_yaml(path: str | Path) -> CycleStateModel:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.load(handle, Loader=yaml.CBaseLoader) or {}
    data = _normalize_base_loader_nulls(data)
    return ghost_state_from_dict(data)
