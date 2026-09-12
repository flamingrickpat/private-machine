from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Type

import yaml

from pm_core.model.ghost_state import CycleStateModel
from pm_core.model.knoxel_core import KnoxelBase
from pm_core.model.knoxel_common import (
    Action,
    ActionSchemaKnoxel,
    CauseEffectKnoxel,
    CoversTicksEventsKnoxel,
    DeclarativeFactKnoxel,
    Entity,
    Feature,
    MemoryClusterKnoxel,
    Narrative,
    RealityKnoxel,
    SemanticRecordKnoxel,
    Stimulus,
)
from pm_core.model.knoxel_codelets import CodeletPercept, Structure
from pm_core.model.knoxel_codelets import ModuleOutput
from pm_core.model.knoxel_embodiment import (
    BodyEpisodeKnoxel,
    ExecutableSkillKnoxel,
    WorldObjectKnoxel,
    WorldPlaceKnoxel,
)
from pm_core.model.knoxel_graph import ConceptNode, GraphEdge, GraphNode
from pm_core.model.knoxel_qualia import QualeKnoxel
from pm_core.model.knoxel_state_change import (
    AttentionFrameKnoxel,
    AttentionKnoxel,
    BeliefKnoxel,
    BeliefShiftKnoxel,
    Intention,
    OntologyBindingKnoxel,
    OntologyTermKnoxel,
)

KNOXEL_TYPE_REGISTRY: Dict[str, Type[KnoxelBase]] = {
    cls.__name__: cls
    for cls in [
        Action,
        ActionSchemaKnoxel,
        AttentionFrameKnoxel,
        AttentionKnoxel,
        BeliefKnoxel,
        BeliefShiftKnoxel,
        BodyEpisodeKnoxel,
        CauseEffectKnoxel,
        CodeletPercept,
        ConceptNode,
        CoversTicksEventsKnoxel,
        DeclarativeFactKnoxel,
        Entity,
        ExecutableSkillKnoxel,
        Feature,
        GraphEdge,
        GraphNode,
        Intention,
        MemoryClusterKnoxel,
        ModuleOutput,
        Narrative,
        OntologyBindingKnoxel,
        OntologyTermKnoxel,
        QualeKnoxel,
        RealityKnoxel,
        SemanticRecordKnoxel,
        Stimulus,
        Structure,
        WorldObjectKnoxel,
        WorldPlaceKnoxel,
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


def _serialize_knoxel(knoxel: KnoxelBase) -> Dict[str, Any]:
    return {
        "__type__": knoxel.__class__.__name__,
        "data": knoxel.model_dump(mode="json", serialize_as_any=True),
    }


def _deserialize_knoxel(raw: Any) -> KnoxelBase:
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
