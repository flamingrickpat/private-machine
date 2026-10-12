from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Type, Any

from pm.records.cycle_state import CycleState, PersonaConfig
from pm.records.codelet_records import CodeletPercept, Structure
from pm.records.memory_records import Action, CauseEffectRecord, DeclarativeFactRecord, Entity, Feature, MemoryClusterRecord, Narrative, Stimulus
from pm.records.base import RecordBase, RecordOwner, RecordKind
from pm.records.enums import ActionType, ClusterType, EntityClass, FeatureSubtype, FeatureType, RecordSubtypeBase, NarrativeTypes, StimulusType
from pm.records.graph_records import ConceptNode, GraphEdge, GraphNode
from pm.records.cognitive_records import AttentionFrameRecord, AttentionRecord, BeliefRecord, BeliefShiftRecord, Intention, OntologyBindingRecord, OntologyTermRecord
from pm.records.base import get_causal_datetime


class RecordIndex(RecordOwner):
    """
    Runtime owner for knoxels and cycle state.

    `all_knoxels` is canonical. The `all_*` accessors are indexed views kept up
    to date by `add_knoxel`. Use `rebuild_knoxel_indexes` after bulk loading or
    direct replacement of `all_knoxels`.
    """

    _KNOXEL_TYPE_EXPANSIONS: dict[RecordKind, set[RecordKind]] = {
        RecordKind.CoversTicksEventsRecord: {
            RecordKind.CoversTicksEventsRecord,
            RecordKind.MemoryClusterRecord,
            RecordKind.DeclarativeFactRecord,
            RecordKind.CauseEffectRecord,
        },
    }

    _SUBTYPE_PARENT_MAP: dict[type[RecordSubtypeBase], RecordKind] = {
        FeatureType: RecordKind.Feature,
        FeatureSubtype: RecordKind.Feature,
        StimulusType: RecordKind.Stimulus,
        ActionType: RecordKind.Action,
        NarrativeTypes: RecordKind.Narrative,
        EntityClass: RecordKind.Entity,
        ClusterType: RecordKind.MemoryClusterRecord,
    }

    def __init__(self, llm: Any | None, config: PersonaConfig):
        self._all_knoxels: Dict[int, RecordBase] = {}
        self._knoxels_by_type: dict[RecordKind, List[RecordBase]] = defaultdict(list)
        self._knoxels_by_subtype: dict[RecordSubtypeBase, List[RecordBase]] = defaultdict(list)
        self._knoxels_by_class: dict[type[RecordBase], List[RecordBase]] = defaultdict(list)
        super().__init__()

        self.ghost_config: PersonaConfig = config
        self.llm: Any | None = llm
        self.current_db_path: str = ""

        self.current_tick_id: int = 0
        self.current_knoxel_id: int = 0
        self.states: List[CycleState] = []
        self.simulated_reply: Optional[str] = None

    @property
    def all_knoxels(self) -> Dict[int, RecordBase]:
        return self._all_knoxels

    @all_knoxels.setter
    def all_knoxels(self, value: Dict[int, RecordBase]) -> None:
        self._all_knoxels = dict(value or {})
        if hasattr(self, "_knoxels_by_type"):
            self.rebuild_knoxel_indexes()

    @property
    def current_state(self) -> CycleState | None:
        if self.states:
            return self.states[-1]
        return None

    @property
    def previous_state(self) -> CycleState | None:
        if len(self.states) > 1:
            return self.states[-2]
        return None

    @property
    def all_entities(self) -> List[Entity]:
        return self._knoxels_for_class(Entity)

    @property
    def all_stimuli(self) -> List[Stimulus]:
        return self._knoxels_for_type(RecordKind.Stimulus)

    @property
    def all_features(self) -> List[Feature]:
        return self._knoxels_for_type(RecordKind.Feature)

    @property
    def all_actions(self) -> List[Action]:
        return self._knoxels_for_type(RecordKind.Action)

    @property
    def all_intentions(self) -> List[Intention]:
        return self._knoxels_for_type(RecordKind.Intention)

    @property
    def all_narratives(self) -> List[Narrative]:
        return self._knoxels_for_type(RecordKind.Narrative)

    @property
    def all_episodic_memories(self) -> List[MemoryClusterRecord]:
        return self._knoxels_for_type(RecordKind.MemoryClusterRecord)

    @property
    def all_declarative_facts(self) -> List[DeclarativeFactRecord]:
        return self._knoxels_for_type(RecordKind.DeclarativeFactRecord)

    @property
    def all_cause_effects(self) -> List[CauseEffectRecord]:
        return self._knoxels_for_type(RecordKind.CauseEffectRecord)

    @property
    def all_concepts(self) -> List[ConceptNode]:
        return self._knoxels_for_type(RecordKind.ConceptNode)

    @property
    def all_graph_nodes(self) -> List[GraphNode]:
        return self._knoxels_for_type(RecordKind.GraphNode)

    @property
    def all_graph_edges(self) -> List[GraphEdge]:
        return self._knoxels_for_type(RecordKind.GraphEdge)

    @property
    def all_codelet_percepts(self) -> List[CodeletPercept]:
        return self._knoxels_for_type(RecordKind.CodeletPercept)

    @property
    def all_percept_coalitions(self) -> List[Structure]:
        return self._knoxels_for_class(Structure)

    @property
    def all_beliefs(self) -> List[BeliefRecord]:
        return self._knoxels_for_class(BeliefRecord)

    @property
    def all_belief_shifts(self) -> List[BeliefShiftRecord]:
        return self._knoxels_for_class(BeliefShiftRecord)

    @property
    def all_attention_knoxels(self) -> List[AttentionRecord]:
        return self._knoxels_for_class(AttentionRecord)

    @property
    def all_attention_frames(self) -> List[AttentionFrameRecord]:
        return self._knoxels_for_class(AttentionFrameRecord)

    @property
    def all_ontology_terms(self) -> List[OntologyTermRecord]:
        return self._knoxels_for_class(OntologyTermRecord)

    @property
    def all_ontology_bindings(self) -> List[OntologyBindingRecord]:
        return self._knoxels_for_class(OntologyBindingRecord)

    @property
    def all_structures(self) -> List[Structure]:
        return self._knoxels_for_class(Structure)

    def _get_state_at_tick(self, tick_id: int) -> CycleState | None:
        if tick_id == self.current_tick_id:
            return self.current_state
        for state in self.states:
            if state.tick_id == tick_id:
                return state
        return None

    def _get_next_id(self) -> int:
        self.current_knoxel_id += 1
        return self.current_knoxel_id

    def _get_current_tick_id(self) -> int:
        return self.current_tick_id

    def build_body_and_situation_addendum(self) -> str:
        current_causal_datetime = get_causal_datetime()
        dt_text = current_causal_datetime.strftime("%Y-%m-%d %H:%M:%S UTC")
        return (
            "Body and Situation\n"
            f"Current causal datetime: {dt_text}\n"
            "This is an in-character temporal anchor for reasoning."
        )

    def get_all_knoxels(self) -> List[RecordBase]:
        return list(self.all_knoxels.values())

    def get_simple_character_story_block(self) -> str:
        return self.ghost_config.character_card_story

    def get_companion_name(self) -> str:
        return self.ghost_config.companion_name

    def get_user_name(self) -> str:
        return self.ghost_config.user_name

    def add_knoxel(self, knoxel: RecordBase, generate_embedding: bool = True) -> RecordBase:
        if knoxel.id == -1:
            knoxel.id = self._get_next_id()
        else:
            self.current_knoxel_id = max(self.current_knoxel_id, knoxel.id)

        if knoxel.tick_id == -1:
            knoxel.tick_id = self._get_current_tick_id()

        if isinstance(knoxel, Feature) and not knoxel.entity_id and knoxel.override_source:
            for existing in self.all_entities:
                if existing.content == knoxel.override_source:
                    knoxel.entity_id = existing.id
                    break

        existing = self.all_knoxels.get(knoxel.id)
        if existing is not None:
            if existing is knoxel:
                return knoxel
            raise ValueError(f"Knoxel id {knoxel.id} already exists")

        knoxel._owner = self
        self._all_knoxels[knoxel.id] = knoxel
        self._index_knoxel(knoxel)

        if generate_embedding and not knoxel.embedding and knoxel.content:
            if self.llm:
                knoxel.embedding = self.llm.get_embedding(knoxel.content)
            else:
                knoxel.embedding = [0]

        return knoxel

    def replace_knoxel(self, knoxel: RecordBase) -> RecordBase:
        """Replace the knoxel stored under knoxel.id (external-edit digest)."""
        existing = self.all_knoxels.get(knoxel.id)
        if existing is None:
            raise ValueError(f"Knoxel id {knoxel.id} does not exist")
        if existing is knoxel:
            return knoxel
        self._deindex_knoxel(existing)
        knoxel._owner = self
        self._all_knoxels[knoxel.id] = knoxel
        self._index_knoxel(knoxel)
        return knoxel

    def get_knoxel_by_id(self, knoxel_id: int) -> Optional[RecordBase]:
        return self.all_knoxels.get(knoxel_id, None)

    def get_entity_id(self, name: str) -> int | None:
        for entity in self.all_entities:
            if name == entity.content or name in entity.aliases:
                return entity.id
        return None

    def get_entity_name(self, entity_id: int) -> str | None:
        for entity in self.all_entities:
            if entity_id == entity.id:
                return entity.content
        return None

    def _reset_internal_state(self) -> None:
        self.current_tick_id = 0
        self.current_knoxel_id = 0
        self._all_knoxels = {}
        self._clear_knoxel_indexes()
        self.states = []
        self.simulated_reply = None

    def rebuild_knoxel_indexes(self) -> None:
        self._clear_knoxel_indexes()
        current_knoxel_id = getattr(self, "current_knoxel_id", 0)
        self.current_knoxel_id = max(current_knoxel_id, max(self.all_knoxels.keys(), default=0))
        for knoxel in self.all_knoxels.values():
            knoxel._owner = self
            self._index_knoxel(knoxel)

    def _rebuild_specific_lists(self) -> None:
        self.rebuild_knoxel_indexes()

    def _clear_knoxel_indexes(self) -> None:
        self._knoxels_by_type = defaultdict(list)
        self._knoxels_by_subtype = defaultdict(list)
        self._knoxels_by_class = defaultdict(list)

    def _index_knoxel(self, knoxel: RecordBase) -> None:
        for bucket in self._iter_index_buckets(knoxel):
            bucket.append(knoxel)

    def _deindex_knoxel(self, knoxel: RecordBase) -> None:
        for cls in type(knoxel).mro():
            if not isinstance(cls, type) or not issubclass(cls, RecordBase) or cls is RecordBase:
                continue
            bucket = self._knoxels_by_class.get(cls)
            if bucket and knoxel in bucket:
                bucket.remove(knoxel)

        knoxel_type = self._knoxel_type_of(knoxel)
        if knoxel_type is not None:
            bucket = self._knoxels_by_type.get(knoxel_type)
            if bucket and knoxel in bucket:
                bucket.remove(knoxel)

        subtype = self._knoxel_subtype_of(knoxel)
        if subtype is not None:
            bucket = self._knoxels_by_subtype.get(subtype)
            if bucket and knoxel in bucket:
                bucket.remove(knoxel)

        feature_subtype = getattr(knoxel, "feature_subtype", None)
        if isinstance(feature_subtype, RecordSubtypeBase):
            bucket = self._knoxels_by_subtype.get(feature_subtype)
            if bucket and knoxel in bucket:
                bucket.remove(knoxel)

    def _iter_index_buckets(self, knoxel: RecordBase):
        for cls in type(knoxel).mro():
            if not isinstance(cls, type) or not issubclass(cls, RecordBase) or cls is RecordBase:
                continue
            yield self._knoxels_by_class[cls]

        knoxel_type = self._knoxel_type_of(knoxel)
        if knoxel_type is not None:
            yield self._knoxels_by_type[knoxel_type]

        subtype = self._knoxel_subtype_of(knoxel)
        if subtype is not None:
            yield self._knoxels_by_subtype[subtype]

        feature_subtype = getattr(knoxel, "feature_subtype", None)
        if isinstance(feature_subtype, RecordSubtypeBase):
            yield self._knoxels_by_subtype[feature_subtype]

    @staticmethod
    def _knoxel_type_of(knoxel: RecordBase) -> RecordKind | None:
        try:
            return knoxel.type
        except (KeyError, ValueError):
            return None

    @staticmethod
    def _knoxel_subtype_of(knoxel: RecordBase) -> RecordSubtypeBase | None:
        try:
            return knoxel.subtype
        except (KeyError, ValueError):
            return None

    def _knoxels_for_type(self, knoxel_type: RecordKind) -> List:
        return self._knoxels_by_type.get(knoxel_type, [])

    def _knoxels_for_class(self, cls: Type[RecordBase]) -> List:
        return self._knoxels_by_class.get(cls, [])

    def knoxels_by_type(self, knoxel_type: RecordKind, *, needs_causal: bool = False) -> List[RecordBase]:
        return [k for k in self._knoxels_for_type(knoxel_type) if self._passes_causal_filter(k, needs_causal)]

    def knoxels_by_subtype(self, subtype: RecordSubtypeBase, *, needs_causal: bool = False) -> List[RecordBase]:
        return [k for k in self._knoxels_by_subtype.get(subtype, []) if self._passes_causal_filter(k, needs_causal)]

    def knoxels_by_class(self, cls: Type[RecordBase], *, needs_causal: bool = False) -> List[RecordBase]:
        return [k for k in self._knoxels_for_class(cls) if self._passes_causal_filter(k, needs_causal)]

    def _flatten_type_filters(self, *type_filters):
        for item in type_filters:
            if item is None:
                continue

            if isinstance(item, (list, tuple, set, frozenset)):
                yield from self._flatten_type_filters(*item)
                continue

            if isinstance(item, (RecordKind, RecordSubtypeBase)):
                yield item
                continue

            if isinstance(item, type) and issubclass(item, RecordBase):
                yield item
                continue

            raise TypeError(
                f"Unsupported type filter {item!r} "
                f"(expected KnoxelType / KnoxelSubtypeBase / KnoxelBase subclass / nested list/tuple/set)"
            )

    @classmethod
    def _expand_knoxel_types(cls, knoxel_types: set[RecordKind]) -> set[RecordKind]:
        expanded: set[RecordKind] = set()
        for knoxel_type in knoxel_types:
            expanded.update(cls._KNOXEL_TYPE_EXPANSIONS.get(knoxel_type, {knoxel_type}))
        return expanded

    @classmethod
    def _parent_knoxel_type_for_subtype(cls, subtype: RecordSubtypeBase) -> RecordKind:
        for subtype_cls, parent_knoxel_type in cls._SUBTYPE_PARENT_MAP.items():
            if isinstance(subtype, subtype_cls):
                return parent_knoxel_type
        raise TypeError(f"Unsupported subtype filter: {subtype!r} ({type(subtype).__name__})")

    @staticmethod
    def _passes_causal_filter(knoxel: RecordBase, needs_causal: bool) -> bool:
        if not needs_causal:
            return True
        if RecordIndex._knoxel_type_of(knoxel) != RecordKind.Feature:
            return True
        return bool(getattr(knoxel, "causal", False))

    def get_knoxels_of_types(
        self,
        *type_filters,
        needs_causal: bool = True,
    ) -> List[RecordBase]:
        requested_knoxel_types: set[RecordKind] = set()
        requested_subtypes: dict[RecordKind, set[RecordSubtypeBase]] = {}
        requested_classes: set[type[RecordBase]] = set()

        for item in self._flatten_type_filters(*type_filters):
            if isinstance(item, RecordKind):
                requested_knoxel_types.add(item)
            elif isinstance(item, type) and issubclass(item, RecordBase):
                requested_classes.add(item)
            else:
                parent_knoxel_type = self._parent_knoxel_type_for_subtype(item)
                requested_subtypes.setdefault(parent_knoxel_type, set()).add(item)

        requested_knoxel_types = self._expand_knoxel_types(requested_knoxel_types)

        if not requested_knoxel_types and not requested_subtypes and not requested_classes:
            candidates: Iterable[RecordBase] = self.all_knoxels.values()
        else:
            seen: set[int] = set()
            selected: List[RecordBase] = []

            for knoxel_type in requested_knoxel_types:
                self._append_unique(selected, seen, self._knoxels_for_type(knoxel_type))

            for subtype_set in requested_subtypes.values():
                for subtype in subtype_set:
                    self._append_unique(selected, seen, self._knoxels_by_subtype.get(subtype, []))

            for cls in requested_classes:
                self._append_unique(selected, seen, self._knoxels_for_class(cls))

            candidates = selected

        result = [knoxel for knoxel in candidates if self._passes_causal_filter(knoxel, needs_causal)]
        result.sort(key=lambda item: (item.timestamp_world_begin, item.sub_tick_id, item.id))
        return result

    @staticmethod
    def _append_unique(target: List[RecordBase], seen: set[int], items: Iterable[RecordBase]) -> None:
        for item in items:
            if item.id in seen:
                continue
            seen.add(item.id)
            target.append(item)

    def get_ids_of_types(
        self,
        *type_filters,
        needs_causal: bool = True,
        exclude_ids: List[int] | None = None,
    ) -> List[int]:
        ids = [
            knoxel.id
            for knoxel in self.get_knoxels_of_types(
                *type_filters,
                needs_causal=needs_causal,
            )
        ]
        if exclude_ids:
            ids = [knoxel_id for knoxel_id in ids if knoxel_id not in exclude_ids]
        return ids


if __name__ == "__main__":
    gs = CycleState(tick_id=1)
    print(gs.tick_id)
