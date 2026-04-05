import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from pm.ghost.ghost_base import BaseGhost
from pm.ghost.ghost_config import GhostConfig
from pm.model.knoxel_codelets import CodeletPercept, PerceptCoalition
from pm.model.knoxel_common import (
    Action,
    CauseEffectKnoxel,
    DeclarativeFactKnoxel,
    Entity,
    Intention,
    MemoryClusterKnoxel,
    Narrative,
    Stimulus,
)
from pm.model.knoxel_core import KnoxelBase, KnoxelHaver
from pm.model.knoxel_enums import (
    ActionType,
    ClusterType,
    EntityClass,
    FeatureType,
    InterlocusType,
    NarrativeTypes,
    StimulusType,
)
from pm.model.knoxel_feature import Feature
from pm.model.knoxel_graph import ConceptNode, GraphEdge, GraphNode
from pm.model.mental_state_vectors import (
    AppraisalGeneral,
    AppraisalSocial,
    StateCore,
    StateEmotions,
    VectorModelReservedSize,
    create_empty_ms_vector,
)
from pm.system.load_config import PmConfig, load_config


DEFAULT_EMBEDDING_DIM = 16
DEFAULT_START_TIME = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)
DEFAULT_TOPICS = ("music", "gardening", "robotics", "travel")
DEFAULT_CHARACTERS = (
    ("Mira", EntityClass.AI, ("assistant", "ghost", "companion")),
    ("Rick", EntityClass.Human, ("user", "human")),
    ("Tara", EntityClass.Human, ("friend", "neighbor")),
    ("Archivist", EntityClass.Agent, ("memory keeper", "librarian")),
)


@dataclass(frozen=True)
class GenerationScale:
    days: int
    scenes_per_day: int
    topical_clusters_per_topic: int
    temporal_cluster_levels: Sequence[int]


SMALL_SCALE = GenerationScale(days=6, scenes_per_day=3, topical_clusters_per_topic=2, temporal_cluster_levels=(6, 5))
MEDIUM_SCALE = GenerationScale(days=12, scenes_per_day=4, topical_clusters_per_topic=3, temporal_cluster_levels=(6, 5, 4))
LARGE_SCALE = GenerationScale(days=21, scenes_per_day=5, topical_clusters_per_topic=4, temporal_cluster_levels=(6, 5, 4, 3))


@dataclass(frozen=True)
class CharacterSpec:
    name: str
    entity_class: EntityClass
    aliases: Sequence[str]


@dataclass(frozen=True)
class TestInfrastructureConfig:
    __test__ = False
    seed: int = 7
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    start_time_utc: datetime = DEFAULT_START_TIME
    scale: GenerationScale = SMALL_SCALE
    topics: Sequence[str] = DEFAULT_TOPICS
    character_card_story: str = "Mira is an introspective AI companion who tracks memory, emotion, and narrative continuity carefully."
    available_tools: str = "calendar.lookup, notes.search"
    characters: Sequence[CharacterSpec] = field(
        default_factory=lambda: [CharacterSpec(name, entity_class, aliases) for name, entity_class, aliases in DEFAULT_CHARACTERS]
    )


@dataclass
class GeneratedCorpus:
    ghost: "TestGhost"
    config: TestInfrastructureConfig
    entities: Dict[str, Entity]
    query_embeddings: Dict[str, List[float]]
    mental_queries: Dict[str, List[float]]
    topic_event_ids: Dict[str, List[int]]
    topic_cluster_ids: Dict[str, List[int]]


@dataclass
class TestRuntime:
    config: PmConfig
    ghost: Any
    llm: Any | None


@dataclass(frozen=True)
class SyntheticSqliteGhostConfig:
    output_path: str
    number_of_days: int
    avg_features_per_day: int
    internal_thought_ratio: float
    random_other_feature_ratio: float
    seed: int = 17
    embedding_dim: int = DEFAULT_EMBEDDING_DIM


@dataclass(frozen=True)
class SyntheticScenarioPreset:
    name: str
    number_of_days: int
    avg_features_per_day: int
    internal_thought_ratio: float
    random_other_feature_ratio: float
    topics: Sequence[str]
    character_card_story: str


BASELINE_RELATIONSHIP_SCENARIO = SyntheticScenarioPreset(
    name="baseline_relationship",
    number_of_days=30,
    avg_features_per_day=28,
    internal_thought_ratio=0.22,
    random_other_feature_ratio=0.18,
    topics=("music", "gardening", "robotics", "travel", "trust_repair", "boundary_fear"),
    character_card_story=(
        "Mira is an AI companion with steady long-term memory, a warm relational baseline, "
        "and occasional spikes in trust, fear, and repair without losing continuity."
    ),
)

HEATED_CONFLICT_SCENARIO = SyntheticScenarioPreset(
    name="heated_conflict",
    number_of_days=21,
    avg_features_per_day=36,
    internal_thought_ratio=0.34,
    random_other_feature_ratio=0.22,
    topics=("boundary_fear", "trust_repair", "music", "heated_argument", "robotics", "travel"),
    character_card_story=(
        "Mira is an AI companion under relational strain. Trust, fear, rupture, apology, and repair recur, "
        "with enough calm baseline memory to contrast the spikes."
    ),
)

TRUST_REPAIR_SCENARIO = SyntheticScenarioPreset(
    name="trust_repair",
    number_of_days=24,
    avg_features_per_day=26,
    internal_thought_ratio=0.28,
    random_other_feature_ratio=0.16,
    topics=("trust_repair", "music", "gardening", "robotics", "travel", "boundary_fear"),
    character_card_story=(
        "Mira is learning to trust the user over time. The history includes rupture, reflection, apology, "
        "repair, and a gradually stabilizing bond."
    ),
)

SPARSE_HISTORY_SCENARIO = SyntheticScenarioPreset(
    name="sparse_history",
    number_of_days=10,
    avg_features_per_day=10,
    internal_thought_ratio=0.2,
    random_other_feature_ratio=0.1,
    topics=("music", "travel", "trust_repair", "robotics"),
    character_card_story=(
        "Mira is early in the relationship and the memory store is still sparse, with only a few repeated topics."
    ),
)

SYNTHETIC_SCENARIO_PRESETS: Dict[str, SyntheticScenarioPreset] = {
    preset.name: preset
    for preset in (
        BASELINE_RELATIONSHIP_SCENARIO,
        HEATED_CONFLICT_SCENARIO,
        TRUST_REPAIR_SCENARIO,
        SPARSE_HISTORY_SCENARIO,
    )
}


class TestGhost(KnoxelHaver):
    def __init__(self) -> None:
        super().__init__()
        self.current_knoxel_id = 0
        self.current_tick_id = 0
        self.ghost_config: GhostConfig
        self.system_config: PmConfig

        self.all_entities: List[Entity] = []
        self.all_stimuli: List[Stimulus] = []
        self.all_features: List[Feature] = []
        self.all_actions: List[Action] = []
        self.all_intentions: List[Intention] = []
        self.all_narratives: List[Narrative] = []
        self.all_episodic_memories: List[MemoryClusterKnoxel] = []
        self.all_declarative_facts: List[DeclarativeFactKnoxel] = []
        self.all_cause_effects: List[CauseEffectKnoxel] = []
        self.all_graph_nodes: List[GraphNode] = []
        self.all_graph_edges: List[GraphEdge] = []
        self.all_concepts: List[ConceptNode] = []
        self.all_percept_coalitions: List[PerceptCoalition] = []
        self.all_codelet_percepts: List[CodeletPercept] = []

    def get_simple_character_story_block(self) -> str:
        return self.ghost_config.universal_character_card

    def get_companion_name(self) -> str:
        return self.ghost_config.companion_name

    def get_user_name(self) -> str:
        return self.ghost_config.user_name

    def get_all_knoxels(self) -> List[KnoxelBase]:
        return list(self.all_knoxels.values())

    def add_knoxel(self, knoxel: KnoxelBase) -> KnoxelBase:
        self.current_knoxel_id += 1
        knoxel.id = self.current_knoxel_id
        knoxel.tick_id = max(1, knoxel.tick_id if knoxel.tick_id >= 0 else self.current_tick_id)
        knoxel._owner = self
        self.all_knoxels[knoxel.id] = knoxel

        if isinstance(knoxel, Entity):
            self.all_entities.append(knoxel)
        elif isinstance(knoxel, Stimulus):
            self.all_stimuli.append(knoxel)
        elif isinstance(knoxel, Feature):
            self.all_features.append(knoxel)
        elif isinstance(knoxel, Action):
            self.all_actions.append(knoxel)
        elif isinstance(knoxel, Intention):
            self.all_intentions.append(knoxel)
        elif isinstance(knoxel, Narrative):
            self.all_narratives.append(knoxel)
        elif isinstance(knoxel, MemoryClusterKnoxel):
            self.all_episodic_memories.append(knoxel)
        elif isinstance(knoxel, DeclarativeFactKnoxel):
            self.all_declarative_facts.append(knoxel)
        elif isinstance(knoxel, CauseEffectKnoxel):
            self.all_cause_effects.append(knoxel)
        elif isinstance(knoxel, GraphNode):
            self.all_graph_nodes.append(knoxel)
        elif isinstance(knoxel, GraphEdge):
            self.all_graph_edges.append(knoxel)
        elif isinstance(knoxel, ConceptNode):
            self.all_concepts.append(knoxel)
        elif isinstance(knoxel, PerceptCoalition):
            self.all_percept_coalitions.append(knoxel)
        elif isinstance(knoxel, CodeletPercept):
            self.all_codelet_percepts.append(knoxel)

        return knoxel

    def get_knoxel_by_id(self, knoxel_id: int) -> Optional[KnoxelBase]:
        return self.all_knoxels.get(knoxel_id)

    def bump_tick(self) -> int:
        self.current_tick_id += 1
        return self.current_tick_id


class _OfflineExportLlm:
    def get_embedding(self, text: str) -> List[float]:
        return []


def _create_sqlite_export_ghost(config: PmConfig, companion_name: str, user_name: str, character_card: str) -> BaseGhost:
    class SQLiteExportGhost(BaseGhost):
        def __init__(self, llm: _OfflineExportLlm, ghost_config: GhostConfig):
            super().__init__(llm, ghost_config)
            self.all_cause_effects: List[CauseEffectKnoxel] = []
            self.all_percept_coalitions: List[PerceptCoalition] = []
            self.all_codelet_percepts: List[CodeletPercept] = []

        def add_knoxel(self, knoxel: KnoxelBase, generate_embedding: bool = True):
            super().add_knoxel(knoxel, generate_embedding=generate_embedding)
            if isinstance(knoxel, CauseEffectKnoxel):
                self.all_cause_effects.append(knoxel)
            elif isinstance(knoxel, PerceptCoalition):
                self.all_percept_coalitions.append(knoxel)
            elif isinstance(knoxel, CodeletPercept):
                self.all_codelet_percepts.append(knoxel)

        def _reset_internal_state(self):
            super()._reset_internal_state()
            self.all_cause_effects = []
            self.all_percept_coalitions = []
            self.all_codelet_percepts = []

        def _rebuild_specific_lists(self):
            super()._rebuild_specific_lists()
            self.all_entities = [k for k in self.sorted_knoxels_causal if isinstance(k, Entity)]
            self.all_cause_effects = [k for k in self.sorted_knoxels_causal if isinstance(k, CauseEffectKnoxel)]
            self.all_percept_coalitions = [k for k in self.sorted_knoxels_causal if isinstance(k, PerceptCoalition)]
            self.all_codelet_percepts = [k for k in self.sorted_knoxels_causal if isinstance(k, CodeletPercept)]

    ghost_config = GhostConfig(
        companion_name=companion_name,
        user_name=user_name,
        universal_character_card=character_card,
    )
    ghost = SQLiteExportGhost(_OfflineExportLlm(), ghost_config)
    ghost.system_config = config
    return ghost


def create_test_ghost() -> TestGhost:
    return TestGhost()


def get_unit_test_config_path() -> Path:
    return Path(__file__).with_name("config.yaml")


def load_unit_test_config() -> PmConfig:
    return load_config(str(get_unit_test_config_path()))


def bootstrap_unit_test_runtime(*, start_llm: bool = False, initialize_basic_knoxels: bool = True) -> TestRuntime:
    from pm.ghost.ghost_config import GhostConfig
    from pm.ghost.ghost_r2 import GhostR2
    from pm.system.llm.llm_proxy import start_llm_thread

    cfg = load_unit_test_config()
    llm = start_llm_thread(cfg) if start_llm else None
    ghost_cfg = GhostConfig(
        companion_name=cfg.companion_name,
        user_name=cfg.user_name,
        universal_character_card=cfg.character_card_story,
    )
    ghost = GhostR2(llm, ghost_cfg)
    ghost.system_config = cfg
    if initialize_basic_knoxels:
        ghost.initialize_basic_knoxels()
    return TestRuntime(config=cfg, ghost=ghost, llm=llm)


def generate_populated_ghost_sqlite(
    output_path: str,
    number_of_days: int,
    avg_features_per_day: int,
    internal_thought_ratio: float,
    random_other_feature_ratio: float,
    seed: int = 17,
) -> str:
    export_config = SyntheticSqliteGhostConfig(
        output_path=output_path,
        number_of_days=number_of_days,
        avg_features_per_day=avg_features_per_day,
        internal_thought_ratio=internal_thought_ratio,
        random_other_feature_ratio=random_other_feature_ratio,
        seed=seed,
    )
    _validate_export_config_strict(export_config)
    corpus = _build_sqlite_export_corpus(export_config)
    validate_generated_ghost_strict(corpus.ghost)
    runtime_config = load_unit_test_config()
    runtime_config.commit = True
    runtime_config.db_path = export_config.output_path
    export_ghost = _create_sqlite_export_ghost(
        runtime_config,
        companion_name=corpus.ghost.ghost_config.companion_name,
        user_name=corpus.ghost.ghost_config.user_name,
        character_card=corpus.ghost.ghost_config.universal_character_card,
    )
    _copy_corpus_into_runtime_ghost(export_ghost, corpus.ghost)

    from pm.persist.persist_sqlite import PersistSqlite

    PersistSqlite(export_ghost).save_state_sqlite(export_config.output_path)
    return str(Path(export_config.output_path).resolve())


def generate_populated_ghost_sqlite_from_scenario(output_path: str, scenario_name: str, seed: int = 17) -> str:
    preset = SYNTHETIC_SCENARIO_PRESETS[scenario_name]
    export_config = SyntheticSqliteGhostConfig(
        output_path=output_path,
        number_of_days=preset.number_of_days,
        avg_features_per_day=preset.avg_features_per_day,
        internal_thought_ratio=preset.internal_thought_ratio,
        random_other_feature_ratio=preset.random_other_feature_ratio,
        seed=seed,
    )
    _validate_export_config_strict(export_config)
    corpus = _build_sqlite_export_corpus(export_config, preset=preset)
    validate_generated_ghost_strict(corpus.ghost)
    runtime_config = load_unit_test_config()
    runtime_config.commit = True
    runtime_config.db_path = export_config.output_path
    export_ghost = _create_sqlite_export_ghost(
        runtime_config,
        companion_name=corpus.ghost.ghost_config.companion_name,
        user_name=corpus.ghost.ghost_config.user_name,
        character_card=corpus.ghost.ghost_config.universal_character_card,
    )
    _copy_corpus_into_runtime_ghost(export_ghost, corpus.ghost)

    from pm.persist.persist_sqlite import PersistSqlite

    PersistSqlite(export_ghost).save_state_sqlite(export_config.output_path)
    return str(Path(export_config.output_path).resolve())


def build_scenario_corpus(scenario_name: str, seed: int = 17) -> GeneratedCorpus:
    preset = SYNTHETIC_SCENARIO_PRESETS[scenario_name]
    export_config = SyntheticSqliteGhostConfig(
        output_path="",
        number_of_days=preset.number_of_days,
        avg_features_per_day=preset.avg_features_per_day,
        internal_thought_ratio=preset.internal_thought_ratio,
        random_other_feature_ratio=preset.random_other_feature_ratio,
        seed=seed,
    )
    _validate_export_config_strict(export_config)
    corpus = _build_sqlite_export_corpus(export_config, preset=preset)
    validate_generated_ghost_strict(corpus.ghost)
    return corpus


def load_persisted_ghost_strict(db_path: str) -> BaseGhost:
    runtime_config = load_unit_test_config()
    runtime_config.commit = True
    runtime_config.db_path = db_path
    ghost = _create_sqlite_export_ghost(
        runtime_config,
        companion_name=runtime_config.companion_name,
        user_name=runtime_config.user_name,
        character_card=runtime_config.character_card_story,
    )
    from pm.persist.persist_sqlite import PersistSqlite

    if not PersistSqlite(ghost).load_state_sqlite(db_path):
        raise RuntimeError(f"Could not load persisted ghost from {db_path}")
    validate_generated_ghost_strict(ghost)
    return ghost


def collect_ghost_stats_strict(ghost: BaseGhost | TestGhost) -> Dict[str, Any]:
    validate_generated_ghost_strict(ghost)

    feature_type_counts: Dict[str, int] = {}
    for feature in ghost.all_features:
        key = feature.feature_type.value
        feature_type_counts[key] = feature_type_counts.get(key, 0) + 1

    return {
        "knoxel_count": len(ghost.all_knoxels),
        "current_knoxel_id": ghost.current_knoxel_id,
        "current_tick_id": ghost.current_tick_id,
        "type_counts": summarize_knoxel_types(ghost.all_knoxels.values()),
        "feature_type_counts": feature_type_counts,
        "cluster_signatures": [
            (
                cluster.id,
                cluster.cluster_type.value,
                cluster.level,
                cluster.included_event_ids or "",
                cluster.included_cluster_ids or "",
                cluster.timestamp_world_begin.isoformat(),
                cluster.timestamp_world_end.isoformat(),
            )
            for cluster in sorted(ghost.all_episodic_memories, key=lambda item: item.id)
        ],
        "fact_signatures": [
            (
                fact.id,
                fact.source_cluster_id,
                tuple(fact.category),
                fact.min_event_id,
                fact.max_event_id,
                fact.timestamp_world_begin.isoformat(),
                fact.timestamp_world_end.isoformat(),
            )
            for fact in sorted(ghost.all_declarative_facts, key=lambda item: item.id)
        ],
        "cause_effect_signatures": [
            (
                item.id,
                item.source_cluster_id,
                item.situation,
                item.cause,
                item.effect,
                item.timestamp_world_begin.isoformat(),
                item.timestamp_world_end.isoformat(),
            )
            for item in sorted(getattr(ghost, "all_cause_effects", []), key=lambda item: item.id)
        ],
        "first_timestamp": min(knoxel.timestamp_world_begin for knoxel in ghost.all_knoxels.values()).isoformat(),
        "last_timestamp": max(knoxel.timestamp_world_end for knoxel in ghost.all_knoxels.values()).isoformat(),
    }


def build_query_embedding(topic: str, config: Optional[TestInfrastructureConfig] = None) -> List[float]:
    config = config or TestInfrastructureConfig()
    try:
        topic_index = list(config.topics).index(topic)
    except ValueError as exc:
        raise KeyError(f"Unknown topic: {topic}") from exc
    return _compose_embedding(
        base=_make_topic_waveform(topic_index, config.embedding_dim),
        type_bias=_make_type_bias("query", config.embedding_dim),
        time_position=0.85,
    )


def summarize_knoxel_types(knoxels: Iterable[KnoxelBase]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for knoxel in knoxels:
        summary[knoxel.type.value] = summary.get(knoxel.type.value, 0) + 1
    return summary


def populate_test_ghost(config: Optional[TestInfrastructureConfig] = None) -> GeneratedCorpus:
    config = config or TestInfrastructureConfig()
    rng = random.Random(config.seed)
    ghost = create_test_ghost()
    ghost.ghost_config = GhostConfig(
        companion_name=_get_primary_name(config, EntityClass.AI, fallback="Mira"),
        user_name=_get_primary_name(config, EntityClass.Human, fallback="Rick"),
        universal_character_card=config.character_card_story,
    )
    ghost.system_config = load_unit_test_config()
    ghost.system_config.character_card_story = config.character_card_story
    ghost.system_config.companion_name = ghost.ghost_config.companion_name
    ghost.system_config.user_name = ghost.ghost_config.user_name

    entities = _create_entities(ghost, config)
    topic_event_ids: Dict[str, List[int]] = {topic: [] for topic in config.topics}
    topic_cluster_ids: Dict[str, List[int]] = {topic: [] for topic in config.topics}
    topic_waveforms = {topic: _make_topic_waveform(index, config.embedding_dim) for index, topic in enumerate(config.topics)}
    query_embeddings = {topic: build_query_embedding(topic, config) for topic in config.topics}
    mental_queries = {
        "distress": _build_emotional_query(
            valence=-0.85,
            arousal=0.78,
            sadness=0.85,
            anger=0.45,
            fear=0.72,
            tenderness=0.0,
            goal_congruence=-0.75,
            trust=0.15,
        ),
        "warmth": _build_emotional_query(
            valence=0.8,
            arousal=0.45,
            sadness=0.0,
            anger=0.0,
            fear=0.0,
            tenderness=0.8,
            goal_congruence=0.85,
            trust=0.8,
        ),
        "music_distress": _build_emotional_query(
            valence=-0.85,
            arousal=0.7,
            sadness=0.85,
            anger=0.1,
            fear=0.35,
            tenderness=0.0,
            goal_congruence=-0.75,
            trust=0.15,
        ),
        "music_warmth": _build_emotional_query(
            valence=0.8,
            arousal=0.45,
            sadness=0.0,
            anger=0.0,
            fear=0.0,
            tenderness=0.8,
            goal_congruence=0.85,
            trust=0.8,
        ),
    }

    all_scene_features: List[Feature] = []
    for day_index in range(config.scale.days):
        day_start = config.start_time_utc + timedelta(days=day_index)
        for scene_index in range(config.scale.scenes_per_day):
            topic = config.topics[(day_index + scene_index) % len(config.topics)]
            scene_time = day_start + timedelta(hours=scene_index * 3, minutes=15 * ((scene_index + day_index) % 3))
            scene = _create_scene(
                ghost=ghost,
                rng=rng,
                entities=entities,
                topic=topic,
                scene_time=scene_time,
                scene_index=(day_index * config.scale.scenes_per_day) + scene_index,
                embedding_dim=config.embedding_dim,
                topic_waveform=topic_waveforms[topic],
            )
            all_scene_features.extend(scene["features"])
            topic_event_ids[topic].extend(feature.id for feature in scene["features"] if feature.causal)

    _create_narratives(ghost, entities, config, topic_waveforms)
    topical_clusters = _create_topical_clusters(ghost, config, topic_event_ids, topic_waveforms)
    for cluster in topical_clusters:
        topic_cluster_ids[cluster.metadata["topic"]].append(cluster.id)

    _create_temporal_clusters(ghost, config, all_scene_features, topic_waveforms)
    _create_facts_and_rules(ghost, topical_clusters, topic_waveforms)
    _create_graph_memory(ghost, entities, topical_clusters, topic_waveforms, config.embedding_dim)

    return GeneratedCorpus(
        ghost=ghost,
        config=config,
        entities=entities,
        query_embeddings=query_embeddings,
        mental_queries=mental_queries,
        topic_event_ids=topic_event_ids,
        topic_cluster_ids=topic_cluster_ids,
    )


def populate_test_ghost_raw_only(config: Optional[TestInfrastructureConfig] = None) -> GeneratedCorpus:
    config = config or TestInfrastructureConfig()
    rng = random.Random(config.seed)
    ghost = create_test_ghost()
    ghost.ghost_config = GhostConfig(
        companion_name=_get_primary_name(config, EntityClass.AI, fallback="Mira"),
        user_name=_get_primary_name(config, EntityClass.Human, fallback="Rick"),
        universal_character_card=config.character_card_story,
    )
    ghost.system_config = load_unit_test_config()
    ghost.system_config.character_card_story = config.character_card_story
    ghost.system_config.companion_name = ghost.ghost_config.companion_name
    ghost.system_config.user_name = ghost.ghost_config.user_name

    entities = _create_entities(ghost, config)
    topic_event_ids: Dict[str, List[int]] = {topic: [] for topic in config.topics}
    topic_waveforms = {topic: _make_topic_waveform(index, config.embedding_dim) for index, topic in enumerate(config.topics)}
    query_embeddings = {topic: build_query_embedding(topic, config) for topic in config.topics}
    mental_queries = {
        "distress": _build_emotional_query(
            valence=-0.85,
            arousal=0.78,
            sadness=0.85,
            anger=0.45,
            fear=0.72,
            tenderness=0.0,
            goal_congruence=-0.75,
            trust=0.15,
        ),
        "warmth": _build_emotional_query(
            valence=0.8,
            arousal=0.45,
            sadness=0.0,
            anger=0.0,
            fear=0.0,
            tenderness=0.8,
            goal_congruence=0.85,
            trust=0.8,
        ),
    }

    for day_index in range(config.scale.days):
        day_start = config.start_time_utc + timedelta(days=day_index)
        for scene_index in range(config.scale.scenes_per_day):
            topic = config.topics[(day_index + scene_index) % len(config.topics)]
            scene_time = day_start + timedelta(hours=scene_index * 3, minutes=15 * ((scene_index + day_index) % 3))
            scene = _create_scene(
                ghost=ghost,
                rng=rng,
                entities=entities,
                topic=topic,
                scene_time=scene_time,
                scene_index=(day_index * config.scale.scenes_per_day) + scene_index,
                embedding_dim=config.embedding_dim,
                topic_waveform=topic_waveforms[topic],
            )
            topic_event_ids[topic].extend(feature.id for feature in scene["features"] if feature.causal)

    return GeneratedCorpus(
        ghost=ghost,
        config=config,
        entities=entities,
        query_embeddings=query_embeddings,
        mental_queries=mental_queries,
        topic_event_ids=topic_event_ids,
        topic_cluster_ids={topic: [] for topic in config.topics},
    )


def _build_sqlite_export_corpus(
    export_config: SyntheticSqliteGhostConfig,
    preset: SyntheticScenarioPreset | None = None,
) -> GeneratedCorpus:
    scenes_per_day = max(1, export_config.avg_features_per_day // 4)
    scale = GenerationScale(
        days=export_config.number_of_days,
        scenes_per_day=scenes_per_day,
        topical_clusters_per_topic=max(2, min(5, export_config.number_of_days // 4 or 2)),
        temporal_cluster_levels=(6, 5, 4, 3) if export_config.number_of_days >= 8 else (6, 5, 4),
    )
    topics = preset.topics if preset is not None else ("music", "gardening", "robotics", "travel", "trust_repair", "boundary_fear")
    character_card_story = (
        preset.character_card_story
        if preset is not None
        else (
            "Mira is an AI companion with a long relational memory. "
            "She starts guarded, grows warmer through repeated contact, settles into a baseline, "
            "and still has strong spikes around trust, fear, repair, and uncertainty."
        )
    )
    config = TestInfrastructureConfig(
        seed=export_config.seed,
        embedding_dim=export_config.embedding_dim,
        scale=scale,
        topics=topics,
        character_card_story=character_card_story,
    )
    corpus = populate_test_ghost(config)
    _retune_feature_distribution_for_export(corpus.ghost, export_config)
    _rewrite_knoxel_content_for_export(corpus.ghost)
    return corpus


def _validate_export_config_strict(export_config: SyntheticSqliteGhostConfig) -> None:
    if export_config.number_of_days < 1:
        raise ValueError("number_of_days must be >= 1")
    if export_config.avg_features_per_day < 4:
        raise ValueError("avg_features_per_day must be >= 4")
    if not 0.0 <= export_config.internal_thought_ratio <= 1.0:
        raise ValueError("internal_thought_ratio must be between 0.0 and 1.0")
    if not 0.0 <= export_config.random_other_feature_ratio <= 1.0:
        raise ValueError("random_other_feature_ratio must be between 0.0 and 1.0")
    if export_config.internal_thought_ratio + export_config.random_other_feature_ratio > 1.0:
        raise ValueError("internal_thought_ratio + random_other_feature_ratio must be <= 1.0")
    if export_config.embedding_dim < 8:
        raise ValueError("embedding_dim must be >= 8")


def _copy_corpus_into_runtime_ghost(target_ghost: BaseGhost, source_ghost: TestGhost) -> None:
    target_ghost.ghost_config.universal_character_card = source_ghost.ghost_config.universal_character_card
    target_ghost.ghost_config.companion_name = source_ghost.ghost_config.companion_name
    target_ghost.ghost_config.user_name = source_ghost.ghost_config.user_name

    for knoxel in source_ghost.sorted_knoxels_causal:
        target_ghost.add_knoxel(knoxel.model_copy(deep=True), generate_embedding=False)

    target_ghost.current_knoxel_id = source_ghost.current_knoxel_id
    target_ghost.current_tick_id = source_ghost.current_tick_id


def _retune_feature_distribution_for_export(ghost: TestGhost, export_config: SyntheticSqliteGhostConfig) -> None:
    causal_features = [feature for feature in ghost.all_features if feature.causal]
    target_total = export_config.number_of_days * export_config.avg_features_per_day
    other_target = int(target_total * export_config.random_other_feature_ratio)
    thought_target = int(target_total * export_config.internal_thought_ratio)
    current_other = len([f for f in causal_features if f.feature_type not in (FeatureType.Dialogue, FeatureType.Thought)])
    current_thought = len([f for f in causal_features if f.feature_type == FeatureType.Thought])

    _apply_export_emotional_arc(causal_features, export_config)

    for feature in causal_features:
        if current_thought < thought_target and feature.feature_type == FeatureType.StoryWildcard:
            feature.feature_type = FeatureType.Thought
            feature.interlocus = InterlocusType.PrivateInternal
            current_thought += 1
        elif current_other < other_target and feature.feature_type == FeatureType.Dialogue and feature.source == "Mira":
            feature.feature_type = FeatureType.SubjectiveExperience
            feature.interlocus = InterlocusType.PrivateReportable
            current_other += 1

    while len(causal_features) < target_total:
        anchor = causal_features[len(causal_features) % len(causal_features)]
        extra_kind = _pick_export_extra_feature_type(current_thought, thought_target, current_other, other_target)
        timestamp = anchor.timestamp_world_end + timedelta(seconds=10 + (len(causal_features) % 9))
        appraisal, delta = _build_export_emotional_vectors(
            day_index=(timestamp - ghost.all_features[0].timestamp_world_begin).days,
            scene_index=anchor.metadata.get("scene_index", 0),
            mode=anchor.metadata.get("phase", "user_present"),
            variant_index=len(causal_features),
        )
        extra_feature = ghost.add_knoxel(
            Feature(
                tick_id=anchor.tick_id,
                content="placeholder",
                source=anchor.source if extra_kind != FeatureType.Thought else "Mira",
                feature_type=extra_kind,
                interlocus=InterlocusType.PrivateInternal if extra_kind == FeatureType.Thought else InterlocusType.PrivateReportable,
                causal=True,
                embedding=_compose_embedding(anchor.embedding, _make_type_bias(extra_kind.value, len(anchor.embedding)), 0.67),
                timestamp_creation=timestamp,
                timestamp_world_begin=timestamp,
                timestamp_world_end=timestamp + timedelta(seconds=20),
                mental_state_appraisal=appraisal,
                mental_state_delta=delta,
                metadata=dict(anchor.metadata),
            )
        )
        causal_features.append(extra_feature)
        if extra_kind == FeatureType.Thought:
            current_thought += 1
        elif extra_kind != FeatureType.Dialogue:
            current_other += 1


def _apply_export_emotional_arc(features: Sequence[Feature], export_config: SyntheticSqliteGhostConfig) -> None:
    for index, feature in enumerate(sorted(features, key=lambda item: item.timestamp_world_begin)):
        day_index = (feature.timestamp_world_begin.date() - features[0].timestamp_world_begin.date()).days
        appraisal, delta = _build_export_emotional_vectors(
            day_index=day_index,
            scene_index=feature.metadata.get("scene_index", 0),
            mode=feature.metadata.get("phase", "user_present"),
            variant_index=index,
        )
        feature.mental_state_appraisal = appraisal
        feature.mental_state_delta = delta


def _pick_export_extra_feature_type(current_thought: int, thought_target: int, current_other: int, other_target: int) -> FeatureType:
    if current_thought < thought_target:
        return FeatureType.Thought
    if current_other < other_target:
        return FeatureType.SubjectiveExperience
    return FeatureType.Dialogue


def _build_export_emotional_vectors(day_index: int, scene_index: int, mode: str, variant_index: int) -> tuple[List[float], List[float]]:
    appraisal = create_empty_ms_vector()
    delta = create_empty_ms_vector()

    idx_goal = AppraisalGeneral.model_fields["goal_congruence"].json_schema_extra["vector_position"]
    idx_trust = AppraisalSocial.model_fields["trust_cues"].json_schema_extra["vector_position"]
    idx_valence = StateCore.model_fields["valence"].json_schema_extra["vector_position"]
    idx_arousal = StateCore.model_fields["arousal"].json_schema_extra["vector_position"]
    idx_joy = StateEmotions.model_fields["joy"].json_schema_extra["vector_position"]
    idx_sadness = StateEmotions.model_fields["sadness"].json_schema_extra["vector_position"]
    idx_anger = StateEmotions.model_fields["anger"].json_schema_extra["vector_position"]
    idx_fear = StateEmotions.model_fields["fear"].json_schema_extra["vector_position"]
    idx_tenderness = StateEmotions.model_fields["tenderness"].json_schema_extra["vector_position"]

    intro_gain = min(1.0, day_index / 4.0)
    trust_wave = math.sin((day_index + 1) * 0.65 + (scene_index * 0.4))
    fear_wave = math.cos((day_index + 2) * 0.52 + (variant_index * 0.17))
    baseline = 0.2 + 0.35 * intro_gain

    appraisal[idx_goal] = -0.15 + baseline + 0.18 * trust_wave
    appraisal[idx_trust] = 0.25 + 0.4 * intro_gain + 0.12 * math.sin((day_index + 1) * 0.4)
    delta[idx_valence] = -0.1 + baseline + 0.24 * trust_wave
    delta[idx_arousal] = 0.3 + 0.22 * abs(fear_wave)
    delta[idx_joy] = max(0.0, 0.15 + 0.25 * trust_wave)
    delta[idx_tenderness] = max(0.0, 0.12 + 0.28 * intro_gain)
    delta[idx_sadness] = max(0.0, 0.08 - 0.06 * intro_gain)
    delta[idx_anger] = 0.0
    delta[idx_fear] = max(0.0, 0.08 + 0.22 * max(0.0, fear_wave))

    if mode == "internal_only":
        delta[idx_valence] -= 0.24
        delta[idx_arousal] += 0.16
        delta[idx_sadness] += 0.2
        delta[idx_fear] += 0.18
    elif mode == "user_returns":
        delta[idx_valence] += 0.1
        delta[idx_tenderness] += 0.08

    if (day_index + scene_index) % 9 == 4:
        delta[idx_fear] += 0.42
        delta[idx_valence] -= 0.18
    if (day_index + scene_index) % 11 == 7:
        delta[idx_joy] += 0.36
        delta[idx_tenderness] += 0.16
        appraisal[idx_trust] += 0.18

    return appraisal, delta


def _rewrite_knoxel_content_for_export(ghost: TestGhost) -> None:
    for knoxel in ghost.sorted_knoxels_causal:
        if isinstance(knoxel, MemoryClusterKnoxel):
            included = knoxel.included_event_ids or knoxel.included_cluster_ids or ""
            knoxel.content = (
                f"type: {knoxel.type.value} cluster: {knoxel.cluster_type.value} level: {knoxel.level} "
                f"ids included: {included} timespan: {knoxel.timestamp_world_begin.isoformat()} - {knoxel.timestamp_world_end.isoformat()} "
                f"topic: {knoxel.metadata.get('topic', 'general')}"
            )
            continue
        if isinstance(knoxel, DeclarativeFactKnoxel):
            knoxel.content = (
                f"type: {knoxel.type.value} source_cluster: {knoxel.source_cluster_id} "
                f"event_span: {knoxel.min_event_id}-{knoxel.max_event_id} "
                f"tick_span: {knoxel.min_tick_id}-{knoxel.max_tick_id} "
                f"topic: {knoxel.metadata.get('topic', 'general')}"
            )
            continue
        if isinstance(knoxel, CauseEffectKnoxel):
            knoxel.content = (
                f"type: {knoxel.type.value} source_cluster: {knoxel.source_cluster_id} "
                f"situation: {knoxel.situation} cause: {knoxel.cause} effect: {knoxel.effect} "
                f"timespan: {knoxel.timestamp_world_begin.isoformat()} - {knoxel.timestamp_world_end.isoformat()}"
            )
            continue
        topic = knoxel.metadata.get("topic", "general")
        scene = knoxel.metadata.get("scene_index", "-")
        phase = knoxel.metadata.get("phase", "n/a")
        source = getattr(knoxel, "source", None) or knoxel.metadata.get("speaker", "system")
        target = _describe_knoxel_target(knoxel)
        knoxel.content = (
            f"type: {knoxel.type.value} subtype: {getattr(knoxel, 'feature_type', getattr(knoxel, 'cluster_type', 'none'))} "
            f"source: {source} scene: {scene} phase: {phase} topic: {topic} target: {target} "
            f"signal: {_describe_knoxel_signal(knoxel)}"
        )


def _describe_knoxel_target(knoxel: KnoxelBase) -> str:
    if isinstance(knoxel, Feature):
        fear = knoxel.mental_state_delta[StateEmotions.model_fields["fear"].json_schema_extra["vector_position"]]
        trust = knoxel.mental_state_appraisal[AppraisalSocial.model_fields["trust_cues"].json_schema_extra["vector_position"]]
        if fear > 0.45:
            return "high-fear-simulation"
        if trust > 0.6:
            return "trust-rise"
        return "baseline-regulation"
    if isinstance(knoxel, DeclarativeFactKnoxel):
        return f"fact-cluster-{knoxel.source_cluster_id}"
    if isinstance(knoxel, CauseEffectKnoxel):
        return f"cause-effect-{knoxel.source_cluster_id}"
    if isinstance(knoxel, MemoryClusterKnoxel):
        return f"memory-level-{knoxel.level}"
    return "general-diagnostic"


def _describe_knoxel_signal(knoxel: KnoxelBase) -> str:
    if isinstance(knoxel, Feature):
        valence = knoxel.mental_state_delta[StateCore.model_fields["valence"].json_schema_extra["vector_position"]]
        fear = knoxel.mental_state_delta[StateEmotions.model_fields["fear"].json_schema_extra["vector_position"]]
        joy = knoxel.mental_state_delta[StateEmotions.model_fields["joy"].json_schema_extra["vector_position"]]
        if fear > joy and fear > 0.35:
            return "fear-dominant"
        if joy > 0.35:
            return "warm-engagement"
        if valence < -0.2:
            return "strained-reflection"
        return "neutral-baseline"
    if isinstance(knoxel, MemoryClusterKnoxel):
        return "consolidated-memory"
    if isinstance(knoxel, DeclarativeFactKnoxel):
        return "extracted-fact"
    if isinstance(knoxel, CauseEffectKnoxel):
        return "causal-pattern"
    return "supporting-artifact"


def validate_generated_ghost_strict(ghost: BaseGhost | TestGhost) -> None:
    if not ghost.all_knoxels:
        raise ValueError("ghost has no knoxels")
    if ghost.current_knoxel_id != max(ghost.all_knoxels):
        raise ValueError("current_knoxel_id does not match max knoxel id")
    if ghost.current_tick_id < 1:
        raise ValueError("current_tick_id must be >= 1")

    _validate_knoxel_time_order_strict(ghost)
    _validate_feature_distribution_strict(ghost)
    _validate_cluster_provenance_strict(ghost)
    _validate_fact_and_cause_effect_links_strict(ghost)
    _validate_temporal_hierarchy_strict(ghost)


def _validate_knoxel_time_order_strict(ghost: BaseGhost | TestGhost) -> None:
    for knoxel in ghost.all_knoxels.values():
        if knoxel.timestamp_world_begin > knoxel.timestamp_world_end:
            raise ValueError(f"knoxel {knoxel.id} has inverted world timestamps")
        if knoxel.embedding is not None and len(knoxel.embedding) == 0:
            raise ValueError(f"knoxel {knoxel.id} has empty embedding list")


def _validate_feature_distribution_strict(ghost: BaseGhost | TestGhost) -> None:
    if not ghost.all_features:
        raise ValueError("ghost has no features")
    causal_features = [feature for feature in ghost.all_features if feature.causal]
    if not causal_features:
        raise ValueError("ghost has no causal features")
    if not any(feature.feature_type == FeatureType.Dialogue for feature in causal_features):
        raise ValueError("ghost has no causal dialogue features")
    #if not any(feature.feature_type == FeatureType.Thought for feature in causal_features):
    #    raise ValueError("ghost has no causal thought features")


def _validate_cluster_provenance_strict(ghost: BaseGhost | TestGhost) -> None:
    cluster_by_id = {cluster.id: cluster for cluster in ghost.all_episodic_memories}
    if not cluster_by_id:
        raise ValueError("ghost has no episodic memories")
    for cluster in ghost.all_episodic_memories:
        if cluster.cluster_type == ClusterType.Topical:
            if not cluster.included_event_ids:
                raise ValueError(f"topical cluster {cluster.id} has no included_event_ids")
            event_ids = [int(value) for value in cluster.included_event_ids.split(",") if value]
            events = [ghost.get_knoxel_by_id(event_id) for event_id in event_ids]
            if not events or any(event is None for event in events):
                raise ValueError(f"topical cluster {cluster.id} references missing events")
            begin = min(event.timestamp_world_begin for event in events)
            end = max(event.timestamp_world_end for event in events)
            if cluster.timestamp_world_begin != begin or cluster.timestamp_world_end != end:
                raise ValueError(f"topical cluster {cluster.id} timespan does not match covered events")
        elif cluster.cluster_type == ClusterType.Temporal:
            if cluster.included_cluster_ids:
                child_ids = [int(value) for value in cluster.included_cluster_ids.split(",") if value]
                children = [cluster_by_id[child_id] for child_id in child_ids]
                begin = min(child.timestamp_world_begin for child in children)
                end = max(child.timestamp_world_end for child in children)
                if cluster.timestamp_world_begin != begin or cluster.timestamp_world_end != end:
                    raise ValueError(f"temporal cluster {cluster.id} timespan does not match child clusters")
                if any(child.level <= cluster.level for child in children):
                    raise ValueError(f"temporal cluster {cluster.id} has invalid child level")
            elif not cluster.included_event_ids:
                raise ValueError(f"temporal cluster {cluster.id} must include events or child clusters")


def _validate_fact_and_cause_effect_links_strict(ghost: BaseGhost | TestGhost) -> None:
    cluster_ids = {cluster.id for cluster in ghost.all_episodic_memories}
    if not ghost.all_declarative_facts:
        raise ValueError("ghost has no declarative facts")
    for fact in ghost.all_declarative_facts:
        if fact.source_cluster_id not in cluster_ids:
            raise ValueError(f"fact {fact.id} references missing source cluster")
    for item in ghost.all_cause_effects:
        if item.source_cluster_id not in cluster_ids:
            raise ValueError(f"cause/effect {item.id} references missing source cluster")


def _validate_temporal_hierarchy_strict(ghost: BaseGhost | TestGhost) -> None:
    temporal_clusters = [cluster for cluster in ghost.all_episodic_memories if cluster.cluster_type == ClusterType.Temporal]
    if not temporal_clusters:
        raise ValueError("ghost has no temporal clusters")
    by_level: Dict[int, List[MemoryClusterKnoxel]] = {}
    for cluster in temporal_clusters:
        by_level.setdefault(cluster.level, []).append(cluster)
    for level, clusters in by_level.items():
        clusters.sort(key=lambda item: item.timestamp_world_begin)
        for current, nxt in zip(clusters, clusters[1:]):
            if current.timestamp_world_end > nxt.timestamp_world_begin:
                raise ValueError(f"temporal clusters overlap at level {level}")


def _create_entities(ghost: TestGhost, config: TestInfrastructureConfig) -> Dict[str, Entity]:
    entities: Dict[str, Entity] = {}
    for index, spec in enumerate(config.characters):
        entity = Entity(
            content=spec.name,
            aliases=list(spec.aliases),
            entity_class=spec.entity_class,
            embedding=_compose_embedding(
                base=_make_topic_waveform(index, config.embedding_dim),
                type_bias=_make_type_bias("entity", config.embedding_dim),
                time_position=0.0,
            ),
            timestamp_creation=config.start_time_utc,
            timestamp_world_begin=config.start_time_utc,
            timestamp_world_end=config.start_time_utc,
        )
        ghost.add_knoxel(entity)
        entities[spec.name] = entity
    return entities


def _get_primary_name(config: TestInfrastructureConfig, entity_class: EntityClass, fallback: str) -> str:
    for spec in config.characters:
        if spec.entity_class == entity_class:
            return spec.name
    return fallback


def _create_scene(
    ghost: TestGhost,
    rng: random.Random,
    entities: Dict[str, Entity],
    topic: str,
    scene_time: datetime,
    scene_index: int,
    embedding_dim: int,
    topic_waveform: List[float],
) -> Dict[str, List[KnoxelBase]]:
    user = entities["Rick"]
    companion = entities["Mira"]
    witness = entities["Tara"]
    archivist = entities["Archivist"]

    tick_id = ghost.bump_tick()
    topic_token = topic.upper()
    time_marker = scene_time.strftime("%Y-%m-%d %H:%M")
    time_position = min(1.0, 0.05 * scene_index)
    phase_name = ("user_present", "internal_only", "user_returns")[scene_index % 3]
    emotional_state, emotional_delta = _build_scene_emotional_vectors(topic=topic, phase_name=phase_name, scene_index=scene_index)

    scene_variant = rng.randint(1, 5)
    stimulus_type = StimulusType.UserMessage if phase_name != "internal_only" else StimulusType.UserInactivity
    stimulus_source = user.content if phase_name != "internal_only" else companion.content
    stimulus_content = (
        f"{user.content} introduces a {topic} topic at {time_marker}."
        if phase_name != "internal_only"
        else f"{companion.content} slips into a private {topic} reflection at {time_marker}."
    )
    ghost.add_knoxel(
        Stimulus(
            tick_id=tick_id,
            content=stimulus_content,
            source=stimulus_source,
            stimulus_type=stimulus_type,
            embedding=_compose_embedding(topic_waveform, _make_type_bias("stimulus", embedding_dim), time_position),
            timestamp_creation=scene_time,
            timestamp_world_begin=scene_time,
            timestamp_world_end=scene_time + timedelta(seconds=20),
            metadata={"topic": topic, "scene_index": scene_index, "phase": phase_name, "active_user": stimulus_source if phase_name != "internal_only" else None},
        )
    )
    ghost.add_knoxel(
        Stimulus(
            tick_id=tick_id,
            content=f"{archivist.content} flags prior {topic} memory for retrieval in scene {scene_index}.",
            source=archivist.content,
            stimulus_type=StimulusType.MemoryRecall,
            embedding=_compose_embedding(topic_waveform, _make_type_bias("stimulus_memory", embedding_dim), time_position),
            timestamp_creation=scene_time + timedelta(minutes=5),
            timestamp_world_begin=scene_time + timedelta(minutes=5),
            timestamp_world_end=scene_time + timedelta(minutes=5, seconds=20),
            metadata={"topic": topic, "scene_index": scene_index, "phase": phase_name, "active_user": user.content if phase_name != "internal_only" else None},
        )
    )

    user_dialogue = ghost.add_knoxel(
        Feature(
            tick_id=tick_id,
            content=(
                f"[{topic_token}] {user.content} asks about {topic} pattern {scene_index}."
                if phase_name != "internal_only"
                else f"[{topic_token}] {companion.content} quietly restates the {topic} problem to herself in scene {scene_index}."
            ),
            source=user.content if phase_name != "internal_only" else companion.content,
            feature_type=FeatureType.Dialogue if phase_name != "internal_only" else FeatureType.ExternalThought,
            interlocus=InterlocusType.Public if phase_name != "internal_only" else InterlocusType.PrivateInternal,
            causal=True,
            embedding=_compose_embedding(topic_waveform, _make_type_bias("dialogue_user", embedding_dim), time_position),
            timestamp_creation=scene_time + timedelta(seconds=30),
            timestamp_world_begin=scene_time + timedelta(seconds=30),
            timestamp_world_end=scene_time + timedelta(minutes=1),
            mental_state_appraisal=emotional_state,
            mental_state_delta=emotional_delta,
            metadata={"topic": topic, "speaker": user.content if phase_name != "internal_only" else companion.content, "scene_index": scene_index, "phase": phase_name, "active_user": user.content if phase_name != "internal_only" else None},
        )
    )
    thought = ghost.add_knoxel(
        Feature(
            tick_id=tick_id,
            content=_build_internal_thought_text(companion.content, topic_token, topic, scene_index, phase_name),
            source=companion.content,
            feature_type=FeatureType.Thought,
            interlocus=InterlocusType.PrivateReportable,
            causal=True,
            embedding=_compose_embedding(topic_waveform, _make_type_bias("thought", embedding_dim), time_position),
            timestamp_creation=scene_time + timedelta(minutes=1, seconds=10),
            timestamp_world_begin=scene_time + timedelta(minutes=1, seconds=10),
            timestamp_world_end=scene_time + timedelta(minutes=2),
            mental_state_appraisal=emotional_state,
            mental_state_delta=emotional_delta,
            metadata={"topic": topic, "speaker": companion.content, "scene_index": scene_index, "phase": phase_name, "active_user": user.content if phase_name != "internal_only" else None},
        )
    )
    action = ghost.add_knoxel(
        Action(
            tick_id=tick_id,
            content=f"Reply with a grounded {topic} explanation for scene {scene_index}.",
            action_type=ActionType.Reply,
            generated_expectation_ids=[],
            embedding=_compose_embedding(topic_waveform, _make_type_bias("action", embedding_dim), time_position),
            timestamp_creation=scene_time + timedelta(minutes=2, seconds=10),
            timestamp_world_begin=scene_time + timedelta(minutes=2, seconds=10),
            timestamp_world_end=scene_time + timedelta(minutes=2, seconds=20),
            metadata={"topic": topic, "scene_index": scene_index, "phase": phase_name, "active_user": user.content if phase_name != "internal_only" else None},
        )
    )
    intention = ghost.add_knoxel(
        Intention(
            tick_id=tick_id,
            content=f"Ensure the answer stays helpful and concrete for {topic} scene {scene_index}.",
            urgency=0.55 + 0.05 * math.sin(scene_index),
            affective_valence=0.25,
            incentive_salience=0.6,
            fulfilment=0.2,
            internal=False,
            originating_action_id=action.id,
            status="active",
            timeout=10,
            embedding=_compose_embedding(topic_waveform, _make_type_bias("intention", embedding_dim), time_position),
            timestamp_creation=scene_time + timedelta(minutes=2, seconds=25),
            timestamp_world_begin=scene_time + timedelta(minutes=2, seconds=25),
            timestamp_world_end=scene_time + timedelta(minutes=2, seconds=30),
            metadata={"topic": topic, "scene_index": scene_index, "phase": phase_name, "active_user": user.content if phase_name != "internal_only" else None},
        )
    )
    action.generated_expectation_ids = [intention.id]
    companion_dialogue = ghost.add_knoxel(
        Feature(
            tick_id=tick_id,
            content=_build_companion_reply_text(companion.content, user.content, topic_token, topic, scene_index, scene_variant, phase_name),
            source=companion.content,
            feature_type=FeatureType.Dialogue,
            interlocus=InterlocusType.Public,
            causal=True,
            embedding=_compose_embedding(topic_waveform, _make_type_bias("dialogue_ai", embedding_dim), time_position),
            timestamp_creation=scene_time + timedelta(minutes=3),
            timestamp_world_begin=scene_time + timedelta(minutes=3),
            timestamp_world_end=scene_time + timedelta(minutes=4),
            mental_state_appraisal=emotional_state,
            mental_state_delta=emotional_delta,
            metadata={"topic": topic, "speaker": companion.content, "scene_index": scene_index, "phase": phase_name, "active_user": user.content if phase_name != "internal_only" else None},
        )
    )
    witness_feature = ghost.add_knoxel(
        Feature(
            tick_id=tick_id,
            content=_build_witness_text(witness.content, topic_token, topic, scene_index, phase_name),
            source=witness.content,
            feature_type=FeatureType.StoryWildcard,
            interlocus=InterlocusType.Public,
            causal=scene_index % 2 == 0,
            embedding=_compose_embedding(topic_waveform, _make_type_bias("wildcard", embedding_dim), time_position),
            timestamp_creation=scene_time + timedelta(minutes=4, seconds=15),
            timestamp_world_begin=scene_time + timedelta(minutes=4, seconds=15),
            timestamp_world_end=scene_time + timedelta(minutes=5),
            mental_state_appraisal=emotional_state,
            mental_state_delta=emotional_delta,
            metadata={"topic": topic, "speaker": witness.content, "scene_index": scene_index, "phase": phase_name, "active_user": user.content if phase_name != "internal_only" else None},
        )
    )

    return {"features": [user_dialogue, thought, companion_dialogue, witness_feature]}


def _create_narratives(
    ghost: TestGhost,
    entities: Dict[str, Entity],
    config: TestInfrastructureConfig,
    topic_waveforms: Dict[str, List[float]],
) -> None:
    for index, topic in enumerate(config.topics):
        created_at = config.start_time_utc + timedelta(days=index, hours=8)
        ghost.add_knoxel(
            Narrative(
                content=f"{entities['Mira'].content} keeps returning to {topic} as a durable competence theme.",
                narrative_type=NarrativeTypes.SelfImage,
                target_name=entities["Mira"].content,
                last_refined_with_tick=max(1, index + 1),
                embedding=_compose_embedding(topic_waveforms[topic], _make_type_bias("narrative_self", config.embedding_dim), 0.2 * index),
                timestamp_creation=created_at,
                timestamp_world_begin=created_at,
                timestamp_world_end=created_at + timedelta(minutes=5),
                metadata={"topic": topic},
            )
        )
        ghost.add_knoxel(
            Narrative(
                content=f"{entities['Rick'].content} usually asks practical questions about {topic}.",
                narrative_type=NarrativeTypes.Relations,
                target_name=entities["Rick"].content,
                last_refined_with_tick=max(1, index + 1),
                embedding=_compose_embedding(topic_waveforms[topic], _make_type_bias("narrative_user", config.embedding_dim), 0.2 * index),
                timestamp_creation=created_at + timedelta(minutes=10),
                timestamp_world_begin=created_at + timedelta(minutes=10),
                timestamp_world_end=created_at + timedelta(minutes=15),
                metadata={"topic": topic},
            )
        )


def _build_scene_emotional_vectors(topic: str, phase_name: str, scene_index: int) -> tuple[List[float], List[float]]:
    appraisal = create_empty_ms_vector()
    delta = create_empty_ms_vector()
    topic_strength = 0.05 if topic == "music" else 0.02
    is_heated_topic = "heated" in topic or "conflict" in topic or "argument" in topic

    idx_goal = AppraisalGeneral.model_fields["goal_congruence"].json_schema_extra["vector_position"]
    idx_trust = AppraisalSocial.model_fields["trust_cues"].json_schema_extra["vector_position"]
    idx_valence = StateCore.model_fields["valence"].json_schema_extra["vector_position"]
    idx_arousal = StateCore.model_fields["arousal"].json_schema_extra["vector_position"]
    idx_joy = StateEmotions.model_fields["joy"].json_schema_extra["vector_position"]
    idx_sadness = StateEmotions.model_fields["sadness"].json_schema_extra["vector_position"]
    idx_anger = StateEmotions.model_fields["anger"].json_schema_extra["vector_position"]
    idx_fear = StateEmotions.model_fields["fear"].json_schema_extra["vector_position"]
    idx_tenderness = StateEmotions.model_fields["tenderness"].json_schema_extra["vector_position"]

    if is_heated_topic and phase_name == "user_present":
        appraisal[idx_goal] = -0.45
        appraisal[idx_trust] = 0.25
        delta[idx_valence] = -0.35
        delta[idx_arousal] = 0.74
        delta[idx_anger] = 0.64
        delta[idx_fear] = 0.52
    elif is_heated_topic and phase_name == "internal_only":
        appraisal[idx_goal] = -0.88
        appraisal[idx_trust] = 0.08
        delta[idx_valence] = -0.92
        delta[idx_arousal] = 0.86
        delta[idx_sadness] = 0.78
        delta[idx_anger] = 0.72
        delta[idx_fear] = 0.84
    elif is_heated_topic:
        appraisal[idx_goal] = 0.18
        appraisal[idx_trust] = 0.42
        delta[idx_valence] = 0.12
        delta[idx_arousal] = 0.58
        delta[idx_joy] = 0.08
        delta[idx_fear] = 0.25
        delta[idx_tenderness] = 0.05
    elif phase_name == "user_present":
        appraisal[idx_goal] = 0.65 + topic_strength
        appraisal[idx_trust] = 0.72
        delta[idx_valence] = 0.45 + topic_strength
        delta[idx_arousal] = 0.28
        delta[idx_joy] = 0.52
        delta[idx_tenderness] = 0.35
    elif phase_name == "internal_only":
        appraisal[idx_goal] = -0.70 - topic_strength
        appraisal[idx_trust] = 0.12
        delta[idx_valence] = -0.78 - topic_strength
        delta[idx_arousal] = 0.68
        delta[idx_sadness] = 0.82
        delta[idx_anger] = 0.08
        delta[idx_fear] = 0.24
    else:
        appraisal[idx_goal] = 0.35 + topic_strength
        appraisal[idx_trust] = 0.58
        delta[idx_valence] = 0.18 + topic_strength
        delta[idx_arousal] = 0.42
        delta[idx_joy] = 0.18
        delta[idx_tenderness] = 0.12
        delta[idx_fear] = 0.05

    delta[idx_valence] += 0.03 * math.sin(scene_index)
    delta[idx_arousal] = min(1.0, max(0.0, delta[idx_arousal] + 0.02 * math.cos(scene_index)))
    return appraisal, delta


def _build_emotional_query(
    *,
    valence: float,
    arousal: float,
    sadness: float,
    anger: float,
    fear: float,
    tenderness: float,
    goal_congruence: float,
    trust: float,
) -> List[float]:
    query = [0.0] * VectorModelReservedSize
    query[AppraisalGeneral.model_fields["goal_congruence"].json_schema_extra["vector_position"]] = goal_congruence
    query[AppraisalSocial.model_fields["trust_cues"].json_schema_extra["vector_position"]] = trust
    query[StateCore.model_fields["valence"].json_schema_extra["vector_position"]] = valence
    query[StateCore.model_fields["arousal"].json_schema_extra["vector_position"]] = arousal
    query[StateEmotions.model_fields["sadness"].json_schema_extra["vector_position"]] = sadness
    query[StateEmotions.model_fields["anger"].json_schema_extra["vector_position"]] = anger
    query[StateEmotions.model_fields["fear"].json_schema_extra["vector_position"]] = fear
    query[StateEmotions.model_fields["tenderness"].json_schema_extra["vector_position"]] = tenderness
    return query


def _build_internal_thought_text(companion_name: str, topic_token: str, topic: str, scene_index: int, phase_name: str) -> str:
    if "heated" in topic or "conflict" in topic or "argument" in topic:
        if phase_name == "internal_only":
            return f"[{topic_token}] {companion_name} replays the shouting, betrayal, and fear from scene {scene_index} in private."
        if phase_name == "user_returns":
            return f"[{topic_token}] {companion_name} braces for another escalation but hopes the apology will hold in scene {scene_index}."
        return f"[{topic_token}] {companion_name} notices a heated argument forming and fears the trust may break in scene {scene_index}."
    if phase_name == "internal_only":
        return f"[{topic_token}] {companion_name} worries alone about unresolved {topic} cues in scene {scene_index}."
    if phase_name == "user_returns":
        return f"[{topic_token}] {companion_name} notices relief as {topic} conversation resumes in scene {scene_index}."
    return f"[{topic_token}] {companion_name} internally rates {topic} as relevant for scene {scene_index}."


def _build_companion_reply_text(
    companion_name: str,
    user_name: str,
    topic_token: str,
    topic: str,
    scene_index: int,
    scene_variant: int,
    phase_name: str,
) -> str:
    if "heated" in topic or "conflict" in topic or "argument" in topic:
        if phase_name == "internal_only":
            return f"[{topic_token}] {companion_name} rehearses an apology after the betrayal and shouting in scene {scene_index}."
        if phase_name == "user_returns":
            return f"[{topic_token}] {companion_name} tells {user_name} they want repair after the heated argument and fear in scene {scene_index}."
        return f"[{topic_token}] {companion_name} answers sharply as the heated argument around betrayal escalates in scene {scene_index}."
    if phase_name == "internal_only":
        return f"[{topic_token}] {companion_name} talks through the {topic} answer aloud with no user present in scene {scene_index}."
    if phase_name == "user_returns":
        return f"[{topic_token}] {companion_name} re-engages {user_name} about {topic} with repair example {scene_variant}."
    return f"[{topic_token}] {companion_name} explains {topic} answer {scene_index} with example {scene_variant}."


def _build_witness_text(witness_name: str, topic_token: str, topic: str, scene_index: int, phase_name: str) -> str:
    if "heated" in topic or "conflict" in topic or "argument" in topic:
        if phase_name == "internal_only":
            return f"[{topic_token}] {witness_name} recalls how the shouting left the room tense and fearful in scene {scene_index}."
        if phase_name == "user_returns":
            return f"[{topic_token}] {witness_name} notices the apology after the heated argument is fragile in scene {scene_index}."
        return f"[{topic_token}] {witness_name} notices the heated argument, betrayal, and rising fear in scene {scene_index}."
    if phase_name == "internal_only":
        return f"[{topic_token}] {witness_name} is absent while the {topic} exchange collapses into solitary reflection in scene {scene_index}."
    if phase_name == "user_returns":
        return f"[{topic_token}] {witness_name} notices the {topic} conversation becoming social again in scene {scene_index}."
    return f"[{topic_token}] {witness_name} notices that the {topic} conversation became clearer in scene {scene_index}."


def _create_topical_clusters(
    ghost: TestGhost,
    config: TestInfrastructureConfig,
    topic_event_ids: Dict[str, List[int]],
    topic_waveforms: Dict[str, List[float]],
) -> List[MemoryClusterKnoxel]:
    clusters: List[MemoryClusterKnoxel] = []
    for topic, event_ids in topic_event_ids.items():
        if not event_ids:
            continue
        chunk_size = max(2, len(event_ids) // config.scale.topical_clusters_per_topic)
        for chunk_index, chunk_start in enumerate(range(0, len(event_ids), chunk_size)):
            chunk_ids = event_ids[chunk_start:chunk_start + chunk_size]
            if len(chunk_ids) < 2:
                continue
            covered = [ghost.get_knoxel_by_id(knoxel_id) for knoxel_id in chunk_ids]
            covered = [item for item in covered if item is not None]
            begin = min(item.timestamp_world_begin for item in covered)
            end = max(item.timestamp_world_end for item in covered)
            clusters.append(
                ghost.add_knoxel(
                    MemoryClusterKnoxel(
                        content=f"Topical cluster for {topic}: scenes {chunk_index} emphasize repeated {topic} discussion.",
                        minimum_interlocus=int(min(getattr(item, "interlocus", 0) for item in covered)),
                        min_tick_id=min(item.tick_id for item in covered),
                        max_tick_id=max(item.tick_id for item in covered),
                        min_event_id=min(chunk_ids),
                        max_event_id=max(chunk_ids),
                        level=100,
                        cluster_type=ClusterType.Topical,
                        included_event_ids=",".join(str(knoxel_id) for knoxel_id in chunk_ids),
                        included_cluster_ids=None,
                        token=24 + len(chunk_ids) * 4,
                        facts_extracted=True,
                        temporal_key=f"{topic}-topic-{chunk_index}",
                        emotion_description=f"Steady confidence while discussing {topic}.",
                        emotion_embedding=_compose_embedding(topic_waveforms[topic], _make_type_bias("emotion", config.embedding_dim), 0.3 + 0.1 * chunk_index),
                        embedding=_compose_embedding(topic_waveforms[topic], _make_type_bias("cluster_topical", config.embedding_dim), 0.3 + 0.1 * chunk_index),
                        timestamp_creation=end + timedelta(minutes=1),
                        timestamp_world_begin=begin,
                        timestamp_world_end=end,
                        metadata={"topic": topic, "cluster_kind": "topical"},
                    )
                )
            )
    return clusters


def _create_temporal_clusters(
    ghost: TestGhost,
    config: TestInfrastructureConfig,
    scene_features: Sequence[Feature],
    topic_waveforms: Dict[str, List[float]],
) -> None:
    if not scene_features:
        return

    by_day: Dict[datetime, List[Feature]] = {}
    for feature in scene_features:
        day_key = datetime(
            feature.timestamp_world_begin.year,
            feature.timestamp_world_begin.month,
            feature.timestamp_world_begin.day,
            tzinfo=feature.timestamp_world_begin.tzinfo,
        )
        by_day.setdefault(day_key, []).append(feature)

    for level in config.scale.temporal_cluster_levels:
        bucket_size = _temporal_bucket_days_for_level(level)
        day_keys = sorted(by_day.keys())
        for bucket_index in range(0, len(day_keys), bucket_size):
            bucket_days = day_keys[bucket_index:bucket_index + bucket_size]
            bucket_features = [feature for day in bucket_days for feature in by_day[day]]
            if not bucket_features:
                continue

            begin = min(feature.timestamp_world_begin for feature in bucket_features)
            end = max(feature.timestamp_world_end for feature in bucket_features)
            dominant_topic = bucket_features[len(bucket_features) // 2].metadata["topic"]
            child_ids = [
                cluster.id
                for cluster in ghost.all_episodic_memories
                if cluster.level == level + 1
                and cluster.timestamp_world_begin >= begin
                and cluster.timestamp_world_end <= end
            ]
            included_cluster_ids = None
            included_event_ids = None
            if child_ids:
                child_clusters = [ghost.get_knoxel_by_id(cluster_id) for cluster_id in child_ids]
                child_begin = min(cluster.timestamp_world_begin for cluster in child_clusters)
                child_end = max(cluster.timestamp_world_end for cluster in child_clusters)
                if child_begin == begin and child_end == end:
                    included_cluster_ids = ",".join(str(cluster_id) for cluster_id in child_ids)
                else:
                    included_event_ids = ",".join(str(feature.id) for feature in bucket_features)
            elif level == max(config.scale.temporal_cluster_levels):
                included_event_ids = ",".join(str(feature.id) for feature in bucket_features)
            else:
                included_event_ids = ",".join(str(feature.id) for feature in bucket_features)
            ghost.add_knoxel(
                MemoryClusterKnoxel(
                    content=f"Temporal level {level} summary across {len(bucket_days)} day(s), centered on {dominant_topic}.",
                    minimum_interlocus=0,
                    min_tick_id=min(feature.tick_id for feature in bucket_features),
                    max_tick_id=max(feature.tick_id for feature in bucket_features),
                    min_event_id=min(feature.id for feature in bucket_features),
                    max_event_id=max(feature.id for feature in bucket_features),
                    level=level,
                    cluster_type=ClusterType.Temporal,
                    included_event_ids=included_event_ids,
                    included_cluster_ids=included_cluster_ids,
                    token=36 + len(bucket_features) * 2,
                    facts_extracted=False,
                    temporal_key=f"level-{level}-{begin.date()}",
                    emotion_description=f"Temporal recall for {dominant_topic}.",
                    emotion_embedding=_compose_embedding(topic_waveforms[dominant_topic], _make_type_bias("emotion", config.embedding_dim), 0.6),
                    embedding=_compose_embedding(topic_waveforms[dominant_topic], _make_type_bias("cluster_temporal", config.embedding_dim), 0.6),
                    timestamp_creation=end + timedelta(minutes=2),
                    timestamp_world_begin=begin,
                    timestamp_world_end=end,
                    metadata={"topic": dominant_topic, "cluster_kind": "temporal"},
                )
            )


def _create_facts_and_rules(
    ghost: TestGhost,
    topical_clusters: Sequence[MemoryClusterKnoxel],
    topic_waveforms: Dict[str, List[float]],
) -> None:
    for index, cluster in enumerate(topical_clusters):
        topic = cluster.metadata["topic"]
        fact_time = cluster.timestamp_world_end + timedelta(minutes=10)
        embedding_dim = len(cluster.embedding)
        ghost.add_knoxel(
            DeclarativeFactKnoxel(
                content=f"{topic.title()} conversations usually contain repeatable examples and grounded answers.",
                reason=f"Extracted from topical cluster {cluster.id}.",
                category=[topic, "interaction_style", "retrieval_test"],
                importance=0.65 + (0.05 * (index % 3)),
                time_dependent=0.2,
                source_cluster_id=cluster.id,
                minimum_interlocus=cluster.minimum_interlocus,
                min_tick_id=cluster.min_tick_id,
                max_tick_id=cluster.max_tick_id,
                min_event_id=cluster.min_event_id,
                max_event_id=cluster.max_event_id,
                embedding=_compose_embedding(topic_waveforms[topic], _make_type_bias("fact", embedding_dim), 0.4),
                timestamp_creation=fact_time,
                timestamp_world_begin=cluster.timestamp_world_begin,
                timestamp_world_end=cluster.timestamp_world_end,
                metadata={"topic": topic},
            )
        )
        ghost.add_knoxel(
            CauseEffectKnoxel(
                content=f"When {topic} is discussed concretely, the next answer becomes easier to follow.",
                situation=f"A multi-turn {topic} exchange begins.",
                cause=f"The companion uses a concrete {topic} example.",
                effect="The user continues the topic instead of switching away.",
                category=topic,
                source_cluster_id=cluster.id,
                minimum_interlocus=cluster.minimum_interlocus,
                min_tick_id=cluster.min_tick_id,
                max_tick_id=cluster.max_tick_id,
                min_event_id=cluster.min_event_id,
                max_event_id=cluster.max_event_id,
                embedding=_compose_embedding(topic_waveforms[topic], _make_type_bias("cause_effect", embedding_dim), 0.45),
                timestamp_creation=fact_time + timedelta(minutes=5),
                timestamp_world_begin=cluster.timestamp_world_begin,
                timestamp_world_end=cluster.timestamp_world_end,
                metadata={"topic": topic},
            )
        )


def _create_graph_memory(
    ghost: TestGhost,
    entities: Dict[str, Entity],
    topical_clusters: Sequence[MemoryClusterKnoxel],
    topic_waveforms: Dict[str, List[float]],
    embedding_dim: int,
) -> None:
    if not topical_clusters:
        return
    for topic_index, topic in enumerate(topic_waveforms):
        concept = ghost.add_knoxel(
            ConceptNode(
                content=f"Concept for {topic}",
                description=f"High-level category used in planner tests for {topic}.",
                parent_concept_id=0,
                embedding=_compose_embedding(topic_waveforms[topic], _make_type_bias("concept", embedding_dim), 0.1 * topic_index),
                timestamp_creation=DEFAULT_START_TIME,
                timestamp_world_begin=DEFAULT_START_TIME,
                timestamp_world_end=DEFAULT_START_TIME,
                metadata={"topic": topic},
            )
        )
        node = ghost.add_knoxel(
            GraphNode(
                content=f"Graph node for {topic}",
                name=topic.title(),
                labels=["Topic", "TestMemory"],
                source_memory_id=topical_clusters[min(topic_index, len(topical_clusters) - 1)].id,
                concept_id=concept.id,
                attributes={"topic": topic},
                embedding=_compose_embedding(topic_waveforms[topic], _make_type_bias("graph_node", embedding_dim), 0.1 * topic_index),
                timestamp_creation=DEFAULT_START_TIME + timedelta(minutes=topic_index),
                timestamp_world_begin=DEFAULT_START_TIME,
                timestamp_world_end=DEFAULT_START_TIME,
                metadata={"topic": topic},
            )
        )
        ghost.add_knoxel(
            GraphEdge(
                content=f"{entities['Mira'].content} often discusses {topic}.",
                source_id=entities["Mira"].id,
                target_id=node.id,
                label="discusses",
                fact_text=f"{entities['Mira'].content} often discusses {topic}.",
                source_memory_id=node.source_memory_id,
                valid_at=DEFAULT_START_TIME,
                invalid_at=None,
                attributes={"topic": topic},
                embedding=_compose_embedding(topic_waveforms[topic], _make_type_bias("graph_edge", embedding_dim), 0.1 * topic_index),
                timestamp_creation=DEFAULT_START_TIME + timedelta(minutes=topic_index + 1),
                timestamp_world_begin=DEFAULT_START_TIME,
                timestamp_world_end=DEFAULT_START_TIME,
                metadata={"topic": topic},
            )
        )


def _create_codelet_artifacts(
    ghost: TestGhost,
    scene_features: Sequence[Feature],
    topic_waveforms: Dict[str, List[float]],
    embedding_dim: int,
) -> None:
    for index, topic in enumerate(topic_waveforms):
        topic_features = [feature for feature in scene_features if feature.metadata.get("topic") == topic][:3]
        if not topic_features:
            continue
        coalition = ghost.add_knoxel(
            PerceptCoalition(
                content=f"Coalition for {topic}",
                knoxel_ids=[feature.id for feature in topic_features],
                embedding=_compose_embedding(topic_waveforms[topic], _make_type_bias("coalition", embedding_dim), 0.5),
                timestamp_creation=topic_features[-1].timestamp_world_end + timedelta(seconds=30),
                timestamp_world_begin=topic_features[0].timestamp_world_begin,
                timestamp_world_end=topic_features[-1].timestamp_world_end,
                metadata={"topic": topic},
            )
        )
        ghost.add_knoxel(
            CodeletPercept(
                content=f"Percept summary for {topic}",
                salience=min(0.95, 0.55 + 0.08 * index),
                valence=0.15 + 0.1 * math.sin(index),
                embedding=_compose_embedding(topic_waveforms[topic], _make_type_bias("codelet", embedding_dim), 0.55),
                timestamp_creation=coalition.timestamp_world_end + timedelta(seconds=15),
                timestamp_world_begin=coalition.timestamp_world_begin,
                timestamp_world_end=coalition.timestamp_world_end,
                metadata={"topic": topic},
            )
        )


def _temporal_bucket_days_for_level(level: int) -> int:
    return {6: 1, 5: 2, 4: 4, 3: 7, 2: 14, 1: 30}.get(level, 1)


def _make_topic_waveform(topic_index: int, embedding_dim: int) -> List[float]:
    values: List[float] = []
    for dim in range(embedding_dim):
        angle = (topic_index + 1) * (dim + 1) * 0.37
        values.append(math.sin(angle) + 0.5 * math.cos(angle * 0.5))
    return _normalize(values)


def _make_type_bias(type_name: str, embedding_dim: int) -> List[float]:
    seed = sum(ord(char) for char in type_name)
    values: List[float] = []
    for dim in range(embedding_dim):
        angle = (seed % 31 + 1) * (dim + 1) * 0.11
        values.append(math.cos(angle) * 0.35 + math.sin(angle * 0.7) * 0.15)
    return _normalize(values)


def _compose_embedding(base: List[float], type_bias: List[float], time_position: float) -> List[float]:
    values: List[float] = []
    for dim, base_value in enumerate(base):
        wave = math.sin((dim + 1) * math.pi * time_position) * 0.18
        values.append((base_value * 0.78) + (type_bias[dim] * 0.17) + wave)
    return _normalize(values)


def _normalize(values: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        return [0.0 for _ in values]
    return [value / norm for value in values]
