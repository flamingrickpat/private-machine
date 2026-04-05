from datetime import timedelta
from pathlib import Path
from typing import Any

from pm.model.knoxel_common import CauseEffectKnoxel, DeclarativeFactKnoxel, MemoryClusterKnoxel, Narrative
from pm.model.knoxel_graph import ConceptNode, GraphEdge, GraphNode
from pm.model.knoxel_enums import FeatureType, InterlocusType
from pm.model.knoxel_feature import Feature
from pm.persist.persist_sqlite import PersistSqlite
from pm.subsystems.context.common_context import plan_context_story_simple
from pm.subsystems.context.render_presets import context_render_story_messages_assistant_for_dialoge
from pm.subsystems.memory.memory_consolidation import MemoryConsolidationConfig
from pm.subsystems.memory.memory_consolidation_test_mode import SyntheticMemoryConsolidator
from pm.utils.token_utils import get_token_count
from test.infrastructure import _create_sqlite_export_ghost, load_unit_test_config


DERIVED_KNOXEL_TYPES = (
    MemoryClusterKnoxel,
    DeclarativeFactKnoxel,
    CauseEffectKnoxel,
    Narrative,
    ConceptNode,
    GraphNode,
    GraphEdge,
)


def resolve_existing_path(path_text: str) -> Path:
    raw_path = Path(path_text)
    candidates = [
        raw_path,
        Path.cwd() / raw_path,
        Path(__file__).resolve().parent / raw_path,
        Path(__file__).resolve().parents[1] / raw_path,
    ]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    return candidates[0].resolve()


def print_console_safe(text: str) -> None:
    print(text.encode("cp1252", errors="replace").decode("cp1252"))


def clear_derived_memory_knoxels(ghost) -> None:
    keep_knoxels = {
        knoxel_id: knoxel
        for knoxel_id, knoxel in ghost.all_knoxels.items()
        if not isinstance(knoxel, DERIVED_KNOXEL_TYPES)
    }
    ghost.all_knoxels = keep_knoxels
    ghost.current_knoxel_id = max(keep_knoxels.keys(), default=0)
    ghost._rebuild_specific_lists()


def load_persisted_ghost_relaxed(db_path: str):
    resolved_db_path = resolve_existing_path(db_path)
    runtime_config = load_unit_test_config()
    runtime_config.commit = True
    runtime_config.db_path = str(resolved_db_path)
    ghost = _create_sqlite_export_ghost(
        runtime_config,
        companion_name=runtime_config.companion_name,
        user_name=runtime_config.user_name,
        character_card=runtime_config.character_card_story,
    )
    if not PersistSqlite(ghost).load_state_sqlite(str(resolved_db_path)):
        raise RuntimeError(f"Could not load persisted ghost from {resolved_db_path}")

    ghost.current_knoxel_id = max(ghost.all_knoxels.keys(), default=0)
    ghost.current_tick_id = max((knoxel.tick_id for knoxel in ghost.all_knoxels.values()), default=ghost.current_tick_id)
    ghost._rebuild_specific_lists()
    return ghost


def collect_ghost_stats_relaxed(ghost) -> dict[str, Any]:
    feature_type_counts: dict[str, int] = {}
    for feature in ghost.all_features:
        key = feature.feature_type.value
        feature_type_counts[key] = feature_type_counts.get(key, 0) + 1

    def _safe_iso(value) -> str | None:
        return value.isoformat() if value is not None else None

    return {
        "knoxel_count": len(ghost.all_knoxels),
        "current_knoxel_id": ghost.current_knoxel_id,
        "current_tick_id": ghost.current_tick_id,
        "feature_count": len(ghost.all_features),
        "causal_feature_count": len([feature for feature in ghost.all_features if feature.causal]),
        "feature_type_counts": feature_type_counts,
        "cluster_signatures": [
            (
                cluster.id,
                cluster.cluster_type.value,
                cluster.level,
                cluster.included_event_ids or "",
                cluster.included_cluster_ids or "",
                _safe_iso(cluster.timestamp_world_begin),
                _safe_iso(cluster.timestamp_world_end),
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
                _safe_iso(fact.timestamp_world_begin),
                _safe_iso(fact.timestamp_world_end),
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
                _safe_iso(item.timestamp_world_begin),
                _safe_iso(item.timestamp_world_end),
            )
            for item in sorted(getattr(ghost, "all_cause_effects", []), key=lambda item: item.id)
        ],
        "first_feature_timestamp": _safe_iso(min((feature.timestamp_world_begin for feature in ghost.all_features), default=None)),
        "last_feature_timestamp": _safe_iso(max((feature.timestamp_world_end for feature in ghost.all_features), default=None)),
        "broken_topical_clusters": [
            cluster.id
            for cluster in ghost.all_episodic_memories
            if cluster.included_event_ids
            and any(ghost.get_knoxel_by_id(int(event_id)) is None for event_id in cluster.included_event_ids.split(",") if event_id)
        ],
    }


def debug_real_db_with_synthetic_memory(
    db_path: str = "./data/main.db",
    *,
    query: str = "cats, kittens and other felines, animals",
    context_budget: int = 16000,
    save_path: str | None = None,
) -> dict[str, Any]:
    resolved_db_path = resolve_existing_path(db_path)
    ghost = load_persisted_ghost_relaxed(str(resolved_db_path))
    before_stats = collect_ghost_stats_relaxed(ghost)

    clear_derived_memory_knoxels(ghost)
    after_clear_stats = collect_ghost_stats_relaxed(ghost)

    consolidator = SyntheticMemoryConsolidator(
        ghost,
        MemoryConsolidationConfig(
            min_event_memory_consolidation_threshold=1,
            min_events_for_processing=3,
            min_split_up_cluster_size=4,
            hierchical_summary_token_limit=300,
            enable_cause_effect_extraction=True,
            duplicate_string_similarity_threshold=0.96,
            duplicate_embedding_similarity_threshold=0.995,
            duplicate_overlap_ratio_threshold=0.9,
        ),
        seed=1701,
        embedding_dim=24,
    )
    consolidator.consolidate_memory_if_needed()

    after_synth_stats = collect_ghost_stats_relaxed(ghost)
    planner_output, settings, _ = plan_context_story_simple(ghost, consolidator.llm.get_embedding(query), context_budget)
    turns = context_render_story_messages_assistant_for_dialoge(ghost, planner_output)
    rendered_prompt = "\n\n".join(f"{role.upper()}:\n{content}" for role, content in turns)

    if save_path:
        PersistSqlite(ghost).save_state_sqlite(save_path)

    result = {
        "db_path": str(resolved_db_path),
        "save_path": str(Path(save_path).resolve()) if save_path else None,
        "before_stats": before_stats,
        "after_clear_stats": after_clear_stats,
        "after_synth_stats": after_synth_stats,
        "planner_output": planner_output,
        "turns": turns,
        "planner_token_budget": settings.token_length,
        "planner_used_tokens": sum(meta["token_budget_used"] for meta in planner_output.metadata["lanes"].values()),
        "rendered_prompt_tokens": get_token_count(rendered_prompt),
        "rendered_prompt_preview": rendered_prompt[:4000],
    }

    print(f"Loaded DB: {result['db_path']}")
    print(f"Before clear: memories={len(before_stats['cluster_signatures'])} facts={len(before_stats['fact_signatures'])} cause_effect={len(before_stats['cause_effect_signatures'])}")
    print(f"Broken topical clusters before clear: {len(before_stats['broken_topical_clusters'])}")
    print(f"After clear: memories={len(after_clear_stats['cluster_signatures'])} facts={len(after_clear_stats['fact_signatures'])} cause_effect={len(after_clear_stats['cause_effect_signatures'])}")
    print(f"After synth: memories={len(after_synth_stats['cluster_signatures'])} facts={len(after_synth_stats['fact_signatures'])} cause_effect={len(after_synth_stats['cause_effect_signatures'])}")
    print(f"Planner used tokens: {result['planner_used_tokens']} / {result['planner_token_budget']}")
    for lane_name, meta in planner_output.metadata["lanes"].items():
        print(
            f"lane={lane_name} candidates={meta['candidate_count']} "
            f"selected={len(meta['selected_source_ids'])} used={meta['token_budget_used']} "
            f"base={meta['token_budget_base']} current={meta['token_budget_current']}"
        )
    print(f"Rendered prompt tokens: {result['rendered_prompt_tokens']}")
    print("=== PROMPT PREVIEW START ===")
    print_console_safe(result["rendered_prompt_preview"])
    print("=== PROMPT PREVIEW END ===")
    if result["save_path"]:
        print(f"Saved synthetic-memory DB copy to: {result['save_path']}")

    print_console_safe(f"BEGIN CONTENT\n{rendered_prompt}\nEND CONTENT")


    return result


def _inject_prompt_role_probe_features(ghost, embedding_fn) -> dict[str, str]:
    latest_feature = max(
        (feature for feature in ghost.all_features if feature.causal),
        key=lambda item: (item.timestamp_world_end, item.id),
    )
    base_time = latest_feature.timestamp_world_end
    next_tick_id = max((feature.tick_id for feature in ghost.all_features), default=0) + 1
    companion_name = ghost.get_companion_name()
    user_name = ghost.get_user_name()

    probe_contents = {
        "dialogue": "ROLE_PROBE_DIALOGUE assistant-dialogue probe line",
        "thought": "ROLE_PROBE_THOUGHT assistant-thought probe line",
        "external_thought": "ROLE_PROBE_EXTERNAL_THOUGHT assistant-external-thought probe line",
        "subjective_experience": "ROLE_PROBE_SUBJECTIVE_EXPERIENCE third-person feeling description probe line",
        "codelet_output": "ROLE_PROBE_CODELET_OUTPUT codelet output probe line",
        "codelet_percept": "ROLE_PROBE_CODELET_PERCEPT codelet percept probe line",
    }

    probe_specs = [
        (FeatureType.Dialogue, companion_name, InterlocusType.Public, probe_contents["dialogue"]),
        (FeatureType.Thought, companion_name, InterlocusType.PrivateReportable, probe_contents["thought"]),
        (FeatureType.ExternalThought, companion_name, InterlocusType.PrivateReportable, probe_contents["external_thought"]),
        (FeatureType.SubjectiveExperience, companion_name, InterlocusType.PrivateInternal, probe_contents["subjective_experience"]),
        (FeatureType.CodeletOutput, "probe-codelet", InterlocusType.PrivateInternal, probe_contents["codelet_output"]),
        (FeatureType.CodeletPercept, "probe-codelet", InterlocusType.PrivateInternal, probe_contents["codelet_percept"]),
        (FeatureType.Dialogue, user_name, InterlocusType.Public, "ROLE_PROBE_USER_DIALOGUE user-dialogue probe line"),
    ]

    for offset, (feature_type, source, interlocus, content) in enumerate(probe_specs, start=1):
        timestamp = base_time + timedelta(seconds=offset)
        ghost.add_knoxel(
            Feature(
                content=content,
                feature_type=feature_type,
                source=source,
                interlocus=interlocus,
                causal=True,
                embedding=embedding_fn(content),
                tick_id=next_tick_id,
                timestamp_creation=timestamp,
                timestamp_world_begin=timestamp,
                timestamp_world_end=timestamp,
                metadata={"debug_probe": True},
            ),
            generate_embedding=False,
        )

    return probe_contents


def _normalize_feature_embeddings_for_synthetic_debug(ghost, embedding_fn) -> None:
    for feature in ghost.all_features:
        feature.embedding = embedding_fn(feature.content)


def debug_real_db_prompt_role_probe(
    db_path: str = "./data/main.db",
    *,
    context_budget: int = 16000,
) -> dict[str, Any]:
    resolved_db_path = resolve_existing_path(db_path)
    ghost = load_persisted_ghost_relaxed(str(resolved_db_path))
    clear_derived_memory_knoxels(ghost)

    consolidator = SyntheticMemoryConsolidator(
        ghost,
        MemoryConsolidationConfig(
            min_event_memory_consolidation_threshold=1,
            min_events_for_processing=3,
            min_split_up_cluster_size=4,
            hierchical_summary_token_limit=300,
            enable_cause_effect_extraction=True,
            duplicate_string_similarity_threshold=0.96,
            duplicate_embedding_similarity_threshold=0.995,
            duplicate_overlap_ratio_threshold=0.9,
        ),
        seed=1701,
        embedding_dim=24,
    )
    _normalize_feature_embeddings_for_synthetic_debug(ghost, consolidator.llm.get_embedding)
    probe_contents = _inject_prompt_role_probe_features(ghost, consolidator.llm.get_embedding)
    consolidator.consolidate_memory_if_needed()

    planner_output, settings, _ = plan_context_story_simple(
        ghost,
        consolidator.llm.get_embedding("ROLE_PROBE dialogue thought codelet subjective experience"),
        context_budget,
    )
    turns = context_render_story_messages_assistant_for_dialoge(ghost, planner_output)

    role_by_probe_key: dict[str, str] = {}
    for role, content in turns:
        for probe_key, probe_text in probe_contents.items():
            if probe_text in content:
                role_by_probe_key[probe_key] = role

    print(f"Loaded DB: {resolved_db_path}")
    print("=== ROLE PROBE RESULTS ===")
    for probe_key, probe_text in probe_contents.items():
        print(f"{probe_key} -> {role_by_probe_key.get(probe_key, 'MISSING')} :: {probe_text}")

    expected_assistant = {"dialogue", "thought", "external_thought"}
    expected_user = {"subjective_experience", "codelet_output", "codelet_percept"}
    for probe_key in expected_assistant:
        if role_by_probe_key.get(probe_key) != "assistant":
            raise Exception(f"prompt role probe failed: {probe_key} was not rendered in assistant turn")
    for probe_key in expected_user:
        if role_by_probe_key.get(probe_key) != "user":
            raise Exception(f"prompt role probe failed: {probe_key} was not rendered in user turn")

    return {
        "db_path": str(resolved_db_path),
        "planner_output": planner_output,
        "turns": turns,
        "role_by_probe_key": role_by_probe_key,
        "planner_token_budget": settings.token_length,
    }


if __name__ == "__main__":
    debug_real_db_with_synthetic_memory(
        db_path="./data/main.db",
        save_path="./data/main_synth_debug.db",
    )
