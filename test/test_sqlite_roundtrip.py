from test.infrastructure import (
    collect_ghost_stats_strict,
    generate_populated_ghost_sqlite_from_scenario,
    load_persisted_ghost_strict,
    SYNTHETIC_SCENARIO_PRESETS,
    build_scenario_corpus,
)


def test_sqlite_roundtrip_preserves_generated_stats(tmp_path):
    scenario_name = "sparse_history"
    assert scenario_name in SYNTHETIC_SCENARIO_PRESETS

    corpus = build_scenario_corpus(scenario_name, seed=41)
    stats_before = collect_ghost_stats_strict(corpus.ghost)

    db_path = tmp_path / "roundtrip_demo.db"
    generate_populated_ghost_sqlite_from_scenario(str(db_path), scenario_name, seed=41)

    loaded_ghost = load_persisted_ghost_strict(str(db_path))
    stats_after = collect_ghost_stats_strict(loaded_ghost)

    assert stats_after == stats_before
