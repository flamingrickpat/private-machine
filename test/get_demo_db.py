from test.infrastructure import generate_populated_ghost_sqlite

path = generate_populated_ghost_sqlite(
    output_path="test_artifacts/synthetic_debug_ghost.db",
    number_of_days=90,
    avg_features_per_day=64,
    internal_thought_ratio=0.25,
    random_other_feature_ratio=0.2,
    seed=1337,
)
print(path)