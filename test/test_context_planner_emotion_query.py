from pm.subsystems.context.context_planner_base import (
    ContextDiversitySettings,
    ContextLaneSettings,
    ContextPlanner,
    ContextPlannerSettings,
    ContextPostProcessSettings,
    ContextPreFilterRules,
    ContextSampleSettings,
    ContextScoreSettings,
    SortingStrategy,
)
from test.infrastructure import populate_test_ghost


def test_single_lane_prefers_emotionally_matching_music_features():
    corpus = populate_test_ghost()
    planner = ContextPlanner()
    music_embedding = corpus.query_embeddings["music"]
    distress_delta = corpus.mental_queries["music_distress"]

    settings = ContextPlannerSettings(
        token_length=40,
        ordered_lanes=[
            ContextLaneSettings(
                name="music_emotion",
                token_budget_pct=1.0,
                sorting_strategy=SortingStrategy.ScoreDesc,
                scoring=ContextScoreSettings(
                    weighted_embeddings=[(1.0, music_embedding)],
                    weighted_mental_delta=[(2.5, distress_delta)],
                ),
            )
        ],
        lane_priority=[],
        pre_filter=ContextPreFilterRules(),
        scoring=ContextScoreSettings(weighted_embeddings=[(1.0, music_embedding)]),
        diversity=ContextDiversitySettings(),
        sampler=ContextSampleSettings(),
        post_process=ContextPostProcessSettings(),
    )

    features = list(corpus.ghost.all_features)
    planner_output = planner.plan(knoxels=features, settings=settings)
    selected = planner_output.lane_data["music_emotion"]
    lane_metadata = planner_output.metadata["lanes"]["music_emotion"]

    assert selected
    assert all(item.metadata["topic"] == "music" for item in selected[:2])
    assert selected[0].metadata["phase"] == "internal_only"
    assert selected[0].source == "Mira"
    assert selected[0].mental_state_delta != selected[0].mental_state_appraisal

    scored_candidates = []
    for feature in features:
        combined_score = (
            planner._cosine_similarity(feature.embedding, music_embedding)
            + 2.5 * planner._cosine_similarity(feature.mental_state_delta, distress_delta)
        )
        scored_candidates.append((feature, combined_score, planner._estimate_knoxel_tokens(feature)))
    scored_candidates.sort(key=lambda item: (-item[1], item[0].id))

    expected_ids = []
    tokens_used = 0
    for feature, score, token_cost in scored_candidates:
        if expected_ids and tokens_used + token_cost > settings.token_length:
            continue
        expected_ids.append(feature.id)
        tokens_used += token_cost
        if tokens_used >= settings.token_length:
            break

    assert [item.id for item in selected] == expected_ids
    assert all(entry["breakdown"]["mental_delta_0"] >= 0.0 for entry in lane_metadata["selected_scores"])
    assert lane_metadata["token_budget_used"] <= settings.token_length

    best_internal_music = next(
        score for feature, score, _ in scored_candidates if feature.metadata["topic"] == "music" and feature.metadata["phase"] == "internal_only"
    )
    best_non_internal_music = next(
        score for feature, score, _ in scored_candidates if feature.metadata["topic"] == "music" and feature.metadata["phase"] != "internal_only"
    )
    assert best_internal_music > best_non_internal_music
