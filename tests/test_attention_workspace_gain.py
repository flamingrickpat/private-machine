from types import SimpleNamespace

from pm.ghosts.procedures.attention import (
    _coalition_size_from_gain,
    _compute_workspace_gain,
    _token_cap_from_gain,
    AttentionProc,
)


def _ghost_for_gain(interlocus=-0.6, arousal=0.8, ego=0.7, aperture=0.5, willpower=0.0, energy=0.4, proc=0.5):
    ms = SimpleNamespace(
        state_core=SimpleNamespace(arousal=arousal),
        state_cognition=SimpleNamespace(
            ego_strength=ego,
            interlocus=interlocus,
            mental_aperture=aperture,
            willpower=willpower,
        ),
        state_needs=SimpleNamespace(energy_stability=energy, processing_power=proc),
    )
    return SimpleNamespace(current_state=SimpleNamespace(latent_mental_state=ms))


def test_workspace_gain_higher_for_tense_state():
    calm = _ghost_for_gain(interlocus=0.0, arousal=0.2, ego=0.3, aperture=0.1, energy=0.8, proc=0.8)
    tense = _ghost_for_gain(interlocus=-0.9, arousal=0.9, ego=0.8, aperture=0.9, energy=0.2, proc=0.2)

    g_calm = _compute_workspace_gain(calm)
    g_tense = _compute_workspace_gain(tense)

    assert g_tense > g_calm
    assert 0.0 <= g_calm <= 1.0
    assert 0.0 <= g_tense <= 1.0


def test_coalition_size_from_gain_bands():
    assert _coalition_size_from_gain(0.8) == 1
    assert _coalition_size_from_gain(0.5) == 2
    assert _coalition_size_from_gain(0.2) == 4


def test_token_cap_from_gain_monotonic():
    ghost = SimpleNamespace(llm=SimpleNamespace(get_max_tokens=lambda _: 4000))
    high = _token_cap_from_gain(ghost, 0.85)
    mid = _token_cap_from_gain(ghost, 0.50)
    low = _token_cap_from_gain(ghost, 0.20)

    assert high < mid < low


def test_attention_run_sets_target_and_cap_and_coalition():
    class _CSM:
        def items(self):
            return [
                SimpleNamespace(knoxel_id=1, activation=0.9),
                SimpleNamespace(knoxel_id=2, activation=0.6),
                SimpleNamespace(knoxel_id=3, activation=0.4),
            ]

    knoxels = {
        1: SimpleNamespace(id=1, content="a", incentive_salience=0.9, affective_valence=0.1),
        2: SimpleNamespace(id=2, content="b", incentive_salience=0.5, affective_valence=-0.2),
        3: SimpleNamespace(id=3, content="c", incentive_salience=0.2, affective_valence=0.0),
    }

    ms = SimpleNamespace(
        state_core=SimpleNamespace(arousal=0.95, valence=-0.2),
        state_cognition=SimpleNamespace(ego_strength=0.8, interlocus=-0.7, mental_aperture=0.8, willpower=-0.2),
        state_needs=SimpleNamespace(energy_stability=0.3, processing_power=0.3),
    )

    ghost = SimpleNamespace(
        current_tick_id=1,
        current_state=SimpleNamespace(latent_mental_state=ms),
        csm_manager=_CSM(),
        broadcast_history=[],
        get_knoxel_by_id=lambda k: knoxels.get(k),
        llm=SimpleNamespace(get_max_tokens=lambda _: 4000),
    )

    AttentionProc.run(ghost)

    assert hasattr(ghost, "workspace_gain")
    assert hasattr(ghost, "coalition_target_count")
    assert hasattr(ghost, "generation_token_cap")
    assert hasattr(ghost, "current_coalition")
    assert len(ghost.current_coalition) >= 1
    assert len(ghost.current_coalition) <= ghost.coalition_target_count


def test_attention_respects_qualia_modulation_tags():
    class _CSM:
        def items(self):
            return [
                SimpleNamespace(knoxel_id=1, activation=0.7),
                SimpleNamespace(knoxel_id=2, activation=0.7),
            ]

    knoxels = {
        1: SimpleNamespace(id=1, content="status defense mode", incentive_salience=0.4, affective_valence=0.1),
        2: SimpleNamespace(id=2, content="generic safe wording", incentive_salience=0.4, affective_valence=0.1),
    }

    ms = SimpleNamespace(
        state_core=SimpleNamespace(arousal=0.5, valence=0.0),
        state_cognition=SimpleNamespace(ego_strength=0.6, interlocus=-0.3, mental_aperture=0.1, willpower=0.0),
        state_needs=SimpleNamespace(energy_stability=0.6, processing_power=0.6),
    )

    ghost = SimpleNamespace(
        current_tick_id=1,
        current_state=SimpleNamespace(latent_mental_state=ms),
        csm_manager=_CSM(),
        broadcast_history=[],
        get_knoxel_by_id=lambda k: knoxels.get(k),
        llm=SimpleNamespace(get_max_tokens=lambda _: 4000),
        qualia_attention_modulation={
            "temperature_delta": -0.05,
            "boost_tags": ["status"],
            "suppress_tags": ["generic"],
            "rationale": "test",
        },
    )

    AttentionProc.run(ghost)
    assert ghost.current_coalition
    assert any(kid == 1 for kid, _ in ghost.current_coalition)
