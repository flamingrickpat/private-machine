from types import SimpleNamespace

from pm.codelets.codelet_definitions import CodeletFamily
from pm.ghosts.schemas import CodeletRunTrace, CodeletTickTrace
from pm.data_structures import ShellCCQUpdate
from pm.ghosts.procedure_main import BaseProcMain, Phase
from pm.ghosts.procedures.codelets import (
    _apply_pathway_boosts,
    _build_meta_reflection_text,
    _seed_pathway_boosts,
    _select_pathway_themes,
    _should_emit_meta_reflection,
    _triage_families,
    render_codelet_timeline,
)


class _DummyGhost:
    def __init__(self):
        self.current_tick_id = 7
        self.all_features = []
        self.all_knoxels = {}
        self.current_state = None

    @property
    def max_knoxel_id(self):
        return -1


def test_procedure_main_traces_failures_and_continues():
    ghost = _DummyGhost()
    called = []

    def ok_phase(g):
        called.append("ok")

    def fail_phase(g):
        called.append("fail")
        raise RuntimeError("boom")

    old_phases = BaseProcMain.PHASES
    try:
        BaseProcMain.PHASES = (
            Phase("ok", ok_phase),
            Phase("fail", fail_phase),
            Phase("ok2", ok_phase),
        )
        ccq = BaseProcMain.cognitive_cycle(ghost)
    finally:
        BaseProcMain.PHASES = old_phases

    assert isinstance(ccq, ShellCCQUpdate)
    assert called == ["ok", "fail", "ok"]
    assert hasattr(ghost, "last_cycle_trace")
    assert len(ghost.last_cycle_trace) == 3
    assert ghost.last_cycle_trace[1]["phase"] == "fail"
    assert ghost.last_cycle_trace[1]["ok"] is False
    assert hasattr(ghost, "runtime_debug_panel")
    assert ghost.runtime_debug_panel["phase"]["err_count"] == 1


def test_procedure_main_builds_runtime_debug_panel():
    ghost = _DummyGhost()
    ghost.current_tick_id = 42
    ghost.workspace_gain = 0.83
    ghost.coalition_target_count = 1
    ghost.generation_token_cap = 512
    ghost.current_coalition = [(101, 0.9)]
    ghost.conscious_candidates = [SimpleNamespace(id=101, content="focus"), SimpleNamespace(id=102, content="alt")]
    ghost.conscious_broadcast = SimpleNamespace(id=101, content="Focused item")
    ghost.codelet_timeline_last = "tick=42 ... runs=2"
    ghost.codelet_trace_last = {
        "selected_families": ["MetaMonitoringIdentity"],
        "pathway_themes": ["anxiety"],
        "pathway_seeded": ["UncertaintyMagnitudeDetector"],
        "precursor_links": {"MetaCoherenceUpdate": [5001]},
        "runs": [{"families": ["MetaMonitoringIdentity"]}, {"families": ["Appraisals"]}],
    }
    ghost.all_features = [SimpleNamespace(id=999, tick_id=42, source="CodeletTraceReflection", content="inner")]

    called = []

    def ok_phase(g):
        called.append("ok")

    old_phases = BaseProcMain.PHASES
    try:
        BaseProcMain.PHASES = (Phase("ok", ok_phase),)
        BaseProcMain.cognitive_cycle(ghost)
    finally:
        BaseProcMain.PHASES = old_phases

    panel = ghost.runtime_debug_panel
    assert panel["tick"] == 42
    assert panel["attention_workspace"]["workspace_gain"] == 0.83
    assert panel["attention_workspace"]["broadcast_id"] == 101
    assert panel["codelets"]["run_count"] == 2
    assert panel["codelets"]["meta_run_count"] == 1
    assert panel["reflection"]["emitted"] is True
    assert isinstance(panel["ccq"]["knoxel_ids"], list)


def test_codelet_family_triage_prefers_high_score_families():
    ex1 = SimpleNamespace(signature=SimpleNamespace(families=[CodeletFamily.Appraisals]))
    ex2 = SimpleNamespace(signature=SimpleNamespace(families=[CodeletFamily.Attention]))
    ex3 = SimpleNamespace(signature=SimpleNamespace(families=[CodeletFamily.Appraisals, CodeletFamily.Narratives]))

    candidates = [
        (ex1, 0.9),
        (ex2, 0.2),
        (ex3, 0.8),
    ]

    families = _triage_families(
        candidates=candidates,
        state_boosts={CodeletFamily.Narratives: 1.5},
        max_families=2,
    )

    assert CodeletFamily.Appraisals in families
    assert CodeletFamily.Narratives in families
    assert CodeletFamily.Attention not in families


def test_pathway_theme_selection_from_context_text():
    ghost = SimpleNamespace(
        primary_stimulus=SimpleNamespace(content="I am anxious and worried but also curious to learn."),
        csm_manager=SimpleNamespace(state=SimpleNamespace(gist="Need to make progress toward this goal.")),
        conscious_broadcast=None,
    )

    themes = _select_pathway_themes(ghost)
    assert "anxiety" in themes
    assert "curiosity" in themes
    assert "persistence_goal_pursuit" in themes


def test_pathway_seeding_boosts_first_codelets():
    class _Reg:
        def __init__(self):
            self.calls = []

        def boost_codelet(self, codelet_type, scalar=0, factor=1):
            self.calls.append((codelet_type.name, factor))

    reg = _Reg()
    boosted = _seed_pathway_boosts(reg, ["anxiety"])
    assert boosted
    assert any(name for name, _ in reg.calls)


def test_pathway_boosts_create_precursor_links():
    class _Reg:
        def __init__(self):
            self.calls = []

        def boost_codelet(self, codelet_type, scalar=0, factor=1):
            self.calls.append((codelet_type.name, factor))

    reg = _Reg()
    precursor = {}
    boosted = _apply_pathway_boosts(
        registry=reg,
        fired_name="UncertaintyMagnitudeDetector",
        precursor_feature_ids=[101, 102],
        precursor_dict=precursor,
    )
    assert boosted
    assert reg.calls
    assert precursor
    assert all(v for v in precursor.values())
    assert all(x in [101, 102] for ids in precursor.values() for x in ids)


def test_codelet_trace_schemas_validate_for_logging():
    run = CodeletRunTrace(
        iteration=1,
        codelet="DemoCodelet",
        families=["Appraisals"],
        score=0.75,
        percepts=["InnerMonologuePercept"],
        feature_ids=[1, 2],
        delta_count=3,
        summary_excerpt="short text",
        output_mode="structured",
        precursor_feature_count=2,
        used_precursor_boost=True,
        boosted_followups=["NextCodelet"],
    )
    tick = CodeletTickTrace(
        tick=10,
        selected_families=["Appraisals"],
        pathway_themes=["anxiety"],
        pathway_seeded=["UncertaintyMagnitudeDetector"],
        runs=[run],
        precursor_links={"NextCodelet": [1, 2]},
    )
    assert tick.runs[0].used_precursor_boost is True
    assert tick.precursor_links["NextCodelet"] == [1, 2]


def test_render_codelet_timeline_compact_readable():
    run = CodeletRunTrace(
        iteration=1,
        codelet="MetaCoherenceUpdate",
        families=["MetaMonitoringIdentity"],
        score=0.81,
        percepts=["MetaMonitorPercept"],
        feature_ids=[11],
        delta_count=4,
        summary_excerpt="coherence",
        output_mode="structured",
        precursor_feature_count=3,
        used_precursor_boost=True,
        boosted_followups=["IdentityConsistencyCheck"],
    )
    tick = CodeletTickTrace(
        tick=33,
        selected_families=["MetaMonitoringIdentity"],
        pathway_themes=["anxiety"],
        pathway_seeded=["UncertaintyMagnitudeDetector"],
        runs=[run],
        precursor_links={"IdentityConsistencyCheck": [11]},
    )

    out = render_codelet_timeline(tick)
    assert "tick=33" in out
    assert "MetaCoherenceUpdate" in out
    assert "score=0.81" in out
    assert "->IdentityConsistencyCheck" in out


def test_meta_reflection_gate_respects_interlocus_and_meta_runs():
    run = CodeletRunTrace(
        iteration=1,
        codelet="MetaCoherenceUpdate",
        families=["MetaMonitoringIdentity"],
        score=0.8,
        percepts=[],
        feature_ids=[],
        delta_count=0,
        summary_excerpt="x",
        output_mode="structured",
        precursor_feature_count=0,
        used_precursor_boost=False,
        boosted_followups=[],
    )
    tick = CodeletTickTrace(
        tick=8,
        selected_families=[],
        pathway_themes=[],
        pathway_seeded=[],
        runs=[run],
        precursor_links={},
    )

    ghost_internal = SimpleNamespace(
        current_state=SimpleNamespace(
            latent_mental_state=SimpleNamespace(
                state_cognition=SimpleNamespace(interlocus=-0.7)
            )
        )
    )
    ghost_external = SimpleNamespace(
        current_state=SimpleNamespace(
            latent_mental_state=SimpleNamespace(
                state_cognition=SimpleNamespace(interlocus=0.2)
            )
        )
    )

    assert _should_emit_meta_reflection(ghost_internal, tick) is True
    assert _should_emit_meta_reflection(ghost_external, tick) is False


def test_meta_reflection_text_uses_trace_content():
    run = CodeletRunTrace(
        iteration=2,
        codelet="MetaCoherenceUpdate",
        families=["MetaMonitoringIdentity"],
        score=0.77,
        percepts=[],
        feature_ids=[],
        delta_count=2,
        summary_excerpt="x",
        output_mode="structured",
        precursor_feature_count=1,
        used_precursor_boost=True,
        boosted_followups=[],
    )
    tick = CodeletTickTrace(
        tick=99,
        selected_families=["MetaMonitoringIdentity"],
        pathway_themes=["anxiety", "curiosity"],
        pathway_seeded=[],
        runs=[run],
        precursor_links={},
    )
    txt = _build_meta_reflection_text(tick)
    assert "tick 99" in txt
    assert "pathway_themes=anxiety,curiosity" in txt
    assert "MetaCoherenceUpdate(0.77" in txt
