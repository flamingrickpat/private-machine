from pathlib import Path
from types import SimpleNamespace

from pm.data_structures import Feature, FeatureType, ShellCCQUpdate
from pm.ghosts.procedure_main import BaseProcMain, Phase
from pm.ghosts.replay import TickReplayHarness, capture_tick_snapshot, load_tick_snapshot, save_tick_snapshot


class _Ghost:
    def __init__(self):
        self.current_tick_id = 12
        self.all_knoxels = {}
        self.all_features = []
        self.current_state = None
        self.simulated_reply = ""
        self._next = 1

    @property
    def max_knoxel_id(self):
        if not self.all_knoxels:
            return -1
        return max(self.all_knoxels.keys())

    def add_knoxel(self, knoxel, generate_embedding=True):
        if getattr(knoxel, "id", -1) == -1:
            knoxel.id = self._next
            self._next += 1
        if getattr(knoxel, "tick_id", -1) == -1:
            knoxel.tick_id = self.current_tick_id
        self.all_knoxels[knoxel.id] = knoxel
        if isinstance(knoxel, Feature):
            self.all_features.append(knoxel)
        return knoxel.id

    def get_knoxel_by_id(self, kid):
        return self.all_knoxels.get(kid)


def test_task12_event_trace_and_replay_snapshot_roundtrip(tmp_path: Path):
    ghost = _Ghost()

    def phase_setup(g):
        seed = Feature(
            content="Need stable next action.",
            feature_type=FeatureType.Thought,
            source="phase_setup",
            interlocus=-1,
            causal=True,
        )
        g.add_knoxel(seed)
        g.current_coalition = [(seed.id, 0.93)]
        g.conscious_candidates = [seed]
        g.conscious_broadcast = seed
        g.workspace_gain = 0.79
        g.coalition_target_count = 2
        g.generation_token_cap = 420

    def phase_arbitration(g):
        g.ego_decision_last = {
            "winner_lens": "world",
            "runner_up_lens": "self",
            "dissonance": 0.22,
        }
        g.ego_directive = "Prefer clear external coordination while staying concise."

    def phase_action(g):
        g.simulated_reply = "Let's do this in two steps: confirm assumptions, then execute."

    old_phases = BaseProcMain.PHASES
    try:
        BaseProcMain.PHASES = (
            Phase("setup", phase_setup),
            Phase("attention", lambda g: None),
            Phase("arbitration", phase_arbitration),
            Phase("action", phase_action),
            Phase("learn_store", lambda g: None),
        )
        ccq = BaseProcMain.cognitive_cycle(ghost)
    finally:
        BaseProcMain.PHASES = old_phases

    assert isinstance(ccq, ShellCCQUpdate)
    assert hasattr(ghost, "last_cycle_events")
    assert len(ghost.last_cycle_events) == 5
    assert ghost.last_cycle_events[1]["phase"] == "attention"
    assert ghost.last_cycle_events[1]["key_decisions"]["workspace_gain"] == 0.79
    assert ghost.last_cycle_events[2]["key_decisions"]["winner_lens"] == "world"
    assert ghost.last_cycle_events[3]["key_decisions"]["simulated_reply_len"] > 10

    snapshot = capture_tick_snapshot(ghost, ccq=ccq, include_tick_knoxels=True)
    assert snapshot.tick == 12
    assert snapshot.key_decisions["winner_lens"] == "world"
    assert snapshot.ccq_story
    assert len(snapshot.tick_knoxels) >= 1

    snap_path = tmp_path / "replay_tick_12.json"
    save_tick_snapshot(snapshot, snap_path)
    loaded = load_tick_snapshot(snap_path)
    harness = TickReplayHarness(loaded)
    summary = harness.summary()

    assert summary["tick"] == 12
    assert summary["phase_count"] == 5
    assert summary["err_count"] == 0
    assert summary["winner_lens"] == "world"
    assert summary["ccq_story_len"] > 0
    assert any(e.phase == "arbitration" for e in harness.iter_events())

