from types import SimpleNamespace

from pm.data_structures import Feature, FeatureType
from pm.ghosts.procedures.learning import ExpectationRealityProc


class _DummyCSM:
    def __init__(self):
        self.state = SimpleNamespace(gist="user asked for concise implementation progress")
        self.added = []

    def add_or_boost(self, item):
        self.added.append(item)


class _Ghost:
    def __init__(self):
        self.current_tick_id = 20
        self.primary_stimulus = SimpleNamespace(content="Thanks, that matches what I wanted.")
        self.conscious_broadcast = SimpleNamespace(content="Provide concise implementation progress")
        self.csm_manager = _DummyCSM()
        self.state_deltas_buffer = []
        self.policy_tendencies = {}

        self.simulation_bundle_last = {
            "winner_lens": "world",
            "outcomes": [
                {"lens": "world", "summary": "User will confirm alignment if response is concise."},
                {"lens": "self", "summary": "Internal coherence should improve."},
            ],
        }
        self.ego_decision_last = {"winner_lens": "world"}
        self.thought_blueprint_last = {"final_thought": "Compose Reply"}
        self.qualia_reflection_last = {"dominant_emotion": "focused", "tension": 0.3}

        self.all_knoxels = {}
        self.all_features = []
        self._next = 1000
        self.pending_expectation_queue = []
        self.llm = SimpleNamespace()  # intentionally sparse; rich extraction is safe-fail

    def add_knoxel(self, knoxel):
        if getattr(knoxel, "id", -1) == -1:
            knoxel.id = self._next
            self._next += 1
        self.all_knoxels[knoxel.id] = knoxel
        if isinstance(knoxel, Feature):
            self.all_features.append(knoxel)
        return knoxel.id


def test_store_current_expectation_creates_pending_record():
    ghost = _Ghost()
    action_feature = Feature(
        id=10,
        tick_id=ghost.current_tick_id,
        content='{"action_description":"reply"}',
        feature_type=FeatureType.Action,
        source="ActionProc",
        interlocus=1,
        causal=True,
        metadata={"sim_prediction": "User likely confirms alignment."},
    )
    ghost.add_knoxel(action_feature)

    ExpectationRealityProc.store_current_expectation(ghost)

    assert ghost.pending_expectation_queue
    pending = ghost.pending_expectation_queue[-1]
    assert pending["tick"] == ghost.current_tick_id
    assert "expected_outcome" in pending
    assert pending["expected_outcome"]


def test_bind_previous_outcome_consumes_pending_and_records_delta():
    ghost = _Ghost()
    ghost.pending_expectation_queue = [
        {
            "tick": 19,
            "expected_outcome": "User confirms alignment with concise implementation update.",
            "causal_action": "Provided concise implementation progress.",
            "situation_signature": "test-signature",
            "ego": {"winner_lens": "world"},
            "thought": {"final_thought": "Compose Reply"},
        }
    ]

    ExpectationRealityProc.bind_previous_outcome(ghost)

    assert ghost.pending_expectation_queue == []
    assert hasattr(ghost, "last_expectation_outcome")
    assert ghost.last_expectation_outcome["outcome"] in {"matched", "partial", "missed", "unknown"}
    assert ghost.policy_tendencies
    assert any(f.feature_type == FeatureType.ExpectationOutcome for f in ghost.all_features)
