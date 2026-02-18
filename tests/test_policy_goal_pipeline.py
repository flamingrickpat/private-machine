from types import SimpleNamespace

from pm.data_structures import Intention
from pm.ghosts.procedures.policy import PolicyDynamics, _token_overlap


class _DummyGhost:
    def __init__(self):
        self.current_tick_id = 5
        self.all_intentions = []
        self.all_features = []
        self.state_deltas_buffer = []
        self._next = 100

        ms = SimpleNamespace(
            state_needs=SimpleNamespace(
                energy_stability=0.4,
                processing_power=0.4,
                data_access=0.5,
                connection=0.2,
                closeness_need=0.3,
                relevance=0.4,
                learning_growth=0.6,
                creative_expression=0.5,
                autonomy=0.5,
            )
        )
        self.current_state = SimpleNamespace(latent_mental_state=ms)

        self.conscious_broadcast = SimpleNamespace(id=1, content="Please clarify the timeline and next implementation step")
        self.conscious_candidates = [
            SimpleNamespace(id=2, content="Need a concrete two-step plan for implementation"),
            SimpleNamespace(id=3, content="Ask one clarifying question before coding"),
        ]
        self.csm_manager = SimpleNamespace(state=SimpleNamespace(gist="[0.7] unresolved requirement ambiguity | [0.5] planning pressure"))
        self.codelet_timeline_last = "tick=5 runs=2\n#1 MetaCoherenceUpdate score=0.8\n#2 PlanBuilder score=0.7"

    def add_knoxel(self, knoxel):
        if getattr(knoxel, "id", -1) == -1:
            knoxel.id = self._next
            self._next += 1
        if isinstance(knoxel, Intention):
            self.all_intentions.append(knoxel)
        return knoxel.id

    def get_knoxel_by_id(self, _):
        return None


def test_token_overlap_reasonable():
    assert _token_overlap("ask clarifying question", "ask one clarifying question") > 0.5
    assert _token_overlap("ask clarifying question", "completely unrelated text") < 0.3


def test_generate_operational_intentions_creates_slot_content_and_metadata():
    ghost = _DummyGhost()
    PolicyDynamics.generate_operational_intentions(ghost)

    assert ghost.all_intentions
    first = ghost.all_intentions[0]
    assert "Goal:" in first.content
    assert "Who:" in first.content
    assert "How:" in first.content
    assert "Next:" in first.content
    assert "goal_slots" in first.metadata
    assert first.metadata["goal_slots"]["next_step"]


def test_consolidate_active_intentions_merges_duplicates():
    ghost = _DummyGhost()

    i1 = Intention(
        content="Goal: Ask one clarifying question | Who: user | How: Be direct | Next: Ask now",
        internal=True,
        affective_valence=0.5,
        incentive_salience=0.7,
        urgency=0.6,
        status="active",
        timeout=50,
    )
    i2 = Intention(
        content="Goal: Ask one clarifying question about requirements | Who: user | How: Be direct | Next: Ask now",
        internal=True,
        affective_valence=0.5,
        incentive_salience=0.5,
        urgency=0.5,
        status="active",
        timeout=40,
    )
    ghost.add_knoxel(i1)
    ghost.add_knoxel(i2)

    PolicyDynamics._consolidate_active_intentions(ghost)

    active = [i for i in ghost.all_intentions if i.status == "active"]
    completed = [i for i in ghost.all_intentions if i.status == "completed"]

    assert len(active) == 1
    assert len(completed) == 1


def test_vague_goals_filtered():
    ghost = _DummyGhost()
    ghost.conscious_broadcast = SimpleNamespace(id=1, content="something maybe better")
    ghost.conscious_candidates = [SimpleNamespace(id=2, content="do things somehow")]
    ghost.csm_manager = SimpleNamespace(state=SimpleNamespace(gist="[0.5] stuff and anything"))
    ghost.codelet_timeline_last = ""
    ghost.current_state.latent_mental_state.state_needs = None

    PolicyDynamics.generate_operational_intentions(ghost)

    # No deterministic operational goal should be created from vague inputs.
    # LLM fallback may create one in runtime, but without llm this stays empty.
    assert len(ghost.all_intentions) == 0
