from types import SimpleNamespace

from pm.csm.csm import CSMItem, CSMManager, CSMState
from pm.data_structures import DeclarativeFactKnoxel, Feature, FeatureType
from pm.ghosts.procedures.pam import PamProc


class _DummyLlm:
    def __init__(self, mapping):
        self._mapping = mapping

    def get_embedding(self, text: str):
        return self._mapping[text]


class _DummyGhost:
    def __init__(self, llm):
        self.llm = llm
        self.current_tick_id = 100
        self.primary_stimulus = SimpleNamespace(content="current situation")

        core = SimpleNamespace(valence=-0.8, arousal=0.7, dominance=0.4)
        ms = SimpleNamespace(state_core=core)
        self.current_state = SimpleNamespace(latent_mental_state=ms)

        self.all_declarative_facts = []
        self.all_episodic_memories = []
        self.all_features = []
        self.all_knoxels = {}
        self._next_id = 1000

        self.csm_manager = CSMManager(self, state=CSMState())

    def add_knoxel(self, knoxel):
        if getattr(knoxel, "id", -1) == -1:
            knoxel.id = self._next_id
            self._next_id += 1
        self.all_knoxels[knoxel.id] = knoxel
        if isinstance(knoxel, Feature):
            self.all_features.append(knoxel)
        return knoxel.id

    def get_knoxel_by_id(self, knoxel_id):
        return self.all_knoxels.get(knoxel_id)


def _mk_fact(fid: int, content: str, emb, tick_id: int, valence: float):
    return DeclarativeFactKnoxel(
        id=fid,
        tick_id=tick_id,
        content=content,
        embedding=emb,
        reason="test",
        category=["test"],
        importance=1.0,
        time_dependent=0.0,
        metadata={
            "mental_state_signature": {
                "valence": valence,
                "arousal": 0.7,
                "dominance": 0.4,
            },
            "valence_hint": valence,
        },
    )


def test_csm_decay_spread_prune_and_gist():
    ghost = SimpleNamespace(all_knoxels={})

    k1 = Feature(id=1, tick_id=10, content="urgent context", feature_type=FeatureType.Dialogue, source="u", interlocus=1, embedding=[1.0, 0.0, 0.0], causal=True)
    k2 = Feature(id=2, tick_id=10, content="related detail", feature_type=FeatureType.Dialogue, source="u", interlocus=1, embedding=[0.95, 0.05, 0.0], causal=False)
    k3 = Feature(id=3, tick_id=10, content="stale noise", feature_type=FeatureType.Dialogue, source="u", interlocus=1, embedding=[0.0, 1.0, 0.0], causal=False)
    ghost.all_knoxels = {1: k1, 2: k2, 3: k3}

    state = CSMState(
        csm_item_states={
            1: CSMItem(knoxel_id=1, first_tick=10, last_tick=10, activation=1.0, peak_activation=1.0),
            2: CSMItem(knoxel_id=2, first_tick=10, last_tick=10, activation=0.2, peak_activation=0.2),
            3: CSMItem(knoxel_id=3, first_tick=10, last_tick=10, activation=0.08, peak_activation=0.08),
        }
    )
    manager = CSMManager(ghost, state=state)

    manager.decay_step(decay_factor=0.9)
    before = manager.state.csm_item_states[2].activation
    manager.spread_activation(min_source_activation=0.5, similarity_threshold=0.5, spread_factor=0.3)
    after = manager.state.csm_item_states[2].activation

    assert after > before

    removed = manager.prune(current_tick=200, min_activation=0.12, max_idle_ticks=20, permanent_max_idle_ticks=30)
    removed_ids = {x.knoxel_id for x in removed}

    assert 3 in removed_ids
    assert 1 in manager.state.csm_item_states

    gist = manager.build_gist(max_items=2)
    assert "urgent context" in gist


def test_pam_blended_ranking_prefers_state_aligned_memory():
    llm = _DummyLlm(
        {
            "current situation": [1.0, 0.0],
        }
    )
    ghost = _DummyGhost(llm)

    best = _mk_fact(11, "storm warning recalled", [0.99, 0.01], tick_id=95, valence=-0.8)
    mismatch = _mk_fact(12, "storm warning but upbeat", [0.99, 0.01], tick_id=99, valence=0.9)
    weak = _mk_fact(13, "unrelated memory", [0.2, 0.8], tick_id=99, valence=-0.8)

    ghost.all_declarative_facts = [mismatch, best, weak]

    ranked = PamProc._rank_memories(ghost, [1.0, 0.0], limit=3)
    assert ranked[0].memory.id == 11

    PamProc.run(ghost)

    recalled = [f for f in ghost.all_features if f.source == "PAM_Recall"]
    assert recalled
    assert recalled[0].content == "storm warning recalled"
    assert ghost.csm_manager.get_active_items(min_activation=0.3)

