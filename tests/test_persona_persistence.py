from types import SimpleNamespace

from pm.data_structures import CauseEffectKnoxel, DeclarativeFactKnoxel, Feature, FeatureType
from pm.ghosts.persona_memory import collect_persona_signals
from pm.ghosts.procedures.persona import PersonaPersistenceProc
from pm.ghosts.schemas import BehaviorOutput


class _Ghost:
    def __init__(self):
        self.current_tick_id = 21
        self.all_knoxels = {}
        self.all_features = []
        self.all_declarative_facts = []
        self.selected_action_schema = BehaviorOutput(
            action_description="Answer with concise directive",
            speech="",
            internal_thought="prioritize directness",
            tool_call=None,
        )
        self.simulated_reply = "Do backup first, then migration, then backfill."
        self.reply_blueprint_last = {"name": "answer_directive"}
        self.ego_decision_last = {"winner_lens": "world"}
        self.simulation_bundle_last = {"winner_lens": "world"}
        self.primary_stimulus = SimpleNamespace(content="Need practical migration help")
        self.conscious_broadcast = SimpleNamespace(content="migration order confusion")
        self.config = SimpleNamespace(companion_name="Companion")
        self.persona_pattern_signatures = set()

        self.add_knoxel(
            Feature(
                content=self.simulated_reply,
                feature_type=FeatureType.Dialogue,
                source="Companion",
                causal=True,
            )
        )

    def add_knoxel(self, knoxel, generate_embedding=True):
        if getattr(knoxel, "id", -1) == -1:
            knoxel.id = len(self.all_knoxels) + 1
        if getattr(knoxel, "tick_id", -1) == -1:
            knoxel.tick_id = self.current_tick_id
        self.all_knoxels[knoxel.id] = knoxel
        if isinstance(knoxel, Feature):
            self.all_features.append(knoxel)
        if isinstance(knoxel, DeclarativeFactKnoxel):
            self.all_declarative_facts.append(knoxel)
        return knoxel.id

    def get_knoxel_by_id(self, kid):
        return self.all_knoxels.get(kid)


def test_persona_persistence_emits_ce_and_fact_signals():
    ghost = _Ghost()
    PersonaPersistenceProc.run(ghost)

    ces = [k for k in ghost.all_knoxels.values() if isinstance(k, CauseEffectKnoxel)]
    facts = [k for k in ghost.all_knoxels.values() if isinstance(k, DeclarativeFactKnoxel)]
    assert ces
    assert facts
    assert ces[-1].category == "persona_action_quirk"
    assert bool((ces[-1].metadata or {}).get("persona_signal", False)) is True
    assert bool((facts[-1].metadata or {}).get("persona_signal", False)) is True


def test_collect_persona_signals_returns_ranked_persona_memory():
    ghost = _Ghost()
    PersonaPersistenceProc.run(ghost)
    out = collect_persona_signals(ghost, query="concise migration steps", limit=3)
    assert out
    assert any(("FACT:" in x or "CE:" in x) for x in out)
