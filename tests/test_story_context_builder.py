from types import SimpleNamespace

from pm.data_structures import DeclarativeFactKnoxel, Feature, FeatureType
from pm.ghosts.story_context import StoryContextBuilder, StorySelectionConfig
from pm.ghosts.base_ghost import BaseGhost, GhostConfig


class _DummyLlm:
    def get_embedding(self, text: str):
        t = (text or "").lower()
        if "migration" in t:
            return [1.0, 0.0, 0.0]
        if "emotion" in t:
            return [0.0, 1.0, 0.0]
        return [0.2, 0.2, 0.2]


class _Ghost:
    def __init__(self):
        self.current_tick_id = 10
        self.llm = _DummyLlm()
        self.all_knoxels = {}
        self.all_features = []
        self.all_declarative_facts = []
        self.all_intentions = []
        self.selected_action_schema = SimpleNamespace(action_description="Give migration steps")
        self.thought_blueprint_directive = "reduce ambiguity"
        self.primary_stimulus = SimpleNamespace(content="Need migration order.")
        self.config = SimpleNamespace(companion_name="Companion", user_name="User")
        core = SimpleNamespace(valence=0.2)
        cog = SimpleNamespace(interlocus=0.2)
        lms = SimpleNamespace(state_core=core, state_cognition=cog)
        self.current_state = SimpleNamespace(latent_mental_state=lms)

    def add_knoxel(self, k):
        if getattr(k, "id", -1) == -1:
            k.id = len(self.all_knoxels) + 1
        if getattr(k, "tick_id", -1) == -1:
            k.tick_id = self.current_tick_id
        self.all_knoxels[k.id] = k
        if isinstance(k, Feature):
            self.all_features.append(k)
        if isinstance(k, DeclarativeFactKnoxel):
            self.all_declarative_facts.append(k)


def test_story_context_builder_selects_relevant_knoxels_with_debug_rows():
    g = _Ghost()
    g.add_knoxel(
        Feature(
            content="User asks for migration order.",
            feature_type=FeatureType.Dialogue,
            source="User",
            causal=True,
            tick_id=9,
            embedding=[1.0, 0.0, 0.0],
            incentive_salience=0.7,
            affective_valence=0.1,
            interlocus=0.9,
        )
    )
    g.add_knoxel(
        Feature(
            content="Random weather chat.",
            feature_type=FeatureType.Dialogue,
            source="User",
            causal=True,
            tick_id=3,
            embedding=[0.0, 1.0, 0.0],
            incentive_salience=0.1,
            affective_valence=0.0,
            interlocus=0.9,
        )
    )
    g.add_knoxel(
        DeclarativeFactKnoxel(
            content="User prefers direct implementation guidance.",
            reason="test",
            category=["pref"],
            importance=0.8,
            time_dependent=0.2,
            tick_id=8,
            embedding=[1.0, 0.0, 0.0],
        )
    )

    packet = StoryContextBuilder.build(
        g,
        focus_text="Need migration implementation steps",
        config=StorySelectionConfig(max_items=10, max_tokens=500),
    )
    assert packet.selected_ids
    assert packet.debug_rows
    assert "migration" in packet.story_text.lower()
    assert any("direct implementation guidance" in x.lower() for x in packet.fact_lines)
    assert getattr(g, "story_context_last", {}).get("selected_count", 0) >= 1


def test_story_context_builder_forces_explicit_recall_old_memory():
    g = _Ghost()
    old = Feature(
        content="Ancient but critical memory about user's migration preference.",
        feature_type=FeatureType.MemoryRecall,
        source="PAM",
        causal=True,
        tick_id=1,
        embedding=[0.0, 1.0, 0.0],  # intentionally low similarity to migration query
        incentive_salience=0.0,
        metadata={"explicit_recall": [{"at_tick": 10, "until_tick": 10, "weight": 1.0, "reason": "must remember"}]},
    )
    g.add_knoxel(old)
    g.add_knoxel(
        Feature(
            content="Recent unrelated chit chat.",
            feature_type=FeatureType.Dialogue,
            source="User",
            causal=True,
            tick_id=9,
            embedding=[0.0, 1.0, 0.0],
            incentive_salience=0.1,
        )
    )
    packet = StoryContextBuilder.build(
        g,
        focus_text="Need migration implementation steps",
        config=StorySelectionConfig(max_items=8, max_tokens=500),
    )
    assert old.id in packet.selected_ids
    row = next((r for r in packet.debug_rows if r["id"] == old.id), {})
    assert row.get("forced", False) is True
    assert row.get("explicit_recall_active", False) is True


def test_baseghost_schedule_explicit_recall_writes_metadata():
    class _Llm:
        def get_embedding(self, text):
            return [0.0, 0.0, 0.0]

    ghost = BaseGhost(_Llm(), GhostConfig())
    f = Feature(content="remember me", feature_type=FeatureType.Thought, source="t")
    ghost.add_knoxel(f)
    ok = ghost.schedule_explicit_recall(f.id, 100, reason="critical callback")
    assert ok is True
    saved = ghost.get_knoxel_by_id(f.id)
    plans = list((saved.metadata or {}).get("explicit_recall", []) or [])
    assert plans
    assert plans[-1]["at_tick"] == 100
    assert plans[-1]["reason"] == "critical callback"
