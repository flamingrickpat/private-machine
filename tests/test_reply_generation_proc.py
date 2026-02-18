from types import SimpleNamespace

from pm.data_structures import DeclarativeFactKnoxel, Feature, FeatureType
from pm.ghosts.procedures.reply import ReplyGenerationProc
from pm.ghosts.schemas import BehaviorOutput


class _DummyLlm:
    def __init__(self, text: str):
        self.text = text
        self.last_msgs = []

    def completion_text(self, preset, inp, discard_thinks=True):
        self.last_msgs = list(inp)
        return self.text


class _Ghost:
    def __init__(self, llm):
        self.llm = llm
        self.current_tick_id = 5
        self.all_knoxels = {}
        self.all_features = []
        self.all_declarative_facts = []
        self.selected_action_schema = None
        self.simulated_reply = None
        self.config = SimpleNamespace(
            companion_name="Companion",
            user_name="User",
            universal_character_card="Sarcastic but precise character.",
        )
        self.conscious_broadcast = SimpleNamespace(content="User asks for concrete next steps.")
        self.conscious_candidates = [
            SimpleNamespace(id=11, content="Need to produce ordered migration checklist."),
        ]

    def add_knoxel(self, knoxel, generate_embedding=True):
        if getattr(knoxel, "id", -1) == -1:
            knoxel.id = len(self.all_knoxels) + 1
        if getattr(knoxel, "tick_id", -1) == -1:
            knoxel.tick_id = self.current_tick_id
        self.all_knoxels[knoxel.id] = knoxel
        if isinstance(knoxel, Feature):
            self.all_features.append(knoxel)
        return knoxel.id

    def get_knoxel_by_id(self, kid):
        return self.all_knoxels.get(kid)


def test_reply_generation_uses_action_schema_and_creates_dialogue_feature():
    llm = _DummyLlm("Do this in order: backup, migrate schema, then backfill.")
    ghost = _Ghost(llm)
    ghost.selected_action_schema = BehaviorOutput(
        action_description="Provide concrete ordered steps",
        speech="(schema hint only)",
        internal_thought="clarity first",
        tool_call=None,
    )
    ghost.all_features = [
        Feature(content="Can you fix this migration plan?", feature_type=FeatureType.Dialogue, source="User", causal=True),
    ]
    ghost.all_declarative_facts = [
        DeclarativeFactKnoxel(
            content="User prefers direct implementation answers.",
            reason="test",
            category=["pref"],
            importance=0.9,
            time_dependent=0.2,
        ),
    ]

    out = ReplyGenerationProc.run(ghost)
    assert out == "Do this in order: backup, migrate schema, then backfill."
    assert ghost.simulated_reply == out
    assert ghost.all_features[-1].feature_type == FeatureType.Dialogue
    assert ghost.all_features[-1].source == "Companion"
    assert ghost.all_features[-1].causal is True

    rendered = "\n".join(m[1] for m in llm.last_msgs if m[0] != "system")
    assert "**Companion's Character:**" in rendered
    assert "**Current Emotional State:**" in rendered
    assert "User prefers direct implementation answers." in rendered
    assert any(m[0] == "user" for m in llm.last_msgs)
    assert llm.last_msgs[-1][0] == "assistant"
    assert llm.last_msgs[-1][1].strip().endswith('Companion says: "')


def test_reply_generation_falls_back_to_schema_speech_when_llm_empty():
    llm = _DummyLlm("")
    ghost = _Ghost(llm)
    ghost.selected_action_schema = BehaviorOutput(
        action_description="Give short answer",
        speech="Short direct fallback.",
        internal_thought="fallback path",
        tool_call=None,
    )

    out = ReplyGenerationProc.run(ghost)
    assert out == "Short direct fallback."
    assert ghost.simulated_reply == "Short direct fallback."
