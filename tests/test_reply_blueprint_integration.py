from types import SimpleNamespace

from pm.ghosts.reply_blueprints import choose_reply_blueprint
from pm.ghosts.procedures.reply import ReplyGenerationProc
from pm.ghosts.schemas import BehaviorOutput


class _DummyLlm:
    def __init__(self):
        self.last_msgs = []

    def completion_text(self, preset, inp, discard_thinks=True):
        self.last_msgs = list(inp)
        return "Blueprint-steered reply."


class _Ghost:
    def __init__(self):
        self.current_tick_id = 9
        self.llm = _DummyLlm()
        self.config = SimpleNamespace(
            companion_name="Companion",
            user_name="User",
            universal_character_card="Brash persona.",
        )
        self.all_features = []
        self.all_declarative_facts = []
        self.conscious_broadcast = SimpleNamespace(content="Need concise direct implementation order.")
        self.conscious_candidates = []
        self.reply_blueprint_last = {}
        self.selected_action_schema = BehaviorOutput(
            action_description="Provide concise migration steps",
            speech="",
            internal_thought="short direct answer",
            tool_call=None,
        )
        self.simulated_reply = ""

    def add_knoxel(self, k, generate_embedding=True):
        self.all_features.append(k)
        return 1


def test_choose_reply_blueprint_prefers_concise_for_directive_language():
    behavior = BehaviorOutput(
        action_description="Give concise direct step-by-step implementation",
        speech="",
        internal_thought="keep it short",
        tool_call=None,
    )
    bp = choose_reply_blueprint(
        tick=3,
        behavior=behavior,
        broadcast="Need brief concrete steps now.",
        sim_prediction="User stays engaged if concise and direct.",
        simulation_bundle={"winner_lens": "world", "consolidated_attractors": ["clarity"]},
        history=[],
    )
    assert bp["name"] in {"answer_concise", "answer_directive"}


def test_reply_proc_appends_assistant_seed_from_blueprint():
    ghost = _Ghost()
    ghost.reply_blueprint_last = {
        "name": "answer_humor",
        "description": "add humor",
        "generation_prefix": "Companion decides to add humor:",
    }

    out = ReplyGenerationProc.run(ghost)
    assert out == "Blueprint-steered reply."
    assert ghost.llm.last_msgs[-1][0] == "assistant"
    assert "add humor" in ghost.llm.last_msgs[-1][1].lower()
