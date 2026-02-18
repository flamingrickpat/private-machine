from types import SimpleNamespace

from pm.data_structures import Feature, FeatureType, StimulusType
from pm.ghosts.procedures.action import ActionSelectionProc
from pm.ghosts.procedures.thought_blueprint import ThoughtBlueprintProc
from pm.ghosts.schemas import BehaviorOutput


class _PromptCaptureLlm:
    def __init__(self):
        self.last_user_prompt = ""

    def completion_tool(self, preset, inp, tools):
        for role, text in inp:
            if role == "user":
                self.last_user_prompt = text
        return None, [
            BehaviorOutput(
                action_description="Reply directly",
                speech="test reply",
                internal_thought="follow thought blueprint",
                tool_call=None,
            )
        ]

    def get_max_tokens(self, _):
        return 4000

    def get_embedding(self, text):
        return [1.0, 0.0]


class _Ghost:
    def __init__(self):
        self.current_tick_id = 14
        self.config = SimpleNamespace(companion_name="Companion")
        self.current_state = SimpleNamespace(
            latent_mental_state=SimpleNamespace(
                state_core=SimpleNamespace(valence=0.1, arousal=0.4),
                state_emotions=SimpleNamespace(
                    affection=0.0,
                    self_worth=0.1,
                    trust=0.1,
                    disgust=0.0,
                    anxiety=0.3,
                ),
                state_needs=SimpleNamespace(
                    energy_stability=0.6,
                    processing_power=0.7,
                    data_access=0.6,
                    connection=0.5,
                    relevance=0.6,
                    learning_growth=0.7,
                    creative_expression=0.5,
                    autonomy=0.6,
                ),
                state_cognition=SimpleNamespace(
                    interlocus=0.2,
                    mental_aperture=0.1,
                    ego_strength=0.7,
                    willpower=0.2,
                ),
            ),
            rating=0,
        )
        self.primary_stimulus = SimpleNamespace(stimulus_type=StimulusType.UserMessage)
        self.conscious_broadcast = SimpleNamespace(id=3, content="User asks for concrete next steps.")
        self.workspace_gain = 0.45
        self.ego_directive = "Prioritize world clarity."
        self.subjective_experience = "I want concise traction."
        self.all_knoxels = {}
        self.all_features = []
        self._next = 100
        self.llm = _PromptCaptureLlm()

    def add_knoxel(self, knoxel):
        if getattr(knoxel, "id", -1) == -1:
            knoxel.id = self._next
            self._next += 1
        self.all_knoxels[knoxel.id] = knoxel
        if isinstance(knoxel, Feature):
            self.all_features.append(knoxel)
        return knoxel.id

    def get_knoxel_by_id(self, kid):
        return self.all_knoxels.get(kid)


def test_thought_blueprint_proc_persists_path_and_directive():
    ghost = _Ghost()

    ThoughtBlueprintProc.run(ghost)

    assert ghost.thought_blueprint_last["path"]
    assert len(ghost.thought_blueprint_last["path"]) >= 2
    assert ghost.thought_blueprint_directive
    assert any(f.source == "ThoughtBlueprintProc" for f in ghost.all_features)


def test_action_selection_prompt_includes_thought_blueprint_directive():
    ghost = _Ghost()
    ghost.thought_blueprint_directive = "Compose a grounded reply from current appraisal."
    ghost.thought_blueprint_action_hints = ["Reply", "LoopBack"]

    ActionSelectionProc.run(ghost)

    assert "Thought Blueprint Directive:" in ghost.llm.last_user_prompt
    assert "Thought Action Hints:" in ghost.llm.last_user_prompt
