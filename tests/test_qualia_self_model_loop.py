from types import SimpleNamespace

from pm.data_structures import Feature, FeatureType, Narrative
from pm.ghosts.base_ghost import SelfModel
from pm.ghosts.procedures.qualia import QualiaProc
from pm.ghosts.schemas import QualiaSelfReflection, SelfTheoryUpdate


class _DummyLlm:
    def __init__(self, fail=False):
        self.fail = fail

    def completion_tool(self, preset, inp, tools):
        if self.fail:
            raise RuntimeError("forced failure")

        tool = tools[0]
        if tool is QualiaSelfReflection:
            return None, [
                QualiaSelfReflection(
                    inner_voice="I do not feel like playing nice; I want control over this turn.",
                    self_narrative="I am acting from pride and impatience, not from helpfulness.",
                    focus_target="control over this turn",
                    dominant_emotion="defiant",
                    tension=0.78,
                    self_consistency=0.70,
                    confidence=0.74,
                    hidden_causes=["status defense", "uncertainty masking"],
                    attention_bias=0.09,
                )
            ]

        if tool is SelfTheoryUpdate:
            return None, [
                SelfTheoryUpdate(
                    revised_biography_line="I defaulted to defensive control under pressure.",
                    retained_theories=["Pressure amplifies status defense."],
                    new_theories=["When uncertain, I become sharper and less cooperative."],
                    discarded_theories=[],
                    confidence=0.73,
                    mismatch=0.28,
                    rationale="Reflection and tension metrics align with defensive behavior.",
                )
            ]

        raise AssertionError("unexpected tool")

    def completion_text(self, preset, inp):
        return "A brief autobiographical summary."

    def get_embedding(self, text: str):
        return [1.0, 0.0]


class _Ghost:
    def __init__(self, llm):
        self.llm = llm
        self.current_tick_id = 9
        self.config = SimpleNamespace(
            companion_name="BenderLike",
            universal_character_card="A blunt, sardonic anti-hero who values pride over politeness.",
        )
        self.self_model = SelfModel(style_anchors=["blunt", "sardonic"])
        self.conscious_broadcast = SimpleNamespace(content="User challenged my competence.")
        self.ego_directive = "Prioritize self lens and preserve authority."
        self.simulation_bundle_last = {
            "outcomes": [
                {"lens": "self", "utility": 0.82, "polarity": "positive", "summary": "assertive response protects identity"},
                {"lens": "meta", "utility": 0.60, "polarity": "neutral", "summary": "stay within persona constraints"},
            ]
        }

        core = SimpleNamespace(valence=-0.35, arousal=0.82)
        cog = SimpleNamespace(interlocus=-0.55, mental_aperture=-0.25)
        self.current_state = SimpleNamespace(latent_mental_state=SimpleNamespace(state_core=core, state_cognition=cog))

        self.all_knoxels = {}
        self.all_features = []
        self.all_narratives = []
        self._next = 100

    def add_knoxel(self, knoxel):
        if getattr(knoxel, "id", -1) == -1:
            knoxel.id = self._next
            self._next += 1
        self.all_knoxels[knoxel.id] = knoxel
        if isinstance(knoxel, Feature):
            self.all_features.append(knoxel)
        if isinstance(knoxel, Narrative):
            self.all_narratives.append(knoxel)
        return knoxel.id


def test_qualia_llm_path_updates_reflection_and_self_model():
    ghost = _Ghost(llm=_DummyLlm(fail=False))

    QualiaProc.generate_subjective_experience(ghost)
    QualiaProc.update_self_model(ghost)
    QualiaProc.apply_self_model_attention(ghost)
    QualiaProc.apply_self_model_action(ghost)

    assert ghost.subjective_experience
    assert ghost.qualia_reflection_last["dominant_emotion"] == "defiant"
    assert ghost.self_model_update_last["new_theories"]
    assert ghost.self_model.active_theories
    assert hasattr(ghost, "qualia_attention_modulation")
    assert hasattr(ghost, "qualia_action_bias")
    assert any(f.source == "QualiaSelfStory" for f in ghost.all_features)


def test_qualia_fallback_path_still_produces_transparent_state():
    ghost = _Ghost(llm=_DummyLlm(fail=True))

    QualiaProc.generate_subjective_experience(ghost)
    QualiaProc.update_self_model(ghost)
    QualiaProc.apply_self_model_attention(ghost)
    QualiaProc.apply_self_model_action(ghost)

    assert ghost.qualia_reflection_last["inner_voice"]
    assert ghost.self_model_update_last["rationale"]
    assert ghost.self_model.theory_confidence >= 0.0
    assert ghost.qualia_attention_modulation["rationale"] == "qualia_self_model_loop"
