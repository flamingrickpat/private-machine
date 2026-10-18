"""Check the rendered grounding of a phase model call."""
from types import SimpleNamespace
import unittest

from pm.context_render import Role, Turn
from workflows.main_cognition.cycle_work import CycleRequest, CycleWork
from workflows.main_cognition.phases import goals
from workflows.main_cognition.schemas import AppraisalAffect, RecordView
from workflows.public_speech.public_context import PublicContext


class PhaseContextRenderingTests(unittest.TestCase):
    def test_current_affect_and_sources_reach_the_model_once(self):
        public = PublicContext(companion_name="Companion", user_name="User", affect="OLDER AFFECT SNAPSHOT",
            memories=["current utterance"], evidence=[{"knoxel_id": 11, "text": "current utterance"}],
            capabilities={"emotion_state": {"shame": 0.18}}, revisions={11: 2})
        captured = {}

        def complete(turns, **generation):
            captured.update(turns=turns, generation=generation)
            return '{"changes":[]}'

        work = CycleWork(request=CycleRequest(
            companion_name="Companion", user_name="User", public_context=public,
            stimuli=[{"knoxel_id": 11, "text": "current utterance", "source": "user_input", "kind": "user.input"}]),
            affect=AppraisalAffect(felt="Companion feels strong shame.", regulation="reflective"),
            records={11: RecordView(record_id=11, kind="Feature", record_class="Feature", source="User",
                                    content="current utterance", feature_type="Dialogue")})
        context = SimpleNamespace(runtime=SimpleNamespace(tools={"llm_complete": complete}, environment=None))
        work = goals.run(context, work)
        self.assertEqual(work.goals.changes, [])
        self.assertTrue(all(isinstance(turn, Turn) for turn in captured["turns"]))
        # The shared story prefix (system, story, acknowledgement), then the
        # grounding turn, then the goal instruction.
        self.assertEqual([turn.role for turn in captured["turns"]],
                         [Role.System, Role.User, Role.Assistant, Role.User, Role.User])
        body = "\n".join(turn.content.text for turn in captured["turns"])
        self.assertNotIn("OLDER AFFECT SNAPSHOT", body)
        self.assertNotIn('"emotion_state"', body)
        self.assertIn("Companion feels strong shame.", captured["turns"][-1].content.text)
        self.assertIn("Regulation mode: reflective", captured["turns"][-1].content.text)
        self.assertEqual(body.count("current utterance"), 1)
        self.assertFalse(captured["generation"]["enable_thinking"])
        self.assertEqual(public.revisions, {11: 2})
