"""Check the rendered stage boundary through the real workflow kernel."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pm.context_render import Role, Turn
from workflows.main_cognition.stage_runner import build_stage_runner
from workflows.public_speech.public_context import PublicContext
from tests.unit.test_design_contracts import small_ghost


class StageContextRenderingTests(unittest.TestCase):
    def test_current_affect_sources_and_thinking_override_reach_the_model(self):
        public = PublicContext(companion_name="Spamuel", user_name="Jesse", affect="OLDER AFFECT SNAPSHOT",
            memories=["current utterance"], evidence=[{"knoxel_id": 11, "text": "current utterance"}],
            capabilities={"emotion_state": {"shame": 0.18}}, revisions={11: 2})
        captured = {}

        def complete(turns, **generation):
            captured.update(turns=turns, generation=generation)
            return '{"changes":[],"summary":"No goal change."}'

        with TemporaryDirectory() as folder:
            run = build_stage_runner(small_ghost(), complete, Path(folder),
                                     generation_overrides={"goals": {"enable_thinking": True}})
            result = run("goals", {"companion_name": "Spamuel", "user_name": "Jesse",
                "public_context": public.model_dump(mode="json"), "affect": {"emotions": {"shame": 0.8932695940694135}},
                "stimuli": [{"knoxel_id": 11, "text": "current utterance", "source": "user_input", "kind": "user.input"}]})
        self.assertEqual(result["changes"], [])
        self.assertTrue(all(isinstance(turn, Turn) for turn in captured["turns"]))
        self.assertEqual([turn.role for turn in captured["turns"]], [Role.System, Role.User, Role.User])
        body = "\n".join(turn.content.text for turn in captured["turns"])
        self.assertNotIn("OLDER AFFECT SNAPSHOT", body)
        self.assertNotIn('"emotion_state"', body)
        self.assertIn('"shame":0.89', captured["turns"][-1].content.text)
        self.assertEqual(body.count("current utterance"), 1)
        self.assertTrue(captured["generation"]["enable_thinking"])
        self.assertEqual(public.revisions, {11: 2})

