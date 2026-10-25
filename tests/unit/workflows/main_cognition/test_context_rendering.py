"""Check the rendered grounding of a phase model call."""
from pathlib import Path
from types import SimpleNamespace
import unittest

from pm.context_render import Role, Turn
from pm.context_render.story import (RecordView as SharedRecordView, fill_story_names, record_view, shard_block_text,
                                     story_line)
from pm.records.workspace import ShardKey
from workflows.main_cognition import codelet_registry, csm_view, model_json, schemas, story_prefix
from workflows.main_cognition.cycle_work import CycleRequest, CycleWork
from workflows.main_cognition.phases import goals
from workflows.main_cognition.schemas import AppraisalAffect, RecordView
from workflows.public_speech.public_context import PublicContext

STORY_PROMPTS = Path("workflows/main_cognition/phases/prompts/story.yaml")


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
        self.assertIn("Companion is reflective:", captured["turns"][-1].content.text)
        self.assertEqual(body.count("current utterance"), 1)
        self.assertFalse(captured["generation"]["enable_thinking"])
        self.assertEqual(public.revisions, {11: 2})


class SharedStoryRenderingTests(unittest.TestCase):
    """The workflow renders must be the shared renderer, not a copy of it.

    The memory inspector in ``pm/tools/inspector`` reads a checkpoint with the
    same functions, so it shows the maintainer what the model read. It may not
    import ``workflows``, so these adapters keep the workflow names while the
    layout lives once, in ``pm.context_render.story``.
    """

    def setUp(self):
        self.prompts = model_json.load_prompts(STORY_PROMPTS)
        self.records = {11: SharedRecordView(record_id=11, kind="Feature", source="Rick",
                                             content="hello there"),
                        12: SharedRecordView(record_id=12, kind="Feature", source="Companion",
                                             content="hi Rick")}

    def test_moved_names_are_the_shared_objects(self):
        self.assertIs(schemas.RecordView, SharedRecordView)
        self.assertIs(model_json.ShardLabels, type(self.prompts.shard_labels))
        self.assertIs(csm_view.story_line, story_line)
        self.assertIs(csm_view.record_view, record_view)

    def test_shard_block_matches_the_shared_layout_exactly(self):
        key = ShardKey(kind="entity", ref="7")
        for verbatim in (False, True):
            with self.subTest(verbatim_tail=verbatim):
                self.assertEqual(
                    story_prefix.shard_block(self.prompts, key, "Rick", "Companion", self.records, [11], [12],
                                             verbatim_tail=verbatim),
                    shard_block_text(key, "Rick", "Companion", self.records, [11], [12],
                                     template=self.prompts.shard_template, labels=self.prompts.shard_labels,
                                     verbatim_tail=verbatim))

    def test_shard_label_names_like_the_inspector_does(self):
        request = SimpleNamespace(user_entity_id=7, user_name="Rick", entity_names={8: "Notch"})
        self.assertEqual(story_prefix.shard_label(ShardKey(kind="entity", ref="8"), request), "Notch")
        self.assertEqual(story_prefix.shard_label(ShardKey(kind="self"), request), "self")
        self.assertEqual(story_prefix.shard_label(ShardKey(kind="workflow", ref="minecraft"), request), "minecraft")

    def test_prompt_name_filling_is_the_shared_one(self):
        text = "{companion_name} answers {entity_name} in {place}"
        self.assertEqual(codelet_registry.fill(text, "Companion", "Rick"),
                         fill_story_names(text, "Companion", "Rick"))
