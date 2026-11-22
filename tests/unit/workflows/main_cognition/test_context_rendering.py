"""Check the rendered grounding of a phase model call."""
import unittest

from pm.context_render import Role
from pm.context_render.prompt import RenderSettings
from pm.context_render.story import (
    RecordView as SharedRecordView, ShardLabels, fill_story_names, record_view, shard_block_text, story_line,
)
from pm.memory.evidence import EvidenceItem
from pm.records.mental_state import FullMentalState
from pm.records.workspace import ShardKey
from workflows.main_cognition import codelet_registry, csm_view, schemas, story_prefix
from workflows.main_cognition.cycle_work import CycleRequest, CycleWork
from workflows.main_cognition.interval_review import IntervalReview
from workflows.main_cognition.phases import goals
from workflows.main_cognition.phases.model import phase_prompts
from workflows.main_cognition.prompt_config import PhasePrompts
from workflows.main_cognition.schemas import AppraisalAffect, RecordView, Stimulus
from workflows.minecraft_body.capabilities import CapabilitySnapshot
from workflows.public_speech.public_context import PublicContext


class PhaseContextRenderingTests(unittest.TestCase):
    def test_goal_prompt_contains_current_affect_and_one_copy_of_the_stimulus(self) -> None:
        accepted_state = FullMentalState()
        accepted_state.state_emotions.shame = 0.18
        public = PublicContext(companion_name="Companion", user_name="User", affect="OLDER AFFECT SNAPSHOT",
            memories=["current utterance"], evidence=[EvidenceItem(knoxel_id=11, text="current utterance")],
            capabilities=CapabilitySnapshot(emotion_state=accepted_state.state_emotions))
        public.revisions[11] = 2
        expected_revisions = public.revisions.copy()
        review = IntervalReview(after_record_id=0, through_record_id=11, first_review=True, moves=[], suspensions=[])
        request = CycleRequest(
            companion_name="Companion", user_name="User", public_context=public, interval_review=review,
            stimuli=[Stimulus(knoxel_id=11, text="current utterance", source="user_input", kind="user.input")])
        affect = AppraisalAffect(felt="Companion feels strong shame.", regulation="reflective")
        utterance = RecordView(record_id=11, kind="Feature", record_class="Feature", source="User",
                               content="current utterance", feature_type="Dialogue")
        work = CycleWork(request=request, affect=affect)
        work.records[utterance.record_id] = utterance
        settings = RenderSettings(context_window=32000, input_fraction=0.8, reserved_output_tokens=1024)
        prefix = story_prefix.StoryPrefix(work, settings=settings)
        turns = goals.build_messages(work, phase_prompts("goals"), prefix)
        self.assertEqual([turn.role for turn in turns], [Role.System, Role.User, Role.Assistant, Role.User])
        body = "\n".join(turn.content.text for turn in turns)
        self.assertNotIn("OLDER AFFECT SNAPSHOT", body)
        self.assertNotIn('"emotion_state"', body)
        self.assertIn("Companion feels strong shame.", turns[-1].content.text)
        self.assertIn("Companion is reflective:", turns[-1].content.text)
        self.assertEqual(body.count("current utterance"), 1)
        self.assertEqual(public.revisions, expected_revisions)


class SharedStoryRenderingTests(unittest.TestCase):
    """The workflow renders must be the shared renderer, not a copy of it.

    The memory inspector in ``pm/tools/inspector`` reads a checkpoint with the
    same functions, so it shows the maintainer what the model read. It may not
    import ``workflows``, so these adapters keep the workflow names while the
    layout lives once, in ``pm.context_render.story``.
    """

    def setUp(self) -> None:
        self.prompts: PhasePrompts = phase_prompts("story")
        self.records: dict[int, SharedRecordView] = {  # pm-lint: allow PM004 -- workspace records keyed by knoxel ID
            11: SharedRecordView(record_id=11, kind="Feature", source="Rick", content="hello there"),
            12: SharedRecordView(record_id=12, kind="Feature", source="Companion", content="hi Rick"),
        }

    def test_moved_names_are_the_shared_objects(self) -> None:
        self.assertIs(schemas.RecordView, SharedRecordView)
        self.assertIs(type(self.prompts.shard_labels), ShardLabels)
        self.assertIs(csm_view.story_line, story_line)
        self.assertIs(csm_view.record_view, record_view)

    def test_shard_block_matches_the_shared_layout_exactly(self) -> None:
        key = ShardKey(kind="entity", ref="7")
        for verbatim in (False, True):
            with self.subTest(verbatim_tail=verbatim):
                self.assertEqual(
                    story_prefix.shard_block(self.prompts, key, "Rick", "Companion", self.records, [11], [12],
                                             verbatim_tail=verbatim),
                    shard_block_text(key, "Rick", "Companion", self.records, [11], [12],
                                     template=self.prompts.shard_template, labels=self.prompts.shard_labels,
                                     verbatim_tail=verbatim))

    def test_shard_label_names_like_the_inspector_does(self) -> None:
        request = CycleRequest(user_entity_id=7, user_name="Rick")
        request.entity_names[8] = "Notch"
        self.assertEqual(story_prefix.shard_label(ShardKey(kind="entity", ref="8"), request), "Notch")
        self.assertEqual(story_prefix.shard_label(ShardKey(kind="self"), request), "self")
        self.assertEqual(story_prefix.shard_label(ShardKey(kind="workflow", ref="minecraft"), request), "minecraft")

    def test_prompt_name_filling_is_the_shared_one(self) -> None:
        text = "{companion_name} answers {entity_name} in {place}"
        self.assertEqual(codelet_registry.fill(text, "Companion", "Rick"),
                         fill_story_names(text, "Companion", "Rick"))
