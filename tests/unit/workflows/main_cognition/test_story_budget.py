from datetime import datetime
import unittest

from pm.context_render.budget import RequiredContextTooLarge
from pm.context_render.prompt import RenderSettings
from pm.context_render.story import RecordView
from pm.records.base import RecordScope
from workflows.main_cognition.cycle_work import CycleRequest, CycleWork
from workflows.main_cognition.schemas import Stimulus
from workflows.main_cognition.story_prefix import StoryPrefix
from workflows.public_speech.situational_guidance import SituationalGuidance, SituationSource


class StoryBudgetTests(unittest.TestCase):
    def test_story_reserves_instruction_and_current_input_before_optional_guidance(self) -> None:
        now = datetime(2026, 6, 8, 21, 30)
        intention = SituationSource(record_id=12, revision=1, occurred_at_utc=now,
            source="Spamuel", kind="intention", text="Earlier intention:" + " return" * 2000,
            scope=RecordScope())
        situation = SituationalGuidance(as_of_utc=now, companion_name="Spamuel", intentions=[intention])
        current = RecordView(record_id=13, source="Jesse", content="Reconsider the last message.")
        stimulus = Stimulus(knoxel_id=current.record_id, source="user_input", text=current.content)
        request = CycleRequest(companion_name="Spamuel", user_name="Jesse", stimuli=[stimulus],
                               situational_guidance=[situation], cycle_time=now)
        work = CycleWork(request=request)
        work.records[current.record_id] = current
        settings = RenderSettings(context_window=32000, input_fraction=0.8, reserved_output_tokens=8192)
        prefix = StoryPrefix(work, settings=settings)
        self.assertIn(intention.text, prefix.ask("Review the goals.")[1].content.text)

        instruction = "Review the goals:" + " evidence" * 14000
        turns = prefix.ask(instruction)
        self.assertEqual(turns[-1].content.text, instruction)
        self.assertIn(current.content, turns[1].content.text)
        self.assertIn("2026-06-08T21:30", turns[1].content.text)
        self.assertNotIn("Earlier intention:", turns[1].content.text)
        with self.assertRaises(RequiredContextTooLarge):
            prefix.ask(instruction * 2)
