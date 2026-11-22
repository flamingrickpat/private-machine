"""Check which managed task evidence reaches the model choosing its next move."""
import unittest

from pm.context_render.prompt import RenderSettings
from pm.records.cycle_records import StimulusGroup
from workflows.main_cognition.cycle_work import CycleRequest, CycleWork
from workflows.main_cognition.phases.decide import build_messages
from workflows.main_cognition.phases.model import phase_prompts
from workflows.main_cognition.schemas import AppraisalAffect, ManagedRunView
from workflows.main_cognition.story_prefix import StoryPrefix


class DecisionTaskReceiptTest(unittest.TestCase):
    def test_choice_sees_the_cancelled_worker_and_its_assignment(self) -> None:
        task = ManagedRunView(run_id="shelter-worker", workflow_name="minecraft_project",
                              state="cancelled", instructions="Gather wood for the shelter.")
        work = CycleWork(request=CycleRequest(managed_runs=[task]), affect=AppraisalAffect(regulation="reflective"))
        group = StimulusGroup()
        settings = RenderSettings(context_window=32000, input_fraction=0.8, reserved_output_tokens=1024)
        prefix = StoryPrefix(work, group=group.key, settings=settings)

        turns = build_messages(work, phase_prompts("decide"), prefix, group)

        self.assertIn("[shelter-worker; minecraft_project; cancelled] Gather wood for the shelter.",
                      turns[-1].content.text)
