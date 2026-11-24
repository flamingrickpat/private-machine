"""Check which managed task evidence reaches the model choosing its next move."""
import unittest

from pm.context_render.prompt import RenderSettings
from pm.records.cycle_records import StimulusGroup
from workflows.main_cognition.cycle_work import CycleRequest, CycleWork
from workflows.main_cognition.phases.decide import build_messages
from workflows.main_cognition.phases.model import phase_prompts
from pydantic import ValidationError
from workflows.main_cognition.schemas import AppraisalAffect, ManagedRunView, RouteAnswer, RouteCheck
from workflows.main_cognition.story_prefix import StoryPrefix


class DecisionTaskReceiptTest(unittest.TestCase):
    def test_pending_revision_needs_a_new_partner_source(self) -> None:
        check = RouteCheck(workflow_names=["self_management"], visible_run_ids=frozenset({"shelter"}),
            updatable_run_ids=frozenset({"shelter"}), pending_revision_run_ids=frozenset({"shelter"}),
            partner_sources=frozenset({30}), change_sources=frozenset({20, 30}))
        for source_id in (10, 20, 30):
            update = {"action_type": "ToolCall", "workflow_name": "self_management", "arguments": {
                "operation": "update", "run_id": "shelter", "change_source_id": source_id,
                "change_reason": "The partner changed the requested site.", "instructions": "Use the west bank."}}
            with self.subTest(source_id=source_id):
                if source_id == 30:
                    self.assertEqual(RouteAnswer.model_validate(update, context=check).arguments.run_id, "shelter")
                else:
                    with self.assertRaises(ValidationError):
                        RouteAnswer.model_validate(update, context=check)

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
