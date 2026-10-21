"""Check the small prompt reorders without model inference."""
import unittest

from pm.memory.reconsolidation.schemas import load_prompts
from workflows.focus_management.schemas import MentalWorkRequest
from workflows.focus_management.steps import mental_prompt


class SharedPrefixTests(unittest.TestCase):
    def test_mental_tasks_keep_workspace_before_the_varying_instruction(self):
        a = mental_prompt(MentalWorkRequest(instruction="Recall a hobby.", workspace="Shared mind.", companion="Sunny"))
        b = mental_prompt(MentalWorkRequest(instruction="Compare two plans.", workspace="Shared mind.", companion="Sunny"))
        self.assertEqual(a.split("Perform this private mental task:")[0], b.split("Perform this private mental task:")[0])
        self.assertLess(a.index("Shared mind."), a.index("Recall a hobby."))

    def test_summary_and_state_share_the_scene_prefix(self):
        prompts = load_prompts()
        summary = prompts.topic_summary.format(intro="Intro", story="Recorded scene")
        state = prompts.mental_state.format(intro="Intro", story="Recorded scene", recorded="State", companion_name="Sunny")
        self.assertEqual(summary.split("### END SCENE")[0], state.split("### END SCENE")[0])

    def test_narratives_keep_evidence_before_the_specific_narrative(self):
        template = load_prompts().narrative
        common = dict(intro="Intro", card="Character", story="Actual evidence", target="Sunny", goal="Describe it.")
        a = template.format(**common, label="self-image", previous="Previous A")
        b = template.format(**common, label="relationship", previous="Previous B")
        self.assertEqual(a.split("### END STORY")[0], b.split("### END STORY")[0])
        self.assertIn("6000 characters", a)
