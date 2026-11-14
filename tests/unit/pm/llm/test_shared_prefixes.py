"""Check the small prompt reorders without model inference."""
import unittest

from pm.memory.reconsolidation.schemas import load_prompts


class SharedPrefixTests(unittest.TestCase):
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
