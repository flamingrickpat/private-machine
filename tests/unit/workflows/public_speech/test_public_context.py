"""Keep the full evidence of a captured Minecraft observation in review."""
import json
from pathlib import Path
import unittest

from workflows.public_speech.public_context import PublicContext, task_lines, ability_lines


class ToolEvidenceTests(unittest.TestCase):
    def test_control_receipt_keeps_the_original_request_and_real_artifacts(self):
        capture = json.loads(Path("test_data/captures/alpha/control-task-receipt.json").read_text(encoding="utf-8"))
        text = "\n".join(task_lines(PublicContext(companion_name="Spamuel", user_name="Jesse",
                                                  task_receipt=capture["task_receipt"])))
        self.assertIn("worker state completed, requested effect not verified", text)
        self.assertIn("scratch/observations.md", text)
        self.assertIn("Do not start note_task for this request", text)

    def test_observation_facts_after_the_camera_reach_review(self):
        capture = json.loads(Path("test_data/captures/alpha/public-observation-review.json").read_text(encoding="utf-8"))
        text = "\n".join(task_lines(PublicContext(companion_name="Spamuel", user_name="Jesse"), capture["tool_receipts"]))
        for field in ("camera", "player", "hotbar", "inventory", "nearby_blocks", "nearby_entities"):
            self.assertIn(f'"{field}"', text)
        self.assertIn('"iron_pickaxe"', text)
        self.assertIn('"food": 20', text)

    def test_capability_text_names_the_current_connection(self):
        for status in ("connected", "disconnected", "unknown"):
            context = PublicContext(companion_name="Spamuel", user_name="Jesse",
                                    capabilities={"minecraft": {"status": status, "visual_interpretation": False}})
            text = "\n".join(ability_lines(context))
            self.assertIn(f"Minecraft connection right now: {status}", text)
            self.assertIn("Screenshot pixels have no interpretation support", text)
