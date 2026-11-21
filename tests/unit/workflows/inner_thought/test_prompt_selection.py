import unittest

from pm.context_render.prompt import RenderSettings
from pm.llm.multimodal import encode_messages
from workflows.inner_thought.prompt_selection import PrivateEvidenceBlock, PrivatePromptInput, select_private_evidence


class PrivatePromptSelectionTests(unittest.TestCase):
    def test_required_image_reference_without_optional_evidence(self) -> None:
        prompt = PrivatePromptInput("private thought", "system", "Origin [image:img-origin]", ())
        settings = RenderSettings(context_window=32000, input_fraction=0.8, reserved_output_tokens=1024)
        rendered = select_private_evidence(prompt, settings)
        messages = encode_messages(rendered.turns)["messages"]
        self.assertEqual(messages[1]["content"], "Origin [image: unavailable]")

    def test_private_image_references_are_resolved_before_transport(self) -> None:
        recalled = PrivateEvidenceBlock("Recalled evidence", "Observation [image:img-recall]", 10, (42,))
        prompt = PrivatePromptInput("private inquiry operation", "system", "Origin [image:img-origin]", (recalled,))
        settings = RenderSettings(context_window=32000, input_fraction=0.8, reserved_output_tokens=1024)
        rendered = select_private_evidence(prompt, settings)
        messages = encode_messages(rendered.turns)["messages"]
        self.assertEqual(messages[1]["content"], "Origin [image: unavailable]\n\nRecalled evidence:\nObservation [image: unavailable]")
        self.assertEqual(rendered.evidence[0].record_id, 42)
        self.assertTrue(rendered.evidence[0].included)
