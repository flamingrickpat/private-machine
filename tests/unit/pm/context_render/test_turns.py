"""Check content preservation across the typed provider boundary."""
import base64
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pm.llm.activation import ChatResult, provider_generation
from pm.llm.config import Connection
from pm.llm.context_window import estimate_tokens
from pm.llm.prompt_encoding import OpenAIChatEncoder
from pm.context_render.turns import Content, Media, MediaKind, Role, Text, ToolCall, Turn, VideoFrame, VideoFrames, format_content


class TurnTests(unittest.TestCase):
    def test_template_preserves_caption_media_and_question_order(self):
        image = Media(MediaKind.Image, "image/png", data=b"original-image")
        body = format_content("Before {record} after", record=Content("caption ") + Content(image))
        with self.assertRaises(TypeError):
            str(body)
        message = OpenAIChatEncoder().encode([Turn(Role.User, body)])["messages"][0]
        parts = message["content"]
        self.assertEqual(parts[0]["text"], "Before caption ")
        self.assertEqual(parts[2]["text"], " after")
        self.assertEqual(base64.b64decode(parts[1]["image_url"]["url"].split(",", 1)[1]), image.data)

    def test_file_capture_does_not_reread_changed_bytes(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "frame.png"
            path.write_bytes(b"first")
            image = Media.from_file(MediaKind.Image, path, "image/png")
            path.write_bytes(b"second")
            self.assertEqual(image.data, b"first")

    def test_audio_and_frame_offsets_survive_encoding(self):
        audio = Media(MediaKind.Audio, "audio/wav", data=b"sound")
        image = Media(MediaKind.Image, "image/png", data=b"frame")
        video = VideoFrames((VideoFrame(0, image), VideoFrame(2.5, image)))
        parts = OpenAIChatEncoder().encode([Turn(Role.User, Content(audio) + Content(video))])["messages"][0]["content"]
        self.assertEqual(base64.b64decode(parts[0]["input_audio"]["data"]), b"sound")
        self.assertIn("2.5 seconds", parts[3]["text"])
        with self.assertRaises(ValueError):
            OpenAIChatEncoder().encode([Turn(Role.System, Content(image))])

    def test_reasoning_stays_separate_and_signed_tool_state_can_replay(self):
        raw = {"model": "test", "usage": {"prompt_tokens": 10}, "choices": [{"finish_reason": "tool_calls",
            "message": {"content": "final", "reasoning_content": "private", "reasoning_details": [{"signature": "opaque"}],
                        "tool_calls": [{"id": "call-1", "type": "function",
                                        "function": {"name": "read", "arguments": '{"id":4}'}}]}}]}
        result = ChatResult.model_validate(raw).completion()
        self.assertEqual(result.text, "final")
        self.assertEqual(result.reasoning.text, "private")
        self.assertEqual(result.turn.tool_calls[0], ToolCall("call-1", "read", '{"id":4}'))
        replay = OpenAIChatEncoder().encode([result.turn, Turn(Role.Tool, "observed", tool_call_id="call-1")])["messages"]
        self.assertEqual(replay[0]["reasoning_details"], [{"signature": "opaque"}])
        self.assertEqual(replay[1]["tool_call_id"], "call-1")
        self.assertNotIn("private", replay[0]["content"])

    def test_thinking_option_remains_per_call(self):
        connection = Connection(base_url="https://openrouter.ai/api/v1", model="test")
        self.assertTrue(provider_generation(connection, {"enable_thinking": True})["reasoning"]["enabled"])
        self.assertFalse(provider_generation(connection, {"enable_thinking": False})["reasoning"]["enabled"])

    def test_media_estimate_is_labelled_and_does_not_count_base64_as_text(self):
        small = Turn(Role.User, Content("image") + Content(Media(MediaKind.Image, "image/png", data=b"a")))
        large = Turn(Role.User, Content("image") + Content(Media(MediaKind.Image, "image/png", data=b"a" * 100000)))
        count, label = estimate_tokens("test", [small])
        self.assertEqual(count, estimate_tokens("test", [large])[0])
        self.assertIn("images_by_constant", label)

