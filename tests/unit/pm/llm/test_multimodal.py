"""The prompt encoder: tags become image parts under the last-N rule, or alt text."""
from __future__ import annotations

import socket
import struct
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest import mock

from pm.utils.operating_system import OperatingSystem
from pm.constants import PROMPT_IMAGE_TOKENS, PROMPT_MAX_IMAGES
from pm.context_render.images import ImageCatalog
from pm.context_render.turns import Role, Turn
from pm.llm.activation import ChatClient, ModelActivation
from pm.llm.config import resolve_connection
from pm.llm.context_window import estimate_tokens
from pm.llm.http_client import ModelError
from pm.llm.multimodal import agent_images, encode_messages, prompt_image_count, resolve_turns
from pm.llm.prompt_encoding import OpenAIChatEncoder
from pm.llm.prompt_log import start_prompt_log
from pm.records.media import image_from_bytes


def png(seed: int) -> bytes:
    """Return a real 1x1 PNG whose pixel value depends on ``seed``."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes([0, seed]))) + chunk(b"IEND", b""))


class EncoderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.images = [image_from_bytes(png(index), alt_text=f"picture {index}", alt_text_source="model")
                       for index in range(10)]
        self.catalog = ImageCatalog(tuple(self.images))

    def test_ten_images_in_three_messages_give_the_last_eight_as_image_parts(self) -> None:
        tags = [image.tag for image in self.images]
        turns = [Turn(Role.System, "You are a careful reader."),
                 Turn(Role.User, "first " + " ".join(tags[:3])),
                 Turn(Role.User, "second " + " ".join(tags[3:7])),
                 Turn(Role.User, "third " + " ".join(tags[7:]))]
        messages = encode_messages(resolve_turns(turns, self.catalog))["messages"]
        self.assertEqual(messages[0]["content"], "You are a careful reader.")
        parts = [part for message in messages[1:] for part in message["content"]]
        urls = [part["image_url"]["url"] for part in parts if part["type"] == "image_url"]
        self.assertEqual(len(urls), PROMPT_MAX_IMAGES)
        self.assertEqual(urls, [image.data_url() for image in self.images[2:]])
        text = "".join(part["text"] for part in parts if part["type"] == "text")
        self.assertIn("[image: picture 0]", text)
        self.assertIn("[image: picture 1]", text)
        self.assertNotIn("[image: picture 2]", text)
        self.assertNotIn("[image:img-", text)
        self.assertNotIn(self.images[0].data_b64, text)

    def test_a_repeated_image_is_sent_once_at_its_first_tag(self) -> None:
        image = self.images[0]
        turns = [Turn(Role.User, f"story {image.tag}"), Turn(Role.Assistant, "ok"),
                 Turn(Role.User, f"new event {image.tag}"), Turn(Role.User, f"task {image.tag}")]
        messages = encode_messages(resolve_turns(turns, self.catalog))["messages"]
        self.assertEqual(messages[0]["content"][1]["image_url"]["url"], image.data_url())
        self.assertEqual(messages[2]["content"], "new event [image: the same image as shown earlier]")
        self.assertEqual(messages[3]["content"], "task [image: the same image as shown earlier]")
        # Repeats do not use up the limit: nine distinct images still give eight parts.
        many = [Turn(Role.User, " ".join(item.tag for item in self.images[:9])), Turn(Role.User, self.images[8].tag)]
        parts = [part for message in encode_messages(resolve_turns(many, self.catalog))["messages"] if isinstance(message["content"], list)
                 for part in message["content"] if part["type"] == "image_url"]
        self.assertEqual([part["image_url"]["url"] for part in parts], [item.data_url() for item in self.images[1:9]])

    def test_tagless_prompts_stay_byte_identical(self) -> None:
        turns = [Turn(Role.System, "sys"), Turn(Role.User, "hello"), Turn(Role.Assistant, "hi")]
        encoded = encode_messages(resolve_turns(turns, self.catalog))
        self.assertEqual(encoded, OpenAIChatEncoder().encode(turns))
        self.assertTrue(all(isinstance(message["content"], str) for message in encoded["messages"]))
        mixed = encode_messages(resolve_turns([Turn(Role.System, "plain"), Turn(Role.User, self.images[0].tag)], self.catalog))
        self.assertEqual(mixed["messages"][0]["content"], "plain")

    def test_missing_alt_text_and_unknown_ids_have_fallback_text(self) -> None:
        bare = image_from_bytes(png(99))
        catalog = ImageCatalog((bare,))
        messages = encode_messages(resolve_turns([Turn(Role.User, f"{bare.tag} and [image:img-gone]")], catalog, limit=0))["messages"]
        self.assertEqual(messages[0]["content"], "[image: no description yet] and [image: unavailable]")

    def test_an_image_in_an_assistant_prefill_moves_to_the_earlier_user_turn(self) -> None:
        image = self.images[0]
        turns = [Turn(Role.System, "sys"), Turn(Role.User, "context"),
                 Turn(Role.Assistant, f'Jesse says: "what is this?\n{image.tag}"\nSpamuel says: "')]
        messages = encode_messages(resolve_turns(turns, self.catalog))["messages"]
        self.assertEqual(messages[2]["content"], turns[2].content.text.replace(image.tag, image.alt_label()))
        parts = messages[1]["content"]
        self.assertEqual(parts[0], {"type": "text",
                                    "text": f"context\n{image.alt_label()} (shown later in this conversation):\n"})
        self.assertEqual(parts[1]["image_url"]["url"], image.data_url())
        self.assertEqual(len(parts), 2)
        alone = encode_messages(resolve_turns([Turn(Role.System, image.tag)], self.catalog))["messages"]
        self.assertEqual(alone[0]["content"], "[image: picture 0]")

    def test_token_estimate_adds_a_constant_per_image(self) -> None:
        prompt = resolve_turns([Turn(Role.User, "look " + " ".join(i.tag for i in self.images))], self.catalog)
        # Only the newest PROMPT_MAX_IMAGES become parts; the rest stay alt text.
        self.assertEqual(prompt_image_count(prompt), PROMPT_MAX_IMAGES)
        plain, _ = estimate_tokens("qwen", [Turn(Role.User, "look")])
        tagged, label = estimate_tokens("qwen", prompt)
        allowance = PROMPT_MAX_IMAGES * PROMPT_IMAGE_TOKENS
        self.assertGreaterEqual(tagged - plain, allowance)
        # The estimate applies its safety margin to text and allowance together.
        self.assertLess(tagged - plain, 2 * (allowance + 200))
        self.assertTrue(label.endswith(".images_by_constant"))


class AgentImageTests(unittest.TestCase):
    """A pm-coder session gets the pictures of its prompt as attachments.

    The session sends its prompt through its own model client, which does not
    resolve image tags. ``agent_images`` returns the bytes for the tags the
    text carries, so the session reads the record text and still sees pixels.
    """

    def setUp(self) -> None:
        self.images = [image_from_bytes(png(index), alt_text=f"picture {index}", alt_text_source="model")
                       for index in range(1, 5)]
        self.catalog = ImageCatalog(tuple(self.images))

    def test_tags_become_attachments_in_text_order(self) -> None:
        text = "\n".join(f'Jesse says: "look [{image.alt_text}]" {image.tag}' for image in self.images[:2])
        parts = agent_images(text, catalog=self.catalog)
        self.assertEqual([part[1] for part in parts], ["image/png", "image/png"])
        self.assertEqual(parts[0][0], self.images[0].data())
        self.assertEqual(parts[1][0], self.images[1].data())

    def test_only_the_newest_images_are_kept(self) -> None:
        """The rule matches a direct call: newest ids, counted from the end."""
        text = "\n".join(image.tag for image in self.images)
        parts = agent_images(text, catalog=self.catalog, limit=2)
        self.assertEqual(parts, [(self.images[2].data(), "image/png"), (self.images[3].data(), "image/png")])

    def test_prompt_without_a_resolvable_tag_gets_nothing(self) -> None:
        self.assertEqual(agent_images("Jesse says: hello", catalog=self.catalog), [])
        self.assertEqual(agent_images("[image: img-gone]", catalog=self.catalog), [])


class ChatClientHookTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_chat_client_sends_image_parts(self) -> None:
        image = image_from_bytes(png(7), alt_text="dot", alt_text_source="model")
        catalog = ImageCatalog((image,))
        with tempfile.TemporaryDirectory() as directory, socket.socket() as reserved:
            reserved.bind(("0.0.0.0", 0))
            connection = resolve_connection(base_url=f"http://127.0.0.1:{reserved.getsockname()[1]}/v1",
                                            model="qwen", api_key="")
            owner = ModelActivation()
            owner._ready[connection.base_url] = connection.model
            client = ChatClient(connection, owner, live_test=True)
            with mock.patch.dict(OperatingSystem.environment, {"PM_PROMPT_LOG_DIR": directory, "PM_PROMPT_LOG": "1"}), \
                 mock.patch("pm.llm.prompt_log.start_prompt_log", wraps=start_prompt_log) as log:
                with self.assertRaises(ModelError):
                    await client.complete(resolve_turns([Turn(Role.User, f"what is this?\n{image.tag}")], catalog))
            payload = log.call_args.kwargs["payload"]
            # The exact payload is retained beside the readable transcript.
            prompts = sorted(Path(directory).glob("*.md"))
            self.assertEqual(len(prompts), 1)
            self.assertTrue(prompts[0].with_suffix(".json").exists())
        parts = payload["messages"][0]["content"]
        self.assertEqual(parts[0], {"type": "text", "text": "what is this?\n"})
        self.assertEqual(parts[1]["image_url"]["url"], image.data_url())


if __name__ == "__main__":
    unittest.main()
