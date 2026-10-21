"""Serve the web chat over real HTTP from a real MemoryStore.

The submit callable is a small stub that commits the message like the
dialogue input does. No model, bus, or application runs.
"""
from __future__ import annotations

import base64
import json
import shutil
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from pm.app.character_home import CharacterHome
from pm.app.web_chat import STATIC_ROOT, WebChatServer, read_messages
from pm.paths import REPO_ROOT
from pm.records.enums import FeatureType, InterlocusType
from pm.records.media import attach_images, record_media, with_tags
from pm.records.memory_records import new_feature

PERSONA = REPO_ROOT / "test_data" / "personas" / "cheerful" / "config.yaml"
#: A real 1x1 PNG.
PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


def make_home(root: Path) -> CharacterHome:
    """Create a small character home with the cheerful test persona and an empty checkpoint."""
    root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PERSONA, root / "character.yaml")
    home = CharacterHome(root)
    home.create_store()
    return home


class WebChatServerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.store = make_home(self.tmp / "home").open_store()
        self.calls: list[tuple[str, str]] = []
        self.errors: list[BaseException] = []
        self.server = WebChatServer(self.store, self.submit, port=0, on_error=self.errors.append)
        self.server.start()
        self.addCleanup(self.server.stop)

    def submit(self, text: str, *, speaker: str, images=()) -> str:
        """Commit the message as a pending public exchange, like the dialogue input."""
        self.calls.append((speaker, text))
        metadata = {"exchange_correlation": f"c{len(self.calls)}", "exchange_status": "pending"}
        text = with_tags(text, attach_images(metadata, images)) if images else text
        new_feature(self.store, text, FeatureType.Dialogue, source=speaker, interlocus=InterlocusType.Public,
                    metadata=metadata)
        return f"c{len(self.calls)}"

    def reply(self, text: str) -> int:
        return new_feature(self.store, text, FeatureType.Dialogue, source=self.store.get_companion_name(),
                           interlocus=InterlocusType.Public).id

    def request(self, path: str, body: dict | None = None, headers: dict | None = None):
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(self.server.url.rstrip("/") + path, data=data,
                                         headers={"Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, response.headers, response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.headers, exc.read()

    def get_json(self, path: str):
        status, _, body = self.request(path)
        self.assertEqual(status, 200)
        return json.loads(body)

    def test_login_name_becomes_the_speaker(self):
        status, _, body = self.request("/api/messages", {"name": "  Jesse ", "text": "hi"})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["correlation_id"], "c1")
        self.request("/api/messages", {"name": "Alex", "text": "hello too"})
        self.assertEqual(self.calls, [("Jesse", "hi"), ("Alex", "hello too")])
        messages = self.get_json("/api/messages")["messages"]
        self.assertEqual([(m["speaker"], m["text"], m["from_companion"]) for m in messages],
                         [("Jesse", "hi", False), ("Alex", "hello too", False)])

    def test_the_companion_name_and_empty_text_are_rejected(self):
        status, _, _ = self.request("/api/messages", {"name": "Sunny", "text": "I am you"})
        self.assertEqual(status, 400)
        status, _, _ = self.request("/api/messages", {"name": "Jesse", "text": "   "})
        self.assertEqual(status, 400)
        status, _, _ = self.request("/api/messages", {"name": "Jesse", "text": "x"},
                                    headers={"Origin": "http://evil.example"})
        self.assertEqual(status, 403)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.errors, [])

    def test_history_is_in_time_order_and_after_returns_only_newer_lines(self):
        self.submit("first", speaker="Jesse")
        first_reply = self.reply("answer one")
        self.submit("second", speaker="Jesse")
        page = self.get_json("/api/messages")
        self.assertEqual([m["text"] for m in page["messages"]], ["first", "answer one", "second"])
        self.assertEqual(page["last_id"], page["messages"][-1]["id"])
        later = self.get_json(f"/api/messages?after={first_reply}")
        self.assertEqual([m["text"] for m in later["messages"]], ["second"])
        empty = self.get_json(f"/api/messages?after={page['last_id']}")
        self.assertEqual((empty["messages"], empty["last_id"]), ([], page["last_id"]))
        self.assertEqual(self.request("/api/messages?after=x")[0], 400)

    def test_thinking_lasts_until_a_companion_line_follows(self):
        self.assertFalse(self.get_json("/api/messages")["thinking"])
        self.request("/api/messages", {"name": "Jesse", "text": "are you there?"})
        page = self.get_json("/api/messages")
        self.assertTrue(page["thinking"])
        self.assertEqual(page["messages"][-1]["exchange_status"], "pending")
        self.reply("yes!")
        page = self.get_json("/api/messages")
        self.assertFalse(page["thinking"])
        self.assertTrue(page["messages"][-1]["from_companion"])
        self.assertEqual(page["messages"][-1]["speaker"], "Sunny")

    def test_markup_stays_text(self):
        hostile = '<img src=x onerror="alert(1)"><script>alert(2)</script> & done'
        self.request("/api/messages", {"name": "<b>Mallory</b>", "text": hostile})
        status, headers, body = self.request("/api/messages")
        self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
        message = json.loads(body)["messages"][0]
        self.assertEqual((message["speaker"], message["text"]), ("<b>Mallory</b>", hostile))
        # The page writes message text only through textContent.
        script = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
        self.assertNotIn("innerHTML", script)
        self.assertNotIn("insertAdjacentHTML", script)
        status, headers, page = self.request("/")
        self.assertEqual(status, 200)
        self.assertIn("script-src 'self'", headers["Content-Security-Policy"])
        self.assertIn(b'<script src="/app.js"', page)

    def test_an_image_message_is_stored_served_and_listed_without_base64(self):
        encoded = base64.b64encode(PNG).decode("ascii")
        status, _, _ = self.request("/api/messages", {"name": "Jesse", "text": "what is this?",
                                                      "images": [{"media_type": "image/png", "data_b64": encoded}]})
        self.assertEqual(status, 200)
        status, _, raw = self.request("/api/messages")
        self.assertNotIn(encoded.encode("ascii"), raw)
        message = json.loads(raw)["messages"][0]
        self.assertEqual(message["text"], "what is this?")
        self.assertEqual(len(message["images"]), 1)
        record = self.store.get_knoxel_by_id(message["id"])
        image = record_media(record).images[0]
        self.assertEqual(record.content, "what is this?\n" + image.tag)
        self.assertEqual((image.id, image.media_type, image.provenance.source),
                         (message["images"][0], "image/png", "user_upload"))
        self.assertEqual((image.width, image.height), (1, 1))
        status, headers, body = self.request(f"/api/media/{message['id']}/{image.id}")
        self.assertEqual((status, headers["Content-Type"], body), (200, "image/png", PNG))
        self.assertEqual(self.request(f"/api/media/{message['id']}/img-unknown")[0], 404)
        self.assertEqual(self.request(f"/api/media/x/{image.id}")[0], 404)

    def test_an_image_only_message_is_accepted_and_bad_images_are_rejected(self):
        encoded = base64.b64encode(PNG).decode("ascii")
        status, _, _ = self.request("/api/messages", {"name": "Jesse",
                                                      "images": [{"media_type": "image/png", "data_b64": encoded}]})
        self.assertEqual(status, 200)
        self.assertEqual(self.calls[-1][1], "")
        junk = base64.b64encode(b"not an image at all").decode("ascii")
        status, _, body = self.request("/api/messages", {"name": "Jesse", "text": "x",
                                                         "images": [{"media_type": "image/png", "data_b64": junk}]})
        self.assertEqual(status, 400)
        self.assertIn(b"Invalid image", body)
        status, _, _ = self.request("/api/messages", {"name": "Jesse", "text": "x",
                                                      "images": [{"media_type": "image/jpeg", "data_b64": encoded}]})
        self.assertEqual(status, 400)
        status, _, _ = self.request("/api/messages", {"name": "Jesse", "text": "see [image:img-0000]"})
        self.assertEqual(status, 400)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.errors, [])

    def test_session_names_the_companion_and_the_user(self):
        session = self.get_json("/api/session")
        self.assertEqual((session["companion_name"], session["user_name"]), ("Sunny", "Jesse"))
        self.assertGreater(session["poll_ms"], 0)

    def test_read_messages_matches_the_store_after_a_restart(self):
        self.submit("persist me", speaker="Jesse")
        self.reply("persisted")
        before = read_messages(self.store)
        reopened = CharacterHome(self.tmp / "home").open_store()
        self.assertEqual(read_messages(reopened), before)


if __name__ == "__main__":
    unittest.main()
