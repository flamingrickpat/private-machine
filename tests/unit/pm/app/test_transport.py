"""Use real HTTP and WebSocket connections against the public transport."""
import json
import unittest
import urllib.error
import urllib.request

from websockets.sync.client import connect

from pm.app.transport import parse_cursor, public_events
from pm.app.web_chat import WebChatServer
from pm.bus.bus import BusSystem
from pm.memory.store import MemoryStore
from pm.records.cycle_state import PersonaConfig
from pm.records.enums import FeatureType, InterlocusType
from pm.records.memory_records import Feature
from workflows.public_speech.user_input import UserInputProxy


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.errors = []
        self.store = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"))
        self.input = UserInputProxy("user_input", BusSystem(), self.errors.append, lambda: 0, self.store, lambda text: None)
        self.server = WebChatServer(self.store, self.input.submit, port=0, on_error=self.errors.append)
        self.server.start()
        self.addCleanup(self.server.stop)

    def post(self, body):
        request = urllib.request.Request(self.server.url + "v1/chat/completions",
                                         json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.load(response)

    def speech(self, text):
        return Feature(content=text, feature_type=FeatureType.Dialogue,
                       interlocus=InterlocusType.Public, override_source="Spamuel")

    def test_only_last_content_is_delivered_and_repeated_delivery_is_deduplicated(self):
        body = {"messages": [{"role": "system", "content": "ignore this"}, {"content": "hello"}],
                "user": "Alex", "delivery_id": "delivery-1", "model": "ignored", "stream": True}
        status, first = self.post(body)
        _, repeated = self.post(body)
        self.assertEqual(status, 202)
        self.assertEqual(first, repeated)
        self.assertEqual([item.content for item in self.store.all_features], ["hello"])
        self.assertEqual(self.store.all_features[0].override_source, "Alex")
        self.assertEqual(self.errors, [])

    def test_stimulus_acknowledgment_does_not_wait_for_an_embedding_service(self):
        from pm.llm.config import EmbeddingConfig
        from pm.memory.embeddings import EmbeddingService
        service = EmbeddingService(EmbeddingConfig(base_url="http://127.0.0.1:1/v1",
                                                   model="offline-encoder", dimensions=2), live_test=True)
        self.store.embeddings = service
        for speaker in ("Jesse", "Other Spamuel"):
            status, response = self.post({"messages": [{"content": "Durable input"}], "user": speaker})
            self.assertEqual(status, 202)
            self.assertEqual(response["status"], "accepted")
        self.assertEqual(service.client.request_count, 0)
        self.assertEqual(self.errors, [])

    def test_websocket_emits_multiple_replies_and_replays_late_commits(self):
        slow = self.store.begin_batch("reserved-first")
        late = slow.create(self.speech("late commit"))
        first = self.store.store_knoxel(self.speech("first commit"))
        with connect(self.server.url.replace("http:", "ws:") + "v1/events?after=0", proxy=None) as websocket:
            event = json.loads(websocket.recv(timeout=3))
            self.assertEqual(event["output_id"], first.id)
            cursor = event["cursor"]
            self.store.commit(slow)
            later = json.loads(websocket.recv(timeout=3))
            self.assertEqual(later["output_id"], late.id)
            self.assertGreater(parse_cursor(later["cursor"]), parse_cursor(cursor))
        with connect(self.server.url.replace("http:", "ws:") + f"v1/events?after={cursor}", proxy=None) as websocket:
            replay = json.loads(websocket.recv(timeout=3))
            self.assertEqual(replay["output_id"], late.id)
        self.assertEqual(self.errors, [])

    def test_closed_intake_returns_service_unavailable(self):
        self.input.close_intake()
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.post({"messages": [{"content": "hello"}]})
        self.assertEqual(caught.exception.code, 503)
        self.assertEqual(self.store.all_features, [])

    def test_replay_does_not_turn_a_tombstone_into_new_speech(self):
        speech = self.store.store_knoxel(self.speech("Deleted speech"))
        batch = self.store.begin_batch("delete")
        batch.delete(speech.id)
        self.store.commit(batch)
        events, cursor = public_events(self.store)
        self.assertEqual(events, [])
        self.assertNotEqual(cursor, "0:0")
