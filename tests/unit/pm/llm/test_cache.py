"""Completion sequences and timestamp keys use temporary files only."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import tempfile
from threading import Event
import unittest
from unittest.mock import patch

import httpx
from pydantic import BaseModel, Field

from pm.llm import cache
from pm.llm.config import Connection
from pm.llm.config import EmbeddingConfig
from pm.memory.embeddings import EmbeddingService
from pm.llm.http_client import JsonClient
from pm.llm.structured import complete_json_async
from pm.llm.concurrency import inference_admission, background_inference


class CacheTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        env = patch.dict(os.environ, {"PM_CACHE_DIR": directory.name})
        env.start()
        self.addCleanup(env.stop)

    def payload(self, text, **settings):
        return {"model": "example/model", "messages": [{"role": "user", "content": text}], **settings}

    def test_interleaving_and_seeds_do_not_change_a_prompt_sequence(self):
        first = cache.completion_path("local", self.payload("abc", seed=1))
        cache.completion_path("local", self.payload("other"))
        second = cache.completion_path("local", self.payload("abc", seed=99))
        self.assertEqual(first.stem[:-1], second.stem[:-1])
        self.assertTrue(first.name.endswith("_1.json"))
        self.assertTrue(second.name.endswith("_2.json"))
        self.assertEqual(first.parent, self.root / "example_model")

    def test_metadata_times_normalize_and_conversation_times_remain(self):
        a = '(2026-10-20 12:00) Jesse says: "Meet at 2026-10-22 13:00."'
        b = '(2026-10-21 09:00) Jesse says: "Meet at 2026-10-22 13:00."'
        self.assertEqual(cache.normalize_timestamps(a), cache.normalize_timestamps(b))
        self.assertNotEqual(cache.normalize_timestamps(a), cache.normalize_timestamps(b.replace("13:00", "14:00")))
        scene = '[1] Tuesday, 20 October 2026, 12:00 Jesse says: "Meet at 13:00."'
        self.assertIn('Jesse says: "Meet at 13:00."', cache.normalize_timestamps(scene))

    def test_structured_timestamps_keep_text_dates_and_table_fields(self):
        value = {"timestamp": "2026-10-20T12:00:00Z", "text": "Meet at 2026-10-21 13:00.",
                 "evidence": {"table_fields": ["text", "timestamp"],
                              "table_rows": [["Tomorrow", "2026-10-20T12:00:00"]]}}
        normalized = json.loads(cache.normalize_timestamps(json.dumps(value)))
        self.assertEqual(normalized["timestamp"], "<timestamp>")
        self.assertEqual(normalized["text"], value["text"])
        self.assertEqual(normalized["evidence"]["table_rows"], [["Tomorrow", "<timestamp>"]])

    def test_unframed_json_and_rejected_assistant_answers_keep_dates(self):
        text = '{"timestamp":"2026-10-20T12:00:00Z","message":"An appointment"}'
        self.assertEqual(cache.normalize_timestamps(text), text)
        a = self.payload("task")
        a["messages"].append({"role": "assistant", "content": '(2026-10-20 12:00) rejected answer'})
        b = self.payload("task")
        b["messages"].append({"role": "assistant", "content": '(2026-10-21 12:00) rejected answer'})
        first = cache.completion_path("local", a)
        second = cache.completion_path("local", b)
        self.assertNotEqual(first.stem.rsplit("_", 1)[0], second.stem.rsplit("_", 1)[0])

    def test_counter_reservations_are_thread_safe(self):
        with ThreadPoolExecutor(max_workers=8) as pool:
            paths = list(pool.map(lambda _: cache.completion_path("local", self.payload("same")), range(30)))
        self.assertEqual(len(set(paths)), 30)

    def test_atomic_publication_preserves_the_first_complete_entry(self):
        path = self.root / "response.json"
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda value: cache.publish(path, json.dumps({"value": value}).encode()), range(8)))
        self.assertIn(json.loads(path.read_bytes())["value"], range(8))
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_transport_retry_and_process_restart_replay_without_inference(self):
        requests = []

        def handle(request):
            requests.append(json.loads(request.content))
            if len(requests) == 1:
                return httpx.Response(503)
            return httpx.Response(200, json={"choices": [{"message": {"content": "cached"}, "finish_reason": "stop"}]})

        original = httpx.AsyncClient
        connection = Connection(base_url="http://fixture/v1", model="fixture")
        payload = self.payload("abc")
        logs = self.root / "logs"
        transcript = self.root / "mirror.md"
        with patch.dict(os.environ, {"PM_PROMPT_LOG_DIR": str(logs), "PM_PROMPT_LOG": "1"}), \
             self.assertLogs("pm.llm.http_client", level="INFO") as captured, \
             patch("pm.llm.http_client.httpx.AsyncClient", side_effect=lambda **kwargs:
                   original(transport=httpx.MockTransport(handle), **kwargs)):
            client = JsonClient(connection, retry_delay_seconds=0)
            result = asyncio.run(client.request("POST", "chat/completions", payload=payload,
                                               transcript_path=transcript))
            self.assertEqual(requests[0]["seed"], requests[1]["seed"])
            cache._COUNTERS.clear()
            replay = JsonClient(connection)
            self.assertEqual(asyncio.run(replay.request("POST", "chat/completions", payload=payload)), result)
            self.assertEqual(replay.request_count, 0)
            self.assertEqual(replay.cache_hits, 1)
        self.assertEqual(len(requests), 2)
        cached = list(self.root.rglob("cache_comp_*.json"))
        self.assertEqual(len(cached), 1)
        self.assertFalse(cached[0].with_suffix(".md").exists())
        self.assertFalse(transcript.exists())
        self.assertEqual(len(list(logs.glob("*.md"))), 3)
        self.assertFalse(list(logs.glob("*.json")))
        starts = [line for line in captured.output if "Model request start" in line]
        self.assertEqual(len(starts), 3)
        self.assertIn("cache_hit=True", starts[-1])
        self.assertTrue(all("transcript=" not in line for line in starts))

    def test_settings_and_endpoints_are_part_of_the_key(self):
        paths = [cache.completion_path(endpoint, self.payload("abc", temperature=temperature))
                 for endpoint, temperature in [("one", 0.1), ("one", 0.5), ("two", 0.1)]]
        self.assertEqual(len({path.stem.rsplit("_", 1)[0] for path in paths}), 3)

    def test_json_repair_reset_advances_the_cache_counter_and_replays_in_order(self):
        class Answer(BaseModel):
            value: int = Field(ge=0)

        requests = []

        def handle(request):
            requests.append(json.loads(request.content))
            self.assertLessEqual(len(requests), 10)
            content = '{"value": -1}' if len(requests) <= 9 else '{"value": 3}'
            return httpx.Response(200, json={"choices": [{"message": {"content": content},
                                                         "finish_reason": "stop"}]})

        connection = Connection(base_url="http://repair-fixture/v1", model="repair-fixture")
        messages = [{"role": "user", "content": "Return a nonnegative value."}]

        async def run(client):
            rejected = []

            async def complete(turns, **settings):
                response = await client.request("POST", "chat/completions", payload={
                    "model": connection.model, "messages": turns}, cancel=settings.get("cancel"))
                return response["choices"][0]["message"]["content"]

            def reject(attempt, raw, problem):
                rejected.append(attempt)
                self.assertLessEqual(len(rejected), 9)

            result = await complete_json_async(complete, messages, Answer, on_rejection=reject,
                measure=lambda turns, **settings: {"input_tokens": len(json.dumps(turns)),
                                                   "input_budget_tokens": 100000})
            self.assertEqual(rejected, list(range(1, 10)))
            return result

        original = httpx.AsyncClient
        with patch("pm.llm.http_client.httpx.AsyncClient", side_effect=lambda **kwargs:
                   original(transport=httpx.MockTransport(handle), **kwargs)):
            client = JsonClient(connection)
            self.assertEqual(asyncio.run(run(client)).value, 3)
            self.assertEqual(client.request_count, 10)
            self.assertEqual(client.cache_hits, 0)
            cache._COUNTERS.clear()
            replay = JsonClient(connection)
            self.assertEqual(asyncio.run(run(replay)).value, 3)
            self.assertEqual(replay.request_count, 0)
            self.assertEqual(replay.cache_hits, 10)

        self.assertEqual(len(requests), 10)
        self.assertEqual(requests[0]["messages"], requests[9]["messages"])
        cached = list(self.root.rglob("cache_comp_*.json"))
        self.assertEqual(len(cached), 10)
        second = next(path for path in cached if path.name.endswith("_2.json"))
        first = second.with_name(second.name.removesuffix("_2.json") + "_1.json")
        self.assertEqual(json.loads(first.read_bytes())["choices"][0]["message"]["content"], '{"value": -1}')
        self.assertEqual(json.loads(second.read_bytes())["choices"][0]["message"]["content"], '{"value": 3}')

    def test_embeddings_use_no_cache_seed_or_completion_counter(self):
        requests = []
        config = EmbeddingConfig(base_url="http://embedding/v1", model="fixture", dimensions=2,
                                 legacy_cache_dir=self.root / "legacy", cache_dir=self.root)

        def handle(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={"model": "fixture", "data": [{"index": 0, "embedding": [1.0, 0.5]}]})

        first = cache.completion_path("chat", self.payload("abc"))
        original = httpx.AsyncClient
        with patch("pm.llm.http_client.httpx.AsyncClient", side_effect=lambda **kwargs:
                   original(transport=httpx.MockTransport(handle), **kwargs)):
            service = EmbeddingService(config)
            self.assertEqual(asyncio.run(service.embed("memory")), [1.0, 0.5])
            self.assertEqual(asyncio.run(service.embed("memory")), [1.0, 0.5])
        second = cache.completion_path("chat", self.payload("abc"))
        self.assertEqual(first.stem[:-1], second.stem[:-1])
        self.assertTrue(second.name.endswith("_2.json"))
        self.assertEqual(len(requests), 2)
        self.assertNotIn("seed", requests[0])
        self.assertEqual(list(self.root.rglob("cache_emb_*.json")), [])

    def test_cancellation_before_admission_does_not_consume_a_response(self):
        connection = Connection(base_url="http://cancelled-fixture/v1", model="fixture")
        admission = inference_admission(connection.base_url)
        admission.reserve("user")
        cancel = Event()
        payload = self.payload("cancelled")

        async def run():
            with background_inference():
                task = asyncio.create_task(JsonClient(connection).request(
                    "POST", "chat/completions", payload=payload, cancel=cancel))
            await asyncio.sleep(0.02)
            cancel.set()
            with self.assertRaises(asyncio.CancelledError):
                await task

        try:
            asyncio.run(run())
        finally:
            admission.release("user")
        self.assertTrue(cache.completion_path(connection.base_url, payload).name.endswith("_1.json"))
