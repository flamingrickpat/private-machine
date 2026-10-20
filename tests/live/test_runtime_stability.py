"""Exercise reply preparation and JSON repair against the local LLM.

Run this module explicitly. The default endpoint is port 8080 with alias qwen.
Missing services fail. Every completion bypasses the response cache.
Receipts and full prompts stay under a fresh ignored runtime directory.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

from pm.bus.bus import BusSystem
from pm.bus.envelopes import new_envelope
from pm.context_render.turns import Role, Turn
from pm.llm.activation import ModelActivation
from pm.llm.config import Connection
from pm.llm.stage_model import LazyStageModel
from pm.llm.structured import complete_json_async
from pm.memory.bash_view import MemoryBashEnvironment, MemoryBashTarget
from pm.records.enums import FeatureType
from pm.records.memory_records import Feature, new_feature
from pm.records.workspace import SELF_SHARD, ShardKey, WorkspaceShard
from tests.support import Recorder, fresh_ghost, load_seed_state
from workflows.dialogue_story.host import StoryDialogue, build_story_tools
from workflows.inner_thought.schemas import ThoughtResult
from workflows.main_cognition.host import build_main_cognition_tools
from workflows.main_cognition.model_json import complete_json
from workflows.main_cognition.schemas import ActionRequest, RequestShard, Stimulus
from workflows.public_speech.publication import exchange_record
from workflows.public_speech.public_context import build_public_context
from workflows.public_speech.speech_review import review_speech, select_spoken_text
from workflows.public_speech.user_input import UserInputProxy

ROOT = Path(__file__).resolve().parents[2]


class LiveRuntimeStabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8]
        cls.out = Path(os.environ.get("PM_STABILITY_EVIDENCE", ROOT / ".runtime/stability" / stamp))
        cls.out.mkdir(parents=True, exist_ok=False)
        cls.enterClassContext(patch.dict(os.environ, {
            "PM_COMPLETION_CACHE": "0", "PM_PROMPT_LOG": "1",
            "PM_PROMPT_LOG_DIR": str(cls.out / "prompts"),
        }))
        cls.connection = Connection(
            base_url=os.environ.get("PM_STABILITY_LLM_URL", "http://127.0.0.1:8080/v1"),
            model=os.environ.get("PM_STABILITY_LLM_MODEL", "qwen"),
        )
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        (cls.out / "run.json").write_text(json.dumps({
            "revision": revision, "base_url": cls.connection.base_url, "model": cls.connection.model,
            "completion_cache": False, "scope": "Engineering checks. No human quality rating.",
        }, indent=2), encoding="utf-8")
        (cls.out / "source.patch").write_bytes(subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=ROOT))
        (cls.out / "test_runtime_stability.py").write_bytes(Path(__file__).read_bytes())
        print(f"Runtime stability evidence: {cls.out}", flush=True)

    def setUp(self):
        self.model = LazyStageModel(self.connection, ModelActivation(), live_test=True)
        self.case = self.out / self._testMethodName
        self.case.mkdir()
        self.addCleanup(self.record_transport)

    def save(self, name: str, payload) -> None:
        """Save one diagnostic payload without changing accepted state."""
        (self.case / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def record_transport(self) -> None:
        """Keep request counts after success or failure, including activation failure."""
        client = self.model._client
        self.save("transport.json", {
            "requests": self.model.request_count,
            "cache_hits": client.client.cache_hits if client is not None else 0,
        })

    def assert_real_inference(self, minimum: int = 1) -> None:
        self.assertGreaterEqual(self.model.request_count, minimum)
        self.assertEqual(self.model.client.client.cache_hits, 0)

    def test_captured_syntax_repair_uses_the_granted_real_model(self):
        capture = json.loads((ROOT / "test_data/captures/alpha/cognitive-format-recovery.json").read_text(encoding="utf-8"))
        raw = capture["malformed_thought"]["text"]
        environment = MemoryBashEnvironment({}, {}, "read")
        tools = build_main_cognition_tools(None, self.case, self.model.complete)(environment, None)
        context = SimpleNamespace(runtime=SimpleNamespace(environment=environment, tools=tools, stop_event=Event()))
        turns = [{"role": "system", "content":
                  "Correct JSON syntax only. Preserve every fact and evidence ID. Return one JSON object. "
                  "The schema is: " + json.dumps(ThoughtResult.model_json_schema())},
                 {"role": "user", "content": raw}]
        result = complete_json(context, turns, ThoughtResult, initial=raw, temperature=0, enable_thinking=False)
        rejected = json.loads(environment.read_text("scratch/model-output-rejection-01.json"))
        self.save("rejection.json", rejected)
        self.save("repaired.json", result.model_dump(mode="json"))
        self.assertEqual(rejected["raw"], raw)
        self.assertEqual(result.evidence_ids, [11323, 11374, 11387])
        self.assert_real_inference()

    def test_async_schema_repair_keeps_the_original_evidence_ids(self):
        initial = {"thought": "A quiet moment.", "evidence_ids": "17, 23"}
        messages = [{"role": "system", "content":
                     "Return one JSON object for a private thought. Preserve the thought and evidence IDs 17 and 23. "
                     "The schema is: " + json.dumps(ThoughtResult.model_json_schema())},
                    {"role": "user", "content": json.dumps(initial)}]
        complete = self.model.build_async_tool()
        result = asyncio.run(complete_json_async(
            complete, messages, ThoughtResult, initial=initial, temperature=0, enable_thinking=False,
            on_rejection=lambda attempt, raw, error: self.save(f"rejection-{attempt:02}.json", {"raw": raw, "error": error}),
        ))
        self.save("repaired.json", result.model_dump(mode="json"))
        self.assertEqual(result.evidence_ids, [17, 23])
        self.assertTrue(result.thought.strip())
        self.assert_real_inference()

    def test_normal_story_commits_and_delivers_a_real_reply(self):
        self.run_story(empty_history=False)

    def test_empty_history_still_commits_and_delivers_a_real_reply(self):
        self.run_story(empty_history=True)

    def test_empty_prefill_recovers_with_real_inference(self):
        self.run_story(empty_history=False, reproduce_empty_prefill=True)

    def test_invalid_review_span_is_repaired_by_the_real_model(self):
        self.run_review_span_repair(self.model.build_async_tool())

    def test_standalone_review_repair_keeps_its_live_measurement(self):
        from workflows.dialogue_story.standalone import build_story_tools
        self.run_review_span_repair(build_story_tools(self.model.client)["public_complete"])

    def run_review_span_repair(self, complete) -> None:
        """Repair one injected review fault through the supplied real adapter."""
        ghost = fresh_ghost(load_seed_state())
        context = build_public_context(ghost)
        candidate = "Your duck drawing deserves a frame."
        calls = []

        async def faulty_first_review(messages, **generation):
            calls.append(messages)
            if len(calls) == 1:
                messages = [Turn(Role.System, 'Return exactly this JSON object: '
                    '{"reason":"Protocol fault for a recovery test.","false_task_result":false,'
                    '"spoken_text":"These words are absent from the draft."}'), *messages[1:]]
            raw = await complete(messages, **generation)
            self.save(f"review-{len(calls):02}.json", {"raw": raw, "fault_injected": len(calls) == 1})
            return raw

        faulty_first_review.measure_context = complete.measure_context
        result = asyncio.run(review_speech(faulty_first_review, candidate, context))
        self.save("review.json", result.model_dump(mode="json"))
        self.assertGreaterEqual(len(calls), 2)
        self.assertIn(select_spoken_text(candidate, result), candidate)
        self.assert_real_inference(2)

    def run_story(self, *, empty_history: bool, reproduce_empty_prefill: bool = False) -> None:
        """Run the real Story host and kernel over a fresh full-canon store."""
        ghost = fresh_ghost(load_seed_state())
        bus = BusSystem()
        delivered = []
        proxy = UserInputProxy("user_input", bus, self.fail, lambda: 0, ghost, delivered.append)
        correlation = proxy.submit("I drew a duck drinking cold brew from a World's Okayest Employee mug. What do you think?")
        current = exchange_record(ghost, correlation)
        partner_id = ghost.get_entity_id(ghost.get_user_name())
        self.assertIsNotNone(partner_id)
        partner = ShardKey(kind="entity", ref=str(partner_id))
        history = sorted((record for record in ghost.all_knoxels.values()
                          if isinstance(record, Feature) and record.feature_type == FeatureType.Dialogue
                          and record.source in {ghost.get_user_name(), ghost.get_companion_name()}
                          and record.id != current.id), key=lambda record: (record.timestamp_world_begin, record.id))[-6:]
        self.assertTrue(history)
        thought = new_feature(ghost, "Jesse wants a playful response to the duck drawing.",
                              FeatureType.Thought, source=ghost.get_companion_name())
        dialogue_ids = [record.id for record in history]
        thought_ids = [thought.id]
        blanks = []
        if empty_history:
            for text, kind in (("", FeatureType.Dialogue), (" \t\n", FeatureType.Thought),
                               ("\u00a0\u2003", FeatureType.InnerMonologue)):
                record = new_feature(ghost, text, kind, source=ghost.get_companion_name())
                blanks.append(record)
                (dialogue_ids if kind == FeatureType.Dialogue else thought_ids).append(record.id)
        request = ActionRequest(content="Reply playfully to Jesse's duck drawing.", partner_name=ghost.get_user_name(),
            partner_entity_id=partner_id, group=partner,
            stimuli=[Stimulus(knoxel_id=current.id, text=current.content)],
            shards=[RequestShard(name="self", shard=WorkspaceShard(key=SELF_SHARD, tail_ids=thought_ids)),
                    RequestShard(name=ghost.get_user_name(), shard=WorkspaceShard(key=partner, tail_ids=dialogue_ids))])
        real_factory = build_story_tools(self.model)

        def factory(environment, batch):
            tools = real_factory(environment, batch)
            if reproduce_empty_prefill:
                complete = tools["story_complete"]
                calls = []

                async def faulty_prefill(messages, **generation):
                    calls.append(messages)
                    if len(calls) == 1:
                        # This old public instruction reproduced an empty Qwen
                        # continuation with Spamuel. Every attempt still uses inference.
                        old = ("Your task: write the next spoken line of the companion character, and only\n"
                               "that line. The transcript ends with an open quotation mark. Continue\n"
                               "directly after it with the words the character says aloud. End with a\n"
                               "closing quotation mark. Then stop. Do not write the other character's next\nline.")
                        current = ("Write the reply to the current user message.\n"
                                   "Continue the unfinished last line in your previous turn.\n"
                                   "Return only the newly written words.")
                        system = messages[0].content.text
                        self.assertIn(current, system)
                        messages = [Turn(Role.System, system.replace(current, old)), *messages[1:]]
                        self.save("fault.json", {"kind": "Old public prefill instruction; Spamuel fixture only."})
                    return await complete(messages, **generation)

                tools["story_complete"] = faulty_prefill
            return tools

        module = StoryDialogue("dialogue", bus, self.fail, lambda: 0, None,
            working_target=MemoryBashTarget(ghost, "read"),
            workflow=ROOT / "workflows/dialogue_story/dialogue_story.workflow.md",
            tools_factory=factory, runtime_root=self.case / "runs", ghost=ghost)
        bus.subscribe(module.module_id, module.accepts, module.deliver)
        bus.subscribe(proxy.module_id, proxy.accepts, proxy.deliver)
        replies = Recorder(bus, "dialogue.reply")
        event = new_envelope("live_stability", "data", "action.request", correlation,
                             request.model_dump(mode="json"), target="dialogue")
        before = {record.id: record.model_dump(mode="json") for record in blanks}
        self.save("request.json", request.model_dump(mode="json"))
        self.save("empty-records.json", before)
        bus.put(event)
        self.assertTrue(module.step(block=False))
        self.assertTrue(proxy.step(block=False))
        context = module._public_context
        self.save("context.json", context.model_dump(mode="json"))
        self.save("report.json", module.last_result.report)
        self.save("replies.json", [json.loads(event.to_json()) for event in replies.events])
        self.assertEqual(module.last_result.status.value, "completed")
        self.assertEqual(len(replies.events), 1)
        self.assertEqual(replies.events[0].correlation_id, correlation)
        self.assertEqual(len(delivered), 1)
        self.assertTrue(delivered[0].strip())
        self.assertFalse(delivered[0].startswith("*("), delivered[0])
        self.assertNotIn(f"{ghost.get_user_name()} says:", delivered[0])
        self.assertNotIn(f"{ghost.get_companion_name()} thinks:", delivered[0])
        self.assertEqual(delivered[0], module.last_result.report["candidate"])
        self.assertEqual(exchange_record(ghost, correlation).metadata["exchange_status"], "resolved")
        reply_id = exchange_record(ghost, correlation).metadata["public_reply_id"]
        accepted = ghost.get_knoxel_by_id(reply_id)
        self.save("accepted-reply.json", accepted.model_dump(mode="json"))
        self.assertEqual(accepted.content, delivered[0])
        if reproduce_empty_prefill:
            attempts = module.last_result.report["attempts"]
            self.assertEqual(len(attempts), 2)
            self.assertTrue(attempts[0]["generation_error"])
            self.assertEqual(attempts[-1]["message_roles"], ["system", "user", "assistant", "user"])
        self.assertEqual(set(context.plan["empty_transcript_ids"]), set(before))
        self.assertTrue(set(before) <= {item.knoxel_id for item in context.evidence})
        self.assertEqual({key: ghost.get_knoxel_by_id(key).model_dump(mode="json") for key in before}, before)
        self.assert_real_inference(2)


if __name__ == "__main__":
    unittest.main()
