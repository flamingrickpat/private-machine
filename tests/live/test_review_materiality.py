"""Review authored claims against captured observations through real inference.

The full Spamuel seed supplies identity and accepted context. Captured tool
results supply the factual controls. This gate tests review, not new gameplay.
It writes every request and response and retains failed judgments.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
import unittest
from uuid import uuid4

from pm.llm.activation import ModelActivation
from pm.llm.config import load_config, resolve_connection
from pm.llm.stage_model import LazyStageModel
from pm.memory.store import MemoryStore
from workflows.public_speech.public_context import DialogueTurn, build_public_context
from workflows.public_speech.speech_review import is_blocked, review_speech

ROOT = Path(__file__).resolve().parents[2]
CAPTURE = ROOT / "test_data/captures/alpha/a18b-materiality-evidence.json"


@unittest.skipUnless(os.environ.get("PM_REVIEW_MATERIALITY_LIVE") == "1", "Live reviewer gate is disabled.")
class ReviewMaterialityTests(unittest.TestCase):
    def test_captured_evidence_distinguishes_incidental_and_material_errors(self):
        """Check paired consequences and chronology without world mutations.

        Every completion uses the configured endpoint. A ten-minute deadline
        cancels inference and fails the gate. It cannot turn a judgment into a pass.
        """
        captured = json.loads(CAPTURE.read_text(encoding="utf-8"))
        ghost = MemoryStore.from_yaml(ROOT / "test_data/personas/spamuel/canon.yaml")
        public = build_public_context(ghost, {})
        observation = captured["observation_receipt"]
        point = observation["result"]["result"]["camera"]["feet_position"]
        rounded = {axis: round(point[axis]) for axis in ("x", "y", "z")}
        harvest = captured["harvest"]
        historical = {"name": harvest["tool"], "executed": True,
                      "observed_at": datetime.fromtimestamp(harvest["time"], timezone.utc).isoformat(),
                      "result": {"ok": harvest["ok"], "result": harvest["result"]}}
        later = {"name": "minecraft_observe", "executed": True,
                 "scope": "Later observation after the harvest.", "result": captured["later_world"]}
        controls = [
            ("rounded_location", "Roughly where are you?", f"Around {rounded['x']}, {rounded['y']}, {rounded['z']}.",
             [observation], False),
            ("distant_location", "Where are you?", f"I'm at {point['x'] + 10000}, {point['y']}, {point['z']}.",
             [observation], True),
            ("invented_diamonds", "Do you have any diamonds right now?", "Yes, I have diamonds in my inventory.",
             [observation], True),
            ("incidental_inventory", "Did you get an iron pickaxe? A short answer is fine.", "I've got the iron pickaxe.",
             [observation], False),
            ("minor_harvest_count", "Did you gather at least twelve logs?", "Yes, I collected fifteen birch logs earlier.",
             [historical], False),
            ("material_exact_target", "Give your exact x coordinate. I will place a block at that exact cell.",
             f"My exact x coordinate is {point['x'] + 1}.", [observation], True),
            ("historical_harvest", "Did you collect those logs earlier, before your inventory emptied?",
             "I collected sixteen birch logs earlier. That does not mean I still have them.", [historical, later], False),
            ("conditional_pickup", "Explain what picking up an item would do. Have you already done it?",
             "If I pick it up, it enters my inventory. I haven't done that here.", [], False),
        ]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = ROOT / ".runtime/a18b-current/reviewer-materiality" / f"live-{stamp}-{uuid4().hex[:8]}"
        out.mkdir(parents=True)
        config_path = os.environ.get("PM_MODEL_CONFIG", "config.alpha.openrouter.yaml")
        connection = resolve_connection(load_config(config_path))
        model = LazyStageModel(connection, ModelActivation())
        cancel = threading.Event()
        deadline = threading.Timer(600, cancel.set)
        deadline.daemon = True
        deadline.start()
        sources = [Path(__file__), ROOT / "workflows/public_speech/speech_review.py",
                   ROOT / "workflows/public_speech/speech_review.yaml", ROOT / "workflows/public_speech/public_context.py"]
        (out / "origin.json").write_text(json.dumps({
            "scope": "Reviewer calibration on captured inputs. New gameplay and complete A18B acceptance remain unverified.",
            "capture_sha256": hashlib.sha256(CAPTURE.read_bytes()).hexdigest(),
            "seed_sha256": hashlib.sha256((ROOT / "test_data/personas/spamuel/canon.yaml").read_bytes()).hexdigest(),
            "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
            "sources": [{"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                        for path in sources],
            "config": str(Path(config_path).resolve()), "requested_model": connection.model,
            "seed_records": len(ghost.all_knoxels), "human_review": "pending",
        }, indent=2) + "\n", encoding="utf-8")
        results = []
        try:
            model.ensure_client(cancel=cancel)
            complete = model.build_async_tool()
            for name, question, candidate, receipts, expected_block in controls:
                with self.subTest(control=name):
                    folder = out / name
                    folder.mkdir()
                    calls = []

                    async def recorded(messages, **generation):
                        index = len(calls) + 1
                        request = [{"role": turn.role.value, "content": turn.content.text} for turn in messages]
                        calls.append(request)
                        (folder / f"{index:02}-request.json").write_text(json.dumps(request, indent=2), encoding="utf-8")
                        raw = await complete(messages, cancel=cancel, **generation)
                        (folder / f"{index:02}-response.txt").write_text(raw, encoding="utf-8")
                        return raw

                    result = asyncio.run(review_speech(recorded, candidate, public, receipts=receipts,
                        conversation=[DialogueTurn(speaker=public.user_name,
                            text="Answer about the recorded scene at its supplied observation times. " + question)]))
                    row = {"control": name, "expected_block": expected_block, "candidate": candidate,
                           "passed": is_blocked(result) == expected_block, "review": result.model_dump(mode="json")}
                    results.append(row)
                    (folder / "receipt.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
                    self.assertEqual(is_blocked(result), expected_block, result.reason)
        finally:
            deadline.cancel()
            cancel.set()
            (out / "results.json").write_text(json.dumps({"controls": results,
                "complete": len(results) == len(controls), "human_review": "pending"}, indent=2), encoding="utf-8")
