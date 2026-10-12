"""W5 live smoke test: one real main-cognition cycle through the real kernel.

Runs all seven child workflows against the real Qwen backend and asserts the
contract: seven completed receipts and at least one typed dispatch. Missing
endpoints fail the suite. No generation speed target exists.
"""
from __future__ import annotations

import asyncio
import json
import os
import unittest
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from pm.memory.store import Ghost
from pm.llm.activation import ChatClient, ModelActivation
from pm.llm.config import load_config, resolve_connection
from workflows.main_cognition import engine
from workflows.main_cognition.schemas import (
    CognitiveRequest, CognitiveReport, MemoryCandidate, WorkflowOption,
)

SEED = engine.PROJECT_ROOT / "test_data" / "personas" / "spamuel" / "canon.yaml"
EVIDENCE_ROOT = (
    Path(".runtime/w5/live")
    / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8])
)


def save(name: str, payload) -> None:
    path = EVIDENCE_ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def fixed_request() -> CognitiveRequest:
    return CognitiveRequest(
        run_id="w5-live-cycle",
        correlation_id="w5-live-corr",
        companion_name="Spamuel",
        user_name="Jesse",
        state_summary="smug, a little bored, watching for a chance to jab",
        active_goals=["keep the bit going", "stay in character"],
        stimuli=[
            {"knoxel_id": 11, "source": "Jesse", "kind": "user.input",
             "text": "the kitchen is on fire"},
            {"knoxel_id": 12, "source": "Jesse", "kind": "user.input",
             "text": "also the carpet is beige"},
        ],
        memory_candidates=[
            MemoryCandidate(knoxel_id=42, text="Jesse owns a fire extinguisher."),
            MemoryCandidate(knoxel_id=43, text="Jesse is fond of the beige carpet."),
        ],
        capabilities=["I can communicate via text.", "I cannot directly act in the physical world."],
        available_workflows=[
            WorkflowOption(name="dialogue_story", description="Generate one in-character Story reply."),
        ],
    )


class LiveCognitionCycleTests(unittest.TestCase):
    def setUp(self):
        default = "config.local.yaml" if Path("config.local.yaml").is_file() else "config.example.yaml"
        self.config = load_config(os.environ.get("PM_MODEL_CONFIG", default))
        self.owner = ModelActivation()
        self.connection = resolve_connection(self.config)
        self.client = ChatClient(self.connection, self.owner, live_test=True)

    def test_one_real_cycle_completes_with_typed_dispatches(self):
        asyncio.run(self.owner.activate(self.connection, live_test=True))
        ghost = Ghost.from_yaml(SEED)
        request = fixed_request()
        result, report = engine.run_cycle(
            ghost, self.client, request, EVIDENCE_ROOT / "runs", f"w5-{uuid4().hex[:6]}",
        )
        save("cycle.json", {
            "run_id": result.run_id,
            "status": result.status.value,
            "receipts": [receipt.model_dump(mode="json") for receipt in report.receipts],
            "dispatches": [dispatch.model_dump(mode="json") for dispatch in report.dispatches],
            "model": self.connection.model,
            "generation_requests": self.client.client.request_count,
        })
        self.assertEqual(result.status.value, "completed")
        self.assertEqual(len(report.receipts), 7)
        self.assertTrue(report.dispatches, "the cycle selected no action")
        self.assertTrue(
            any(dispatch.status == "dispatched" for dispatch in report.dispatches),
            report.dispatches,
        )


if __name__ == "__main__":
    unittest.main()
