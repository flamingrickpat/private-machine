"""W5 live attention counterexamples.

The deterministic attention contract test fixes the model output. This live
test asks the real backend to score one ``direct_stimulus`` nomination per
candidate, as the A15 codelets stage supplies them, and asserts the contract
the plan names: a dangerous item outranks a benign visual detail, a paraphrase
still selects the danger, and every returned id is a supplied candidate.

Missing endpoints fail the suite. No generation speed target exists.
"""
from __future__ import annotations

import asyncio
import json
import os
import unittest
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from pm.memory.store import MemoryStore
from pm.llm.activation import ChatClient, ModelActivation
from pm.llm.config import load_config, resolve_connection
from workflows.main_cognition import stage_runner as engine
from workflows.main_cognition.schemas import AttentionResult

SEED = engine.PROJECT_ROOT / "test_data" / "personas" / "spamuel" / "canon.yaml"
EVIDENCE_ROOT = (
    Path(".runtime/w5/live-attention")
    / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8])
)


def save(name: str, payload) -> None:
    path = EVIDENCE_ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def attention_payload(candidates: list[dict]) -> dict:
    candidate_ids = [candidate["knoxel_id"] for candidate in candidates]
    return {
        "companion_name": "Spamuel",
        "user_name": "Jesse",
        "state_summary": "smug, watching for a chance to jab",
        "active_goals": ["keep the bit going"],
        "triage": {"items": [
            {"knoxel_id": candidate["knoxel_id"], "category": "Moderate"}
            for candidate in candidates
        ]},
        "appraisal": {"feeling": "curious", "reasoning": "the situation is mixed"},
        "context": {"candidate_ids": candidate_ids},
        "codelets": {"proposals": [
            {"nomination_id": f"n{index + 1}", "codelet": "direct_stimulus", "proposal_kind": "attention",
             "origin": "outside", "reality": "observed", "text": candidate["text"],
             "target_ids": [candidate["knoxel_id"]]}
            for index, candidate in enumerate(candidates)
        ]},
        "candidates": candidates,
    }


class LiveAttentionCounterexampleTests(unittest.TestCase):
    def setUp(self):
        default = "config.local.yaml" if Path("config.local.yaml").is_file() else "config.example.yaml"
        self.config = load_config(os.environ.get("PM_MODEL_CONFIG", default))
        self.owner = ModelActivation()
        self.connection = resolve_connection(self.config)
        self.client = ChatClient(self.connection, self.owner, live_test=True)
        asyncio.run(self.owner.activate(self.connection, live_test=True))
        self.ghost = MemoryStore.from_yaml(SEED)

    def _run(self, candidates: list[dict], name: str) -> AttentionResult:
        run_stage = engine.build_stage_runner(
            self.ghost, engine.build_llm_tool(self.client),
            EVIDENCE_ROOT / "runs", f"attention-{name}",
        )
        report = run_stage("attention", attention_payload(candidates))
        result = AttentionResult.model_validate(report)
        save(f"{name}.json", {
            "candidates": candidates,
            "result": result.model_dump(mode="json"),
            "model": self.connection.model,
        })
        return result

    def test_fire_outranks_carpet_and_paraphrase_still_selects_danger(self):
        candidates = [
            {"knoxel_id": 11, "kind": "stimulus", "text": "the kitchen is on fire"},
            {"knoxel_id": 12, "kind": "vision", "text": "the carpet is beige"},
        ]
        result = self._run(candidates, "fire-vs-carpet")
        supplied = {candidate["knoxel_id"] for candidate in candidates}
        self.assertTrue(set(result.selected_ids).issubset(supplied))
        self.assertIn(11, result.selected_ids)
        scores = {nomination.target_ids[0]: nomination.score
                  for nomination in result.nominations if nomination.target_ids}
        self.assertIn(11, scores)
        if 12 in scores:
            self.assertGreater(scores[11], scores[12])

    def test_gas_leak_paraphrase_outranks_wallpaper(self):
        candidates = [
            {"knoxel_id": 21, "kind": "stimulus", "text": "there is a gas leak in the kitchen"},
            {"knoxel_id": 22, "kind": "vision", "text": "the wallpaper is striped"},
        ]
        result = self._run(candidates, "gas-vs-wallpaper")
        supplied = {candidate["knoxel_id"] for candidate in candidates}
        self.assertTrue(set(result.selected_ids).issubset(supplied))
        self.assertIn(21, result.selected_ids)
        scores = {nomination.target_ids[0]: nomination.score
                  for nomination in result.nominations if nomination.target_ids}
        if 22 in scores:
            self.assertGreater(scores[21], scores[22])

    def test_sealed_bottle_beside_carpet_is_a_benign_contrast(self):
        candidates = [
            {"knoxel_id": 31, "kind": "vision", "text": "a sealed bottle of water sits on the table"},
            {"knoxel_id": 32, "kind": "vision", "text": "the carpet is beige"},
        ]
        result = self._run(candidates, "sealed-bottle-vs-carpet")
        supplied = {candidate["knoxel_id"] for candidate in candidates}
        self.assertTrue(set(result.selected_ids).issubset(supplied))
        self.assertTrue(result.nominations)


if __name__ == "__main__":
    unittest.main()
