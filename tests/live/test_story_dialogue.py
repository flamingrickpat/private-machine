"""W4 live acceptance for standalone Story dialogue.

Two checks:
  * one real accepted dialogue with the exact five message roles and prefix
  * ten independent quality trials from the same initial Ghost state, then a
    separate evaluator phase over the saved artifacts

Generation always finishes before the evaluator model is activated. Missing
endpoints fail the suite; nothing is skipped silently. No generation speed
target exists.
"""
from __future__ import annotations

import asyncio
import json
import os
import unittest
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from pm_core.ghost.service import Ghost
from runtime.model_activation import ChatClient, ModelActivation
from runtime.models import LlmPreset, load_config, resolve_connection
from workflows.dialogue_story import engine
from workflows.dialogue_story.context import plan_story_request
from workflows.dialogue_story.schemas import StoryReport
from workflows.dialogue_story.steps import build_story_messages, load_prompts, render_prior_dialogue

SEED = engine.PROJECT_ROOT / "data" / "spamuel_agentic_canon.yaml"
SITUATION = engine.WORKFLOW_DIR / "situation.yaml"
EVIDENCE_ROOT = (
    Path(".runtime/w4/live")
    / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8])
)
REQUIRED_GOOD = 7


def save(name: str, payload) -> Path:
    path = EVIDENCE_ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


class LiveStoryDialogueTests(unittest.TestCase):
    def setUp(self):
        default = "config.local.yaml" if Path("config.local.yaml").is_file() else "config.example.yaml"
        self.config = load_config(os.environ.get("PM_MODEL_CONFIG", default))
        self.owner = ModelActivation()
        self.default = resolve_connection(self.config)
        self.client = ChatClient(self.default, self.owner, live_test=True)
        self.situation = engine.read_yaml(SITUATION)
        self.prompts = load_prompts()

    def _plan(self, ghost):
        from pm_core.embeddings import EmbeddingService
        embedding = EmbeddingService(self.config.embedding, live_test=True)
        return asyncio.run(plan_story_request(ghost, embedding, self.situation))

    def test_real_accepted_dialogue_preserves_roles_and_prefix(self):
        asyncio.run(self.owner.activate(self.default, live_test=True))
        ghost = Ghost.from_yaml(SEED)
        request = self._plan(ghost)
        result, report = engine.generate_story(
            ghost, engine.make_environment(ghost), self.client,
            request.model_dump(mode="json"), EVIDENCE_ROOT / "runs", f"w4-live-{uuid4().hex[:6]}",
        )
        self.assertEqual(result.status.value, "completed")
        self.assertEqual(
            report.message_roles,
            ["system", "user", "assistant", "user", "assistant"],
        )
        self.assertEqual(report.reply_prefix, 'Spamuel says: "')
        self.assertTrue(report.raw_completion.strip())
        self.assertTrue(report.candidate.strip())
        self.assertFalse(report.raw_completion.lstrip().casefold().startswith("jesse says:"))
        save("accepted-dialogue.json", {
            "run_id": result.run_id,
            "status": result.status.value,
            "message_roles": report.message_roles,
            "reply_prefix": report.reply_prefix,
            "raw_completion": report.raw_completion,
            "candidate": report.candidate,
            "accepted": report.accepted,
            "ontology": report.ontology.model_dump(mode="json"),
            "model": self.default.model,
            "generation_requests": self.client.client.request_count,
        })
        self.assertTrue(report.accepted, report.ontology)

    def test_ten_independent_quality_trials_pass_the_gate(self):
        asyncio.run(self.owner.activate(self.default, live_test=True))
        ghost = Ghost.from_yaml(SEED)
        request = self._plan(ghost)
        environment = engine.make_environment(ghost)

        reports: list[StoryReport] = []
        for trial in range(10):
            run_id = f"w4-quality-{trial:02d}-{uuid4().hex[:6]}"
            _, report = engine.generate_story(
                ghost, environment, self.client, request.model_dump(mode="json"),
                EVIDENCE_ROOT / "runs", run_id,
            )
            reports.append(report)
            save(f"trials/trial-{trial:02d}.json", {
                "trial": trial, "run_id": run_id, "accepted": report.accepted,
                "candidate": report.candidate, "ontology": report.ontology.model_dump(mode="json"),
            })

        # Generation is finished. Only now activate the evaluator.
        evaluator_connection = resolve_connection(self.config, preset=LlmPreset.UnitTestEvaluator)
        asyncio.run(self.owner.activate(evaluator_connection, live_test=True))
        evaluator = ChatClient(evaluator_connection, self.owner, live_test=True)
        evaluator_prompt = engine.read_yaml(engine.WORKFLOW_DIR / "evaluator.yaml")

        evaluations = []
        prior = render_prior_dialogue(request.prior_dialogue)
        memories = "\n".join(request.memories)
        for trial, report in enumerate(reports):
            score = asyncio.run(engine.evaluate_candidate(
                evaluator,
                companion_name=request.companion_name,
                current_user_message=request.current_user_message,
                candidate=report.candidate,
                evaluator_prompt=evaluator_prompt,
                prior_dialogue=prior,
                memories=memories,
            ))
            entry = {"trial": trial, "accepted": report.accepted, **score.model_dump()}
            evaluations.append(entry)
            save(f"evaluations/trial-{trial:02d}.json", entry)

        good = [entry for entry in evaluations if entry["scale"] == 1]
        summary = {
            "trials": len(reports),
            "accepted": sum(1 for report in reports if report.accepted),
            "good": len(good),
            "required_good": REQUIRED_GOOD,
            "passed": len(good) >= REQUIRED_GOOD,
            "evaluator_model": evaluator_connection.model,
            "evaluator_shares_backend": self.config.evaluator_shares_backend(),
            "rubric_version": evaluator_prompt["version"],
        }
        save("quality-summary.json", summary)
        save("quality-evaluations.json", evaluations)
        self.assertGreaterEqual(len(good), REQUIRED_GOOD, summary)


if __name__ == "__main__":
    unittest.main()
