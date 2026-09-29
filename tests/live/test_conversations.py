"""W7 ten-scenario conversation acceptance.

Ten scenarios, twenty exchanges each, all generation first, then a separate
evaluator phase over saved artifacts. Requires seven good trajectories out of
ten and no hard contract violation. Missing endpoints fail; nothing is skipped
silently.

Set ``PM_W7_SCENARIOS`` / ``PM_W7_EXCHANGES`` for a smaller smoke run.
``PM_W7_EVALUATE=0`` skips evaluation; it is never the acceptance default.
"""
from __future__ import annotations

import asyncio
import json
import os
import statistics
import subprocess
import unittest
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from runtime.cognitive_model import LazyStageModel
from runtime.model_activation import ChatClient, ModelActivation
from runtime.models import LlmPreset, load_config, resolve_connection
from tests.live import conversation as conv

EVIDENCE_ROOT = (
    Path(".runtime/w7/conversations")
    / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8])
)


def _save(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


class ConversationAcceptanceTests(unittest.TestCase):
    def setUp(self):
        default = "config.local.yaml" if Path("config.local.yaml").is_file() else "config.example.yaml"
        self.config = load_config(os.environ.get("PM_MODEL_CONFIG", default))
        self.owner = ModelActivation()
        self.connection = resolve_connection(self.config)
        self.stage_model = LazyStageModel(self.connection, self.owner, live_test=True)
        self.jesse = ChatClient(self.connection, self.owner, live_test=True)

    def test_ten_scenarios_twenty_exchanges(self):
        selected = os.environ.get("PM_W7_SCENARIOS")
        names = [name for name in selected.split(",") if name] if selected else None
        scenarios = [s for s in conv.SCENARIOS if names is None or s.name in names]
        exchanges = int(os.environ.get("PM_W7_EXCHANGES", "20"))
        evaluate = os.environ.get("PM_W7_EVALUATE", "1") != "0"

        results: list[conv.ScenarioResult] = []
        for scenario in scenarios:
            root = EVIDENCE_ROOT / f"scenario-{scenario.name}"
            result = conv.run_scenario(
                scenario,
                root=root,
                stage_model=self.stage_model,
                jesse_client=self.jesse,
                exchanges=exchanges,
            )
            results.append(result)
            _save(root / "exchanges.json", result.exchanges)
            _save(root / "transcript.json", result.transcript)
            _save(root / "violations.json", result.violations)
            print(f"scenario {scenario.name}: {len(result.exchanges)} exchanges, "
                  f"{len(result.violations)} violations")

        hard_violations: list[str] = []
        for result in results:
            for violation in result.violations:
                if "no committed reply" in violation:
                    # An explicitly rejected claim is a correctly enforced
                    # contract, not a crash. It is recorded but is not a hard
                    # violation.
                    continue
                hard_violations.append(f"{result.name}: {violation}")

        evaluations: list[dict] = []
        trajectory_scores: list[int] = []
        evaluator_model = ""
        rubric_version = ""
        shares_backend = False
        if evaluate:
            evaluator_connection = resolve_connection(self.config, preset=LlmPreset.UnitTestEvaluator)
            asyncio.run(self.owner.activate(evaluator_connection, live_test=True))
            evaluator = ChatClient(evaluator_connection, self.owner, live_test=True)
            prompt = conv.load_evaluator_prompt()
            evaluator_model = evaluator_connection.model
            rubric_version = str(prompt["version"])
            shares_backend = self.config.evaluator_shares_backend()
            for result in results:
                scenario = conv.scenario_by_name(result.name)
                transcript = conv.render_transcript(result.transcript)
                scores: list[int] = []
                scenario_evaluations: list[dict] = []
                for exchange in result.exchanges:
                    reply = exchange.get("companion", "")
                    try:
                        score = asyncio.run(conv.evaluate_reply(
                            evaluator, scenario=scenario, transcript=transcript,
                            reply=reply, prompt=prompt,
                        ))
                    except Exception as error:  # evaluator error, not a quality failure
                        score = {"error": f"{type(error).__name__}: {error}"}
                    scenario_evaluations.append({"index": exchange.get("index"), **score})
                    if isinstance(score.get("scale"), int):
                        scores.append(score["scale"])
                if scores:
                    trajectory_scores.append(int(statistics.median(scores)))
                else:
                    trajectory_scores.append(-1)
                evaluations.append({"scenario": result.name, "replies": scenario_evaluations})
                _save(EVIDENCE_ROOT / f"scenario-{result.name}" / "evaluations.json", scenario_evaluations)

        good = sum(1 for score in trajectory_scores if score == 1)
        summary = {
            "scenarios": [result.name for result in results],
            "exchanges_per_scenario": exchanges,
            "total_exchanges": sum(len(result.exchanges) for result in results),
            "hard_violations": hard_violations,
            "rejections": [
                violation for result in results for violation in result.violations
                if "no committed reply" in violation
            ],
            "evaluated": evaluate,
            "trajectory_scores": trajectory_scores,
            "good_trajectories": good,
            "required_good": conv.REQUIRED_GOOD,
            "passed": good >= conv.REQUIRED_GOOD and not hard_violations,
            "evaluator_model": evaluator_model,
            "rubric_version": rubric_version,
            "evaluator_shares_backend": shares_backend,
            "application_version": "0.9.0",
            "generation_model": self.connection.model,
            "git_revision": _git_revision(),
        }
        _save(EVIDENCE_ROOT / "evaluations.json", evaluations)
        _save(EVIDENCE_ROOT / "summary.json", summary)
        print(f"run root: {EVIDENCE_ROOT}")
        print(json.dumps(summary, indent=2))

        self.assertEqual(hard_violations, [], f"hard contract violations: {hard_violations}")
        if evaluate:
            self.assertGreaterEqual(
                good, conv.REQUIRED_GOOD,
                f"only {good}/{len(results)} good trajectories",
            )


if __name__ == "__main__":
    unittest.main()
