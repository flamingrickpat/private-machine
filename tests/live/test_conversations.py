"""W7 ten-scenario conversation acceptance.

Ten scenarios, twenty exchanges each, all generation first, then a separate
evaluator phase over saved artifacts. Requires seven good trajectories out of
ten and no hard contract violation. Missing endpoints fail; nothing is skipped
silently.

Set ``PM_W7_SCENARIOS`` / ``PM_W7_EXCHANGES`` for a smaller smoke run.
``PM_W7_EVALUATE=0`` skips evaluation; it is never the acceptance default.
"""
from __future__ import annotations

import json
import os
import subprocess
import tomllib
import unittest
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from pm.llm.stage_model import LazyStageModel
from pm.llm.activation import ChatClient, ModelActivation
from pm.llm.config import load_config, resolve_connection
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
        self.assertTrue(scenarios, "No conversation scenarios were selected.")
        if names is not None:
            self.assertEqual(set(names), {s.name for s in scenarios}, "Unknown scenario name.")
        exchanges = int(os.environ.get("PM_W7_EXCHANGES", "20"))
        self.assertGreater(exchanges, 0)
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
        infrastructure_errors: list[str] = []
        for result in results:
            for violation in result.violations:
                if "no committed reply" in violation:
                    # An explicitly rejected claim is a correctly enforced
                    # contract, not a crash. It is recorded but is not a hard
                    # violation.
                    continue
                if "ModelError" in violation or "HTTP" in violation:
                    # A transient backend rejection after the bounded retries
                    # is an infrastructure error, reported separately from a
                    # companion contract violation.
                    infrastructure_errors.append(f"{result.name}: {violation}")
                    continue
                hard_violations.append(f"{result.name}: {violation}")

        generation_fields = {
            "scenarios": [result.name for result in results],
            "exchanges_per_scenario": exchanges,
            "total_exchanges": sum(len(result.exchanges) for result in results),
            "completed_exchanges": sum(
                bool(exchange.get("completed")) for result in results for exchange in result.exchanges
            ),
            "execution_path": "application",
            "hard_violations": hard_violations,
            "infrastructure_errors": infrastructure_errors,
            "rejections": [
                violation for result in results for violation in result.violations
                if "no committed reply" in violation
            ],
            "application_version": tomllib.loads(Path("pyproject.toml").read_text())["project"]["version"],
            "generation_model": self.connection.model,
            "git_revision": _git_revision(),
        }
        _save(EVIDENCE_ROOT / "summary.json", generation_fields)
        if evaluate:
            summary = conv.run_evaluation(
                EVIDENCE_ROOT,
                config=self.config,
                owner=self.owner,
                selected=[result.name for result in results],
            )
        else:
            summary = {**generation_fields, "evaluated": False}
            _save(EVIDENCE_ROOT / "summary.json", summary)
        print(f"run root: {EVIDENCE_ROOT}")
        print(json.dumps(summary, indent=2))

        self.assertEqual(hard_violations, [], f"hard contract violations: {hard_violations}")
        self.assertEqual(infrastructure_errors, [], f"incomplete exchanges: {infrastructure_errors}")
        self.assertEqual(generation_fields["completed_exchanges"], len(scenarios) * exchanges)
        if evaluate:
            self.assertTrue(summary["passed"], "Full acceptance requires all 200 application response cycles.")
            # Evaluator refusal or malformed output is an evaluation error, not
            # a poor companion score. It is reported separately. The gate runs
            # on the trajectories the evaluator could actually score.
            self.assertGreaterEqual(
                summary.get("evaluated_trajectories", 0), conv.REQUIRED_GOOD,
                f"only {summary.get('evaluated_trajectories')} trajectories were evaluable",
            )
            self.assertGreaterEqual(
                summary["good_trajectories"], conv.REQUIRED_GOOD,
                f"only {summary['good_trajectories']}/{summary.get('evaluated_trajectories')} good trajectories",
            )


if __name__ == "__main__":
    unittest.main()
