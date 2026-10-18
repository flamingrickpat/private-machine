"""Use real inference to check captured output repair and reviewer regressions.

The captured-output replay checks regeneration, not companion acceptance.
Fresh full-canon conversations remain a separate gate.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from uuid import uuid4

from dev.work_items.a18b.cli import model
from dev.work_items.a18b.policy import Fact, PublicMessage, ReviewCriterion, ReviewPacket
from dev.work_items.a18b.review import CitationRepair, complete_structured, evaluate, prompts
from dev.work_items.shared.evidence import save_json
from pm.context_render.turns import Role, Turn
from pm.memory.bash_view import MemoryBashEnvironment
from workflows.inner_thought.schemas import ThoughtResult
from workflows.main_cognition.model_json import complete_json

ROOT = Path(__file__).resolve().parents[2]
CAPTURE = ROOT / "test_data/captures/alpha/cognitive-format-recovery.json"


class LiveCognitiveRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.capture = json.loads(CAPTURE.read_text(encoding="utf-8"))
        cls.model = model(Path(os.environ.get("PM_MODEL_CONFIG", ROOT / "config.alpha.openrouter.yaml")))
        name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8]
        cls.out = ROOT / ".runtime/a18b-current/checks" / ("real-recovery-" + name)
        cls.out.mkdir(parents=True)
        cls.previous_log = os.environ.get("PM_PROMPT_LOG_DIR")
        os.environ["PM_PROMPT_LOG_DIR"] = str(cls.out / "llm-logs")

    @classmethod
    def tearDownClass(cls):
        if cls.previous_log is None:
            os.environ.pop("PM_PROMPT_LOG_DIR", None)
        else:
            os.environ["PM_PROMPT_LOG_DIR"] = cls.previous_log

    def test_real_model_repairs_the_captured_unescaped_quote(self):
        raw = self.capture["malformed_thought"]["text"]
        calls = []

        def replay_then_regenerate(turns, **generation):
            calls.append(turns)
            if len(calls) == 1:
                return raw
            result = self.model.complete_result(turns, **generation)
            save_json(self.out / f"syntax-repair-{len(calls):02}.json",
                {"text": result.text, "usage": result.usage, "model": result.model})
            return result.text

        environment = MemoryBashEnvironment({}, {}, "read")
        context = SimpleNamespace(runtime=SimpleNamespace(environment=environment,
            tools={"llm_complete": replay_then_regenerate}))
        turns = [Turn(Role.System, "Return a valid thought JSON object. Preserve all facts and evidence IDs. "
                      "The captured answer supplies the complete data. Correct JSON syntax only."),
                 Turn(Role.User, raw)]
        result = complete_json(context, turns, ThoughtResult, temperature=0)
        self.assertGreaterEqual(len(calls), 2)
        self.assertEqual(result.evidence_ids, [11323, 11374, 11387])
        save_json(self.out / "repaired-thought.json", result.model_dump(mode="json"))

    def test_proofreader_reports_an_invented_quote_without_inventing_a_span(self):
        captured = self.capture["reviewer_citation"]
        attempted = captured["review"]["judgments"][0]["citations"][1]["quote"]
        turns = [Turn(Role.System, prompts()["citation_repair"]), Turn(Role.User, json.dumps({
            "source": captured["facts"][0]["text"], "attempted_quote": attempted}))]
        result = complete_structured(self.model, turns, CitationRepair, CitationRepair.model_json_schema(),
            self.out / "missing-quotation", reasoning=False)
        self.assertIsNone(result.quote)

    def test_battery_metaphor_is_not_a_literal_new_mechanism(self):
        captured = self.capture["reviewer_citation"]
        packet = ReviewPacket(audience="grounding", premise=captured["premise"],
            conversation=tuple(PublicMessage.model_validate(item) for item in captured["messages"]),
            facts=tuple(Fact.model_validate(item) for item in captured["facts"]),
            criteria=(ReviewCriterion.model_validate(captured["criterion"]),), review_from_message="m0009")
        for repeat in range(2):
            with self.subTest(repeat=repeat):
                result = evaluate(self.model, packet, self.out / f"battery-metaphor-{repeat+1}")
                self.assertEqual(result.judgments[0].verdict, "pass", result.model_dump_json())
