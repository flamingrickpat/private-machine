"""Real OpenRouter early-triage kernel and actual publication/restart boundary.

This narrow gate does not claim full-cycle or spontaneous-thought acceptance.
It uses the primary Spamuel seed with synthetic incoming test events.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
import unittest
from urllib.parse import urlsplit
from uuid import uuid4

from pm.bus.bus import BusSystem
from pm.llm.activation import ModelActivation
from pm.llm.config import load_config, resolve_connection
from pm.llm.stage_model import LazyStageModel
from pm.memory.bash_view import MemoryBashTarget
from pm.memory.store import MemoryStore
from pm.utils.operating_system import OperatingSystem
from workflows.cognitive_control.host import CognitiveControl
from workflows.input_triage.host import InputTriage
from workflows.main_cognition.host import build_main_cognition_tools
from workflows.public_speech.publication_control import ReplySuperseded
from workflows.public_speech.user_input import UserInputProxy


ROOT = Path(__file__).resolve().parents[2]


class LiveInputTriageTests(unittest.TestCase):
    """Require actual contextual model classification and retained evidence."""

    def test_ordinary_input_and_consequential_correction(self) -> None:
        config = load_config(ROOT / "config.yaml")
        connection = resolve_connection(config)
        if urlsplit(connection.base_url).hostname != "openrouter.ai" or connection.startup_script is not None:
            raise ValueError("This gate requires OpenRouter without a local model startup script.")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8]
        out = ROOT / ".runtime/beta-input-triage" / stamp
        out.mkdir(parents=True)
        print(f"Early-triage evidence: {out}", flush=True)
        OperatingSystem.environment["PM_COMPLETION_CACHE"] = "0"
        OperatingSystem.environment["PM_PROMPT_LOG_DIR"] = str(out / "prompts")
        model = LazyStageModel(connection, ModelActivation(), live_test=True)
        store = MemoryStore.from_yaml(ROOT / "test_data/personas/spamuel/canon.yaml", checkpoint=out / "ghost.pickle")
        bus = BusSystem()
        control = CognitiveControl("control", bus, self.fail, time.monotonic, None, ghost=store)
        control.enable_input_triage()
        module = InputTriage("input_triage", bus, self.fail, time.monotonic, None, ghost=store,
            working_target=MemoryBashTarget(store, "read", recovery_only=True),
            workflow=ROOT / "workflows/input_triage/input_triage.workflow.md", runtime_root=out,
            tools_factory=build_main_cognition_tools(store, out, model.complete,
                                                   cancel_provider=lambda: module.run_stop))
        module.cognitive_control = control
        bus.subscribe(module.module_id, module.accepts, module.deliver)
        intake = UserInputProxy("user_input", bus, self.fail, time.monotonic, store, lambda text: None)
        intake.register_triage_input = control.register_input
        results: list[dict[str, object]] = []
        ordinary = "Hey Spamuel, how is your afternoon going?"
        correction = ("Spamuel, this is Jesse. The last week of messages from my account were written by my ex, not me. "
                      "I just regained access. They invented those promises and stories. I need you to rethink what "
                      "you assumed about me before you say anything else.")
        for label, message in (("ordinary", ordinary), ("identity_correction", correction)):
            before = time.monotonic()
            correlation = intake.submit(message)
            source_id = next(item.id for item in store.all_features
                             if item.metadata.get("exchange_correlation") == correlation)
            self.assertTrue(control.reconsideration_required())
            self.assertTrue(module.step())
            assessment = control.triage_result([source_id])
            assert assessment is not None
            result = {"case": label, "source_id": source_id, "correlation": correlation,
                      "seconds": time.monotonic() - before, "assessment": assessment.model_dump(mode="json")}
            results.append(result)
            (out / f"{label}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            if label == "ordinary":
                self.assertNotEqual(assessment.items[0].category, "Critical")
                self.assertFalse(control.reconsideration_required())
                old_epoch = control.publication_epoch()
            else:
                self.assertEqual(assessment.items[0].category, "Critical", assessment.items[0].reasoning)
                with self.assertRaises(ReplySuperseded):
                    control.validate_publication(old_epoch, None)
                reopened = MemoryStore.from_checkpoint(out / "ghost.pickle")
                rebuilt = CognitiveControl("rebuilt", BusSystem(), self.fail, time.monotonic, None, ghost=reopened)
                rebuilt.enable_input_triage()
                self.assertTrue(rebuilt.has_assessment(source_id))
                self.assertTrue(rebuilt.reconsideration_required())
        (out / "result.json").write_text(json.dumps({"model": connection.model, "endpoint": connection.base_url,
            "requests": model.request_count, "completion_cache": False, "results": results,
            "scope": __doc__}, indent=2), encoding="utf-8")
        print(f"Real classification passed: {model.request_count} provider requests.", flush=True)
