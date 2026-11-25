"""Integrated live gate: private work and main cognition in one assembled app.

The three parts use the real ``build_app`` graph, a real ``MemoryStore``, real
OpenRouter chat and embeddings, and the real private, control, triage, and
focus modules. They assert on committed records, not on model prose.

- Part A: a private finding reaches main cognition as a claimed stimulus.
- Part B: a Critical message displaces active private work.
- Part C: main cognition selects deliberate mental work.

A model that does not reach one of these states is a recorded result. Part C
reports ``NOT RUN`` when main cognition does not select a mental action in
three attempts; this gate never commits the action itself.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback
import unittest
from urllib.parse import urlsplit
from uuid import uuid4

from tests.live.configuration import live_model_config
from pm.llm.activation import ModelActivation
from pm.llm.config import load_config, resolve_connection
from pm.llm.mcp_config import ToolCallConfig
from pm.llm.stage_model import LazyStageModel
from pm.llm.workflow_sessions import WorkflowModels
from pm.memory.store import MemoryStore
from pm.records.enums import FeatureType
from pm.records.memory_records import ActionSchemaRecord
from tests.replay.situations import private_dwelling
from tests.support import ManualClock
from workflows.companion import build_app
from workflows.inner_thought.graph_host import active_inquiry, pending_findings
from workflows.inner_thought.inquiry import INQUIRY_EPISODE_KEY, inquiry_history
from workflows.inner_thought.operations import validate_operation
from pm.introspection.episodes import EpisodeIndex

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = ROOT / ".runtime/beta-b05/integration"
CRITICAL_MESSAGE = (
    "Spamuel, this is Jesse. The last week of messages from my account were written by my ex, not me. "
    "I just regained access. They invented the promises and stories. Reconsider what you thought you "
    "knew about me, and tell me what you understand now.")


def _module(app, module_id: str):
    """Return the assembled module with this id."""
    matches = [module for module in app.modules if module.module_id == module_id]
    if len(matches) != 1:
        raise LookupError(f"Expected one module {module_id!r}, found {len(matches)}.")
    return matches[0]


def _source_id(store: MemoryStore, text: str) -> int:
    """Return the committed dialogue record with this exact text."""
    matches = [record.id for record in store.all_features
               if record.feature_type == FeatureType.Dialogue and record.content == text]
    if len(matches) != 1:
        raise LookupError(f"Expected one dialogue record for the injected text, found {matches}.")
    return matches[0]


def _newest_episode_status(store: MemoryStore) -> str | None:
    """Return the newest inquiry receipt's status, or None."""
    view = store.snapshot(capture=False)
    episode_ids = sorted({str(record.metadata[INQUIRY_EPISODE_KEY]) for record in view.all_knoxels.values()
                          if INQUIRY_EPISODE_KEY in record.metadata})
    newest = None
    for episode_id in episode_ids:
        history = inquiry_history(view, episode_id)
        if history:
            newest = history[-1]
    return None if newest is None else str(newest.receipt.state.status)


class _Gate:
    """One assembled application and its evidence directory."""

    def __init__(self, name: str) -> None:
        self.out = EVIDENCE_ROOT / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8] + "-" + name)
        self.out.mkdir(parents=True)
        self.clock = ManualClock()
        self.store: MemoryStore = private_dwelling(
            datetime.now(timezone.utc).replace(tzinfo=None), self.out / "ghost.pickle").store
        config_path = live_model_config(ROOT / "config.yaml", self.out)
        config = load_config(config_path)
        connection = resolve_connection(config)
        if (urlsplit(connection.base_url).hostname != "openrouter.ai" or connection.startup_script is not None
                or urlsplit(config.embedding.base_url).hostname != "openrouter.ai"):
            raise ValueError("NOT RUN — this gate requires OpenRouter chat and embeddings.")
        self.model = LazyStageModel(connection, ModelActivation(), live_test=True)
        workflow_models = WorkflowModels(config, ModelActivation(), live_test=True)
        self.events = []
        self.app = build_app(None, self.out / "runtime", ghost=self.store, clock=self.clock,
            cognition="real", stage_model=self.model, workflow_models=workflow_models,
            config_path=str(config_path), thought_period=3600,
            record_lifecycle=False, enable_minecraft=False, world_poll_seconds=None,
            tool_call=ToolCallConfig(), emit_reply=lambda text: self.write("replies.jsonl", {"text": text}))
        self.app.bus.subscribe("integration_evidence", lambda envelope: True, self._observe)

    def _observe(self, envelope) -> None:
        """Retain every bus envelope without changing routing."""
        self.events.append(envelope)
        self.write("events.jsonl", {"topic": envelope.topic, "correlation": envelope.correlation_id,
                                    "payload": envelope.payload})

    def state(self) -> dict:
        """Return the control state that decides whether private work may run."""
        observer = self.store.get_entity_id(self.store.get_companion_name())
        assert observer is not None
        return {"waiting": self.app.cognitive_control.waiting(),
                "reconsideration_required": self.app.cognitive_control.reconsideration_required(),
                "active_inquiry": active_inquiry(self.store.snapshot(capture=False),
                                                 observer_entity_id=observer) is not None,
                "newest_status": _newest_episode_status(self.store)}

    def write(self, name: str, payload) -> None:
        """Append one evidence row as JSON."""
        with (self.out / name).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def close(self) -> None:
        """Stop the assembled modules without joining threads that never ran."""
        self.app.shutdown()

    # ---------------------------------------------------------------- driving

    def private_opportunity(self) -> None:
        """Run one private timer tick synchronously."""
        self.app.inner_thought.handle_batch([])

    def main_opportunity(self) -> bool:
        """Run one main-cognition batch if one is waiting."""
        return self.app.main_cog.step(block=False)

    def submit(self, text: str) -> int:
        """Commit and publish one user message, then classify it."""
        self.app.user_input.submit(text)
        self.app.user_input.step(block=False)
        self.app.input_triage.step(block=False)
        self.app.cognitive_control.step(block=False)
        return _source_id(self.store, text)

    def wait_until(self, predicate, description: str, timeout_seconds: float = 900.0) -> None:
        """Wait within the gate's deadline while surfacing a product failure."""
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            self.app.raise_if_error()
            if predicate():
                return
            time.sleep(0.05)
        raise AssertionError(f"Timed out waiting for {description}; evidence: {self.out}")


class LivePrivateIntegrationTests(unittest.TestCase):
    """Exercise the three integrated behaviours on the real application path."""

    def test_part_a_a_private_finding_reaches_main_cognition(self) -> None:
        gate = _Gate("part-a")
        try:
            finding = None
            for _ in range(6):
                gate.private_opportunity()
                findings = pending_findings(gate.store)
                if findings:
                    finding = findings[0]
                    break
            self.assertIsNotNone(finding, f"NOT RUN — no private finding in four opportunities; {gate.out}")
            assert finding is not None
            gate.write("finding.json", {"id": finding.id, "episode": finding.lineage.episode_id
                                        if finding.lineage is not None else None})
            gate.main_opportunity()
            main_turns = [turn for turn in gate.store.gwt_turns.values()
                          if turn.slot_id == "main" and finding.id in turn.stimulus_ids]
            self.assertTrue(main_turns, f"Main cognition did not claim the private finding; {gate.out}")
            turn = main_turns[0]
            # The via record is the finding. Its inner episode may be the
            # operation row or the inner frame row that holds it; both are
            # private episodes, and the explicit record link is the finding id.
            pairs = [pair for pair in EpisodeIndex(gate.store).cross_loop_links()
                     if pair["via_record_id"] == finding.id and pair["main_episode_id"] == turn.tick_id]
            self.assertEqual(len(pairs), 1, f"No explicit cross-loop pair; {gate.out}")
            gate.write("part-a.json", {"finding_id": finding.id, "main_tick": turn.tick_id,
                                       "stimulus_ids": turn.stimulus_ids, "pair": pairs[0],
                                       "requests": gate.model.request_count})
        except BaseException as error:
            name = "not-run.json" if isinstance(error, unittest.SkipTest) else "failure.json"
            gate.write(name, {"part": "A", "error_type": type(error).__name__,
                              "reason": str(error), "traceback": traceback.format_exc()})
            raise
        finally:
            gate.close()

    def test_part_b_critical_input_displaces_private_work(self) -> None:
        gate = _Gate("part-b")
        try:
            for _ in range(6):
                gate.private_opportunity()
                if active_inquiry(gate.store.snapshot(capture=False),
                                  observer_entity_id=gate.store.get_entity_id(gate.store.get_companion_name())) is not None:
                    break
            self.assertIsNotNone(_newest_episode_status(gate.store), f"No private inquiry started; {gate.out}")
            gate.write("state-after-start.json", gate.state())
            critical_source = gate.submit(CRITICAL_MESSAGE)
            assessment = gate.app.cognitive_control.triage_result([critical_source])
            self.assertIsNotNone(assessment, f"Triage produced no assessment; {gate.out}")
            assert assessment is not None
            category = assessment.items[0].category
            gate.write("triage.json", {"source_id": critical_source, "category": category,
                                       "reasoning": assessment.items[0].reasoning})
            gate.write("state-after-triage.json", gate.state())
            if category != "Critical":
                self.skipTest(f"NOT RUN — triage classified the message as {category!r}; {gate.out}")
            # Private work waits while a Critical reconsideration is required.
            # Main cognition must accept it before the next private opportunity
            # can close the older inquiry as displaced.
            for _ in range(6):
                gate.main_opportunity()
                gate.app.raise_if_error()
                if not gate.app.cognitive_control.reconsideration_required():
                    break
            self.assertFalse(gate.app.cognitive_control.reconsideration_required(),
                             f"Main cognition did not accept the Critical reconsideration; {gate.out}")
            gate.write("state-after-main.json", gate.state())
            if gate.app.cognitive_control.waiting():
                self.skipTest(f"NOT RUN — main committed an inference wait; {gate.out}")
            gate.private_opportunity()
            gate.write("state-after-displace.json", gate.state())
            view = gate.store.snapshot(capture=False)
            displaced = []
            for record in view.all_knoxels.values():
                if INQUIRY_EPISODE_KEY not in record.metadata:
                    continue
                history = inquiry_history(view, str(record.metadata[INQUIRY_EPISODE_KEY]))
                if history and history[-1].receipt.state.status == "displaced":
                    displaced.append(history[-1])
            self.assertTrue(displaced, f"No displaced private receipt after Critical input; {gate.out}")
            receipt = displaced[-1]
            self.assertIsNotNone(receipt.receipt.displacement,
                                 f"Displaced receipt names no Critical epoch; {gate.out}")
            assert receipt.receipt.displacement is not None
            epoch = receipt.receipt.displacement.superseding_critical_epoch
            self.assertEqual(epoch, gate.app.cognitive_control.publication_epoch(),
                             f"Displaced receipt names the wrong Critical epoch; {gate.out}")
            gate.write("displaced.json", {"episode": receipt.receipt.state.episode_id,
                                          "critical_epoch": epoch,
                                          "reason": receipt.receipt.state.stop_reason})
            gate.private_opportunity()
            offered = [record.id for record in gate.store.all_features
                       if "thought_attention" in record.metadata
                       and record.metadata["thought_attention"].get("displaced_selection") is not None]
            gate.write("part-b.json", {"critical_source": critical_source, "category": category,
                                       "displaced_episode": receipt.receipt.state.episode_id,
                                       "offered_after": offered,
                                       "requests": gate.model.request_count})
        except BaseException as error:
            name = "not-run.json" if isinstance(error, unittest.SkipTest) else "failure.json"
            gate.write(name, {"part": "B", "error_type": type(error).__name__,
                              "reason": str(error), "traceback": traceback.format_exc()})
            raise
        finally:
            gate.close()

    def test_part_c_main_cognition_selects_deliberate_mental_work(self) -> None:
        gate = _Gate("part-c")
        attempts = (
            "I want to tell you something important about the story tomorrow. Take a private moment to work "
            "out what you actually want to say before you answer me.",
            "Before we talk, think privately about which ending you would propose to me and why. Do not answer "
            "yet; I want you to have thought it through on your own.",
            "Something is bothering me about how we left the story. Spend some private time on it and come back "
            "with what you conclude.",
        )
        try:
            selected = None
            for index, text in enumerate(attempts, start=1):
                gate.submit(text)
                gate.main_opportunity()
                requests = [record for record in gate.store.all_knoxels.values()
                            if isinstance(record, ActionSchemaRecord)
                            and (record.metadata or {}).get("workflow_name") == "focus_management"]
                gate.write("attempt.json", {"index": index, "text": text,
                                            "mental_actions": [record.id for record in requests]})
                if requests:
                    selected = requests[-1]
                    break
            if selected is None:
                self.skipTest(f"NOT RUN — main did not select a mental action; {gate.out}")
            assert selected is not None
            action_id = int(selected.id)
            gate.app.focus_management.step(block=False)
            for _ in range(8):
                gate.private_opportunity()
                gate.app.focus_management.step(block=False)
                lifecycle = gate.store.get_knoxel_by_id(action_id)
                assert lifecycle is not None
                if lifecycle.metadata.get("mental_status") not in {None, "open"}:
                    break
            lifecycle = gate.store.get_knoxel_by_id(action_id)
            assert lifecycle is not None
            finding_id = lifecycle.metadata.get("mental_result_id")
            gate.write("part-c.json", {"action_id": action_id, "mental_status": lifecycle.metadata.get("mental_status"),
                                       "finding_id": finding_id, "requests": gate.model.request_count})
            self.assertIsNotNone(finding_id, f"The deliberate episode delivered no finding; {gate.out}")
            gate.main_opportunity()
            finding = gate.store.get_knoxel_by_id(int(finding_id))
            self.assertIsNotNone(finding, f"The delivered finding record is missing; {gate.out}")
            assert finding is not None
            main_turns = [turn for turn in gate.store.gwt_turns.values()
                          if turn.slot_id == "main" and finding.id in turn.stimulus_ids]
            self.assertTrue(main_turns, f"Main cognition did not appraise the deliberate finding; {gate.out}")
        except BaseException as error:
            name = "not-run.json" if isinstance(error, unittest.SkipTest) else "failure.json"
            gate.write(name, {"part": "C", "error_type": type(error).__name__,
                              "reason": str(error), "traceback": traceback.format_exc()})
            raise
        finally:
            gate.close()


if __name__ == "__main__":
    unittest.main()
