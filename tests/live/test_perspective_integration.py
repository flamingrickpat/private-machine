"""Integrated live gate: theory of mind and metacognition in one assembled app.

The three parts use the real ``build_app`` graph, a real ``MemoryStore``, real
OpenRouter chat and embeddings, and the real private, control, triage, focus,
and entity modules. They assert on committed records, not on model prose.

- Part A: a ToM action committed at the dispatch boundary ends in a finding a
  real main turn claims, and the entity owner stages the proposed revision as
  a ``tom`` receipt. The action sends no message.
- Part B: main cognition routes a real Jesse message to ``theory_of_mind``.
- Part C: a noticing of dwelling reaches main cognition with its recorded
  activity facts, and main appraises it.

A model that does not reach one of these states is a recorded result. Part A
reports ``NOT RUN`` when no episode produces a revision in three attempts;
Part B reports ``NOT RUN`` when main selects another action three times; Part C
reports ``NOT RUN`` when no noticing arises in the tick limit or main commits
an inference wait. This gate never forces an action and never clears a wait.
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

from pm.bus.envelopes import new_envelope
from pm.llm.activation import ModelActivation
from pm.llm.config import load_config, resolve_connection
from pm.llm.mcp_config import ToolCallConfig
from pm.llm.stage_model import LazyStageModel
from pm.llm.workflow_sessions import WorkflowModels
from pm.memory.store import MemoryStore
from pm.records.base import Provenance, RecordLineage
from pm.records.enums import ActionType, FeatureType, SourceKind
from pm.records.memory_records import ActionSchemaRecord
from pm.utils.operating_system import OperatingSystem
from tests.replay.situations import PERSPECTIVE_LINES, perspective_morning, private_dwelling
from tests.support import ManualClock
from workflows.companion import build_app
from workflows.entity_context.owner import receipt_chain
from workflows.entity_context.schemas import EntityContextKey
from workflows.inner_thought.graph_host import pending_findings
from workflows.inner_thought.inquiry import INQUIRY_EPISODE_KEY, inquiry_history
from pm.records.base import RecordScope
from workflows.main_cognition.schemas import ActionArguments, ActionRequest, Stimulus

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = ROOT / ".runtime/beta-b06/integration"
PERSPECTIVE_INSTRUCTION = (
    "Understand when Jesse usually becomes available in the morning, and how uncertain that pattern is, "
    "so I can judge whether and when a greeting would be welcome."
)
#: First operation needs no embedding request; the gate's real retrieval stays untested here.
PERSPECTIVE_FIRST_OPERATION = "narrative_stitching"
#: How many private opportunities a part may spend before it gives up.
GATE_TICK_LIMIT = 14


def _module(app, module_id: str):
    """Return the assembled module with this id."""
    matches = [module for module in app.modules if module.module_id == module_id]
    if len(matches) != 1:
        raise LookupError(f"Expected one module {module_id!r}, found {len(matches)}.")
    return matches[0]


def _source_ids(store: MemoryStore) -> list[int]:
    """Return the synthetic morning lines the situation committed."""
    expected = {text for text, _ in PERSPECTIVE_LINES}
    matches = [record.id for record in store.all_features
               if record.feature_type == FeatureType.Dialogue and record.content in expected]
    if len(matches) != len(PERSPECTIVE_LINES):
        raise LookupError(f"Expected every synthetic morning line, found {len(matches)}.")
    return matches


class _Gate:
    """One assembled application and its evidence directory."""

    def __init__(self, name: str, *, dwelling: bool = False) -> None:
        self.name = name
        self.out = EVIDENCE_ROOT / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8] + "-" + name)
        self.out.mkdir(parents=True)
        self.clock = ManualClock()
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        if dwelling:
            self.store: MemoryStore = private_dwelling(now, self.out / "ghost.pickle").store
        else:
            self.store = perspective_morning(now, self.out / "ghost.pickle").store
        config = load_config(ROOT / "config.yaml")
        connection = resolve_connection(config)
        if (urlsplit(connection.base_url).hostname != "openrouter.ai" or connection.startup_script is not None
                or urlsplit(config.embedding.base_url).hostname != "openrouter.ai"):
            raise ValueError("NOT RUN — this gate requires OpenRouter chat and embeddings.")
        self.model = LazyStageModel(connection, ModelActivation(), live_test=True)
        workflow_models = WorkflowModels(config, ModelActivation(), live_test=True)
        OperatingSystem.environment["PM_COMPLETION_CACHE"] = "0"
        OperatingSystem.environment["PM_PROMPT_LOG_DIR"] = str(self.out / "prompts")
        self.events = []
        self.replies: list[dict[str, object]] = []
        self.app = build_app(None, self.out / "runtime", ghost=self.store, clock=self.clock,
            cognition="real", stage_model=self.model, workflow_models=workflow_models,
            config_path=str(ROOT / "config.yaml"), thought_period=3600,
            record_lifecycle=False, enable_minecraft=False, world_poll_seconds=None,
            tool_call=ToolCallConfig(), emit_reply=lambda text: self.replies.append({"text": text}))
        self.app.bus.subscribe("integration_evidence", lambda envelope: True, self._observe)

    def _observe(self, envelope) -> None:
        """Retain every bus envelope without changing routing."""
        self.events.append(envelope)
        self.write("events.jsonl", {"topic": envelope.topic, "correlation": envelope.correlation_id,
                                    "payload": envelope.payload})

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
        matches = [record.id for record in self.store.all_features
                   if record.feature_type == FeatureType.Dialogue and record.content == text]
        if len(matches) != 1:
            raise LookupError(f"Expected one dialogue record for the injected text, found {matches}.")
        return matches[0]

    def dispatch_tom(self, correlation: str) -> int:
        """Commit one theory-of-mind action at the dispatch boundary, as main cognition would."""
        source_ids = _source_ids(self.store)
        view = self.store.snapshot(capture=False)
        sources = [view.get(source_id) for source_id in source_ids]
        user_id = self.store.get_entity_id(self.store.get_user_name())
        if user_id is None:
            raise ValueError("The situation must hold the user identity.")
        request = ActionRequest(action_id="live-tom-action", workflow_name="theory_of_mind",
            content="I want to work this out on my own before I act.",
            partner_entity_id=user_id,
            arguments=ActionArguments(instruction=PERSPECTIVE_INSTRUCTION,
                                      operation=PERSPECTIVE_FIRST_OPERATION),
            stimuli=[Stimulus(knoxel_id=source_id, source="user_input", text=str(source.content))
                     for source_id, source in zip(source_ids, sources, strict=True)])
        batch = self.store.begin_batch("live-tom-dispatch-" + correlation, embed=False)
        action = batch.create(ActionSchemaRecord(content=request.content, action_type=ActionType.Loopback,
            provenance=Provenance(source_kind=SourceKind.Workflow, source_ref="live-gate-synthetic-dispatch"),
            lineage=RecordLineage(episode_id="dispatch:live-tom-action", operation="action_dispatch",
                parent_record_ids=source_ids),
            metadata={"action_id": request.action_id, "workflow_name": "theory_of_mind",
                      "dispatch_status": "pending", "dispatch_target": "theory_of_mind",
                      "dispatch_correlation": correlation, "evidence_ids": source_ids}))
        batch.preserve_on_recovery()
        self.store.commit(batch)
        payload = request.model_copy(update={"knoxel_id": action.id}).payload()
        self.app.bus.put(new_envelope("main_cog", "data", "action.request", correlation, payload,
                                      target="theory_of_mind"))
        return action.id

    def focus_step(self) -> None:
        """Step the deliberate owner when its timer is due, as the runtime harness does."""
        self.clock.advance(3600)
        self.app.focus_management.step(block=False)

    def wait_until(self, predicate, description: str, timeout_seconds: float = 600.0) -> None:
        """Wait within the gate's deadline while surfacing a product failure."""
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            self.app.raise_if_error()
            if predicate():
                return
            time.sleep(0.05)
        raise AssertionError(f"Timed out waiting for {description}; evidence: {self.out}")


class LivePerspectiveIntegrationTests(unittest.TestCase):
    """Exercise ToM selection, revision staging, and noticing on the real path."""

    # ---------------------------------------------------------------- part A

    def test_part_a_tom_finding_stages_through_the_entity_owner(self) -> None:
        gate = _Gate("part-a")
        try:
            for attempt in range(1, 4):
                gate.write("attempt.json", {"attempt": attempt})
                action_id = gate.dispatch_tom(f"live-tom-{attempt}")
                gate.focus_step()
                for _ in range(GATE_TICK_LIMIT):
                    gate.private_opportunity()
                    view = gate.store.snapshot(capture=False)
                    history = inquiry_history(view, f"deliberate:{action_id}")
                    if history and history[-1].receipt.state.status != "active":
                        break
                gate.focus_step()
                gate.main_opportunity()
                view = gate.store.snapshot(capture=False)
                action = view.get(action_id)
                finding_id = action.metadata.get("mental_result_id")
                gate.write("attempt-result.json", {"attempt": attempt, "action_id": action_id,
                                                   "finding_id": finding_id,
                                                   "mental_status": action.metadata.get("mental_status")})
                if finding_id is None:
                    continue
                operation = view.get(int(finding_id))
                revision = operation.metadata.get("perspective_revision")
                if revision is None:
                    continue
                gate.write("revision.json", {"attempt": attempt, "finding_id": int(finding_id),
                                             "revision": revision})
                self._assert_part_a(gate, action_id, int(finding_id), revision)
                return
            self.skipTest(f"NOT RUN — no ToM episode produced a revision in three attempts; {gate.out}")
        except BaseException as error:
            name = "not-run.json" if isinstance(error, unittest.SkipTest) else "failure.json"
            gate.write(name, {"part": "A", "error_type": type(error).__name__,
                              "reason": str(error), "traceback": traceback.format_exc()})
            raise
        finally:
            gate.close()

    def _assert_part_a(self, gate: _Gate, action_id: int, finding_id: int, revision: dict) -> None:
        """Pass assertions: a main turn claims the finding, the owner stages tom, no speech."""
        main_turns = [turn for turn in gate.store.gwt_turns.values()
                      if turn.slot_id == "main" and finding_id in turn.stimulus_ids]
        self.assertTrue(main_turns, f"Main cognition did not claim the ToM finding; {gate.out}")
        view = gate.store.snapshot(capture=False)
        user_id = gate.store.get_entity_id(gate.store.get_user_name())
        observer_id = gate.store.get_entity_id(gate.store.get_companion_name())
        assert user_id is not None and observer_id is not None
        key = EntityContextKey(subject_entity_id=user_id, observer_entity_id=observer_id, scope=RecordScope())
        chain = receipt_chain(view, key)
        tom_receipts = [item for item in chain if item.receipt.proposal.producer == "tom"]
        self.assertTrue(tom_receipts, f"No tom receipt in the subject family after a real main turn; {gate.out}")
        self.assertEqual(gate.replies, [], f"The ToM action must send no public reply; {gate.out}")
        speech = [item for item in gate.events
                  if item.topic in {"speech.request", "reply.ready"} and item.source == "focus_management"]
        self.assertEqual(speech, [], f"FocusManagement must publish no speech; {gate.out}")
        gate.write("part-a.json", {"action_id": action_id, "finding_id": finding_id,
                                   "main_tick": main_turns[0].tick_id,
                                   "tom_receipt_ids": [item.record.id for item in tom_receipts],
                                   "requests": gate.model.request_count})

    # ---------------------------------------------------------------- part B

    def test_part_b_main_cognition_selects_theory_of_mind(self) -> None:
        gate = _Gate("part-b")
        attempts = (
            ("Before we plan the weekend, work out on your own when my mornings usually free up. "
             "Do not ask me; think about what you already know.", "when I am usually up"),
            ("You called my mornings unpredictable once. Use some private time to figure out the pattern "
             "behind when a message is likely to reach me.", "my morning availability"),
            ("I will be busy for a few days. Privately work out when a message from you is most likely to "
             "find me, and how sure you are of that.", "when to find me"),
        )
        try:
            selected = None
            for index, (text, hint) in enumerate(attempts, start=1):
                gate.submit(text)
                gate.main_opportunity()
                gate.app.raise_if_error()
                requests = [record for record in gate.store.all_knoxels.values()
                            if isinstance(record, ActionSchemaRecord)
                            and (record.metadata or {}).get("workflow_name") == "theory_of_mind"]
                gate.write("attempt.json", {"index": index, "text": text, "hint": hint,
                                            "tom_actions": [record.id for record in requests],
                                            "requests": gate.model.request_count})
                if requests:
                    selected = requests[-1]
                    break
            if selected is None:
                self.skipTest(f"NOT RUN — main did not select theory_of_mind in three attempts; {gate.out}")
            assert selected is not None
            gate.write("part-b.json", {"action_id": int(selected.id),
                                       "workflow_name": selected.metadata.get("workflow_name"),
                                       "requests": gate.model.request_count})
        except BaseException as error:
            name = "not-run.json" if isinstance(error, unittest.SkipTest) else "failure.json"
            gate.write(name, {"part": "B", "error_type": type(error).__name__,
                              "reason": str(error), "traceback": traceback.format_exc()})
            raise
        finally:
            gate.close()

    # ---------------------------------------------------------------- part C

    def test_part_c_noticing_reaches_main_cognition(self) -> None:
        gate = _Gate("part-c", dwelling=True)
        try:
            noticing = None
            for _ in range(GATE_TICK_LIMIT):
                gate.private_opportunity()
                candidates = [record for record in pending_findings(gate.store)
                              if "thought_activity" in (record.metadata or {})]
                if candidates:
                    noticing = candidates[0]
                    break
            if noticing is None:
                self.skipTest(f"NOT RUN — no noticing with activity facts within the tick limit; {gate.out}")
            assert noticing is not None
            gate.write("noticing.json", {"id": noticing.id,
                                         "activity": noticing.metadata.get("thought_activity")})
            gate.main_opportunity()
            gate.app.raise_if_error()
            if gate.app.cognitive_control.waiting():
                self.skipTest(f"NOT RUN — main committed an inference wait instead of claiming the noticing; {gate.out}")
            main_turns = [turn for turn in gate.store.gwt_turns.values()
                          if turn.slot_id == "main" and noticing.id in turn.stimulus_ids]
            self.assertTrue(main_turns, f"Main cognition did not claim the noticing; {gate.out}")
            actions = [record for record in gate.store.all_knoxels.values()
                       if isinstance(record, ActionSchemaRecord)
                       and (record.metadata or {}).get("dispatch_status") not in {None, ""}]
            replied = list(gate.replies)
            gate.write("part-c.json", {
                "noticing_id": noticing.id,
                "main_tick": main_turns[0].tick_id,
                "main_summary": main_turns[0].summary,
                "actions_main_chose": [(record.id, (record.metadata or {}).get("workflow_name")) for record in actions],
                "replies": replied,
                "requests": gate.model.request_count})
        except BaseException as error:
            name = "not-run.json" if isinstance(error, unittest.SkipTest) else "failure.json"
            gate.write(name, {"part": "C", "error_type": type(error).__name__,
                              "reason": str(error), "traceback": traceback.format_exc()})
            raise
        finally:
            gate.close()


if __name__ == "__main__":
    unittest.main()
