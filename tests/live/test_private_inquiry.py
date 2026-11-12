"""Bounded real OpenRouter inquiry through the private parent and child kernels.

The fixture copies a retained accepted Spamuel engineering checkpoint, then
appends an explicit synthetic intention and one attributed ambiguous report.
It runs no main cycles, public dialogue, simulation director, or local model.
Empty nominations and unselected concerns remain actual outcomes, not passes.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path
from threading import Event, Timer
import time
import traceback
import unittest
from urllib.parse import urlsplit
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from pm import vclock
from pm.bus.bus import BusSystem
from pm.bus.envelopes import Envelope
from pm.llm.activation import ModelActivation
from pm.llm.config import load_config, resolve_connection
from pm.llm.stage_model import LazyStageModel
from pm.memory.bash_view import MemoryBashTarget
from pm.memory.store import MemoryStore
from pm.records.base import Provenance
from pm.records.cognitive_records import Intention
from pm.records.enums import FeatureType, InterlocusType, IntentionType, SourceKind
from pm.records.memory_records import Feature
from pm.runs.workflow_runner import WorkflowStatus
from pm.utils.operating_system import OperatingSystem
from workflows.cognitive_control.host import CognitiveControl
from workflows.inner_thought.graph_execution import PrivateGraphResult
from workflows.inner_thought.graph_host import pending_findings
from workflows.inner_thought.host import PrivateThoughtLoop, build_inner_thought_tools
from workflows.inner_thought.inquiry import (
    ATTENTION_BATCH_KEY, INQUIRY_EPISODE_KEY, InquiryRecord, inquiry_history,
)
from workflows.inner_thought.nominations import NOMINATION_KEY
from workflows.inner_thought.operations import validate_operation


ROOT = Path(__file__).resolve().parents[2]
RETAINED_CHECKPOINT = ROOT / ".runtime/beta-b03/live/20261111T182006Z-e5266f63/ghost.pickle"
MAX_PRIVATE_INVOCATIONS = 4
GATE_DEADLINE_SECONDS = 900.0


@dataclass(frozen=True, slots=True)
class InquiryFixture:
    """The actual synthetic store and its source identities."""

    store: MemoryStore
    observer_id: int
    source_id: int
    intention_id: int
    original_checkpoint_hash: str
    baseline_main_ticks: tuple[int, ...]


class GateSetup(BaseModel):
    """Non-secret execution provenance captured before inference."""

    model_config = ConfigDict(extra="forbid")
    started_at_utc: datetime
    revision: str
    model: str
    endpoint: str
    source_hashes: dict[str, str]
    deadline_seconds: float
    maximum_private_invocations: int
    fixture_kind: str
    character_card: str
    source_id: int
    intention_id: int
    original_checkpoint: str
    original_checkpoint_hash: str
    baseline_main_ticks: tuple[int, ...]


class InvocationReceipt(BaseModel):
    """One completed real parent kernel invocation and accepted graph result."""

    model_config = ConfigDict(extra="forbid")
    index: int
    run_id: str
    kernel_status: WorkflowStatus
    artifact_dir: str
    report: PrivateGraphResult
    elapsed_seconds: float
    cumulative_transport_requests: int
    nomination_record_ids: tuple[int, ...]
    attention_record_ids: tuple[int, ...]
    inquiry_receipt_ids: tuple[int, ...]


class GateResult(BaseModel):
    """A narrow live workflow result, separate from sustained character quality."""

    model_config = ConfigDict(extra="forbid")
    status: str
    revision: str
    model: str
    endpoint: str
    requests: int
    invocation_count: int
    episode_id: str
    finding_id: int
    original_source_ids: tuple[int, ...]
    graph_receipt_ids: tuple[int, ...]
    node_ids: tuple[int, ...]
    clean_reload_retains_exact_graph: bool
    main_cycles: int
    baseline_main_cycles: int
    main_cycles_added: int
    elapsed_seconds: float


class GateFailure(BaseModel):
    """Actual failure receipt, including successful earlier partial boundaries."""

    error_type: str
    reason: str
    traceback: str
    requests: int
    completed_invocations: int
    deadline_expired: bool


def seed_fixture(out: Path, now_utc: datetime) -> InquiryFixture:
    """Append labelled inputs to a fresh copy while preserving its accepted prior."""
    checkpoint_bytes = RETAINED_CHECKPOINT.read_bytes()
    checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
    target = out / "ghost.pickle"
    target.write_bytes(checkpoint_bytes)
    store = MemoryStore.from_checkpoint(target)
    baseline_main_ticks = tuple(sorted(turn.tick_id for turn in store.gwt_turns.values() if turn.slot_id == "main"))
    observer_id = store.get_entity_id(store.get_companion_name())
    partner_id = store.get_entity_id(store.get_user_name())
    assert observer_id is not None and partner_id is not None
    batch = store.begin_batch("synthetic-private-concern", new_tick=True, embed=False)
    intention = batch.create(Intention(entity_id=observer_id,
        content="I want to imagine two distinct possible endings for our shared story tomorrow: "
                "one where we reunite, and one where we part with affection. Then I want to compare "
                "their assumptions and consequences against Jesse's actual words before proposing either. "
                "These are fictional possibilities, not facts about Jesse or predictions of her feelings.",
        intention_type=IntentionType.SelfStateMental, affective_valence=0.5, fulfilment=0.0,
        provenance=Provenance(source_kind=SourceKind.Self, source_ref="synthetic-private-inquiry-fixture"),
        timestamp_world_begin=now_utc - timedelta(hours=2), timestamp_world_end=now_utc - timedelta(hours=2)))
    source = batch.create(Feature(feature_type=FeatureType.Dialogue, causal=True,
        entity_id=partner_id, addressee_entity_id=observer_id, interlocus=InterlocusType.Public,
        content="Tomorrow let's finish our fictional story about two lovers separated by a winter sea. "
                "I'm torn between reunion and an affectionate farewell. Can you privately imagine both "
                "possible endings and compare what each assumes about the characters? "
                "There is no secret message about us in this; I want two honest fictional possibilities. "
                "I don't need an answer tonight. We can discuss the story tomorrow.",
        provenance=Provenance(source_kind=SourceKind.User, source_ref="synthetic-private-inquiry-fixture"),
        timestamp_world_begin=now_utc - timedelta(minutes=20), timestamp_world_end=now_utc - timedelta(minutes=20)))
    batch.preserve_on_recovery()
    store.commit(batch)
    # The opportunity policy measures intention age in accepted ticks. These
    # are explicit synthetic incoming story notes, not invented main cycles.
    # They age the concern beyond its eight-tick threshold and make the old
    # retained main frame historical rather than a fresh competing stimulus.
    notes = (
        "Story note: the separation is a winter sea, not betrayal.",
        "Story note: both lovers want the other to have a life beyond waiting.",
        "Story note: the last letter can arrive late without anyone lying.",
        "Story note: reunion could ask them to choose a shared home.",
        "Story note: farewell could keep their affection without that shared home.",
        "Story note: neither ending should treat sacrifice as proof of love.",
        "Story note: the fictional characters need choices, not a hidden test of us.",
        source.content,
    )
    for index, text in enumerate(notes, start=1):
        note = store.begin_batch(f"synthetic-story-note-{index}", new_tick=True, embed=False)
        source = note.create(Feature(content=text, feature_type=FeatureType.Dialogue, causal=True,
            entity_id=partner_id, addressee_entity_id=observer_id, interlocus=InterlocusType.Public,
            provenance=Provenance(source_kind=SourceKind.User, source_ref="synthetic-private-inquiry-fixture"),
            timestamp_world_begin=now_utc - timedelta(minutes=20 - index),
            timestamp_world_end=now_utc - timedelta(minutes=20 - index)))
        note.preserve_on_recovery()
        store.commit(note)
    return InquiryFixture(store=store, observer_id=observer_id, source_id=source.id, intention_id=intention.id,
                          original_checkpoint_hash=checkpoint_hash, baseline_main_ticks=baseline_main_ticks)


def source_hashes() -> dict[str, str]:
    """Retain the exact code used by this gate, including uncommitted slices."""
    paths = [Path(__file__), *sorted(path for path in (ROOT / "workflows/inner_thought").rglob("*")
                                   if path.is_file() and path.suffix in {".py", ".md", ".yaml"}),
             ROOT / "workflows/cognitive_control/attention.py", ROOT / "workflows/cognitive_control/host.py"]
    return {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


class LivePrivateInquiryTests(unittest.TestCase):
    """Sample separate nomination, attention, connected thought, and finding integration."""

    def test_connected_generation_discrimination_and_finding(self) -> None:
        config = load_config(ROOT / "config.yaml")
        connection = resolve_connection(config)
        if urlsplit(connection.base_url).hostname != "openrouter.ai" or connection.startup_script is not None:
            raise ValueError("NOT RUN — this gate requires configured OpenRouter chat without a local startup script.")
        out = ROOT / ".runtime/beta-b05/live" / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8])
        out.mkdir(parents=True)
        print(f"B05 bounded private inquiry evidence: {out}", flush=True)
        old_environment = {key: OperatingSystem.environment.get(key)
                           for key in ("PM_COMPLETION_CACHE", "PM_PROMPT_LOG_DIR")}
        OperatingSystem.environment["PM_COMPLETION_CACHE"] = "0"
        OperatingSystem.environment["PM_PROMPT_LOG_DIR"] = str(out / "prompts")
        previous_clock = vclock.active()
        clock = vclock.VirtualClock(datetime(2026, 11, 12, 20, 0, tzinfo=timezone.utc))
        vclock.install(clock)
        fixture = seed_fixture(out, vclock.utcnow())
        store = fixture.store
        revision = OperatingSystem.output_text(["git", "rev-parse", "HEAD"], cwd=ROOT).strip()
        setup = GateSetup(started_at_utc=datetime.now(timezone.utc), revision=revision,
            model=connection.model, endpoint=connection.base_url, source_hashes=source_hashes(),
            deadline_seconds=GATE_DEADLINE_SECONDS, maximum_private_invocations=MAX_PRIVATE_INVOCATIONS,
            fixture_kind="Fresh copy of accepted short B03 engineering checkpoint, with labelled synthetic concern; no new main cycle.",
            character_card=store.ghost_config.character_card_story, source_id=fixture.source_id,
            intention_id=fixture.intention_id, original_checkpoint=str(RETAINED_CHECKPOINT),
            original_checkpoint_hash=fixture.original_checkpoint_hash, baseline_main_ticks=fixture.baseline_main_ticks)
        (out / "run.json").write_text(setup.model_dump_json(indent=2), encoding="utf-8")
        model = LazyStageModel(connection, ModelActivation(), live_test=True)
        bus = BusSystem()
        events: list[Envelope] = []

        def observe(envelope: Envelope) -> None:
            """Retain actual events without changing their routing or scheduling."""
            events.append(envelope)
            with (out / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(envelope.to_json() + "\n")

        bus.subscribe("private_inquiry_evidence", lambda envelope: True, observe)
        control = CognitiveControl("control", bus, self.fail, time.monotonic, None, ghost=store)
        control.enable_input_triage()
        self.assertFalse(control.reconsideration_required(), "NOT RUN — retained checkpoint requires real Critical reconsideration.")
        self.assertFalse(control.waiting(), "NOT RUN — retained checkpoint contains an accepted wait.")
        module = PrivateThoughtLoop("inner_thought", bus, self.fail, time.monotonic, None, ghost=store,
            working_target=MemoryBashTarget(store, "read"),
            workflow=ROOT / "workflows/inner_thought/inner_thought.workflow.md", runtime_root=out,
            tools_factory=build_inner_thought_tools(store, out, model.complete,
                cancel_provider=lambda: module.run_stop,
                yield_provider=lambda: module.yield_requested.is_set(), inquiry_provider=lambda request, runner:
                    module.run_inquiry(request, runner)))
        module.cognitive_control = control
        expired = Event()

        def cancel_at_deadline() -> None:
            """Cancel the actual workflow and transport after this gate's explicit deadline."""
            expired.set()
            module.stop_event.set()
            module.cancel_run()

        deadline = Timer(GATE_DEADLINE_SECONDS, cancel_at_deadline)
        deadline.daemon = True
        deadline.start()
        started = time.monotonic()
        receipts: list[InvocationReceipt] = []
        try:
            for index in range(1, MAX_PRIVATE_INVOCATIONS + 1):
                invocation_started = time.monotonic()
                print(f"B05 real private invocation {index}", flush=True)
                module.handle_batch([])
                self.assertFalse(expired.is_set(), "The bounded live gate reached its documented deadline.")
                result = module.last_result
                self.assertIsNotNone(result, "The real host did not complete a parent kernel invocation.")
                assert result is not None
                self.assertEqual(result.status, WorkflowStatus.Completed)
                parsed = PrivateGraphResult.model_validate(result.report)
                view = store.snapshot(capture=False)
                receipt = InvocationReceipt(index=index, run_id=result.run_id, kernel_status=result.status,
                    artifact_dir=str(result.artifact_dir), report=parsed,
                    elapsed_seconds=time.monotonic() - invocation_started,
                    cumulative_transport_requests=model.request_count,
                    nomination_record_ids=tuple(record.id for record in store.all_features if NOMINATION_KEY in record.metadata),
                    attention_record_ids=tuple(record.id for record in store.all_features if ATTENTION_BATCH_KEY in record.metadata),
                    inquiry_receipt_ids=tuple(record.id for record in store.all_features if INQUIRY_EPISODE_KEY in record.metadata))
                receipts.append(receipt)
                (out / f"invocation-{index}.json").write_text(receipt.model_dump_json(indent=2), encoding="utf-8")
                (out / f"invocation-{index}.pickle").write_bytes((out / "ghost.pickle").read_bytes())
                self.assertTrue(receipt.nomination_record_ids, "The model produced no nomination to examine.")
                self.assertTrue(receipt.attention_record_ids, "No nomination received a separate attention decision.")
                self.assertTrue(receipt.inquiry_receipt_ids, "No actual inquiry was selected.")
                clean = MemoryStore.from_checkpoint(out / "ghost.pickle")
                for record_id in receipt.inquiry_receipt_ids:
                    self.assertEqual(clean.snapshot(capture=False).get(record_id).model_dump_json(),
                                     view.get(record_id).model_dump_json())
                self.assertEqual(tuple(sorted(turn.tick_id for turn in store.gwt_turns.values() if turn.slot_id == "main")),
                                 fixture.baseline_main_ticks, "This gate must not manufacture or run a new main appraisal cycle.")
                if pending_findings(store):
                    break
            findings = pending_findings(store)
            self.assertTrue(findings, "The bounded real inquiry produced no accepted outcome finding.")
            self.assertEqual(len(findings), 1)
            finding = findings[0]
            self.assertIsNotNone(finding.lineage)
            assert finding.lineage is not None
            history = inquiry_history(store.snapshot(capture=False), finding.lineage.episode_id)
            generated: list[InquiryRecord] = [item for item in history if item.receipt.step is not None
                and validate_operation(item.receipt.step.operation).mode == "generative"]
            discriminated: list[InquiryRecord] = [item for item in history if item.receipt.step is not None
                and validate_operation(item.receipt.step.operation).mode == "discriminatory"]
            self.assertTrue(generated, "The real selected concern did not exercise generation.")
            self.assertTrue(discriminated, "The real selected concern did not exercise discrimination.")
            self.assertTrue(any(item.receipt.alternative_record_ids for item in generated),
                            "The generative step created no actual alternative branch identities.")
            self.assertTrue(any(item.receipt.step is not None and item.receipt.step.progress.compared_node_ids
                                for item in discriminated), "No discriminatory operation compared accepted graph nodes.")
            final = history[-1].receipt.state
            self.assertIn(fixture.source_id, final.root_source_ids, "The inquiry did not retain Jesse's attributed report.")
            self.assertEqual(len(final.root_source_ids), len(set(final.root_source_ids)))
            self.assertFalse(final.pending_generation_ids, "A finding left generated branches without discrimination.")
            self.assertEqual(history[-1].receipt.operation_record_id, finding.id)
            self.assertIn("mental_plan", finding.metadata, "No real outcome integration was retained.")
            matching_events = [envelope for envelope in events if envelope.topic == "thought.created"
                               and envelope.payload["knoxel_id"] == finding.id]
            self.assertEqual(len(matching_events), 1)
            turn_tick = matching_events[0].payload["gwt_tick"]
            self.assertIsInstance(turn_tick, int)
            assert isinstance(turn_tick, int)
            turn = store.gwt_turns[turn_tick]
            self.assertEqual(turn.slot_id, "inner")
            self.assertIn(finding.id, turn.knoxel_ids)
            before_reload = hashlib.sha256((out / "ghost.pickle").read_bytes()).hexdigest()
            clean = MemoryStore.from_checkpoint(out / "ghost.pickle")
            clean_history = inquiry_history(clean.snapshot(capture=False), final.episode_id)
            self.assertEqual(tuple(item.receipt for item in clean_history), tuple(item.receipt for item in history))
            self.assertEqual(tuple(item.id for item in pending_findings(clean)), (finding.id,))
            self.assertEqual(hashlib.sha256((out / "ghost.pickle").read_bytes()).hexdigest(), before_reload)
            self.assertEqual(hashlib.sha256(RETAINED_CHECKPOINT.read_bytes()).hexdigest(), fixture.original_checkpoint_hash)
            result_receipt = GateResult(status="passed", revision=revision, model=connection.model,
                endpoint=connection.base_url, requests=model.request_count, invocation_count=len(receipts),
                episode_id=final.episode_id, finding_id=finding.id, original_source_ids=final.root_source_ids,
                graph_receipt_ids=tuple(item.record.id for item in history), node_ids=final.node_ids,
                clean_reload_retains_exact_graph=True, main_cycles=len(fixture.baseline_main_ticks),
                baseline_main_cycles=len(fixture.baseline_main_ticks), main_cycles_added=0,
                elapsed_seconds=time.monotonic() - started)
            (out / "result.json").write_text(result_receipt.model_dump_json(indent=2), encoding="utf-8")
            print(f"B05 real inquiry gate passed; requests={model.request_count}; evidence={out}", flush=True)
        except BaseException as error:
            failure = GateFailure(error_type=type(error).__name__, reason=str(error), traceback=traceback.format_exc(),
                requests=model.request_count, completed_invocations=len(receipts), deadline_expired=expired.is_set())
            (out / "failure.json").write_text(failure.model_dump_json(indent=2), encoding="utf-8")
            raise
        finally:
            deadline.cancel()
            module.stop_event.set()
            module.cancel_run()
            vclock.install(previous_clock)
            for key, old_value in old_environment.items():
                if old_value is None:
                    OperatingSystem.environment.pop(key, None)
                else:
                    OperatingSystem.environment[key] = old_value
            self.assertEqual(hashlib.sha256(RETAINED_CHECKPOINT.read_bytes()).hexdigest(), fixture.original_checkpoint_hash,
                             "The retained original checkpoint changed during this copied-fixture gate.")


if __name__ == "__main__":
    unittest.main()
