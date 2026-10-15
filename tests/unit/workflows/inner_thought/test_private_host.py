"""Private loop host tests against the real MemoryStore commit boundary.

No model endpoint and no module thread. A deterministic ``run_stage`` stub
stands in for the kernel, exactly as the main cognition state tests do.
"""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pm.bus.bus import BusSystem
from pm.memory.bash_view import MemoryBashTarget
from pm.memory.snapshot import GhostConflict
from pm.memory.store import MemoryStore
from pm.records.cognitive_records import ExpectationRecord
from pm.records.enums import FeatureType, SourceKind
from pm.records.global_workspace import GwtTurnRecord
from pm.records.memory_records import new_feature
from pm.runs.workflow_module import PROJECT_ROOT

from tests.support import ManualClock, Recorder
from workflows.inner_thought.host import PrivateThoughtLoop
from workflows.inner_thought.schemas import InnerReport

MANIFEST = PROJECT_ROOT / "workflows" / "inner_thought" / "inner_thought.workflow.md"


def small_ghost() -> MemoryStore:
    from tests.unit.test_design_contracts import small_ghost as make
    return make()


def _reports(*, operation="curiosity_spark", expectation=None):
    return {
        "trigger": {"trigger": "idle_opportunity", "operation": operation,
                    "temporal_focus": "future", "candidate_ids": [], "urgency": 0.25,
                    "reason": "idle time"},
        "thought": {"thought": "I could reread the old story draft.",
                    "evidence_ids": [], "evidence_summary": "(no cited records)",
                    "interpretation": "the draft matters to me", "uncertainty": 0.4,
                    "reality": "internal", "alternatives": [],
                    "continuation": "summarize", "continuation_reason": "complete for now"},
        "integrate": {"goal_changes": [{"change": "add", "goal": "reread the story draft",
                                        "goal_key": "reread-draft", "reason": "it keeps returning",
                                        "priority": 0.6}],
                      "expectation": expectation or {"action": "none", "expectation_id": None,
                                                     "anticipation": "", "expected_outcome": "",
                                                     "expected_valence": 0.0, "uncertainty": 0.5,
                                                     "goal_ids": [], "status": "pending",
                                                     "observation_ids": [], "reason": ""},
                      "noticing": "the story topic keeps returning",
                      "disclosure": {"mode": "withhold", "shareable_summary": "",
                                     "reason": "not ready to share"},
                      "summary": "a private plan formed"},
    }


def _module(ghost, reports, tmp, stage_called=None):
    def run_stage(name, payload):
        if stage_called is not None:
            stage_called.append(name)
        return dict(reports[name])

    bus = BusSystem()
    errors: list[BaseException] = []
    module = PrivateThoughtLoop(
        "inner_thought", bus, errors.append, ManualClock(), None,
        working_target=MemoryBashTarget(ghost, "read"),
        workflow=MANIFEST,
        tools_factory=lambda environment, batch: {"run_stage": run_stage},
        runtime_root=Path(tmp) / "runtime",
        ghost=ghost,
    )
    bus.subscribe(module.module_id, module.accepts, module.deliver)
    return module, bus, errors


def _report(reports) -> dict:
    """Run the pure pipeline over the stub reports to get a valid parent report."""
    from workflows.inner_thought.pipeline import run_cycle
    from workflows.inner_thought.schemas import InnerRequest
    request = InnerRequest(trigger_candidates=[{"name": "idle_opportunity", "cause": "idle",
                                                "urgency": 0.25, "operations": ["curiosity_spark"],
                                                "reason": "test"}])
    return run_cycle(request, lambda name, payload: reports[name]).model_dump(mode="json")


class IdleCycleTests(unittest.TestCase):
    def test_no_trigger_calls_no_stage_and_commits_nothing(self):
        ghost = small_ghost()
        # A fresh user message suppresses the idle trigger; nothing else fires.
        new_feature(ghost, "hello again", FeatureType.Dialogue, source=ghost.get_user_name())
        stage_called: list[str] = []
        with TemporaryDirectory() as tmp:
            module, bus, errors = _module(ghost, _reports(), tmp, stage_called)
            idle = Recorder(bus, "thought.idle")
            before = ghost.state_dict()
            module.handle_batch([])
            self.assertEqual(stage_called, [])
            self.assertEqual(len(idle.events), 1)
            self.assertEqual(ghost.state_dict(), before)
            self.assertEqual(errors, [])


class AcceptedCycleTests(unittest.TestCase):
    def test_accepted_cycle_commits_inner_shard_and_self_origin_event(self):
        ghost = small_ghost()
        with TemporaryDirectory() as tmp:
            ghost.checkpoint = Path(tmp) / "ghost.json"
            module, bus, errors = _module(ghost, _reports(), tmp)
            created = Recorder(bus, "thought.created")
            updated = Recorder(bus, "gwt.updated")
            module.handle_batch([])
            self.assertEqual(errors, [])
            self.assertEqual(len(created.events), 1)
            thought_id = created.events[0].payload["knoxel_id"]
            thought = ghost.get_knoxel_by_id(thought_id)
            self.assertEqual(thought.feature_type, FeatureType.InnerMonologue)
            self.assertEqual(thought.provenance.source_kind, SourceKind.Self)
            self.assertEqual(thought.metadata["loop"], "inner")
            self.assertEqual(thought.metadata["reality"], "internal")
            tick = created.events[0].payload["gwt_tick"]
            turn = ghost.gwt_turns[tick]
            self.assertEqual(turn.slot_id, "inner")
            self.assertEqual(turn.attention_frame_id, thought.metadata.get("frame") or turn.attention_frame_id)
            cycle = next(item for item in ghost.states if item.tick_id == tick)
            self.assertEqual(cycle.published_slots, ["inner"])
            self.assertEqual(cycle.internal_data["operation"], "curiosity_spark")
            self.assertIn("qualia_channels", cycle.internal_data)
            self.assertEqual(updated.events[0].payload["slot"], "inner")
            # The goal change went through the shared owner with private origin.
            goals = [record for record in ghost.all_knoxels.values()
                     if type(record).__name__ == "Intention" and record.active]
            self.assertTrue(any(record.metadata.get("origin") == "private_loop" for record in goals))

    def test_main_shard_stays_untouched_by_the_inner_loop(self):
        ghost = small_ghost()
        with TemporaryDirectory() as tmp:
            module, bus, errors = _module(ghost, _reports(), tmp)
            module.handle_batch([])
            self.assertEqual(errors, [])
            self.assertEqual(ghost.gwt_shard("main"), [])
            self.assertEqual([item.slot_id for item in ghost.gwt_shard("inner")], ["inner"])

    def test_expectation_open_and_resolve_use_the_lifecycle_helpers(self):
        ghost = small_ghost()
        batch = ghost.begin_batch("seed-expectation")
        seeded = ExpectationRecord(content="Jesse replies tonight", anticipation="a reply",
                                   expected_outcome="a friendly reply", status="pending")
        seeded.tick_id = batch.tick_id
        seeded = batch.create(seeded)
        ghost.commit(batch)
        with TemporaryDirectory() as tmp:
            module, bus, errors = _module(ghost, _reports(), tmp)
            report = InnerReport.model_validate(_report(_reports()))
            report.expectation = type(report.expectation)(
                action="resolve", expectation_id=seeded.id, status="contradicted",
                observation_ids=[seeded.id], reason="no reply came")
            module.last_environment = module.working_target.open("inv-resolve")
            module.prepare_input([])
            module.publish_output(report.model_dump(mode="json"), [])
            self.assertEqual(errors, [])
            stored = ghost.get_knoxel_by_id(seeded.id)
            self.assertEqual(stored.status, "contradicted")
            self.assertEqual(stored.observation_ids, [seeded.id])


class RejectionTests(unittest.TestCase):
    def test_stale_source_rejects_the_cycle_with_a_diagnostic(self):
        ghost = small_ghost()
        # A user message gives the loop a citable candidate, so its revision
        # can be changed mid-flight by another writer.
        new_feature(ghost, "hello again", FeatureType.Dialogue, source=ghost.get_user_name())
        with TemporaryDirectory() as tmp:
            module, bus, errors = _module(ghost, _reports(), tmp)
            rejected = Recorder(bus, "thought.rejected")
            report = InnerReport.model_validate(_report(_reports()))
            module.last_environment = module.working_target.open("inv-stale")
            module.prepare_input([])
            # Another writer changes one of the records the cycle read.
            for record_id in module._read_revisions:
                replaced = ghost.get_knoxel_by_id(record_id)
                replaced.metadata = {**(replaced.metadata or {}), "changed": "mid-flight"}
                other = ghost.begin_batch("mid-flight-writer")
                other.expect_revision(record_id, module._read_revisions[record_id])
                other.replace(replaced)
                ghost.commit(other)
                break
            module.publish_output(report.model_dump(mode="json"), [])
            self.assertEqual(len(rejected.events), 1)
            self.assertIn("GhostConflict", rejected.events[0].payload["reason"])
            self.assertEqual(errors, [])

    def test_cancellation_discards_the_cycle_silently(self):
        ghost = small_ghost()
        with TemporaryDirectory() as tmp:
            module, bus, errors = _module(ghost, _reports(), tmp)
            rejected = Recorder(bus, "thought.rejected")
            created = Recorder(bus, "thought.created")
            report = InnerReport.model_validate(_report(_reports()))
            module.last_environment = module.working_target.open("inv-cancel")
            module.prepare_input([])
            module.stop_event.set()
            module.publish_output(report.model_dump(mode="json"), [])
            self.assertEqual(rejected.events, [])
            self.assertEqual(created.events, [])
            self.assertEqual(ghost.gwt_shard("inner"), [])
            self.assertEqual(errors, [])


class RestartTests(unittest.TestCase):
    def test_inner_shard_and_channels_survive_checkpoint_reopen(self):
        ghost = small_ghost()
        with TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "ghost.json"
            ghost.checkpoint = checkpoint
            module, bus, errors = _module(ghost, _reports(), tmp)
            module.handle_batch([])
            self.assertEqual(errors, [])
            reopened = MemoryStore.from_checkpoint(checkpoint)
            inner = reopened.gwt_shard("inner")
            self.assertEqual(len(inner), 1)
            self.assertEqual(inner[0].slot_id, "inner")
            cycle = next(item for item in reopened.states
                         if item.tick_id == inner[0].tick_id)
            self.assertIn("qualia_channels", cycle.internal_data)
            thoughts = [record for record in reopened.all_features
                        if record.feature_type == FeatureType.InnerMonologue
                        and (record.metadata or {}).get("loop") == "inner"]
            self.assertEqual(len(thoughts), 1)


if __name__ == "__main__":
    unittest.main()
