"""A06 state-path tests: affect, goals, evidence, identity, and lifecycle.

No live model, module thread, or Minecraft server. The real parent kernel, the
real module adapter, and the real MemoryStore commit boundary are exercised with a
deterministic ``run_stage`` stub.
"""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pm.records.memory_records import Feature
from pm.records.enums import FeatureSubtype, FeatureType, SourceKind
from pm.records.mental_state import FullMentalState

from pm.bus.bus import BusSystem
from pm.bus.envelopes import Envelope, new_id
from workflows.main_cognition.host import InnerThoughtLoop, MainCogLoop
from pm.records.memory_records import new_feature
from pm.memory.snapshot import GhostConflict
from pm.memory.store import MemoryStore
from pm.records.cycle_state import PersonaConfig
from pm.app.lifecycle_records import CLEAN_END, UNCERTAIN_END, record_shutdown, record_startup
from tests.support import ManualClock, Recorder
from pm.runs.workflow_module import PROJECT_ROOT
from workflows.main_cognition.dispatch import resolve_prerequisites
from workflows.main_cognition.pipeline import run_cycle
from workflows.main_cognition.schemas import ActionDispatch
from workflows.main_cognition.stage_commits import find_intention, normalize_goal_key
from pm.memory.bash_view import MemoryBashTarget

REAL_MANIFEST = PROJECT_ROOT / "workflows" / "main_cognition" / "main_cognition.workflow.md"
INNER_MANIFEST = PROJECT_ROOT / "workflows" / "inner_thought.workflow.md"


def _small_ghost():
    from tests.unit.test_design_contracts import small_ghost
    return small_ghost()


def _store(ghost, content: str) -> int:
    return ghost.store_knoxel(Feature(
        content=content, feature_type=FeatureType.Dialogue, causal=True,
        override_source=ghost.get_user_name(),
    )).id


def _reports(stimulus_id: int, *, emotion=None, need=None, goals=None, actions=None) -> dict:
    return {
        "triage": {"items": [{"knoxel_id": stimulus_id, "category": "Moderate"}], "summary": "t"},
        "appraisal": {
            "feeling": "warm",
            "emotion_delta": dict(emotion or {}),
            "need_delta": dict(need or {}),
            "reasoning": "Jesse shared something good.",
            "evidence_ids": [stimulus_id],
            "summary": "a",
        },
        "context": {"memories": [], "candidate_ids": [], "summary": "c"},
        "attention": {"selected_ids": [stimulus_id], "nominations": [], "summary": "n"},
        "goals": {"changes": list(goals or []), "summary": "g"},
        "workspace": {"frame_reason": "reply", "focus_ids": [stimulus_id], "summary": "w"},
        "action_selection": {
            "actions": list(actions or [{
                "action_type": "Reply", "workflow_name": "dialogue_story",
                "content": "reply to Jesse", "arguments": {}, "depends_on": [],
            }]),
            "summary": "s",
        },
    }


def _module(ghost, reports, tmp, module_id="main_cog"):
    def run_stage(name, payload):
        return dict(reports[name])

    bus = BusSystem()
    errors: list[BaseException] = []
    module = MainCogLoop(
        module_id, bus, errors.append, ManualClock(), None,
        working_target=MemoryBashTarget(ghost, "read"),
        workflow=REAL_MANIFEST,
        tools_factory=lambda environment, batch: {"run_stage": run_stage},
        runtime_root=Path(tmp) / "runtime",
        ghost=ghost,
    )
    bus.subscribe(module.module_id, module.accepts, module.deliver)
    return module, bus, errors


class AppraisalCommitTests(unittest.TestCase):
    def test_real_appraisal_changes_persisted_affect(self):
        ghost = _small_ghost()
        stimulus_id = _store(ghost, "I finished the build!")
        reports = _reports(stimulus_id, emotion={"joy": 0.6, "pride": 0.4}, need={"connection": 0.3})
        with TemporaryDirectory() as tmp:
            module, bus, errors = _module(ghost, reports, tmp)
            requests = Recorder(bus, "action.request")
            bus.put(Envelope(id=new_id(), source="user_input", kind="data", topic="user.input",
                             correlation_id="corr-joy", payload={"knoxel_id": stimulus_id}))
            self.assertTrue(module.step(block=False))

            self.assertEqual(errors, [])
            self.assertEqual(len(requests.events), 1)
            state = ghost.states[-1].mental_state_latent_vec
            mental = FullMentalState.init_from_list(state)
            self.assertGreater(mental.state_emotions.joy, 0.0)
            self.assertAlmostEqual(mental.state_needs.connection, 0.3)
            appraisals = [
                knoxel for knoxel in ghost.all_features
                if knoxel.feature_subtype == FeatureSubtype.Appraisal
            ]
            self.assertEqual(len(appraisals), 1)
            self.assertEqual(appraisals[0].provenance.source_kind, SourceKind.Self)
            self.assertGreater(appraisals[0].mental_state_appraisal.state_emotions.joy, 0.0)

    def test_repeated_real_publish_reuses_committed_ids(self):
        ghost = _small_ghost()
        stimulus_id = _store(ghost, "hello")
        reports = _reports(stimulus_id, emotion={"joy": 0.4},
                           goals=[{"change": "add", "goal": "build a house"}])
        with TemporaryDirectory() as tmp:
            module, bus, errors = _module(ghost, reports, tmp)
            requests = Recorder(bus, "action.request")
            envelope = Envelope(
                id=new_id(), source="user_input", kind="data", topic="user.input",
                correlation_id="corr-repeat", payload={"knoxel_id": stimulus_id},
            )
            request = module.prepare_input([envelope])

            def run_stage(name, payload):
                return dict(reports[name])

            report = run_cycle(request, run_stage).model_dump(mode="json")
            module.invocation_id = "same-real-turn"
            module.publish_output(report, [envelope])
            accepted = set(ghost.all_knoxels)
            module.publish_output(report, [envelope])
            self.assertEqual(set(ghost.all_knoxels), accepted)
            for event in requests.events:
                self.assertIn(event.payload["knoxel_id"], accepted)

    def test_pipeline_report_carries_appraisal_and_goal_changes(self):
        stimulus_id = 11
        reports = _reports(stimulus_id, goals=[{"change": "add", "goal": "build a house"}])

        def run_stage(name, payload):
            return dict(reports[name])

        report = run_cycle({"stimuli": [{"knoxel_id": stimulus_id, "kind": "user.input"}]}, run_stage)
        self.assertIsNotNone(report.appraisal)
        self.assertEqual(report.appraisal.feeling, "warm")
        self.assertEqual([change.goal for change in report.goal_changes], ["build a house"])
        self.assertEqual(report.event_origins[str(stimulus_id)], "user.input")


class GoalLifecycleTests(unittest.TestCase):
    def test_goal_add_then_complete_changes_active_lifecycle(self):
        ghost = _small_ghost()
        stimulus_id = _store(ghost, "let us build a house")
        add = _reports(stimulus_id, goals=[{"change": "add", "goal": "build a house"}])
        with TemporaryDirectory() as tmp:
            module, bus, errors = _module(ghost, add, tmp)
            bus.put(Envelope(id=new_id(), source="user_input", kind="data", topic="user.input",
                             correlation_id="corr-add", payload={"knoxel_id": stimulus_id}))
            self.assertTrue(module.step(block=False))

            key = normalize_goal_key("build a house")
            intention = find_intention(ghost, key)
            self.assertIsNotNone(intention)
            self.assertEqual(intention.status, "active")
            self.assertIn(intention.id, ghost.gwt_turns[ghost.current_tick_id].knoxel_ids)

        complete = _reports(stimulus_id, goals=[{"change": "complete", "goal": "build a house"}])
        with TemporaryDirectory() as tmp:
            module, bus, errors = _module(ghost, complete, tmp)
            bus.put(Envelope(id=new_id(), source="user_input", kind="data", topic="user.input",
                             correlation_id="corr-done", payload={"knoxel_id": stimulus_id}))
            self.assertTrue(module.step(block=False))

            self.assertEqual(errors, [])
            self.assertIsNone(find_intention(ghost, key))
            stored = ghost.get_knoxel_by_id(intention.id)
            self.assertFalse(stored.active)
            self.assertEqual(stored.status, "completed")

    def test_refuse_and_defer_are_distinct_durable_lifecycle_states(self):
        from workflows.main_cognition.schemas import GoalChange
        from workflows.main_cognition.stage_commits import apply_goal_changes

        ghost = _small_ghost()
        batch = ghost.begin_batch("lifecycle")
        records = apply_goal_changes(batch, batch.view(), [
            GoalChange(change="refuse", goal="give away my diamonds", reason="they are mine"),
            GoalChange(change="defer", goal="explore the cave", reason="it is night"),
        ])
        ghost.commit(batch)
        self.assertEqual([record["result"] for record in records], ["refused", "deferred"])
        by_status = {item.status: item for item in ghost.all_intentions}
        self.assertEqual(set(by_status), {"refused", "deferred"})
        self.assertTrue(all(not item.active for item in ghost.all_intentions))
        # A refused goal keeps its reason as evidence.
        self.assertEqual(by_status["refused"].metadata["reason"], "they are mine")

    def test_old_intention_without_key_resolves_by_text(self):
        ghost = _small_ghost()
        from pm.records.enums import IntentionType
        from pm.records.cognitive_records import Intention
        stored = ghost.store_knoxel(Intention(
            content="build a house", entity_id=0, affective_valence=0.0,
            intention_type=IntentionType.WorldState,
        ))
        found = find_intention(ghost, normalize_goal_key("Build  A House"))
        self.assertIsNotNone(found)
        self.assertEqual(found.id, stored.id)

    def test_unknown_change_kind_does_not_mutate_state(self):
        from workflows.main_cognition.schemas import GoalChange
        from workflows.main_cognition.stage_commits import apply_goal_changes
        ghost = _small_ghost()
        batch = ghost.begin_batch("bad-goal")
        records = apply_goal_changes(batch, batch.view(), [GoalChange(change="delete", goal="anything")])
        self.assertEqual(records[0]["result"], "missing")
        ghost.commit(batch)
        self.assertEqual(ghost.all_intentions, [])

    def test_stale_batch_does_not_partly_apply_affect_or_goals(self):
        from pm.records.enums import IntentionType
        from pm.records.cognitive_records import Intention
        from workflows.main_cognition.schemas import AppraisalResult, GoalChange
        from workflows.main_cognition.stage_commits import apply_appraisal, apply_goal_changes

        ghost = _small_ghost()
        goal = ghost.store_knoxel(Intention(
            content="build a house", entity_id=0, affective_valence=0.0,
            intention_type=IntentionType.WorldState,
        ))
        stale = ghost.begin_batch("stale-turn", new_tick=True)
        apply_appraisal(
            stale, stale.view(), AppraisalResult(feeling="proud", emotion_delta={"pride": 0.7}),
            previous_vec=None, evidence_ids=[goal.id], companion_name="Spamuel",
        )
        apply_goal_changes(stale, stale.view(), [GoalChange(change="complete", goal="build a house")])

        # A newer batch changes the same intention before the stale batch commits.
        newer = ghost.begin_batch("newer-turn")
        current = ghost.get_knoxel_by_id(goal.id)
        current.incentive_salience = 0.9
        newer.replace(current)
        ghost.commit(newer)

        with self.assertRaises(GhostConflict):
            ghost.commit(stale)
        stored = ghost.get_knoxel_by_id(goal.id)
        self.assertTrue(stored.active)
        self.assertEqual(stored.status, "active")
        self.assertEqual(len(ghost.states), 0)
        self.assertEqual(
            [k for k in ghost.all_features if k.feature_subtype == FeatureSubtype.Appraisal], []
        )

    def test_active_intention_survives_checkpoint_restart(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "ghost.json"
            ghost = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"), checkpoint=path)
            stimulus_id = _store(ghost, "let us build a house")
            reports = _reports(stimulus_id, goals=[{"change": "add", "goal": "build a house"}])
            module, bus, errors = _module(ghost, reports, tmp)
            bus.put(Envelope(id=new_id(), source="user_input", kind="data", topic="user.input",
                             correlation_id="corr-save", payload={"knoxel_id": stimulus_id}))
            self.assertTrue(module.step(block=False))
            self.assertEqual(errors, [])
            self.assertIsNotNone(find_intention(ghost, normalize_goal_key("build a house")))

            reopened = MemoryStore.from_checkpoint(path)
            found = find_intention(reopened, normalize_goal_key("build a house"))
            self.assertIsNotNone(found)
            self.assertEqual(found.status, "active")


class PrerequisiteTests(unittest.TestCase):
    def test_queued_prerequisite_is_not_released_by_dispatch(self):
        reports = _reports(11, actions=[
            {"action_id": "start", "action_type": "ToolCall", "workflow_name": "self_management",
             "content": "start the house", "arguments": {}, "depends_on": []},
            {"action_id": "celebrate", "action_type": "Reply", "workflow_name": "dialogue_story",
             "content": "the house is done", "arguments": {}, "requires_completed": ["start"]},
        ])

        def run_stage(name, payload):
            return dict(reports[name])

        report = run_cycle({"stimuli": [{"knoxel_id": 11, "kind": "user.input"}]}, run_stage)
        by_id = {dispatch.action_id: dispatch for dispatch in report.dispatches}
        self.assertEqual(by_id["start"].status, "dispatched")
        self.assertEqual(by_id["celebrate"].status, "queued")

        dispatch = ActionDispatch(action_id="celebrate", workflow_name="dialogue_story",
                                  content="the house is done", requires_completed=["start"])
        released = resolve_prerequisites([dispatch], {"start": "completed"})
        self.assertEqual(released[0].status, "proposed")
        rejected = resolve_prerequisites([dispatch], {"start": "failed"})
        self.assertEqual(rejected[0].status, "rejected")


class DuplicateIdentityTests(unittest.TestCase):
    def test_repeated_start_identity_does_not_start_another_task(self):
        ghost = _small_ghost()
        stimulus_id = _store(ghost, "start the house task")
        action = {
            "action_id": "start-house", "action_type": "ToolCall",
            "workflow_name": "self_management", "content": "start house task",
            "arguments": {"task_id": "house"}, "depends_on": [],
        }
        reports = _reports(stimulus_id, actions=[action])
        with TemporaryDirectory() as tmp:
            module, bus, errors = _module(ghost, reports, tmp)
            requests = Recorder(bus, "action.request")
            receipts = Recorder(bus, "action.receipt")
            for _ in range(2):
                bus.put(Envelope(id=new_id(), source="user_input", kind="data", topic="user.input",
                                 correlation_id="corr-start", payload={"knoxel_id": stimulus_id}))
                self.assertTrue(module.step(block=False))

            self.assertEqual(errors, [])
            self.assertEqual(len(requests.events), 1)
            self.assertTrue(any(event.payload.get("status") == "duplicate" for event in receipts.events))


class EventOriginTests(unittest.TestCase):
    def test_workflow_event_keeps_progress_and_evidence(self):
        ghost = _small_ghost()
        with TemporaryDirectory() as tmp:
            module, bus, errors = _module(ghost, _reports(0), tmp)
            envelope = Envelope(
                id=new_id(), source="managed_run", kind="telemetry", topic="workflow.event",
                correlation_id="corr-task",
                payload={"run_id": "run-7", "event": "completed", "detail": "placed 4 blocks",
                         "artifact_path": "evidence/run-7.txt"},
            )
            knoxel_id, text, kind = module._stimulus_from(envelope)
            self.assertEqual(kind, "workflow.event")
            self.assertIn("placed 4 blocks", text)
            feature = ghost.get_knoxel_by_id(knoxel_id)
            self.assertEqual(feature.metadata["run_id"], "run-7")
            self.assertEqual(feature.metadata["artifact_path"], "evidence/run-7.txt")
            self.assertEqual(feature.provenance.source_kind, SourceKind.Workflow)

    def test_thought_event_is_self_origin(self):
        ghost = _small_ghost()
        with TemporaryDirectory() as tmp:
            bus = BusSystem()
            errors: list[BaseException] = []
            module = InnerThoughtLoop(
                "inner_thought", bus, errors.append, ManualClock(), None,
                working_target=MemoryBashTarget(ghost, "read"),
                workflow=INNER_MANIFEST,
                tools_factory=lambda environment, batch: {},
                runtime_root=Path(tmp) / "runtime",
                ghost=ghost,
            )
            module.publish_output({"thought": "I wonder whether Jesse is proud."}, [])
            thought = [
                knoxel for knoxel in ghost.all_features
                if knoxel.feature_type == FeatureType.InnerMonologue
            ][0]
            self.assertEqual(thought.provenance.source_kind, SourceKind.Self)


class LifecycleTests(unittest.TestCase):
    def test_abrupt_startup_reports_uncertain_gap(self):
        ghost = _small_ghost()
        boot = record_startup(ghost, boot_id="boot-a")
        self.assertEqual(boot["end_kind"], UNCERTAIN_END)
        feature = ghost.get_knoxel_by_id(boot["knoxel_id"])
        self.assertEqual(feature.feature_type, FeatureType.SystemMessage)
        self.assertEqual(feature.provenance.source_kind, SourceKind.System)
        self.assertIn("uncertain", feature.content)

    def test_clean_stop_then_restart_reports_clean_gap(self):
        ghost = _small_ghost()
        record_startup(ghost, boot_id="boot-a")
        record_shutdown(ghost, boot_id="boot-a")
        restart = record_startup(ghost, boot_id="boot-b")
        self.assertEqual(restart["end_kind"], CLEAN_END)
        self.assertIn("clean stop", ghost.get_knoxel_by_id(restart["knoxel_id"]).content)

    def test_repeated_boot_identity_does_not_duplicate_events(self):
        ghost = _small_ghost()
        first = record_startup(ghost, boot_id="boot-a")
        before = len(ghost.all_features)
        second = record_startup(ghost, boot_id="boot-a")
        self.assertTrue(second["duplicate"])
        self.assertEqual(second["knoxel_id"], first["knoxel_id"])
        self.assertEqual(len(ghost.all_features), before)

    def test_lifecycle_stimulus_enters_the_first_cycle(self):
        ghost = _small_ghost()
        boot = record_startup(ghost, boot_id="boot-cycle")
        with TemporaryDirectory() as tmp:
            module, bus, errors = _module(ghost, _reports(boot["knoxel_id"]), tmp)
            envelope = Envelope(
                id=new_id(), source="system", kind="data", topic="system.lifecycle",
                correlation_id="", payload={"knoxel_id": boot["knoxel_id"]},
            )
            knoxel_id, text, kind = module._stimulus_from(envelope)
            self.assertEqual(kind, "system.lifecycle")
            self.assertEqual(knoxel_id, boot["knoxel_id"])
            self.assertIn("started again", text)


if __name__ == "__main__":
    unittest.main()
