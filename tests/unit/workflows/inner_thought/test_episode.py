"""Bounded private-episode tests against the real store and wait controller.

No model endpoint and no module thread. A deterministic ``run_stage`` stub
supplies the stage reports; the memory store, checkpoint, and wait controller
are real. The stub is a protocol boundary, not behavioral acceptance.
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta
from tempfile import TemporaryDirectory
import unittest

from pm.bus.bus import BusSystem
from pm.memory.bash_view import MemoryBashTarget
from pm.records.cognitive_records import AttentionFrameRecord
from pm.records.cycle_state import CycleState
from pm.records.enums import FeatureType
from pm.records.global_workspace import GwtTurnRecord
from pm.records.memory_records import new_feature

from tests.support import ManualClock, Recorder
from tests.unit.test_design_contracts import small_ghost
from workflows.cognitive_control.host import CognitiveControl
from workflows.cognitive_control.schemas import ControlConfig
from workflows.inner_thought.host import PrivateThoughtLoop, build_inner_thought_tools
from workflows.inner_thought.triggers import evaluate_triggers
from workflows.inner_thought.schemas import InnerRequest

MANIFEST = Path("workflows/inner_thought/inner_thought.workflow.md").resolve()


def stage_reports(*, continuation="continue", next_operation="plan_next_step",
                  progress="none", alternatives=None, evidence=None, goal_changes=None):
    """Return one complete set of deterministic stage reports."""
    return {
        "trigger": {"trigger": "idle_opportunity", "operation": "curiosity_spark",
                    "temporal_focus": "future", "candidate_ids": [], "urgency": 0.25,
                    "reason": "idle time"},
        "thought": {"thought": "I could reread the old story draft.",
                    "evidence_ids": list(evidence or []), "evidence_summary": "(none)",
                    "interpretation": "the draft matters to me", "uncertainty": 0.4,
                    "reality": "internal", "alternatives": list(alternatives or []),
                    "continuation": continuation, "continuation_reason": "test",
                    "next_operation": next_operation, "topic_hints": []},
        "integrate": {"goal_changes": list(goal_changes or []),
                      "expectation": {"action": "none", "expectation_id": None,
                                      "anticipation": "", "expected_outcome": "",
                                      "expected_valence": 0.0, "uncertainty": 0.5,
                                      "goal_ids": [], "status": "pending",
                                      "observation_ids": [], "reason": ""},
                      "noticing": "", "disclosure": {"mode": "withhold",
                                                     "shareable_summary": "", "reason": "private"},
                      "summary": "a private step", "progress": progress,
                      "progress_reason": "test"},
    }


def _module(ghost, tmp, reports, config=None):
    def run_stage(name, payload):
        return dict(reports()[name])

    bus = BusSystem()
    errors: list[BaseException] = []
    control = CognitiveControl("cognitive_control", bus, errors.append, ManualClock(), None,
                               ghost=ghost, config=config or ControlConfig())
    module = PrivateThoughtLoop(
        "inner_thought", bus, errors.append, ManualClock(), None,
        working_target=MemoryBashTarget(ghost, "read"), workflow=MANIFEST,
        tools_factory=lambda environment, batch: {
            **build_inner_thought_tools(ghost, Path(tmp) / "runtime", None)(environment, batch),
            "run_stage": run_stage},
        runtime_root=Path(tmp) / "runtime", ghost=ghost,
    )
    module.cognitive_control = control
    bus.subscribe(module.module_id, module.accepts, module.deliver)
    return module, bus, errors, control


def _acknowledge(ghost, thought_id: int) -> None:
    """Commit one accepted main cycle whose stimuli include the private thought."""
    batch = ghost.begin_batch(f"ack-{thought_id}", new_tick=True)
    frame = batch.create(AttentionFrameRecord(content="main frame", tick_id=batch.tick_id))
    batch.set_gwt(
        GwtTurnRecord(tick_id=batch.tick_id, attention_frame_id=frame.id, knoxel_ids=[frame.id]),
        CycleState(tick_id=batch.tick_id, attention_frame_id=frame.id,
                   published_slots=["main"], stimuli_ids=[thought_id]),
    )
    ghost.commit(batch)


def _inner_thoughts(ghost):
    return [record for record in ghost.all_features
            if record.feature_type == FeatureType.InnerMonologue
            and (record.metadata or {}).get("loop") == "inner"]


class EpisodeLifecycleTests(unittest.TestCase):
    def test_continuation_reuses_identity_and_records_lineage(self):
        ghost = small_ghost()
        reports = stage_reports(alternatives=[{"text": "reread it", "disposition": "chosen",
                                               "reason": "it matters"}])
        with TemporaryDirectory() as tmp:
            module, bus, errors, _ = _module(ghost, tmp, lambda: reports)
            created = Recorder(bus, "thought.created")
            module.handle_batch([])
            first = ghost.get_knoxel_by_id(created.events[0].payload["knoxel_id"])
            self.assertFalse(first.metadata["episode"]["closed"])
            self.assertEqual(first.metadata["episode"]["sequence"], 0)
            episode_id = first.metadata["episode"]["episode_id"]
            self.assertEqual(first.lineage.episode_id, episode_id)
            self.assertEqual(first.lineage.sequence, 0)

            _acknowledge(ghost, first.id)
            reports = stage_reports(continuation="summarize", next_operation="", progress="none")
            module.handle_batch([])
            second = ghost.get_knoxel_by_id(created.events[1].payload["knoxel_id"])
            self.assertEqual(second.metadata["episode"]["episode_id"], episode_id)
            self.assertEqual(second.metadata["episode"]["sequence"], 1)
            self.assertTrue(second.metadata["episode"]["closed"])
            self.assertEqual(second.metadata["episode"]["closed_reason"], "summarize")
            self.assertEqual(second.metadata["episode"]["parent_thought_id"], first.id)
            self.assertEqual(second.lineage.parent_record_ids, [first.id])
            self.assertEqual(second.lineage.episode_id, episode_id)
            self.assertEqual(errors, [])

    def test_continuation_waits_for_main_acknowledgment(self):
        ghost = small_ghost()
        with TemporaryDirectory() as tmp:
            module, bus, errors, _ = _module(ghost, tmp, lambda: stage_reports())
            created = Recorder(bus, "thought.created")
            idle = Recorder(bus, "thought.idle")
            module.handle_batch([])
            first_id = created.events[0].payload["knoxel_id"]
            # No main cycle acknowledged the thought yet: the next tick waits.
            module.handle_batch([])
            self.assertEqual([record.id for record in _inner_thoughts(ghost)], [first_id])
            self.assertEqual(len(idle.events), 1)
            _acknowledge(ghost, first_id)
            module.handle_batch([])
            self.assertEqual(len(_inner_thoughts(ghost)), 2)
            self.assertEqual(errors, [])

    def test_new_generation_interrupts_the_open_episode(self):
        ghost = small_ghost()
        with TemporaryDirectory() as tmp:
            module, bus, errors, control = _module(ghost, tmp, lambda: stage_reports())
            created = Recorder(bus, "thought.created")
            module.handle_batch([])
            first = ghost.get_knoxel_by_id(created.events[0].payload["knoxel_id"])
            control.generation += 1  # a new outside input supersedes the episode
            module.handle_batch([])
            second = ghost.get_knoxel_by_id(created.events[1].payload["knoxel_id"])
            self.assertNotEqual(second.metadata["episode"]["episode_id"],
                                first.metadata["episode"]["episode_id"])
            self.assertEqual(second.metadata["episode"]["sequence"], 0)
            self.assertEqual(errors, [])

    def test_step_limit_closes_the_episode_explicitly(self):
        ghost = small_ghost()
        with TemporaryDirectory() as tmp:
            module, bus, errors, _ = _module(ghost, tmp, lambda: stage_reports(),
                                             config=ControlConfig(episode_steps=1))
            created = Recorder(bus, "thought.created")
            module.handle_batch([])
            only = ghost.get_knoxel_by_id(created.events[0].payload["knoxel_id"])
            self.assertTrue(only.metadata["episode"]["closed"])
            self.assertEqual(only.metadata["episode"]["closed_reason"], "step limit")
            self.assertEqual(errors, [])

    def test_stalled_progress_closes_and_parks_the_sources(self):
        ghost = small_ghost()
        with TemporaryDirectory() as tmp:
            module, bus, errors, _ = _module(
                ghost, tmp, lambda: stage_reports(progress="none"),
                config=ControlConfig(stalled_steps=2))
            created = Recorder(bus, "thought.created")
            module.handle_batch([])
            first = ghost.get_knoxel_by_id(created.events[0].payload["knoxel_id"])
            self.assertFalse(first.metadata["episode"]["closed"])
            _acknowledge(ghost, first.id)
            module.handle_batch([])
            second = ghost.get_knoxel_by_id(created.events[1].payload["knoxel_id"])
            self.assertTrue(second.metadata["episode"]["closed"])
            self.assertEqual(second.metadata["episode"]["closed_reason"], "stalled progress")
            self.assertIsNotNone(second.metadata["episode"]["parked_until"])
            self.assertEqual(errors, [])

    def test_parked_source_suppresses_the_unresolved_goal_trigger(self):
        request = InnerRequest(
            tick_id=100,
            active_goal_records=[{"knoxel_id": 5, "text": "finish the story", "tick_id": 80,
                                  "fulfilment": 0.1, "affective_valence": -0.2, "revision": 1}],
            episode={"suppressed_source_ids": [5]},
        )
        names = [item.name for item in evaluate_triggers(request)]
        self.assertNotIn("unresolved_commitment", names)

    def test_closed_episode_keeps_parked_sources_in_the_next_host_request(self):
        ghost = small_ghost()
        source = new_feature(ghost, "the old draft is still unfinished", FeatureType.Dialogue,
                             source=ghost.get_user_name())
        aging = ghost.begin_batch("older-dialogue")
        older = aging.view().get(source.id)
        older.timestamp_world_begin = datetime.utcnow() - timedelta(minutes=10)
        aging.replace(older)
        ghost.commit(aging)
        with TemporaryDirectory() as tmp:
            module, _, errors, _ = _module(
                ghost, tmp, lambda: stage_reports(continuation="suspend", evidence=[source.id]))
            module.handle_batch([])
            module.last_environment = MemoryBashTarget(ghost, "read").open("after-parking")
            request = module.prepare_input([])
            self.assertEqual(request["episode"]["suppressed_source_ids"], [source.id])
            self.assertNotIn("episode_id", request["episode"])
            self.assertEqual(errors, [])

    def test_episode_survives_checkpoint_reopen(self):
        ghost = small_ghost()
        with TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "ghost.json"
            ghost.checkpoint = checkpoint
            module, bus, errors, _ = _module(ghost, tmp, lambda: stage_reports())
            module.cognitive_control.generation = 7
            created = Recorder(bus, "thought.created")
            module.handle_batch([])
            thought_id = created.events[0].payload["knoxel_id"]
            reopened = type(ghost).from_checkpoint(checkpoint)
            stored = reopened.get_knoxel_by_id(thought_id)
            self.assertFalse(stored.metadata["episode"]["closed"])
            self.assertEqual(stored.metadata["episode"]["sequence"], 0)
            pending, pending_bus, _, _ = _module(reopened, tmp, lambda: stage_reports())
            replayed = Recorder(pending_bus, "thought.created")
            pending.resume()
            pending.resume()
            self.assertEqual([event.payload["knoxel_id"] for event in replayed.events], [thought_id])
            _acknowledge(reopened, thought_id)
            pending.release_results(replayed.events)
            pending.resume()
            self.assertEqual(len(replayed.events), 1)
            resumed, _, _, restored_control = _module(reopened, tmp, lambda: stage_reports())
            self.assertEqual(restored_control.generation, 7)
            resumed.handle_batch([])
            latest = _inner_thoughts(reopened)[-1]
            self.assertEqual(latest.metadata["episode"]["episode_id"], stored.metadata["episode"]["episode_id"])
            self.assertEqual(latest.metadata["episode"]["sequence"], 1)

    def test_acknowledged_episode_can_continue_without_another_trigger(self):
        request = InnerRequest(seconds_since_user_input=1,
            episode={"episode_id": "continued", "next_operation": "plan_next_step",
                     "awaiting_acknowledgment": False})
        triggers = evaluate_triggers(request)
        self.assertEqual([item.name for item in triggers], ["episode_continuation"])
        self.assertEqual(triggers[0].operations, ["plan_next_step"])

    def test_time_limit_closes_the_durable_state_before_new_work(self):
        ghost = small_ghost()
        with TemporaryDirectory() as tmp:
            module, _, _, _ = _module(ghost, tmp, lambda: stage_reports())
            module.handle_batch([])
            first = _inner_thoughts(ghost)[-1]
            aging = ghost.begin_batch("expired-episode")
            expired = aging.view().get(first.id)
            expired.metadata["episode"]["started_at"] = (datetime.utcnow() - timedelta(hours=1)).isoformat()
            aging.replace(expired)
            ghost.commit(aging)
            module.handle_batch([])
            closed = ghost.get_knoxel_by_id(first.id).metadata["episode"]
            self.assertTrue(closed["closed"])
            self.assertEqual(closed["closed_reason"], "time limit")


if __name__ == "__main__":
    unittest.main()
