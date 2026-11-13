"""Bounded private-episode tests against the real store and wait controller.

No model endpoint and no module thread. A deterministic ``run_stage`` stub
supplies the stage reports; the memory store, checkpoint, and wait controller
are real. The stub is a protocol boundary, not behavioral acceptance.

B05 moved fresh private work to the inquiry graph, so each cycle here continues
a retained pre-graph episode seeded by ``tests.support.seed_retained_episode``.
Where a case used to end the linear episode and start a fresh one, it now pins
the routing boundary instead: the cycle must leave the linear pipeline.
:mod:`tests.live.test_private_inquiry` and ``test_graph_host`` cover the graph
side.
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta
from tempfile import TemporaryDirectory
import unittest

from pm.bus.bus import BusSystem
from pm.memory.bash_view import MemoryBashTarget
from pm.records.cycle_records import StimulusGroup
from pm.records.cycle_state import CycleState
from pm.records.enums import FeatureType
from pm.records.global_workspace import GwtTurnRecord
from pm.records.memory_records import new_feature

from tests.support import (ManualClock, PrivateInquiryRouteReached, Recorder,
                            refuse_private_inquiry, seed_retained_episode)
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


def _module(ghost, tmp, reports, config=None, *, retained=True, episode=None):
    if retained:
        seed_retained_episode(ghost, **(episode or {}))

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
            **build_inner_thought_tools(ghost, Path(tmp) / "runtime", None,
                                        inquiry_provider=refuse_private_inquiry)(environment, batch),
            "run_stage": run_stage},
        runtime_root=Path(tmp) / "runtime", ghost=ghost,
    )
    module.cognitive_control = control
    bus.subscribe(module.module_id, module.accepts, module.deliver)
    return module, bus, errors, control


def _acknowledge(ghost, thought_id: int) -> None:
    """Commit one accepted main cycle whose stimuli include the private thought."""
    batch = ghost.begin_batch(f"ack-{thought_id}", new_tick=True)
    batch.set_gwt(
        GwtTurnRecord(tick_id=batch.tick_id, stimulus_ids=[thought_id], knoxel_ids=[thought_id]),
        CycleState(tick_id=batch.tick_id, stimulus_groups=[StimulusGroup(stimulus_ids=[thought_id])]),
    )
    ghost.commit(batch)


def _inner_thoughts(ghost):
    return [record for record in ghost.all_features
            if record.feature_type == FeatureType.InnerMonologue
            and (record.metadata or {}).get("loop") == "inner"]


class EpisodeLifecycleTests(unittest.TestCase):
    def test_continuation_reuses_identity_and_records_lineage(self):
        ghost = small_ghost()
        reports = stage_reports(continuation="summarize", next_operation="", progress="none",
                                alternatives=[{"text": "reread it", "disposition": "chosen",
                                               "reason": "it matters"}])
        with TemporaryDirectory() as tmp:
            seeded = seed_retained_episode(ghost)
            module, bus, errors, _ = _module(ghost, tmp, lambda: reports, retained=False)
            created = Recorder(bus, "thought.created")
            module.handle_batch([])
            continued = ghost.get_knoxel_by_id(created.events[0].payload["knoxel_id"])
            self.assertEqual(continued.metadata["episode"]["episode_id"], "retained-linear")
            self.assertEqual(continued.metadata["episode"]["sequence"], 1)
            self.assertTrue(continued.metadata["episode"]["closed"])
            self.assertEqual(continued.metadata["episode"]["closed_reason"], "summarize")
            self.assertEqual(continued.metadata["episode"]["parent_thought_id"], seeded)
            self.assertEqual(continued.lineage.parent_record_ids, [seeded])
            self.assertEqual(continued.lineage.episode_id, "retained-linear")
            self.assertEqual(errors, [])

    def test_continuation_waits_for_main_acknowledgment(self):
        ghost = small_ghost()
        with TemporaryDirectory() as tmp:
            seeded = seed_retained_episode(ghost, acknowledged=False)
            module, bus, errors, _ = _module(ghost, tmp, lambda: stage_reports(), retained=False)
            # No main cycle acknowledged the retained thought yet, so the host
            # hands the pipeline an episode that must not start new work.
            waiting = module.prepare_input([])
            self.assertTrue(waiting["episode"]["awaiting_acknowledgment"])
            self.assertEqual(waiting["trigger_candidates"], [])
            self.assertEqual([record.id for record in _inner_thoughts(ghost)], [seeded])
            _acknowledge(ghost, seeded)
            released = module.prepare_input([])
            self.assertFalse(released["episode"]["awaiting_acknowledgment"])
            self.assertEqual(errors, [])

    def test_new_control_generation_ends_the_linear_episode_for_good(self):
        ghost = small_ghost()
        with TemporaryDirectory() as tmp:
            seeded = seed_retained_episode(ghost, control_generation=0)
            module, bus, errors, control = _module(ghost, tmp, lambda: stage_reports(),
                                                   retained=False)
            created = Recorder(bus, "thought.created")
            control.generation = 5  # a new outside input supersedes the retained episode
            with self.assertRaises(PrivateInquiryRouteReached):
                module.handle_batch([])
            # The superseded episode cannot continue, and no fresh linear episode starts.
            self.assertEqual(created.events, [])
            self.assertEqual([record.id for record in _inner_thoughts(ghost)], [seeded])
            self.assertEqual(errors, [])

    def test_step_limit_closes_the_episode_explicitly(self):
        ghost = small_ghost()
        with TemporaryDirectory() as tmp:
            seed_retained_episode(ghost, sequence=0)
            module, bus, errors, _ = _module(ghost, tmp, lambda: stage_reports(),
                                             config=ControlConfig(episode_steps=1), retained=False)
            created = Recorder(bus, "thought.created")
            module.handle_batch([])
            only = ghost.get_knoxel_by_id(created.events[0].payload["knoxel_id"])
            self.assertTrue(only.metadata["episode"]["closed"])
            self.assertEqual(only.metadata["episode"]["closed_reason"], "step limit")
            self.assertEqual(errors, [])

    def test_stalled_progress_closes_and_parks_the_sources(self):
        ghost = small_ghost()
        with TemporaryDirectory() as tmp:
            seed_retained_episode(ghost, sequence=0, stalled=1)
            module, bus, errors, _ = _module(
                ghost, tmp, lambda: stage_reports(progress="none"),
                config=ControlConfig(stalled_steps=2), retained=False)
            created = Recorder(bus, "thought.created")
            module.handle_batch([])
            second = ghost.get_knoxel_by_id(created.events[0].payload["knoxel_id"])
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
        older = aging.view().get(source.id).copy_for_edit()
        older.timestamp_world_begin = datetime.utcnow() - timedelta(minutes=10)
        aging.replace(older)
        ghost.commit(aging)
        with TemporaryDirectory() as tmp:
            seed_retained_episode(ghost, closed=True, parked_sources=(source.id,),
                                  parked_until_seconds=600.0)
            module, _, errors, _ = _module(ghost, tmp, lambda: stage_reports(), retained=False)
            module.last_environment = MemoryBashTarget(ghost, "read").open("after-parking")
            request = module.prepare_input([])
            self.assertEqual(request["episode"]["suppressed_source_ids"], [source.id])
            self.assertNotIn("episode_id", request["episode"])
            self.assertEqual(errors, [])

    def test_episode_survives_checkpoint_reopen(self):
        ghost = small_ghost()
        with TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "ghost.pickle"
            ghost.checkpoint = checkpoint
            thought_id = seed_retained_episode(ghost, control_generation=7)
            reopened = type(ghost).from_checkpoint(checkpoint)
            stored = reopened.get_knoxel_by_id(thought_id)
            self.assertFalse(stored.metadata["episode"]["closed"])
            self.assertEqual(stored.metadata["episode"]["sequence"], 0)
            # A restarted loop reads the same retained episode and continues it,
            # so a stop never restarts the episode identity or loses its step.
            resumed, resumed_bus, resumed_errors, _ = _module(reopened, tmp, lambda: stage_reports(),
                                                              retained=False)
            self.assertEqual(resumed.cognitive_control.generation, 7)
            continued = Recorder(resumed_bus, "thought.created")
            resumed.handle_batch([])
            latest = reopened.get_knoxel_by_id(continued.events[-1].payload["knoxel_id"])
            self.assertEqual(latest.metadata["episode"]["episode_id"], stored.metadata["episode"]["episode_id"])
            self.assertEqual(latest.metadata["episode"]["sequence"], 1)
            self.assertEqual(resumed_errors, [])

    def test_acknowledged_episode_can_continue_without_another_trigger(self):
        request = InnerRequest(seconds_since_user_input=1,
            episode={"provenance": "legacy_linear", "awaiting_acknowledgment": False,
                     "state": {"episode_id": "continued", "sequence": 0,
                               "started_at": datetime.utcnow(), "elapsed_seconds": 1.0,
                               "closed": False, "closed_reason": None, "operations": [],
                               "evidence_ids": [], "alternatives": [], "thoughts": [],
                               "progress": [], "stalled": 0, "continuation": "continue",
                               "next_operation": "plan_next_step", "parked_until": None,
                               "parked_sources": [], "parent_thought_id": None}})
        triggers = evaluate_triggers(request)
        self.assertEqual([item.name for item in triggers], ["episode_continuation"])
        self.assertEqual(triggers[0].operations, ["plan_next_step"])

    def test_time_limit_closes_the_durable_state_before_new_work(self):
        ghost = small_ghost()
        with TemporaryDirectory() as tmp:
            first_id = seed_retained_episode(ghost)
            module, _, _, _ = _module(ghost, tmp, lambda: stage_reports(), retained=False)
            aging = ghost.begin_batch("expired-episode")
            aged_episode = dict(aging.view().get(first_id).metadata["episode"])
            aged_episode["started_at"] = (datetime.utcnow() - timedelta(hours=1)).isoformat()
            aging.set_lifecycle(first_id, episode=aged_episode)
            ghost.commit(aging)
            # The expired episode is closed first, so the cycle has no linear work.
            with self.assertRaises(PrivateInquiryRouteReached):
                module.handle_batch([])
            closed = ghost.get_knoxel_by_id(first_id).metadata["episode"]
            self.assertTrue(closed["closed"])
            self.assertEqual(closed["closed_reason"], "time limit")
            self.assertEqual([record.id for record in _inner_thoughts(ghost)], [first_id])


if __name__ == "__main__":
    unittest.main()
