"""A18 private-loop inspection: trigger, prompt rendering, and belief commit.

No model endpoint. The capability overlay is synthetic; the records and the
commit boundary are real.
"""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pm.bus.bus import BusSystem
from pm.memory.bash_view import MemoryBashTarget
from pm.memory.store import MemoryStore
from pm.records.cognitive_records import BeliefRecord
from pm.runs.workflow_module import PROJECT_ROOT

from tests.support import ManualClock
from workflows.inner_thought.host import PrivateThoughtLoop
from workflows.inner_thought.schemas import InnerReport, InnerRequest
from workflows.inner_thought.stages.thought.steps import _render_inspection
from workflows.inner_thought.triggers import evaluate_triggers

MANIFEST = PROJECT_ROOT / "workflows" / "inner_thought" / "inner_thought.workflow.md"


def small_ghost() -> MemoryStore:
    from tests.unit.test_design_contracts import small_ghost as make
    return make()


def _inspection(*, stale=False, state="implemented"):
    return {
        "status": "ok",
        "capability_overlay": {
            "available": True,
            "capabilities": [
                {"name": "source_inspection", "state": state, "summary": "Read selected source.",
                 "revision": "rev1"},
                {"name": "minecraft_body", "state": "observed", "summary": "Act in the world.",
                 "revision": "rev2"},
            ],
        },
        "episodes": [{"episode_id": 7, "loop": "inner", "operation": "memory_search",
                      "outcome": "committed", "summary": "a private recollection"}],
        "newest_episode_answers": {},
        "stale": stale,
    }


class CapabilityDoubtTriggerTests(unittest.TestCase):
    def test_doubt_fires_when_a_capability_is_only_planned(self):
        request = InnerRequest(tick_id=10, inspection=_inspection(state="implemented"))
        names = {item.name for item in evaluate_triggers(request)}
        self.assertIn("capability_doubt", names)

    def test_doubt_stays_silent_when_every_capability_is_observed(self):
        inspection = _inspection(state="observed")
        request = InnerRequest(tick_id=10, inspection=inspection)
        names = {item.name for item in evaluate_triggers(request)}
        self.assertNotIn("capability_doubt", names)

    def test_no_inspection_means_no_doubt(self):
        request = InnerRequest(tick_id=10)
        names = {item.name for item in evaluate_triggers(request)}
        self.assertNotIn("capability_doubt", names)


class InspectionRenderTests(unittest.TestCase):
    def test_render_lists_capabilities_and_episodes(self):
        rendered = _render_inspection(InnerRequest(inspection=_inspection()))
        self.assertIn("capability=source_inspection", rendered)
        self.assertIn("episode=7", rendered)

    def test_missing_inspection_is_an_explicit_limit(self):
        self.assertIn("no implementation evidence", _render_inspection(InnerRequest()))

    def test_failed_inspection_is_reported(self):
        rendered = _render_inspection(InnerRequest(inspection={"status": "unavailable",
                                                               "reason": "SourceDenied: no grant"}))
        self.assertIn("inspection unavailable", rendered)


class CapabilityBeliefCommitTests(unittest.TestCase):
    def _module(self, ghost, tmp, overlay):
        bus = BusSystem()
        errors: list[BaseException] = []
        module = PrivateThoughtLoop(
            "inner_thought", bus, errors.append, ManualClock(), None,
            working_target=MemoryBashTarget(ghost, "read"),
            workflow=MANIFEST,
            tools_factory=lambda environment, batch: {},
            runtime_root=Path(tmp) / "runtime",
            ghost=ghost,
        )
        module.inspector_provider = lambda: overlay
        bus.subscribe(module.module_id, module.accepts, module.deliver)
        return module, bus, errors

    def test_capability_inspection_commits_a_revisable_belief(self):
        from pm.introspection.capabilities import (
            CapabilityContext, CapabilityDeclaration, build_overlay,
        )
        ghost = small_ghost()
        overlay = build_overlay(
            [CapabilityDeclaration(name="source_inspection", summary="Read selected source.")],
            CapabilityContext(),
        )
        with TemporaryDirectory() as tmp:
            module, bus, errors = self._module(ghost, tmp, overlay)
            module.last_environment = module.working_target.open("inv-cap-commit")
            module.prepare_input([])
            report = InnerReport(run_id="r", operation="capability_inspection",
                                 thought="I can read my own wiring, and I can see what stays unobserved.",
                                 summary="capability inspection", reality="inferred")
            module.publish_output(report.model_dump(mode="json"), [])
            self.assertEqual(errors, [])
            beliefs = [record for record in ghost.all_knoxels.values()
                       if isinstance(record, BeliefRecord)
                       and (record.metadata or {}).get("memory_kind") == "capability_belief"]
            self.assertEqual(len(beliefs), 1)
            self.assertEqual(beliefs[0].metadata["state"], "uncertain")
            thought = next(record for record in ghost.all_features
                           if record.feature_type.value == "InnerMonologue")
            self.assertIn("capability_states", thought.metadata)

    def test_a_different_operation_commits_no_capability_belief(self):
        from pm.introspection.capabilities import (
            CapabilityContext, CapabilityDeclaration, build_overlay,
        )
        ghost = small_ghost()
        overlay = build_overlay(
            [CapabilityDeclaration(name="source_inspection", summary="Read selected source.")],
            CapabilityContext(),
        )
        with TemporaryDirectory() as tmp:
            module, bus, errors = self._module(ghost, tmp, overlay)
            module.last_environment = module.working_target.open("inv-cap-skip")
            module.prepare_input([])
            report = InnerReport(run_id="r", operation="curiosity_spark",
                                 thought="I want to reread the draft.", summary="a plan",
                                 reality="internal")
            module.publish_output(report.model_dump(mode="json"), [])
            self.assertEqual(errors, [])
            beliefs = [record for record in ghost.all_knoxels.values()
                       if isinstance(record, BeliefRecord)
                       and (record.metadata or {}).get("memory_kind") == "capability_belief"]
            self.assertEqual(beliefs, [])


if __name__ == "__main__":
    unittest.main()
