"""Metacognition activity evidence for noticing steps, against real stores.

These deterministic boundary inputs count the recorded facts of recent private
thinking without a model call, and prove that only a noticing step's request
carries them. They make no claim about generated noticing quality; the
re-recorded ``private-dwelling`` scenario covers the real child kernels.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest

from pydantic import JsonValue

from pm import vclock
from pm.memory.store import MemoryStore
from pm.memory.write_batch import WriteBatch
from pm.records.base import Provenance, RecordScope
from pm.records.cycle_state import PersonaConfig
from pm.records.enums import EntityClass, FeatureType, InterlocusType, SourceKind
from pm.records.memory_records import Entity, Feature
from pm.records.mental_state import FullMentalState
from workflows.cognitive_control.attention import captured_private_focus
from workflows.cognitive_control.schemas import ControlConfig
from workflows.inner_thought.graph_execution import PrivateRecall, run_graph_opportunity
from workflows.inner_thought.graph_host import PrivateInquiryOwner
from workflows.inner_thought.inquiry import InquiryRecord, stage_attention
from workflows.inner_thought.metacognition import (
    METACOGNITION_OPERATIONS, activity_rule, render_thought_activity, thought_activity,
)
from workflows.inner_thought.nominations import capture_inquiry_grant, stage_nominations
from workflows.inner_thought.request_views import InquiryThoughtView, RecentThoughtView
from workflows.inner_thought.schemas import (
    InnerRequest, InquiryProgress, InquiryStep, ThoughtAttentionDecision, ThoughtJudgments, ThoughtNomination,
)


def step_view(*, knoxel_id: int, source_ids: tuple[int, ...], loop: str = "inquiry",
              operation: str = "perspective_shift", progress: str = "none",
              outcome: str = "continue", episode: str = "ep-a") -> RecentThoughtView:
    """One typed recent-thought view, newest first in the caller's window."""
    inquiry = InquiryThoughtView(episode_id=episode, outcome=outcome, reason="A recorded step.",
        information_gap="", progress_kind=progress) if loop == "inquiry" else None
    return RecentThoughtView(knoxel_id=knoxel_id, text="A recent private step.", revision=1,
        operation=operation, source_ids=list(source_ids), loop=loop, inquiry=inquiry)


class MetacognitionFixture(unittest.TestCase):
    """A real store with one observer, one source, and the private owner."""

    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path: Path = Path(directory.name) / "metacognition.pickle"
        self.store: MemoryStore = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"),
                                              checkpoint=self.path)
        self.now: datetime = datetime(2026, 11, 14, 9, 0)
        previous_clock = vclock.active()
        self.addCleanup(vclock.install, previous_clock)
        vclock.install(vclock.VirtualClock(self.now))
        anchors = self.store.begin_batch("anchors", embed=False)
        self.observer: Entity = anchors.create(Entity(content="Spamuel", aliases=[], entity_class=EntityClass.Self))
        self.source: Feature = anchors.create(Feature(content='Jesse answers: "Yeah. Maybe."', causal=True,
            feature_type=FeatureType.Dialogue, entity_id=self.observer.id, interlocus=InterlocusType.Public,
            provenance=Provenance(source_kind=SourceKind.User)))
        anchors.preserve_on_recovery()
        self.store.commit(anchors)
        self.owner: PrivateInquiryOwner = PrivateInquiryOwner(self.store, commit=self.commit, cancel=Event(),
                                                              limits=ControlConfig())
        self.judgments: ThoughtJudgments = ThoughtJudgments(interestingness=0.5, character_relevance=0.7,
            plausibility=0.2, consequence=0.4, desirability=0.4, action_suitability=0.2,
            reason="A short reply leaves a question, not an observed motive.")
        self.thought_payload: dict[str, JsonValue] | None = None

    def commit(self, batch: WriteBatch) -> None:
        """Use actual durable production acceptance; the owner marks its own recovery roots."""
        self.store.commit(batch)

    def admit(self, suggested_operation: str) -> InquiryRecord:
        """Admit one spontaneous inquiry whose first operation is the suggested one."""
        grant = capture_inquiry_grant(self.store.snapshot(capture=False), observer_entity_id=self.observer.id,
            scope=RecordScope(), source_ids=[self.source.id], critical_epoch=0)
        batch = self.store.begin_batch("nomination", embed=False)
        staged = stage_nominations(batch, [ThoughtNomination(text="The short reply keeps returning.",
            question="Why does the reply keep returning without new evidence?", evidence_ids=(self.source.id,),
            reality="inferred", judgments=self.judgments, suggested_operation=suggested_operation)],
            grant=grant, batch_id="meta-batch")
        batch.preserve_on_recovery()
        self.commit(batch)
        attention = self.store.begin_batch("attention", embed=False)
        selected = stage_attention(attention, [staged[0].record.id],
            ThoughtAttentionDecision(nomination_id=staged[0].record.id, disposition="select",
                reason="Inspect the returning reply."), episode_id="inquiry:meta-batch", now_utc=self.now)
        attention.preserve_on_recovery()
        self.commit(attention)
        assert selected.inquiry is not None
        return selected.inquiry

    def run_step(self, head: InquiryRecord, *, recent: list[RecentThoughtView]) -> None:
        """Run one real graph opportunity with explicit boundary stage inputs."""
        view = self.store.snapshot(capture=False)
        opportunity_grant = capture_inquiry_grant(view, observer_entity_id=self.observer.id,
            scope=RecordScope(), source_ids=[self.source.id], critical_epoch=0)
        operation = head.receipt.state.next_operation

        def stages(name: str, payload: dict[str, JsonValue]) -> dict[str, JsonValue]:
            if name != "thought":
                raise AssertionError(f"Unexpected stage {name!r}.")
            self.thought_payload = payload
            return InquiryStep(operation=operation, thought="The reply returns without new material.",
                evidence_ids=(self.source.id,), assumptions=(), reality="inferred", judgments=self.judgments,
                relations=(), alternatives=(),
                progress=InquiryProgress(kind="none", inspected_source_ids=(),
                    compared_node_ids=(), meaningful_difference=False, reason="No new original evidence yet."),
                outcome="continue", reason="The step read its recorded activity.", information_gap="",
                next_operation="future_projection").model_dump(mode="json")

        result = run_graph_opportunity(InnerRequest(run_id="meta-run", companion_name="Spamuel",
            recent_thoughts=recent), owner=self.owner, grant=opportunity_grant, run_stage=stages,
            mental_state=FullMentalState(), control_generation=0, pending_input_ids=(),
            focus=captured_private_focus(self.store.snapshot(capture=False), self.now),
            recall_memory=lambda question: PrivateRecall(matches=(), sources=(), trace={}))
        self.assertIsNotNone(result.receipt_id)


class ThoughtActivityTests(unittest.TestCase):
    """Pure counting of a mixed window; no store needed."""

    def test_mixed_window_counts_and_repeats(self) -> None:
        recent = [
            step_view(knoxel_id=500, source_ids=(12,), operation="future_projection", progress="discrimination", episode="ep-c"),
            step_view(knoxel_id=400, source_ids=(10,), loop="inner", operation="recall"),
            step_view(knoxel_id=300, source_ids=(10,), progress="none", episode="ep-a"),
            step_view(knoxel_id=200, source_ids=(11,), progress="new_evidence", episode="ep-b"),
            step_view(knoxel_id=100, source_ids=(10,), progress="none", episode="ep-a"),
        ]
        activity = thought_activity(recent)
        self.assertEqual(activity.window_step_ids, (500, 400, 300, 200, 100))
        self.assertEqual(activity.stalled_step_count, 2)  # 300 and 100; the linear 400 was never stalled.
        self.assertEqual(activity.new_evidence_step_count, 1)
        self.assertEqual(len(activity.repeated), 1)
        repeated = activity.repeated[0]
        self.assertEqual(repeated.source_id, 10)
        self.assertEqual(repeated.step_ids, (400, 300, 100))  # newest first.
        self.assertEqual(repeated.stalled_step_ids, (300, 100))
        self.assertEqual(repeated.episode_ids, ("ep-a",))

    def test_rendered_activity_states_facts_without_cause(self) -> None:
        activity = thought_activity([step_view(knoxel_id=100, source_ids=(10,), progress="none", episode="ep-a")])
        text = render_thought_activity(activity)
        self.assertIn("1 of them accepted no new evidence", text)
        self.assertIn("No source returns in more than one", text)
        rule = activity_rule(text)
        self.assertIn("They are not evidence about anyone else.", rule)
        self.assertIn("Returning to a thought adds no support for it.", rule)

    def test_noticing_operations_are_the_delivered_pair(self) -> None:
        self.assertEqual(METACOGNITION_OPERATIONS, frozenset({"rumination_notice", "attention_drift_notice"}))


class ActivityEvidenceTests(MetacognitionFixture):
    """Only a noticing step's request carries the activity block."""

    def test_noticing_request_carries_its_recorded_activity(self) -> None:
        head = self.admit("rumination_notice")
        self.assertEqual(head.receipt.state.next_operation, "rumination_notice")
        window = [
            step_view(knoxel_id=300, source_ids=(self.source.id,), operation="perspective_shift",
                      progress="none", episode="inquiry:meta-batch"),
            step_view(knoxel_id=200, source_ids=(self.source.id,), operation="recall",
                      progress="new_evidence", episode="inquiry:meta-batch"),
            step_view(knoxel_id=100, source_ids=(self.source.id,), operation="emotional_debrief",
                      progress="none", episode="inquiry:meta-batch"),
        ]
        self.run_step(head, recent=window)
        assert self.thought_payload is not None
        evidence = str(self.thought_payload["activity_evidence"])
        self.assertIn("recorded facts about your own recent private thinking", evidence)
        self.assertIn("[300, 200, 100]", evidence)
        # The echoed envelope leaves the rendered material out of the request
        # body; it stays in the instruction only.
        self.assertNotIn("activity_evidence", self.thought_payload["request"])

    def test_memory_search_request_serializes_no_activity_key(self) -> None:
        head = self.admit("memory_search")
        self.assertEqual(head.receipt.state.next_operation, "memory_search")
        self.run_step(head, recent=[step_view(knoxel_id=100, source_ids=(self.source.id,), progress="none")])
        assert self.thought_payload is not None
        self.assertNotIn("activity_evidence", self.thought_payload)
        self.assertNotIn("activity_evidence", self.thought_payload["request"])


if __name__ == "__main__":
    unittest.main()
