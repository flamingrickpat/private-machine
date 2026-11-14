"""Perspective-inquiry evidence against real stores and the real entity owner.

These deterministic boundary inputs exercise the theory-of-mind wiring: the
accepted model of the subject is captured without a model call, the original
messages behind it become citable candidates and extend the step grant, and
every other step's serialized request stays unchanged. They make no claim
about generated thought quality; the recorded ``perspective-morning`` scenario
covers the real child kernels.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from typing import Literal
import unittest

from pydantic import JsonValue

from pm import vclock
from pm.memory.store import MemoryStore
from pm.memory.write_batch import WriteBatch
from pm.records.base import Provenance, RecordScope
from pm.records.cycle_state import PersonaConfig
from pm.records.enums import ActionType, EntityClass, FeatureType, FrameEnum, InterlocusType, SourceKind
from pm.records.memory_records import ActionSchemaRecord, Entity, Feature
from pm.records.mental_state import FullMentalState
from workflows.cognitive_control.attention import captured_private_focus
from workflows.cognitive_control.schemas import ControlConfig
from workflows.entity_context.owner import EntityContextOwner
from workflows.entity_context.schemas import (
    EntityClaimEvidence, EntityClaimView, EntityContextClaim, EntityContextKey, EntityContextProposal,
    EntityProjectionQuery,
)
from workflows.entity_context.projection import project_entity_context
from workflows.inner_thought.graph_execution import (
    PrivateRecall, graph_thought_request, run_graph_opportunity,
)
from workflows.inner_thought.graph_host import PrivateInquiryOwner, accepted_source_ids
from workflows.inner_thought.inquiry import InquiryRecord
from workflows.inner_thought.nominations import capture_inquiry_grant
from workflows.inner_thought.perspective import (
    PerspectiveContext, capture_perspective_context, perspective_rule, render_perspective_context,
)
from workflows.inner_thought.schemas import (
    DeliberateAdmission, InnerRequest, InquiryGrant, InquiryProgress, InquiryStep, ThoughtJudgments,
)
from pm.memory.dependencies import capture_source


class PerspectiveFixture(unittest.TestCase):
    """A real store with one observer, one subject, staged claims, and the private owner."""

    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path: Path = Path(directory.name) / "perspective.pickle"
        self.store: MemoryStore = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"),
                                              checkpoint=self.path)
        self.now: datetime = datetime(2026, 11, 14, 9, 0)
        previous_clock = vclock.active()
        self.addCleanup(vclock.install, previous_clock)
        vclock.install(vclock.VirtualClock(self.now))
        anchors = self.store.begin_batch("anchors", embed=False)
        self.observer: Entity = anchors.create(Entity(content="Spamuel", aliases=[], entity_class=EntityClass.Self))
        self.partner: Entity = anchors.create(Entity(content="Jesse", aliases=[], entity_class=EntityClass.Human))
        self.peer: Entity = anchors.create(Entity(content="Mara", aliases=[], entity_class=EntityClass.AI))
        anchors.preserve_on_recovery()
        self.store.commit(anchors)
        self.entity_owner: EntityContextOwner = EntityContextOwner(self.store)
        self.limits: ControlConfig = ControlConfig()
        self.owner: PrivateInquiryOwner = PrivateInquiryOwner(self.store, commit=self.commit, cancel=Event(),
                                                              limits=self.limits)
        self.key: EntityContextKey = EntityContextKey(subject_entity_id=self.partner.id,
            observer_entity_id=self.observer.id, scope=RecordScope())
        self.thought_payload: dict[str, JsonValue] | None = None
        self.judgments: ThoughtJudgments = ThoughtJudgments(interestingness=0.5, character_relevance=0.6,
            plausibility=0.3, consequence=0.4, desirability=0.4, action_suitability=0.2,
            reason="A morning routine is an estimate, not an observation.")
        #: Two original Jesse lines behind the accepted morning claims.
        self.workday_line = self.report("I am usually up around six on workdays.")
        self.weekend_line = self.report("Weekends do not count, and nothing before coffee.")

    def commit(self, batch: WriteBatch) -> None:
        """Use actual durable production acceptance; the owner marks its own recovery roots."""
        self.store.commit(batch)

    def report(self, text: str, *, when: datetime | None = None,
               scope: RecordScope = RecordScope()) -> Feature:
        """Commit an attributed external utterance through the production store."""
        moment = when if when is not None else self.now
        batch = self.store.begin_batch(f"report-{len(self.store.all_features)}", embed=False)
        record = batch.create(Feature(content=text, feature_type=FeatureType.Dialogue, causal=True,
            entity_id=self.partner.id, addressee_entity_id=self.observer.id, interlocus=InterlocusType.Public,
            scope=scope, timestamp_world_begin=moment, timestamp_world_end=moment,
            provenance=Provenance(source_kind=SourceKind.User)))
        batch.preserve_on_recovery()
        self.store.commit(batch)
        return record

    def claim(self, source: Feature, proposal_id: str, *, claim_key: str, statement: str,
              key: EntityContextKey | None = None) -> EntityContextClaim:
        """Stage one accepted attributed claim through the real owner and store."""
        selected_key = key if key is not None else self.key
        view = self.store.snapshot(capture=False)
        evidence = EntityClaimEvidence(source=capture_source(view, source.id, current_head=True),
                                       exact_quote=source.content)
        claim = EntityContextClaim(layer="pattern", claim_key=claim_key, statement=statement,
            epistemic_kind="report", confidence=0.9, evidence=[evidence],
            observed_at_utc=source.timestamp_world_begin,
            applicable_from_utc=source.timestamp_world_begin)
        proposal = EntityContextProposal(proposal_id=proposal_id, key=selected_key,
            parent_receipt_id=self.entity_owner.current_head(view, selected_key), producer="immediate",
            input_commit_ids=sorted(view.commit_ids), claims=[claim])
        with self.store.lock:
            batch = self.store.begin_batch(f"accept-{proposal_id}", embed=False)
            result = self.entity_owner.stage_update(batch, proposal)
            if result.disposition != "staged":
                raise AssertionError(f"Fixture claim was not staged: {result}")
            batch.preserve_on_recovery()
            self.store.commit(batch)
        return claim

    def admit(self, instruction: str, *, subject_entity_id: int | None, action_id: str = "action-tom",
              correlation: str = "corr-tom") -> InquiryRecord:
        """Open one deliberate private episode through the real owner path."""
        batch = self.store.begin_batch("dispatch", embed=False)
        action = batch.create(ActionSchemaRecord(content="I want to think this through on my own.",
            action_type=ActionType.Loopback, provenance=Provenance(source_kind=SourceKind.Workflow),
            metadata={"action_id": action_id, "workflow_name": "theory_of_mind",
                      "dispatch_status": "pending", "dispatch_target": "theory_of_mind",
                      "dispatch_correlation": correlation, "evidence_ids": [self.workday_line.id]}))
        batch.preserve_on_recovery()
        self.store.commit(batch)
        view = self.store.snapshot(capture=False)
        grant = capture_inquiry_grant(view, observer_entity_id=self.observer.id, scope=RecordScope(),
            source_ids=[action.id, self.workday_line.id], critical_epoch=0)
        admission = DeliberateAdmission(action_record_id=action.id, instruction=instruction,
            evidence_ids=(self.workday_line.id,), operation="memory_search",
            reason="Main cognition selected this private mental work.",
            subject_entity_id=subject_entity_id)
        return self.owner.admit_deliberate_action(admission, grant=grant, now_utc=self.now)

    def run_step(self, head: InquiryRecord, *, evidence_ids: tuple[int, ...]) -> None:
        """Run one real graph opportunity with explicit boundary stage inputs."""
        view = self.store.snapshot(capture=False)
        opportunity_grant = capture_inquiry_grant(view, observer_entity_id=self.observer.id,
            scope=RecordScope(), source_ids=[head.receipt.state.origin_record_id], critical_epoch=0)

        def thought_stage(name: str, payload: dict[str, JsonValue]) -> dict[str, JsonValue]:
            """Record the thought payload and return one explicit boundary step."""
            if name != "thought":
                raise AssertionError(f"Unexpected stage {name!r}.")
            self.thought_payload = payload
            return InquiryStep(operation="memory_search", thought="The morning lines agree so far.",
                evidence_ids=list(evidence_ids), assumptions=(), reality="inferred",
                judgments=self.judgments, relations=(), alternatives=(),
                progress=InquiryProgress(kind="none", inspected_source_ids=(),
                    compared_node_ids=(), meaningful_difference=False, reason="No new original source yet."),
                outcome="continue", reason="A first step reads the subject's accepted model.", information_gap="",
                next_operation="perspective_shift").model_dump(mode="json")

        result = run_graph_opportunity(InnerRequest(run_id="perspective-run", companion_name="Spamuel"),
            owner=self.owner, grant=opportunity_grant, run_stage=thought_stage,
            mental_state=FullMentalState(), control_generation=0, pending_input_ids=(),
            focus=captured_private_focus(self.store.snapshot(capture=False), self.now),
            recall_memory=lambda question: PrivateRecall(matches=(), sources=(), trace={}))
        self.assertIsNotNone(result.receipt_id)

    def grant_source_ids(self) -> set[int]:
        """Return the source ids the recorded thought payload could cite."""
        assert self.thought_payload is not None
        return {int(item["record_id"]) for item in self.thought_payload["grant"]["sources"]}

    def candidate_ids(self) -> set[int]:
        assert self.thought_payload is not None
        return {int(item["knoxel_id"]) for item in self.thought_payload["request"]["candidates"]}


class PerspectiveContextTests(PerspectiveFixture):
    """The accepted model of the subject and the sources behind it."""

    def test_subject_with_two_accepted_claims_supplies_both_lines_and_sources(self) -> None:
        self.claim(self.workday_line, "morning-workday", claim_key="routine.workday",
            statement="Jesse is usually up around six on workdays.")
        self.claim(self.weekend_line, "morning-weekend", claim_key="routine.weekend",
            statement="Weekends do not count and nothing happens before coffee.")
        head = self.admit("Work out when Jesse is usually available in the morning.",
                          subject_entity_id=self.partner.id)
        evidence_ids = (self.workday_line.id, self.weekend_line.id)
        self.run_step(head, evidence_ids=evidence_ids)
        assert self.thought_payload is not None
        perspective_text = str(self.thought_payload["perspective_evidence"])
        self.assertIn("Jesse is usually up around six on workdays.", perspective_text)
        self.assertIn("Weekends do not count and nothing happens before coffee.", perspective_text)
        self.assertIn("Subject Jesse (entity " + str(self.partner.id), perspective_text)
        self.assertTrue(set(evidence_ids).issubset(self.grant_source_ids()))
        self.assertTrue(set(evidence_ids).issubset(self.candidate_ids()))
        kinds = {item["kind"] for item in self.thought_payload["request"]["candidates"]
                 if int(item["knoxel_id"]) in evidence_ids}
        self.assertIn("entity_evidence", kinds)  # The weekend line is only a perspective source.
        self.assertTrue({int(item["knoxel_id"]) for item in self.thought_payload["request"]["candidates"]
                         if item["kind"] == "entity_evidence"}.issubset(evidence_ids))

    def test_subject_without_any_accepted_claim_says_so_and_adds_no_sources(self) -> None:
        head = self.admit("Work out what Mara is usually doing.", subject_entity_id=self.peer.id)
        self.run_step(head, evidence_ids=(self.workday_line.id,))
        assert self.thought_payload is not None
        perspective_text = str(self.thought_payload["perspective_evidence"])
        self.assertIn("No accepted entry exists for this subject.", perspective_text)
        extra = self.candidate_ids() - {self.workday_line.id}
        self.assertEqual(extra, set())

    def test_source_in_another_frame_is_not_evidence(self) -> None:
        foreign = self.report("On the table the meta line stands alone.", scope=RecordScope(frame=FrameEnum.Meta))
        self.claim(foreign, "meta-note", claim_key="meta.note", statement="A meta claim.",
                   key=EntityContextKey(subject_entity_id=self.partner.id, observer_entity_id=self.observer.id,
                                        scope=RecordScope(frame=FrameEnum.Meta)))
        view = self.store.snapshot(capture=False)
        grant = capture_inquiry_grant(view, observer_entity_id=self.observer.id, scope=RecordScope(),
            source_ids=[self.workday_line.id], critical_epoch=0)
        perspective = capture_perspective_context(view, subject_entity_id=self.partner.id, grant=grant,
            accepted_ids=accepted_source_ids(self.store), as_of_utc=self.now)
        self.assertNotIn(foreign.id, perspective.evidence_source_ids)
        self.assertNotIn(self.observer.id, perspective.evidence_source_ids)
        self.assertNotIn(self.partner.id, perspective.evidence_source_ids)

    def test_plain_deliberate_episode_serializes_no_perspective_key(self) -> None:
        head = self.admit("Work out why Jesse stopped describing the story.", subject_entity_id=None,
                          action_id="action-focus", correlation="corr-focus")
        view = self.store.snapshot(capture=False)
        execution_grant = self.owner.execution_grant(head)
        supplied = graph_thought_request(view, InnerRequest(run_id="focus-run", companion_name="Spamuel"),
                                         head, execution_grant)
        dumped = supplied.model_dump(mode="json")
        self.assertNotIn("perspective_evidence", dumped)


class PerspectiveTextTests(unittest.TestCase):
    """Pure text contracts; no store needed."""

    def test_perspective_rule_keeps_roles_and_contact_separate(self) -> None:
        text = perspective_rule("PEGGY: ...", companion_name="Spamuel", subject_name="Jesse")
        self.assertIn("This inquiry models Jesse as Spamuel understands them.", text)
        self.assertIn("A report is what was said.", text)
        self.assertIn("The companion's knowledge is not the subject's knowledge.", text)
        self.assertIn("This inquiry cannot contact the subject.", text)
        self.assertIn("PEGGY: ...", text)


if __name__ == "__main__":
    unittest.main()
