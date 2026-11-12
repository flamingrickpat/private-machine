"""Actual main commit and ordinary reselect boundaries without model substitution.

The CycleWork values are explicit host-boundary inputs. The separate live gate
exercises model extraction; these checks establish atomic state and recovery.
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta
from typing import Literal
from tempfile import TemporaryDirectory
import time
import unittest

from pm import vclock
from pm.bus.bus import BusSystem
from pm.bus.envelopes import Envelope, new_envelope
from pm.context_render.story import record_view
from pm.memory.bash_view import MemoryBashEnvironment, MemoryBashTarget
from pm.memory.dependencies import capture_source
from pm.memory.store import MemoryStore
from pm.memory.store_tools import build_tools
from pm.records.cycle_records import StimulusGroup
from pm.records.cycle_state import CycleState, PersonaConfig
from pm.records.enums import EntityClass
from pm.records.global_workspace import GwtTurnRecord
from pm.records.memory_records import ActionSchemaRecord, Entity
from pm.records.workspace import ShardKey
from workflows.cognitive_control.host import CognitiveControl
from workflows.entity_context.owner import EntityContextOwner, active_claims
from workflows.entity_context.schemas import EntityClaimDraft, EntityExtractionDraft, InteractionClaim
from workflows.input_triage.entity_updates import capture_entity_extraction, hydrate_entity_extraction
from workflows.input_triage.schemas import EarlyTriageAssessment, InputSourcePayload, TriageEvidence
from workflows.interaction.host import Interaction
from workflows.interaction.schemas import InteractionDecision, InteractionOutcome, InteractionReceipt, InteractionRequest
from workflows.main_cognition.cycle_work import CycleRequest, CycleWork
from workflows.main_cognition.entity_updates import EntityActionMetadata
from workflows.main_cognition.host import MainCogLoop
from workflows.main_cognition.schemas import ActionDispatch, BeliefUpdate, PendingDispatch, RouteChoice, Stimulus, TriageItem, TriageResult
from workflows.public_speech.situational_guidance import build_situational_guidance
from workflows.public_speech.user_input import UserInputProxy
from pm.memory.dependencies import source_fingerprint


ROOT = Path(__file__).resolve().parents[4]


class MainBoundary(MainCogLoop):
    """Expose the real host's deterministic finished-kernel acceptance boundary."""

    ghost: MemoryStore

    def accept_boundary(self, work: CycleWork, envelope: Envelope, environment: MemoryBashEnvironment) -> None:
        """Set the exact frozen invocation evidence, then call production publication."""
        assert environment.ghost_view is not None
        self.last_environment = environment
        self._memory_view = environment.ghost_view
        self._stimulus_ids = [item.knoxel_id for item in work.request.stimuli]
        self._stimulus_dependencies = {key: capture_source(environment.ghost_view, key, current_head=True)
                                       for key in self._stimulus_ids}
        self.invocation_id = "main-boundary-" + str(len(self.ghost.states))
        self.publish_output(work.model_dump(mode="json"), [envelope])


class MainAndReselectTests(unittest.TestCase):
    """Exercise the accepted owner and real memory batches at concurrency boundaries."""

    def setUp(self) -> None:
        previous_clock = vclock.active()
        self.clock = vclock.VirtualClock(datetime(2026, 11, 12, 10, 0))
        vclock.install(self.clock)
        self.addCleanup(vclock.install, previous_clock)
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"))
        for name, kind in (("Spamuel", EntityClass.Self), ("Jesse", EntityClass.Human)):
            self.store.store_knoxel(Entity(content=name, aliases=[], entity_class=kind))
        self.self_id = self.store.get_entity_id("Spamuel")
        self.partner_id = self.store.get_entity_id("Jesse")
        assert self.self_id is not None and self.partner_id is not None
        initial = self.store.begin_batch("foundation", new_tick=True, embed=False)
        initial.depend_on(self.store.revisions)
        initial.set_gwt(GwtTurnRecord(tick_id=initial.tick_id), CycleState(tick_id=initial.tick_id))
        self.store.commit(initial)
        self.bus = BusSystem()
        self.events: list[Envelope] = []
        self.bus.subscribe("evidence", lambda envelope: True, self.events.append)
        self.owner = EntityContextOwner(self.store)
        self.input = UserInputProxy("user_input", self.bus, self.fail, time.monotonic, self.store, lambda text: None)
        self.target = MemoryBashTarget(self.store, recovery_only=True)
        self.main = MainBoundary("main_cog", self.bus, self.fail, time.monotonic, None,
            ghost=self.store, working_target=self.target, tools_factory=build_tools,
            workflow=ROOT / "workflows/main_cognition/main_cognition.workflow.md", runtime_root=Path(self.directory.name))
        self.main.entity_context_owner = self.owner

    def submit(self, text: str) -> Envelope:
        correlation = self.input.submit(text)
        source = next(item for item in self.store.all_features if item.metadata.get("exchange_correlation") == correlation)
        return new_envelope("user_input", "data", "user.input", correlation, {"knoxel_id": source.id})

    def maintain(self, envelope: Envelope, statement: str, ability: Literal["can", "cannot", "unknown"], *,
                 review_after_utc: datetime | None = None) -> int:
        """Apply an explicit semantic boundary input through real capture and owner staging."""
        source_id = InputSourcePayload.model_validate(envelope.payload).knoxel_id
        view = self.store.recovery_snapshot(capture=False)
        assert self.partner_id is not None and self.self_id is not None
        extraction = capture_entity_extraction(self.owner, view, subject_ids=[self.partner_id],
            observer_entity_id=self.self_id, source_ids=[source_id], evidence_ids=[source_id],
            as_of_utc=vclock.utcnow(), token_budget=4096)
        previous = active_claims(view, key=extraction.families[0].key)
        draft = EntityClaimDraft(subject_entity_id=self.partner_id, source_ids=[source_id], exact_quote=statement,
            layer="situation", claim_key="availability.respond", statement=statement, epistemic_kind="report",
            confidence=1.0, interaction=InteractionClaim(dimension="respond", ability=ability), review_after_utc=review_after_utc,
            correction_target_ids=[item.claim_id for item in previous])
        proposal = hydrate_entity_extraction(extraction, EntityExtractionDraft(claims=[draft]), producer="immediate")[0]
        batch = self.store.begin_batch("maintain-" + str(source_id), embed=False)
        batch.preserve_on_recovery()
        result = self.owner.stage_update(batch, proposal)
        self.assertEqual(result.disposition, "staged")
        self.store.commit(batch)
        assert result.staged_receipt_id is not None
        return result.staged_receipt_id

    def work(self, envelope: Envelope, environment: MemoryBashEnvironment, *, reply: bool = False) -> CycleWork:
        """Build a typed finished-kernel boundary input without claiming model generation."""
        view = environment.ghost_view
        assert view is not None and self.partner_id is not None
        source_id = InputSourcePayload.model_validate(envelope.payload).knoxel_id
        source = view.get(source_id)
        key = ShardKey(kind="entity", ref=str(self.partner_id))
        state = CycleState(tick_id=0, stimulus_groups=[StimulusGroup(key=key, stimulus_ids=[source_id],
                                                                  correlation_id=envelope.correlation_id)])
        state.workspace.ensure_shard(key)
        request = CycleRequest(companion_name="Spamuel", user_name="Jesse", self_entity_id=self.self_id,
            user_entity_id=self.partner_id, entity_names={self.partner_id: "Jesse"},
            stimuli=[Stimulus(knoxel_id=source_id, source="user_input", text=source.content,
                              correlation_id=envelope.correlation_id, entity_id=self.partner_id)],
            records={source_id: record_view(source)}, situational_guidance=[build_situational_guidance(view,
                as_of_utc=vclock.utcnow(), companion_name="Spamuel", user_name="Jesse")])
        work = CycleWork(request=request, state=state)
        work.affect.felt = "The report matters."
        work.experience.text = "The character considers the new report."
        work.experience.record_id = -2
        if reply:
            work.dispatches = [ActionDispatch(action_id="boundary-public-choice", workflow_name="dialogue_story",
                action_type="Reply", content="The character answers the report.", group=key, stimulus_ids=[source_id],
                correlation_id=envelope.correlation_id, status="dispatched")]
        return work

    def test_stale_entity_proposal_preserves_actual_main_appraisal(self) -> None:
        source = self.submit("I cannot answer for now.")
        frozen = self.target.open("before-direct-update")
        work = self.work(source, frozen)
        source_id = work.request.stimuli[0].knoxel_id
        work.context.belief_updates = [BeliefUpdate(source_id=source_id, quote="I cannot answer for now.",
                                                  belief_key="availability.respond", entity_layer="situation")]
        self.maintain(source, "I cannot answer for now.", "cannot")
        self.main.accept_boundary(work, source, frozen)
        selection = next(item for item in reversed(self.store.all_features) if item.metadata.get("operation") == "action_selection")
        self.assertEqual(selection.metadata["entity_update_outcomes"][0]["disposition"], "deferred")
        self.assertEqual(selection.metadata["entity_update_outcomes"][0]["reason"], "stale_parent")
        self.assertEqual(sum(source_id in state.claimed_ids() for state in self.store.states), 1)
        self.assertTrue(any(item.metadata.get("origin") == "appraisal" for item in self.store.all_features))

    def test_stale_selected_direction_records_new_reconsideration_cause(self) -> None:
        source = self.submit("I cannot answer for now.")
        self.maintain(source, "I cannot answer for now.", "cannot")
        frozen = self.target.open("old-direction")
        work = self.work(source, frozen, reply=True)
        returned = self.submit("I can answer now.")
        self.maintain(returned, "I can answer now.", "can")
        self.main.accept_boundary(work, source, frozen)
        self.assertFalse(any(event.topic == "action.request" for event in self.events))
        markers = [item for item in self.store.all_features if "entity_action_reconsideration" in item.metadata]
        self.assertEqual(len(markers), 1)
        marker = EntityActionMetadata.model_validate(markers[0].metadata).entity_action_reconsideration
        action = self.store.snapshot({marker.action_record_id}).get(marker.action_record_id)
        self.assertIsInstance(action, ActionSchemaRecord)
        self.assertEqual(action.metadata["dispatch_status"], "deferred")
        self.assertEqual(PendingDispatch.model_validate(action.metadata).dispatch_payload.gwt_tick, self.store.states[-1].tick_id)
        self.assertEqual(sum(event.topic == "entity.action_reconsider" for event in self.events), 1)
        self.assertEqual(sum(work.request.stimuli[0].knoxel_id in state.claimed_ids() for state in self.store.states), 1)

    def test_own_staged_entity_correction_defers_old_direction_and_updates_shard(self) -> None:
        first = self.submit("I cannot answer for now.")
        self.maintain(first, "I cannot answer for now.", "cannot")
        returned = self.submit("I can answer now.")
        frozen = self.target.open("before-main-correction")
        work = self.work(returned, frozen, reply=True)
        context = work.request.situational_guidance[0].partner
        assert context is not None
        prior = context.entity_contexts[0]
        work.context.belief_updates = [BeliefUpdate(source_id=work.request.stimuli[0].knoxel_id,
            quote="I can answer now.", belief_key="availability.respond", correction=True,
            correction_target_ids=[item.claim.claim_id for item in prior.claims], entity_layer="situation")]
        self.main.accept_boundary(work, returned, frozen)
        selection = next(item for item in reversed(self.store.all_features) if item.metadata.get("operation") == "action_selection")
        self.assertEqual(selection.metadata["entity_update_outcomes"][0]["disposition"], "staged")
        self.assertTrue(selection.metadata["entity_deferred_actions"])
        self.assertFalse(any(event.topic == "action.request" for event in self.events))
        staged_id = selection.metadata["entity_update_outcomes"][0]["staged_receipt_id"]
        assert self.partner_id is not None
        shard = self.store.states[-1].workspace.entity_shards[self.partner_id]
        self.assertIn(staged_id, shard.item_ids())

    def test_stale_ordinary_reply_gets_new_semantic_selection_identity(self) -> None:
        source = self.submit("I cannot answer for now.")
        self.maintain(source, "I cannot answer for now.", "cannot", review_after_utc=vclock.utcnow() + timedelta(minutes=1))
        control = CognitiveControl("control", self.bus, self.fail, time.monotonic, None, ghost=self.store)
        control.enable_input_triage()
        source_id = InputSourcePayload.model_validate(source.payload).knoxel_id
        view = self.store.recovery_snapshot(capture=False)
        fingerprints = {source_id: source_fingerprint(view.get(source_id))}
        control.accept_triage(EarlyTriageAssessment(source_ids=[source_id], source_fingerprints=fingerprints,
            guidance_tick=self.store.states[-1].tick_id,
            evidence=TriageEvidence(record_fingerprints=fingerprints, commit_ids=sorted(view.commit_ids), current_head_ids=[source_id]),
            result=TriageResult(items=[TriageItem(knoxel_id=source_id, category="Moderate", reasoning="Boundary input.")])))
        module = Interaction("interaction", self.bus, self.fail, time.monotonic, None, ghost=self.store,
            working_target=self.target, tools_factory=build_tools, runtime_root=Path(self.directory.name),
            workflow=ROOT / "workflows/interaction/interaction.workflow.md")
        module.cognitive_control = control
        module.workflow_catalogue_provider = lambda: []
        module.last_environment = self.target.open("ordinary-old")
        request = InteractionRequest.model_validate(module.prepare_input([source]))
        key = ShardKey(kind="entity", ref=str(self.partner_id))
        receipt = InteractionReceipt(selection_id=request.selection_id, source_ids=[source_id], outcome=InteractionOutcome.Reply,
            decision=InteractionDecision(variants=[]), route=RouteChoice(action_type="Reply", workflow_name="dialogue_story"),
            dispatch=ActionDispatch(action_type="Reply", workflow_name="dialogue_story", group=key, stimulus_ids=[source_id]),
            dependencies=request.dependencies)
        module.accept_selection(receipt)
        self.clock.set(vclock.utcnow() + timedelta(minutes=2))
        self.assertEqual(module.remaining_inputs([source]), [source])
        module.refresh_selection(request.selection_id)
        self.assertEqual(module.mailbox.pending(), 1)
        module.last_environment = self.target.open("ordinary-new")
        refreshed = InteractionRequest.model_validate(module.prepare_input([source]))
        self.assertNotEqual(request.selection_id, refreshed.selection_id)
