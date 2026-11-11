"""Real-store selection, source, and restart checks without model substitution.

Receipts are explicit boundary inputs. These checks do not certify generated
choices. The live gate exercises the actual selection and speech kernels.
"""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest

from pm.bus.bus import BusSystem
from pm.bus.envelopes import Envelope, new_envelope
from pm.memory.bash_view import MemoryBashTarget
from pm.memory.snapshot import GhostConflict
from pm.memory.store import MemoryStore
from pm.memory.store_tools import build_tools
from pm.records.cycle_records import StimulusGroup
from pm.records.cycle_state import CycleState, PersonaConfig
from pm.records.enums import EntityClass, FeatureType
from pm.records.global_workspace import GwtTurnRecord
from pm.records.memory_records import Entity, Feature
from workflows.cognitive_control.host import CognitiveControl
from workflows.input_triage.schemas import EarlyTriageAssessment, InputSourcePayload, TriageEvidence
from workflows.interaction.host import Interaction
from workflows.interaction.records import interaction_owned_sources, interaction_receipts
from workflows.interaction.schemas import InteractionDecision, InteractionOutcome, InteractionReceipt, InteractionRequest
from workflows.interaction.steps import prepare_work
from workflows.main_cognition.schemas import ActionDispatch, RouteChoice, TriageItem, TriageResult
from workflows.public_speech.publication_control import ReplySuperseded
from workflows.public_speech.user_input import UserInputProxy
from pm.memory.dependencies import source_fingerprint


ROOT = Path(__file__).resolve().parents[4]


class InteractionBoundaryTests(unittest.TestCase):
    """Exercise actual store commits and checkpoint recovery at the host boundary."""

    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"))
        self.bus = BusSystem()
        for name, kind in (("Spamuel", EntityClass.Self), ("Jesse", EntityClass.Human)):
            self.store.store_knoxel(Entity(content=name, aliases=[], entity_class=kind))
        self.accept_main([], [])
        self.control = CognitiveControl("control", self.bus, self.fail, time.monotonic, None, ghost=self.store)
        self.control.enable_input_triage()
        self.input = UserInputProxy("user_input", self.bus, self.fail, time.monotonic, self.store, lambda text: None)
        self.input.register_triage_input = self.control.register_input
        self.target = MemoryBashTarget(self.store, recovery_only=True)
        self.module = Interaction("interaction", self.bus, self.fail, time.monotonic, None,
            ghost=self.store, working_target=self.target,
            workflow=ROOT / "workflows/interaction/interaction.workflow.md",
            runtime_root=Path(self.directory.name), tools_factory=build_tools)
        self.module.cognitive_control = self.control
        self.module.workflow_catalogue_provider = lambda: []

    def accept_main(self, source_ids: list[int], interaction_ids: list[int]) -> None:
        """Commit a deterministic boundary state without claiming cognitive generation."""
        batch = self.store.begin_batch(f"main-{len(self.store.states)}", new_tick=True, embed=False)
        batch.depend_on(self.store.revisions)
        decision = batch.create(Feature(content="Boundary input ownership", feature_type=FeatureType.StateUpdate,
                                       metadata={"interaction_source_ids": interaction_ids}))
        state = CycleState(tick_id=batch.tick_id, stimulus_groups=[StimulusGroup(stimulus_ids=source_ids)])
        batch.set_gwt(GwtTurnRecord(tick_id=batch.tick_id, stimulus_ids=source_ids,
                                   decision_id=decision.id), state)
        self.store.commit(batch)

    def submit(self, text: str, category: str = "Moderate") -> Envelope:
        """Use actual durable intake and supply classification as a named boundary input."""
        correlation = self.input.submit(text)
        source = next(record for record in self.store.all_features
                      if record.metadata.get("exchange_correlation") == correlation)
        view = self.store.recovery_snapshot(capture=False)
        fingerprints = {source.id: source_fingerprint(source)}
        self.control.accept_triage(EarlyTriageAssessment(source_ids=[source.id], source_fingerprints=fingerprints,
            guidance_tick=self.store.states[-1].tick_id,
            evidence=TriageEvidence(record_fingerprints=fingerprints, commit_ids=sorted(view.commit_ids),
                                    current_head_ids=[source.id]),
            result=TriageResult(items=[TriageItem(knoxel_id=source.id, category=category, reasoning="Boundary input.")])))
        return new_envelope("user_input", "data", "user.input", correlation, {"knoxel_id": source.id})

    def request(self, envelopes: list[Envelope]) -> InteractionRequest:
        """Open the real frozen Bash view, then use production context preparation."""
        self.module.last_environment = self.target.open("boundary-request")
        return InteractionRequest.model_validate(self.module.prepare_input(envelopes))

    def quiet_receipt(self, request: InteractionRequest) -> InteractionReceipt:
        """Construct an explicit valid host-boundary input, not a model result."""
        return InteractionReceipt(selection_id=request.selection_id,
            source_ids=[item.knoxel_id for item in request.cycle.stimuli], outcome=InteractionOutcome.Quiet,
            decision=InteractionDecision(variants=[]), dependencies=request.dependencies)

    def test_quiet_root_survives_default_clean_restart_without_a_main_turn(self) -> None:
        path = Path(self.directory.name) / "ghost.pickle"
        self.store.checkpoint = path
        envelope = self.submit("Ordinary input.")
        request = self.request([envelope])
        tick = self.store.current_tick_id
        result = self.module.accept_selection(self.quiet_receipt(request))
        self.assertEqual(self.store.current_tick_id, tick)
        reopened = MemoryStore.from_checkpoint(path)
        receipts = interaction_receipts(reopened.recovery_snapshot(capture=False))
        self.assertEqual(receipts[request.cycle.stimuli[0].knoxel_id].outcome, InteractionOutcome.Quiet)
        self.assertEqual(reopened.applied_commits[result.commit_id], result)

    def test_completed_main_appraisal_does_not_steal_ordinary_selection(self) -> None:
        envelope = self.submit("New ordinary input.")
        source_id = InputSourcePayload.model_validate(envelope.payload).knoxel_id
        self.accept_main([source_id], [source_id])
        self.assertEqual(self.module.remaining_inputs([envelope]), [envelope])
        self.assertIn(source_id, interaction_owned_sources(self.store.recovery_snapshot(capture=False)))

    def test_cold_start_main_selected_input_is_not_selected_again(self) -> None:
        envelope = self.submit("Cold-start input.")
        source_id = InputSourcePayload.model_validate(envelope.payload).knoxel_id
        self.accept_main([source_id], [])
        self.assertEqual(self.module.remaining_inputs([envelope]), [])

    def test_new_partner_input_rejects_old_selection_without_effects(self) -> None:
        first = self.submit("First ordinary input.")
        request = self.request([first])
        self.submit("Further input before selection acceptance.")
        with self.assertRaises(GhostConflict):
            self.module.accept_selection(self.quiet_receipt(request))
        self.assertEqual(interaction_receipts(self.store.recovery_snapshot(capture=False)), {})

    def test_critical_epoch_rejects_old_ordinary_choice(self) -> None:
        envelope = self.submit("An ordinary input.")
        request = self.request([envelope])
        self.submit("Consequential new evidence.", category="Critical")
        with self.assertRaises(ReplySuperseded):
            self.module.accept_selection(self.quiet_receipt(request))

    def test_prepare_work_preserves_accepted_state_without_appraisal(self) -> None:
        request = self.request([self.submit("Current actual event.")])
        previous = request.cycle.previous_state
        assert previous is not None
        work = prepare_work(request)
        self.assertEqual(work.state.mental_state, previous.mental_state)
        self.assertEqual(work.goals.changes, [])
        self.assertEqual(work.phases, [])
        self.assertEqual(work.request.stimuli[0].text, "Current actual event.")

    def test_new_exchange_coalesces_selected_but_unsent_reply(self) -> None:
        first = self.submit("First input still needs a reply.")
        request = self.request([first])
        group = prepare_work(request).state.stimulus_groups[0]
        receipt = InteractionReceipt(selection_id=request.selection_id,
            source_ids=[item.knoxel_id for item in request.cycle.stimuli], outcome=InteractionOutcome.Reply,
            decision=InteractionDecision(variants=[]), route=RouteChoice(action_type="Reply", workflow_name="dialogue_story"),
            dispatch=ActionDispatch(action_id="boundary-reply", action_type="Reply", workflow_name="dialogue_story",
                group=group.key, stimulus_ids=group.stimulus_ids), dependencies=request.dependencies)
        self.module.accept_selection(receipt)
        second = self.submit("Second input arrives before speech.")
        combined = self.module.coalesce_pending([second])
        self.assertEqual([InputSourcePayload.model_validate(item.payload).knoxel_id for item in combined],
                         [InputSourcePayload.model_validate(first.payload).knoxel_id, InputSourcePayload.model_validate(second.payload).knoxel_id])
        combined_request = self.request(combined)
        self.assertEqual(len(combined_request.dependencies.scopes), 1)
