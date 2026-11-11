"""Deterministic boundaries using real stores, commits, and publication.

Assessment values are explicit boundary inputs. These checks make no claim
about model classification or companion behavior; the live gate covers that.
"""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import time
import unittest

from pm.bus.bus import BusSystem
from pm.bus.envelopes import new_envelope
from pm.memory.bash_view import MemoryBashTarget
from pm.memory.store_tools import build_tools
from pm.memory.snapshot import GhostCancelled
from pm.memory.snapshot import GhostConflict
from pm.memory.store import MemoryStore
from pm.records.cycle_records import StimulusGroup
from pm.records.cycle_state import CycleState, PersonaConfig
from pm.records.global_workspace import GwtTurnRecord
from pm.records.enums import FeatureType, InterlocusType
from pm.records.memory_records import Feature
from workflows.cognitive_control.host import CognitiveControl
from workflows.input_triage.schemas import EarlyTriageAssessment, TriageEvidence
from workflows.main_cognition.schemas import ActionRequest, TriageItem, TriageResult
from workflows.dialogue_story.host import StoryDialogue
from workflows.public_speech.public_context import PublicContext, source_fingerprint
from workflows.public_speech.publication import commit_public_reply
from workflows.public_speech.publication_control import ReplySuperseded
from workflows.public_speech.user_input import UserInputProxy


class EarlyTriageBoundaryTests(unittest.TestCase):
    """Exercise scheduling state at its actual durable boundaries."""

    def setUp(self) -> None:
        self.bus = BusSystem()
        self.store = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"))
        self.control = self.controller(self.store)
        self.input = UserInputProxy("user_input", self.bus, self.fail, time.monotonic, self.store, lambda text: None)
        self.input.register_triage_input = self.control.register_input

    def controller(self, store: MemoryStore) -> CognitiveControl:
        """Construct the real owner and rebuild its accepted control evidence."""
        control = CognitiveControl("control", self.bus, self.fail, time.monotonic, None, ghost=store)
        control.enable_input_triage()
        return control

    def dialogue(self) -> StoryDialogue:
        """Construct the real public module for its deterministic commit boundary."""
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        module = StoryDialogue("dialogue", self.bus, self.fail, time.monotonic, None,
            ghost=self.store, working_target=MemoryBashTarget(self.store, "read"),
            workflow=Path(__file__).resolve().parents[4] / "workflows/dialogue_story/dialogue_story.workflow.md",
            runtime_root=Path(directory.name), tools_factory=build_tools)
        module.cognitive_control = self.control.publication_control()
        self.control.cancel_public_drafts = module.defer_for_critical
        return module

    def submit(self, text: str) -> int:
        """Commit input through actual intake and return its source identity."""
        correlation = self.input.submit(text)
        return next(item.id for item in self.store.all_features
                    if item.metadata.get("exchange_correlation") == correlation)

    def assess(self, source_id: int, category: str) -> EarlyTriageAssessment:
        """Supply an explicitly named classification at the control boundary."""
        view = self.store.recovery_snapshot(capture=False)
        fingerprints = {source_id: source_fingerprint(view.get(source_id))}
        return self.control.accept_triage(EarlyTriageAssessment(
            source_ids=[source_id], source_fingerprints=fingerprints, guidance_tick=None,
            evidence=TriageEvidence(record_fingerprints=fingerprints, commit_ids=sorted(view.commit_ids),
                                    current_head_ids=[source_id]),
            result=TriageResult(items=[TriageItem(knoxel_id=source_id, category=category,
                                                 reasoning="Explicit scheduling-boundary input.")]))
        )

    def accept_cycle(self, source_ids: list[int]) -> int:
        """Accept a real state/write batch without claiming an LLM calculation."""
        batch = self.store.begin_batch(f"boundary-cycle-{len(self.store.states)}", new_tick=True, embed=False)
        batch.depend_on(source_ids)
        state = CycleState(tick_id=batch.tick_id, stimulus_groups=[StimulusGroup(stimulus_ids=source_ids)])
        batch.set_gwt(GwtTurnRecord(tick_id=batch.tick_id, stimulus_ids=source_ids), state)
        self.control.commit(batch, None, self.control.generation, source_ids=source_ids)
        return batch.tick_id

    def test_unknown_input_holds_publication_and_ordinary_assessment_releases_it(self) -> None:
        epoch = self.control.publication_epoch()
        source = self.submit("Hello")
        with self.assertRaises(ReplySuperseded):
            self.control.validate_publication(epoch, None)
        self.assess(source, "Moderate")
        self.control.validate_publication(epoch, None)
        self.assertEqual(self.control.publication_epoch(), epoch)

    def test_critical_invalidates_old_draft_even_after_accepted_reconsideration(self) -> None:
        old_epoch = self.control.publication_epoch()
        source = self.submit("A consequential correction")
        self.assess(source, "Critical")
        with self.assertRaises(ReplySuperseded):
            self.control.validate_publication(old_epoch, None)
        tick = self.accept_cycle([source])
        self.assertFalse(self.control.reconsideration_required())
        with self.assertRaises(ReplySuperseded):
            self.control.validate_publication(old_epoch, tick)
        self.control.validate_publication(self.control.publication_epoch(), tick)
        with self.assertRaises(ReplySuperseded):
            self.control.validate_publication(self.control.publication_epoch(), tick - 1)

    def test_newer_critical_source_cannot_be_cleared_by_older_cycle(self) -> None:
        first = self.submit("First correction")
        self.assess(first, "Critical")
        second = self.submit("Further correction")
        self.assess(second, "Critical")
        with self.assertRaises(GhostCancelled):
            self.accept_cycle([first])
        self.assertTrue(self.control.reconsideration_required())
        self.accept_cycle([first, second])
        self.assertFalse(self.control.reconsideration_required())

    def test_atomic_public_commit_rejects_obsolete_epoch_without_creating_speech(self) -> None:
        old_epoch = self.control.publication_epoch()
        source = self.submit("Critical evidence")
        self.assess(source, "Critical")
        tick = self.accept_cycle([source])
        context = PublicContext(companion_name="Spamuel", user_name="Jesse", reconsideration_epoch=old_epoch,
                                state_tick=tick)
        with self.assertRaises(ReplySuperseded):
            commit_public_reply(self.store, "obsolete draft", "", commit_id="obsolete-draft", context=context,
                                cognitive_control=self.control.publication_control())
        self.assertFalse(any(item.content == "obsolete draft" for item in self.store.all_features))

    def test_clean_restart_preserves_pending_input_and_critical_assessment(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "ghost.pickle"
            self.store.checkpoint = path
            source = self.submit("Critical evidence before interruption")
            reopened = MemoryStore.from_checkpoint(path)
            self.assertTrue(self.controller(reopened).reconsideration_required())
            self.assess(source, "Critical")
            reopened = MemoryStore.from_checkpoint(path)
            restarted = self.controller(reopened)
            self.assertTrue(restarted.has_assessment(source))
            self.assertTrue(restarted.reconsideration_required())

    def test_resolved_historical_input_does_not_create_a_permanent_pending_barrier(self) -> None:
        source = self.submit("Already answered")
        batch = self.store.begin_batch("resolve-historical", embed=False)
        batch.set_lifecycle(source, exchange_status="resolved")
        self.store.commit(batch)
        restarted = self.controller(self.store)
        self.assertFalse(restarted.reconsideration_required())

    def test_cancelled_wait_does_not_hold_a_thread(self) -> None:
        self.submit("Unclassified")
        cancelled = Event()
        cancelled.set()
        self.assertFalse(self.control.wait_for_reconsideration(cancelled))

    def test_accepted_source_successor_rejects_an_old_classification(self) -> None:
        source = self.submit("A message using current guidance")
        baseline = self.store.begin_batch("baseline-guidance", new_tick=True, embed=False)
        guidance = baseline.create(Feature(content="Earlier accepted guidance", feature_type=FeatureType.Thought,
                                           interlocus=InterlocusType.PrivateInternal))
        baseline.set_gwt(GwtTurnRecord(tick_id=baseline.tick_id))
        self.store.commit(baseline)
        view = self.store.recovery_snapshot(capture=False)
        assessment = EarlyTriageAssessment(source_ids=[source],
            source_fingerprints={source: source_fingerprint(view.get(source))}, guidance_tick=baseline.tick_id,
            evidence=TriageEvidence(record_fingerprints={source: source_fingerprint(view.get(source)),
                                                         guidance.id: source_fingerprint(view.get(guidance.id))},
                                    commit_ids=sorted(view.commit_ids), current_head_ids=[source, guidance.id]),
            result=TriageResult(items=[TriageItem(knoxel_id=source, category="Moderate")]))
        correction = self.store.begin_batch("correct-guidance", new_tick=True, embed=False)
        revised = correction.view().get(guidance.id).copy_for_edit()
        revised.content = "Corrected accepted guidance"
        correction.replace(revised)
        correction.set_gwt(GwtTurnRecord(tick_id=correction.tick_id))
        self.store.commit(correction)
        with self.assertRaisesRegex(GhostConflict, "successor"):
            self.control.accept_triage(assessment)
        self.assertFalse(self.control.has_assessment(source))

    def test_critical_cancels_public_calculation_and_keeps_its_input_pending(self) -> None:
        module = self.dialogue()
        module._composing_reply = True
        source = self.submit("New Critical evidence during composition")
        assessment = self.assess(source, "Critical")
        self.assertTrue(module.run_stop.is_set())
        self.assertEqual(module._interrupted_epoch, assessment.reconsideration_epoch)
        self.assertEqual(self.store.snapshot({source}).get(source).metadata["exchange_status"], "pending")

    def test_critical_never_relabels_an_already_committed_utterance_as_unsent(self) -> None:
        module = self.dialogue()
        module._composing_reply = True
        request = ActionRequest()
        module.public_context(request)
        module.accept_reply("Already accepted speech", [new_envelope("main_cog", "data", "action.request", "",
                                                                    request.payload())], recovery_outcome="boundary")
        source = self.submit("Later Critical evidence")
        self.assess(source, "Critical")
        self.assertFalse(module.run_stop.is_set())
        self.assertIsNone(module._interrupted_epoch)
        self.assertTrue(any(record.content == "Already accepted speech" for record in self.store.all_features))

    def test_owner_cancellation_is_not_reported_as_critical_supersession(self) -> None:
        module = self.dialogue()
        module._composing_reply = True
        module.cancel_run()
        self.assertTrue(module.run_stop.is_set())
        self.assertIsNone(module._interrupted_epoch)
