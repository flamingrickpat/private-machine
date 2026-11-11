"""Actual-store checks for ordinary interaction evidence and publication races."""
from __future__ import annotations

import unittest
from uuid import uuid4

from pm.bus.bus import BusSystem
from pm.memory.snapshot import GhostConflict
from pm.memory.store import MemoryStore
from pm.records.base import RecordScope
from pm.records.cycle_state import PersonaConfig
from pm.records.enums import ActionType, EntityClass
from pm.records.memory_records import ActionSchemaRecord, Entity
from workflows.main_cognition.schemas import ActionRequest, Stimulus
from workflows.cognitive_control.host import CognitiveControl
from workflows.public_speech.dependencies import PartnerFrontierChanged, capture_dependencies, validate_dependencies
from workflows.public_speech.public_context import build_public_context
from workflows.public_speech.publication import commit_public_reply, exchange_record, validate_public_context
from workflows.public_speech.user_input import UserInputProxy


class InteractionDependencyTests(unittest.TestCase):
    """Exercise real evidence, lifecycle commits, and competing speech actions."""

    def setUp(self) -> None:
        self.store = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"))
        self.partner_id = self.store.store_knoxel(Entity(content="Jesse", aliases=[], entity_class=EntityClass.Human)).id
        self.store.store_knoxel(Entity(content="Spamuel", aliases=[], entity_class=EntityClass.Self))
        self.store.accept_import()
        self.proxy = UserInputProxy("user_input", BusSystem(), self.fail, lambda: 0,
                                    self.store, lambda text: None)

    def action(self, text: str) -> int:
        """Persist an actual selected action for publication-boundary checks."""
        batch = self.store.begin_batch("local-selection-" + uuid4().hex)
        batch.preserve_on_recovery()
        action = batch.create(ActionSchemaRecord(content=text, action_type=ActionType.Reply))
        self.store.commit(batch)
        return action.id

    def test_missing_selection_owner_fails_before_publication(self) -> None:
        control = CognitiveControl("control", BusSystem(), self.fail, lambda: 0, None, ghost=self.store)
        publication_control = control.publication_control()
        publication_control.validate_selection("")
        with self.assertRaisesRegex(RuntimeError, "injected selection owner"):
            publication_control.validate_selection("ordinary-choice-without-owner")

    def test_lifecycle_does_not_change_source_meaning_or_main_frontier(self) -> None:
        correlation = self.proxy.submit("Keep this in mind.")
        source = exchange_record(self.store, correlation)
        assert source is not None
        dependencies = capture_dependencies(self.store.snapshot(), [source.id], guidance_tick=1,
            critical_epoch=0, partner_entity_id=self.partner_id, scopes=[RecordScope()])
        commit_public_reply(self.store, "I heard you.", correlation, commit_id="first")
        validate_dependencies(self.store, dependencies, current_partner_input=True)
        self.proxy.submit("Here is another ordinary detail.")
        validate_dependencies(self.store, dependencies, current_partner_input=False)
        with self.assertRaises(GhostConflict):
            validate_dependencies(self.store, dependencies, current_partner_input=True)

    def test_another_partner_does_not_invalidate_this_exchange(self) -> None:
        correlation = self.proxy.submit("A message to this conversation.")
        source = exchange_record(self.store, correlation)
        assert source is not None
        dependencies = capture_dependencies(self.store.snapshot(), [source.id], guidance_tick=1,
            critical_epoch=0, partner_entity_id=self.partner_id, scopes=[RecordScope()])
        self.proxy.submit("A separate peer conversation.", speaker="Mara", entity_ref="peer-mara")
        validate_dependencies(self.store, dependencies, current_partner_input=True)

    def test_speech_freezes_new_frontier_and_resolves_peer_input(self) -> None:
        correlation = self.proxy.submit("The peer's first line.", speaker="Mara", entity_ref="peer-mara")
        source = exchange_record(self.store, correlation)
        assert source is not None
        peer_id = source.entity_id
        dependencies = capture_dependencies(self.store.snapshot(), [source.id], guidance_tick=1,
            critical_epoch=0, partner_entity_id=peer_id, scopes=[RecordScope()])
        request = ActionRequest(partner_entity_id=peer_id, partner_name="Mara", action_origin="interaction",
            interaction_dependencies=dependencies, stimuli=[Stimulus(knoxel_id=source.id)])
        original = build_public_context(self.store, request)
        later_correlation = self.proxy.submit("One more detail.", speaker="Mara", entity_ref="peer-mara")
        later = exchange_record(self.store, later_correlation)
        assert later is not None
        with self.assertRaises(PartnerFrontierChanged):
            validate_public_context(self.store, original)
        with self.assertRaises(PartnerFrontierChanged):
            build_public_context(self.store, request, late_ids=[later.id])
        refreshed_dependencies = capture_dependencies(self.store.snapshot(), [source.id, later.id], guidance_tick=1,
            critical_epoch=0, partner_entity_id=peer_id, scopes=[RecordScope(), RecordScope()])
        self.assertEqual(refreshed_dependencies.scopes, [RecordScope()])
        refreshed_request = request.model_copy(update={"interaction_dependencies": refreshed_dependencies})
        refreshed = build_public_context(self.store, refreshed_request, late_ids=[later.id])
        validate_public_context(self.store, refreshed)
        self.assertEqual(set(refreshed.stimulus_user_ids), {source.id, later.id})
        reply_id = commit_public_reply(self.store, "I heard both lines.", later_correlation,
                                      commit_id="peer-answer", context=refreshed)
        self.assertIsNotNone(reply_id)
        for correlation_id in (correlation, later_correlation):
            resolved = exchange_record(self.store, correlation_id)
            assert resolved is not None
            self.assertEqual(resolved.metadata["public_reply_id"], reply_id)

    def test_a_successor_rejects_current_authority(self) -> None:
        correlation = self.proxy.submit("An original statement.")
        source = exchange_record(self.store, correlation)
        assert source is not None
        dependencies = capture_dependencies(self.store.snapshot(), [source.id], guidance_tick=1,
            critical_epoch=0, partner_entity_id=self.partner_id, scopes=[RecordScope()])
        batch = self.store.begin_batch("correct-source")
        corrected = source.copy_for_edit()
        corrected.content = "The corrected statement."
        batch.replace(corrected)
        self.store.commit(batch)
        with self.assertRaises(GhostConflict):
            validate_dependencies(self.store, dependencies, current_partner_input=False)

    def test_different_actions_cannot_answer_the_same_resolved_exchange_twice(self) -> None:
        correlation = self.proxy.submit("One current question.")
        source = exchange_record(self.store, correlation)
        assert source is not None
        first = self.action("Answer the question.")
        second = self.action("A competing answer to the same question.")
        request = ActionRequest(knoxel_id=first, partner_entity_id=self.partner_id,
                                stimuli=[Stimulus(knoxel_id=source.id)])
        context = build_public_context(self.store, request)
        reply_id = commit_public_reply(self.store, "One answer.", correlation,
                                       commit_id="first", action_id=first, context=context)
        duplicate = commit_public_reply(self.store, "Competing answer.", correlation,
                                        commit_id="second", action_id=second, context=context)
        self.assertIsNotNone(reply_id)
        self.assertIsNone(duplicate)

    def test_terminal_task_outcome_can_follow_an_interim_answer(self) -> None:
        correlation = self.proxy.submit("Please do the task and report the result.")
        commit_public_reply(self.store, "I will get back to you.", correlation, commit_id="interim")
        task_action = self.action("Report the actual completed task.")
        reply_id = commit_public_reply(self.store, "The task finished.", correlation,
            commit_id="task-result", action_id=task_action, action_origin="task_reply")
        self.assertIsNotNone(reply_id)
        self.assertIsNone(commit_public_reply(self.store, "Repeated result.", correlation,
            commit_id="duplicate-task-result", action_id=task_action, action_origin="task_reply"))

    def test_new_pending_input_is_not_dropped_with_resolved_prior_input(self) -> None:
        first_correlation = self.proxy.submit("The first message.")
        first = exchange_record(self.store, first_correlation)
        assert first is not None
        commit_public_reply(self.store, "First answer.", first_correlation, commit_id="first-answer")
        current_correlation = self.proxy.submit("A new question.")
        current = exchange_record(self.store, current_correlation)
        assert current is not None
        action_id = self.action("Answer the new question using the existing conversation.")
        context = build_public_context(self.store, ActionRequest(knoxel_id=action_id,
            partner_entity_id=self.partner_id,
            stimuli=[Stimulus(knoxel_id=first.id), Stimulus(knoxel_id=current.id)]))
        reply_id = commit_public_reply(self.store, "The new answer.", current_correlation,
                                      commit_id="current-answer", action_id=action_id, context=context)
        self.assertIsNotNone(reply_id)
        resolved = exchange_record(self.store, current_correlation)
        assert resolved is not None
        self.assertEqual(resolved.metadata["public_reply_id"], reply_id)


if __name__ == "__main__":
    unittest.main()
