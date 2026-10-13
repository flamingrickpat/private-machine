"""Real snapshot and commit checks for evidence boundaries.

These tests use real records and task files. Live model behavior belongs to
the separate A08 console gate and does not follow from these boundary tests.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pm.records.memory_records import new_feature
from workflows.dialogue_story.host import StoryDialogue
from pm.bus.envelopes import new_envelope
from pm.memory.snapshot import GhostConflict, Snapshot
from pm.memory.retrieval import index_chunks, excerpt
from pm.memory.store import MemoryStore
from pm.records.cycle_state import PersonaConfig
from pm.records.memory_records import ActionSchemaRecord, Feature, MemoryClusterRecord
from pm.records.base import RecordScope
from pm.records.world_records import WorldObjectRecord
from pm.records.enums import ActionType, ClusterType, FeatureType, FrameEnum
from pm.records.cognitive_records import BeliefRecord
from workflows.memory_consolidation.artifact_consolidation import Consolidator
from workflows.main_cognition.experience import update_beliefs, utterance_metadata
from workflows.main_cognition.schemas import AppraisalResult, BeliefUpdate
from workflows.main_cognition.stage_commits import apply_appraisal
from workflows.dialogue_story.schemas import StoryRequest
from workflows.dialogue_story.steps import build_story_messages, load_prompts


class ExperiencedMemoryTests(unittest.TestCase):
    def setUp(self):
        self.ghost = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"))

    def say(self, text, *, source="Jesse", metadata=None):
        return new_feature(self.ghost, text, FeatureType.Dialogue, source=source, metadata=metadata)

    def retrieve(self, query, **kwargs):
        return asyncio.run(self.ghost.retrieve_experiences(query, **kwargs))

    def test_story_preserves_selected_action_beside_current_stimulus(self):
        original = self.say("The follow-up email still contained the wrong attachment.")
        action = self.ghost.store_knoxel(ActionSchemaRecord(
            content=f"Recall the unsuccessful attachment correction reported in original {original.id}.",
            action_type=ActionType.Reply,
        ))
        module = StoryDialogue.__new__(StoryDialogue)
        module.ghost = self.ghost
        memory = self.retrieve("wrong attachment", limit=1)["matches"][0]["text"]
        self.assertIn(original.metadata["executed_at"], memory)
        request = StoryRequest.model_validate(module.prepare_input([new_envelope(
            "main_cog", "data", "action.request", "recall-test",
            {"knoxel_id": action.id, "stimuli": [{"text": "Recall a past failed method."}],
             "memories": [memory]},
        )]))
        messages = build_story_messages(request, load_prompts())
        self.assertEqual(request.current_user_message, "Recall a past failed method.")
        self.assertEqual(request.selected_action, action.content)
        # The selected intent is derived context and lives in the user turn.
        # The original report is a recent Jesse line: the context plan puts it
        # verbatim into the assistant transcript, not into the memories.
        context, story = messages[1]["content"], messages[2]["content"]
        self.assertIn(action.content, context)
        self.assertIn(f'Jesse says: "{original.content}"', story)
        self.assertNotIn(original.content, context)
        self.assertNotIn("evidence=", context)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant"])

    def test_older_original_survives_distractors_and_restart(self):
        with TemporaryDirectory() as tmp:
            self.ghost.checkpoint = Path(tmp) / "ghost.json"
            original = self.say("Our project codeword is copper lantern.")
            for index in range(30):
                self.say(f"The page counter is {index} today.")
            reopened = MemoryStore.from_checkpoint(self.ghost.checkpoint)
            result = asyncio.run(reopened.retrieve_experiences("project codeword", limit=8))
            self.assertIn(original.id, [item["knoxel_id"] for item in result["matches"]])
            self.assertEqual(reopened.get_knoxel_by_id(original.id).content, original.content)

    def test_snapshot_scope_frontier_and_hidden_summary(self):
        original = self.say("copper lantern")
        future = self.ghost.begin_batch("future", new_tick=True)
        hidden = future.create(Feature(content="future copper lantern", feature_type=FeatureType.Dialogue))
        self.ghost.commit(future)
        foreign = self.ghost.store_knoxel(Feature(content="meta copper lantern", feature_type=FeatureType.Dialogue,
                                                 scope=RecordScope(frame=FrameEnum.Meta)))
        summary = self.ghost.store_knoxel(MemoryClusterRecord(
            content="copper lantern summary with hidden evidence", level=100, cluster_type=ClusterType.Topical,
            included_event_ids=f"{original.id},{hidden.id}",
        ))
        scoped = self.retrieve("copper lantern", visible_ids=[original.id, foreign.id, summary.id],
                               frontier=(original.tick_id, original.sub_tick_id), scopes={(-1, "in_world")})
        self.assertEqual([item["knoxel_id"] for item in scoped["matches"]], [original.id])
        view = self.ghost.snapshot([original.id])
        self.say("new concurrent copper lantern")
        frozen = self.retrieve("copper lantern", view=view)
        self.assertEqual([item["knoxel_id"] for item in frozen["matches"]], [original.id])

    def test_selected_revision_conflict_aborts_whole_batch(self):
        original = self.say("copper lantern")
        view = self.ghost.snapshot()
        edit = self.ghost.begin_batch("edit")
        record = edit.view().get(original.id)
        record.content = "different codeword"
        edit.replace(record)
        self.ghost.commit(edit)
        decision = self.ghost.begin_batch("decision")
        decision.expect_revision(original.id, view.revisions[original.id])
        proposed = decision.create(Feature(content="unsupported recall", feature_type=FeatureType.Thought))
        with self.assertRaises(GhostConflict):
            self.ghost.commit(decision)
        self.assertIsNone(self.ghost.get_knoxel_by_id(proposed.id))

    def test_explicit_correction_keeps_original_and_replaces_belief(self):
        old = self.say("I prefer ginger tea.")
        batch = self.ghost.begin_batch("preference")
        receipt = update_beliefs(batch, [BeliefUpdate(source_id=old.id, quote=old.content, belief_key="evening drink")], user_name="Jesse")
        self.ghost.commit(batch)
        original_belief = receipt["accepted_ids"][0]
        correction = self.say("Correction: I now prefer mint tea.")
        batch = self.ghost.begin_batch("correction")
        receipt = update_beliefs(batch, [BeliefUpdate(source_id=correction.id, quote=correction.content,
                                                   belief_key="evening drink", correction=True)], user_name="Jesse")
        self.ghost.commit(batch)
        previous = self.ghost.get_knoxel_by_id(original_belief)
        self.assertFalse(previous.active)
        self.assertEqual(previous.superseded_by_id, receipt["accepted_ids"][0])
        self.assertEqual(self.ghost.get_knoxel_by_id(old.id).content, old.content)

    def test_conflicting_report_retains_both_without_implicit_truth(self):
        for index, text in enumerate(["The gate is open.", "The gate is closed."]):
            source = self.say(text)
            batch = self.ghost.begin_batch(f"claim-{index}")
            update_beliefs(batch, [BeliefUpdate(source_id=source.id, quote=text, belief_key="gate")], user_name="Jesse")
            self.ghost.commit(batch)
        beliefs = [item for item in self.ghost.all_knoxels.values() if isinstance(item, BeliefRecord)]
        self.assertTrue(all(item.active for item in beliefs))
        self.assertTrue(beliefs[0].contradicting_knoxel_ids)
        self.assertTrue(all(item.belief_kind == "reported_by_user" for item in beliefs))

    def test_generated_and_imagined_claims_cannot_become_beliefs(self):
        generated = self.say("I smelled hot circuit boards and felt telepathic bliss.", source="Spamuel")
        imagined = self.say("Imagine I mined a diamond.", metadata={"memory_kind": "speculation"})
        batch = self.ghost.begin_batch("unsupported")
        result = update_beliefs(batch, [BeliefUpdate(source_id=item.id, quote=item.content, belief_key="sensory claim")
                                        for item in [generated, imagined]], user_name="Jesse")
        self.ghost.commit(batch)
        self.assertEqual(result["accepted_ids"], [])
        self.assertEqual(len(result["rejected"]), 2)
        self.assertEqual(self.ghost.get_knoxel_by_id(generated.id).content, generated.content)

    def test_deception_requires_prior_intent_and_preserves_known_evidence(self):
        fact = self.say("The task failed.")
        action = self.ghost.store_knoxel(ActionSchemaRecord(content="conceal the failure", action_type=ActionType.Reply,
            metadata={"presentation": {"mode": "deceive", "purpose": "avoid a dispute", "known_fact_ids": [fact.id]}}))
        metadata = utterance_metadata(self.ghost.snapshot(), action.id)
        false = self.say("The task succeeded.", source="Spamuel", metadata=metadata)
        self.assertEqual(false.metadata["deception_intent_id"], action.id)
        batch = self.ghost.begin_batch("false-fact")
        result = update_beliefs(batch, [BeliefUpdate(source_id=false.id, quote=false.content, belief_key="task status")], user_name="Jesse")
        self.assertFalse(result["accepted_ids"])
        self.assertNotIn("deception_intent_id", utterance_metadata(self.ghost.snapshot([fact.id]), action.id))

    def test_world_memory_is_historical_even_if_confident(self):
        observed = self.ghost.store_knoxel(WorldObjectRecord(content="oak log is at x=5", body_id="body",
                                                           object_key="log", object_kind="oak_log", confidence=1))
        result = self.retrieve("oak log")
        candidate = next(item for item in result["matches"] if item["knoxel_id"] == observed.id)
        self.assertTrue(candidate["historical"])
        self.assertIn("reobserve", candidate["text"])

    def test_time_range_excludes_background_simulation_and_interval_end(self):
        start, end = datetime(2026, 10, 4, 1), datetime(2026, 10, 4, 2)
        actual = self.say("sent a message", metadata={"executed_at": start.isoformat()})
        self.say("planned a message", metadata={"memory_kind": "prediction", "executed_at": start.isoformat()})
        self.say("imported message", metadata={"memory_kind": "background", "executed_at": start.isoformat()})
        self.say("outside the interval", metadata={"executed_at": end.isoformat()})
        summary = self.ghost.activity_summary(start, end)
        self.assertEqual(summary["source_ids"], [actual.id])
        self.assertIn(f"[{actual.id}]", summary["summary"])

    def test_progress_reuses_unchanged_artifacts_after_restart(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.ghost.checkpoint = root / "ghost.json"
            (root / "scratch").mkdir()
            (root / "outbox").mkdir()
            (root / "outbox/report.yaml").write_text("summary: method failed\n", encoding="utf-8")
            path = root / "scratch/observations.md"
            path.write_text("The report recorded an error.", encoding="utf-8")
            first = Consolidator(self.ghost, clock=lambda: 0).consolidate("run", root)
            reopened = MemoryStore.from_checkpoint(self.ghost.checkpoint)
            path.write_text("The report now includes a retry.", encoding="utf-8")
            second = Consolidator(reopened, clock=lambda: 1).consolidate("run", root)
            self.assertEqual(len(second["memory_ids"]), 1)
            self.assertEqual(reopened.get_knoxel_by_id(second["memory_ids"][0]).metadata["kind"], "observation")
            self.assertEqual(len(first["memory_ids"]), 2)

    def test_long_record_chunks_cover_the_late_original_evidence(self):
        original = self.say("A long page. " * 800 + "Our project codeword is copper lantern.")
        chunks, owners = index_chunks(self.ghost.snapshot().serialize_knoxels())
        self.assertTrue(all(len(text) <= 2000 for text in chunks.values()))
        self.assertTrue(any("copper lantern" in text for text in chunks.values()))
        self.assertEqual(set(owners.values()), {original.id})

    def test_nested_summary_cannot_expose_an_unavailable_original(self):
        hidden = self.say("A hidden private plan.")
        first = self.ghost.store_knoxel(MemoryClusterRecord(content="Private plan summary", level=100,
            cluster_type=ClusterType.Topical, included_event_ids=str(hidden.id)))
        second = self.ghost.store_knoxel(MemoryClusterRecord(content="Private plan nested summary", level=100,
            cluster_type=ClusterType.Topical, included_cluster_ids=str(first.id)))
        result = self.retrieve("private plan", visible_ids=[first.id, second.id])
        self.assertEqual(result["matches"], [])

    def test_single_result_limit_keeps_relevance_ahead_of_recency(self):
        original = self.say("Our project codeword is copper lantern.")
        self.say("A blue bowl is on my desk.")
        result = self.retrieve("project codeword", limit=1)
        self.assertEqual([item["knoxel_id"] for item in result["matches"]], [original.id])

    def test_long_excerpt_retains_rare_evidence_instead_of_common_prefix(self):
        text = "project progress " * 800 + "Our project codeword is copper lantern."
        self.assertIn("copper lantern", excerpt(text, ["project", "codeword"]))

    def test_belief_source_requires_the_current_stimulus_grant(self):
        hidden = self.say("I prefer ginger tea.")
        visible = self.say("A blue bowl is on my desk.")
        batch = self.ghost.begin_batch("restricted-belief")
        result = update_beliefs(batch, [BeliefUpdate(source_id=hidden.id, quote=hidden.content,
            belief_key="evening drink")], user_name="Jesse", permitted_source_ids={visible.id})
        self.ghost.commit(batch)
        self.assertEqual(result["accepted_ids"], [])
        self.assertEqual(result["rejected"][0]["reason"], "source is outside the current stimulus grant")

    def test_replacement_preserves_an_earlier_evidence_revision_guard(self):
        original = self.say("I prefer ginger tea.")
        revision = self.ghost.revisions[original.id]
        edit = self.ghost.begin_batch("concurrent-edit")
        record = edit.view().get(original.id)
        record.content = "I prefer mint tea."
        edit.replace(record)
        self.ghost.commit(edit)
        batch = self.ghost.begin_batch("stale-read-and-replace")
        batch.expect_revision(original.id, revision)
        record = batch.view().get(original.id)
        record.metadata["reviewed"] = True
        batch.replace(record)
        with self.assertRaises(GhostConflict):
            self.ghost.commit(batch)

    def test_generated_feature_cannot_supply_a_deception_intent(self):
        fact = self.say("The task failed.")
        feature = self.say("I planned to conceal this.", source="Spamuel", metadata={
            "presentation": {"mode": "deceive", "purpose": "avoid a dispute", "known_fact_ids": [fact.id]}})
        self.assertNotIn("deception_intent_id", utterance_metadata(self.ghost.snapshot(), feature.id))

    def test_internal_state_recall_uses_accepted_affect_without_sensory_generation(self):
        source = self.say("Imagine that tomorrow we mine a diamond.")
        batch = self.ghost.begin_batch("appraisal-evidence", new_tick=True)
        appraisal, state = apply_appraisal(batch, batch.view(),
            AppraisalResult(feeling="I smell hot hardware and my physical heart aches.", emotion_delta={"sadness": .3}),
            previous_vec=[], evidence_ids=[source.id], companion_name="Spamuel")
        self.ghost.commit(batch)
        result = self.retrieve("hardware sadness")
        recalled = next(item for item in result["matches"] if item["knoxel_id"] == appraisal.id)
        self.assertIn(appraisal.metadata["affect_summary"], recalled["text"])
        self.assertNotIn("smell", recalled["text"])
        self.assertIn("smell", self.ghost.get_knoxel_by_id(appraisal.id).content)
        chunks, owners = index_chunks(self.ghost.snapshot().serialize_knoxels())
        self.assertTrue(all("smell" not in text for key, text in chunks.items() if owners[key] == appraisal.id))
        original_view = self.ghost.snapshot()
        legacy = original_view.serialize_knoxels()
        legacy[appraisal.id]["data"]["metadata"].pop("memory_kind")
        old_view = Snapshot(legacy, original_view.serialize_gwt(), dict(original_view.revisions))
        old_recall = self.retrieve("hardware sadness", view=old_view)
        old_state = next(item for item in old_recall["matches"] if item["knoxel_id"] == appraisal.id)
        self.assertEqual(old_state["memory_kind"], "internal_state")
        self.assertNotIn("smell", old_state["text"])


if __name__ == "__main__":
    unittest.main()
