"""Cognitive episode queries over committed records.

No model. The tests build real workspace turns, frames, thoughts, and an
action schema through the real MemoryStore commit boundary.
"""
from __future__ import annotations

import unittest

from pm.memory.store import MemoryStore
from pm.records.cognitive_records import AttentionFrameRecord
from pm.records.cycle_state import CycleState, PersonaConfig
from pm.records.enums import ActionType, FeatureType, InterlocusType, SourceKind
from pm.records.global_workspace import GwtTurnRecord
from pm.records.memory_records import ActionSchemaRecord, new_feature
from pm.records.enums import ActionSchemaStatus

from pm.introspection.episodes import EpisodeIndex, answer_questions


def small_ghost() -> MemoryStore:
    from tests.unit.test_design_contracts import small_ghost as make
    return make()


def _commit_thought(ghost, content="I keep wondering about the unfinished story."):
    batch = ghost.begin_batch("thought-1", new_tick=True)
    thought = new_feature(
        ghost, content, FeatureType.InnerMonologue, source=ghost.get_companion_name(),
        interlocus=InterlocusType.PrivateInternal, source_kind=SourceKind.Self, changes=batch,
        metadata={"loop": "inner", "operation": "memory_search", "trigger": "idle_opportunity",
                  "trigger_reason": "idle time", "temporal_focus": "past", "uncertainty": 0.4,
                  "reality": "internal", "evidence_ids": [], "evidence_summary": "(none)",
                  "interpretation": "the story matters to me",
                  "alternatives": [{"text": "drop the topic", "disposition": "rejected"}],
                  "continuation": "summarize", "continuation_reason": "complete",
                  "disclosure": {"mode": "withhold"}, "summary": "a private recollection"},
    )
    frame = batch.create(AttentionFrameRecord(
        content="private focus on the story", tick_id=batch.tick_id, focus_knoxel_ids=[thought.id],
        control_mode="reflective", frame_reason="private operation memory_search"))
    batch.set_gwt(
        GwtTurnRecord(tick_id=batch.tick_id, slot_id="inner", attention_frame_id=frame.id,
                      knoxel_ids=[thought.id, frame.id], summary="private recollection"),
        CycleState(tick_id=batch.tick_id, published_slots=["inner"], attention_frame_id=frame.id),
    )
    ghost.commit(batch)
    return thought.id, frame.id


def _commit_main_frame(ghost, focus_ids):
    batch = ghost.begin_batch("main-1", new_tick=True)
    frame = batch.create(AttentionFrameRecord(
        content="main attention on the private thought", tick_id=batch.tick_id,
        focus_knoxel_ids=list(focus_ids), control_mode="reactive",
        frame_reason="a self-origin thought arrived"))
    batch.set_gwt(
        GwtTurnRecord(tick_id=batch.tick_id, slot_id="main", attention_frame_id=frame.id,
                      knoxel_ids=[frame.id, *focus_ids], summary="appraise the private thought"),
        CycleState(tick_id=batch.tick_id, published_slots=["main"], attention_frame_id=frame.id),
    )
    ghost.commit(batch)
    return frame.id


class EpisodeIndexTests(unittest.TestCase):
    def test_episodes_are_built_from_committed_records(self):
        ghost = small_ghost()
        thought_id, _ = _commit_thought(ghost)
        index = EpisodeIndex(ghost)
        thought_episode = index.by_id(thought_id)
        self.assertIsNotNone(thought_episode)
        self.assertEqual(thought_episode.loop, "inner")
        self.assertEqual(thought_episode.operation, "memory_search")
        self.assertEqual(index.recent(loop="inner")[0].episode_id, thought_id)

    def test_cross_loop_link_uses_explicit_references_not_time(self):
        ghost = small_ghost()
        thought_id, _ = _commit_thought(ghost)
        _commit_main_frame(ghost, [thought_id])
        pairs = EpisodeIndex(ghost).cross_loop_links()
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["via_record_id"], thought_id)
        self.assertEqual(pairs[0]["relation"], "evidence")

    def test_action_schema_is_an_episode_with_its_own_outcome(self):
        ghost = small_ghost()
        batch = ghost.begin_batch("action-1", new_tick=True)
        action = batch.create(ActionSchemaRecord(
            content="invite Jesse to finish the story", action_type=ActionType.Reply,
            status=ActionSchemaStatus.Selected, intention_ids=[], expectation="Jesse says yes"))
        ghost.commit(batch)
        episode = EpisodeIndex(ghost).by_id(action.id)
        self.assertEqual(episode.kind, "action")
        self.assertEqual(episode.outcome, "committed")
        self.assertEqual(episode.metadata["expectation"], "Jesse says yes")

    def test_parallel_loops_keep_distinct_episode_identities(self):
        ghost = small_ghost()
        thought_id, _ = _commit_thought(ghost)
        _commit_main_frame(ghost, [thought_id])
        index = EpisodeIndex(ghost)
        loops = {episode.loop for episode in index.all()}
        self.assertEqual(loops, {"main", "inner"})


class QuestionAnswerTests(unittest.TestCase):
    def test_answers_name_evidence_or_an_explicit_limit(self):
        ghost = small_ghost()
        thought_id, _ = _commit_thought(ghost)
        index = EpisodeIndex(ghost)
        episode = index.by_id(thought_id)
        answers = answer_questions(ghost, episode)
        self.assertIn("information_available", answers)
        self.assertIn("alternatives", answers)
        self.assertEqual(answers["alternatives"]["evidence"][0]["disposition"], "rejected")
        for key, value in answers.items():
            self.assertIn("answer", value, key)
            self.assertIn("limit", value, key)

    def test_a_missing_input_is_an_explicit_limit(self):
        ghost = small_ghost()
        thought_id, _ = _commit_thought(ghost)
        episode = EpisodeIndex(ghost).by_id(thought_id)
        answers = answer_questions(ghost, episode)
        self.assertEqual(answers["information_available"]["limit"],
                         "no input records were retained")
        self.assertEqual(answers["cause"]["limit"], "no producer or cause link was retained")

    def test_capability_questions_require_an_overlay(self):
        ghost = small_ghost()
        thought_id, _ = _commit_thought(ghost)
        episode = EpisodeIndex(ghost).by_id(thought_id)
        self.assertNotIn("capabilities", answer_questions(ghost, episode))
        self.assertIn("capabilities", answer_questions(ghost, episode, overlay=_overlay()))


def _overlay():
    from pm.introspection.capabilities import (
        CapabilityContext, CapabilityDeclaration, build_overlay,
    )
    declaration = CapabilityDeclaration(name="private_thought", summary="Run the inner loop.",
                                        implemented_refs=("workflows/inner_thought/host.py",))
    return build_overlay([declaration], CapabilityContext())


if __name__ == "__main__":
    unittest.main()
