"""Cognitive episode queries over committed records.

No model. The tests build real workspace turns, thoughts, an action schema, and
executed operations of the private inquiry graph through the real MemoryStore
commit boundary.
"""
from __future__ import annotations

import unittest
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event

from pm.memory.store import MemoryStore
from pm.memory.write_batch import WriteBatch
from pm.records.base import Provenance, RecordScope
from pm.records.cycle_records import StimulusGroup
from pm.records.cycle_state import CycleState, PersonaConfig
from pm.records.enums import ActionType, EntityClass, FeatureType, InterlocusType, SourceKind
from pm.records.global_workspace import GwtTurnRecord
from pm.records.memory_records import ActionSchemaRecord, Entity, Feature, new_feature
from pm.records.enums import ActionSchemaStatus
from pm.records.mental_state import FullMentalState

from pm.introspection.episodes import (
    AlternativeRow, EpisodeIndex, InquiryProgressRead, answer_questions,
    recent_without_duplicate_inner_frames,
)
from typing import cast
from workflows.cognitive_control.schemas import ControlConfig
from workflows.inner_thought.graph_host import PrivateInquiryOwner
from workflows.inner_thought.inquiry import InquiryRecord
from workflows.inner_thought.nominations import capture_inquiry_grant, stage_nominations
from workflows.inner_thought.schemas import (
    InquiryAlternative, InquiryProgress, InquiryStep, ThoughtAttentionDecision, ThoughtJudgments, ThoughtNomination,
    ThoughtRelation,
)
from workflows.reflection.experiments import context as reflection_context
from workflows.reflection.rendering import private_material


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
    batch.set_gwt(
        GwtTurnRecord(tick_id=batch.tick_id, slot_id="inner",
                      knoxel_ids=[thought.id], summary="private recollection"),
        CycleState(tick_id=batch.tick_id),
    )
    ghost.commit(batch)
    return thought.id, batch.tick_id


def _commit_main_turn(ghost, stimulus_ids):
    """Commit one main turn that claims ``stimulus_ids``. Return its tick."""
    batch = ghost.begin_batch("main-1", new_tick=True)
    batch.set_gwt(
        GwtTurnRecord(tick_id=batch.tick_id, slot_id="main", stimulus_ids=list(stimulus_ids),
                      knoxel_ids=list(stimulus_ids), summary="appraise the private thought"),
        CycleState(tick_id=batch.tick_id, stimulus_groups=[StimulusGroup(stimulus_ids=list(stimulus_ids))]),
    )
    ghost.commit(batch)
    return batch.tick_id


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

    def test_a_private_step_shows_its_frame_turn_and_its_decision(self):
        """A private step keeps two rows until a turn names its own decision.

        A private turn leaves ``decision_id`` empty, and that field names a
        main-cycle ``StateUpdate`` rather than a private decision, so the index
        cannot merge the frame and the decision without inventing a causal fact. The
        frame row points at the decision row, which is the row that cites evidence.
        """
        ghost = small_ghost()
        thought_id, thought_tick = _commit_thought(ghost)
        rows = EpisodeIndex(ghost).recent(loop="inner")
        self.assertEqual([item.episode_id for item in rows], [thought_tick, thought_id])
        frame = next(item for item in rows if item.kind == "gwt_turn")
        self.assertEqual(frame.outputs, (thought_id,))

    def test_cross_loop_link_uses_explicit_references_not_time(self):
        ghost = small_ghost()
        thought_id, _ = _commit_thought(ghost)
        _commit_main_turn(ghost, [thought_id])
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
        _commit_main_turn(ghost, [thought_id])
        index = EpisodeIndex(ghost)
        loops = {episode.loop for episode in index.all()}
        self.assertEqual(loops, {"main", "inner"})


@dataclass(frozen=True, slots=True)
class _ExecutedStep:
    """One accepted private step and the records that step produced."""

    executed: InquiryRecord
    operation_id: int
    branch_ids: tuple[int, ...]
    nomination_id: int


class InquiryGraphEpisodeTests(unittest.TestCase):
    """Keep the live private graph inspectable as its own episodes.

    The private mind now thinks in the inquiry graph. A mind that cannot see
    its own graph steps cannot notice its recent thinking, cannot answer where
    one of its ideas came from, and cannot see that a main turn was steered by
    private work. These checks use the real selection, execution, and derived
    inner-turn boundaries with no model.
    """

    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store: MemoryStore = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"),
                                             checkpoint=Path(directory.name) / "graph-episodes.pickle")
        batch = self.store.begin_batch("anchors", embed=False)
        self.observer: Entity = batch.create(Entity(content="Spamuel", aliases=[], entity_class=EntityClass.Self))
        self.partner: Entity = batch.create(Entity(content="Jesse", aliases=[], entity_class=EntityClass.Human))
        self.source: Feature = batch.create(Feature(content="I left the sketch unfinished.",
            feature_type=FeatureType.Dialogue, entity_id=self.partner.id,
            provenance=Provenance(source_kind=SourceKind.User)))
        batch.preserve_on_recovery()
        self.store.commit(batch)
        self.owner = PrivateInquiryOwner(self.store, commit=self._commit, cancel=Event(), limits=ControlConfig())
        self.now: datetime = datetime(2026, 11, 12, 17, 0)
        self.judgments = ThoughtJudgments(interestingness=0.8, character_relevance=0.7, plausibility=0.3,
            consequence=0.4, desirability=0.1, action_suitability=0.2, reason="The sketch matters.")
        self.grant = capture_inquiry_grant(self.store.snapshot(capture=False), observer_entity_id=self.observer.id,
            scope=RecordScope(), source_ids=[self.source.id], governing_ids=[self.source.id], critical_epoch=0)

    def _commit(self, batch: WriteBatch) -> None:
        self.store.commit(batch)

    def _step(self, thought: str, *, operation: str = "future_projection",
              next_operation: str = "feasibility_check",
              alternatives: tuple[InquiryAlternative, ...] | None = None,
              relations: tuple[ThoughtRelation, ...] = ()) -> InquiryStep:
        """One possible interpretation under the operation the graph admitted.

        Only a generative operation may branch, so a non-generative step passes
        ``alternatives=()``.
        """
        branches = (InquiryAlternative(text="Jesse revisits the sketch.", evidence_ids=(self.source.id,),
            assumptions=("Interest remains.",), reality="imagined", judgments=self.judgments,
            disposition="open", reason="A possible cause, not observation."),) if alternatives is None else alternatives
        return InquiryStep(operation=operation, thought=thought, evidence_ids=(self.source.id,),
            assumptions=(), reality="inferred", judgments=self.judgments, relations=relations,
            alternatives=branches,
            progress=InquiryProgress(kind="none", inspected_source_ids=(), compared_node_ids=(),
                meaningful_difference=False, reason="Possibilities provide no new outside evidence."),
            outcome="continue", reason="Compare the assumptions before concluding.",
            information_gap="Unknown motive.", next_operation=next_operation)

    def _first_step(self) -> _ExecutedStep:
        """Select one question and execute one real graph operation."""
        batch = self.store.begin_batch("nomination", embed=False)
        nomination = ThoughtNomination(text="Jesse may return to the sketch or drop it.",
            question="Which explanation has evidence?", evidence_ids=(self.source.id,),
            reality="inferred", judgments=self.judgments, suggested_operation="future_projection")
        records = stage_nominations(batch, [nomination], grant=self.grant, batch_id="nomination")
        batch.preserve_on_recovery()
        self.store.commit(batch)
        selected = self.owner.accept_attention([records[0].record.id],
            ThoughtAttentionDecision(nomination_id=records[0].record.id, disposition="select",
                reason="This unresolved question holds private attention."),
            commit_id="attention", episode_id="private-sketch", now_utc=self.now)
        assert selected.inquiry is not None
        executed = self.owner.accept_step(selected.inquiry, self._step("Both explanations remain possible."),
            grant=self.owner.execution_grant(selected.inquiry), commit_id="generation", active_seconds=1.5,
            now_utc=self.now, mental_state=FullMentalState(), control_generation=0)
        operation_id = executed.receipt.operation_record_id
        assert operation_id is not None
        branches = tuple(node_id for node_id in executed.receipt.state.node_ids if node_id != operation_id)
        return _ExecutedStep(executed=executed, operation_id=operation_id, branch_ids=branches,
                             nomination_id=records[0].record.id)

    def test_an_executed_graph_operation_is_an_episode(self) -> None:
        step = self._first_step()
        operation_id, branches = step.operation_id, step.branch_ids
        episode = EpisodeIndex(self.store).by_id(operation_id)
        assert episode is not None
        self.assertEqual(episode.kind, "inquiry_operation")
        self.assertEqual(episode.loop, "inquiry")
        self.assertEqual(episode.owner, "workflows.inner_thought")
        self.assertEqual(episode.operation, "future_projection")
        self.assertEqual(episode.summary, "Both explanations remain possible.")
        # A step that continues leaves its question open, so the episode is not
        # a committed result, and the operation's own word stays in metadata.
        self.assertEqual(episode.outcome, "unresolved")
        self.assertEqual(episode.metadata["inquiry_outcome"], "continue")
        self.assertEqual(episode.metadata["reality"], "inferred")
        self.assertEqual(episode.metadata["information_gap"], "Unknown motive.")
        progress = InquiryProgressRead.model_validate(episode.metadata["progress"])
        self.assertEqual(progress.kind, "none")
        # The first step answers the nomination that opened its episode, not
        # itself. The cause link is the nomination record id.
        self.assertEqual(episode.links["cause"], (step.nomination_id,))
        self.assertEqual(episode.inputs, (self.source.id,))
        self.assertEqual(episode.outputs, (operation_id, *branches))
        self.assertEqual(episode.links["proposal"], branches)
        alternatives = cast("list[AlternativeRow]", episode.metadata["alternatives"])
        self.assertEqual(alternatives, [{"text": "Jesse revisits the sketch.", "disposition": "open"}])
        # A generated branch is a record the step produced, not a decision of
        # its own, so it never anchors a second episode.
        self.assertIsNone(EpisodeIndex(self.store).by_id(branches[0]))

    def test_the_private_loop_sees_its_own_graph_step_as_recent_evidence(self) -> None:
        """A reflection step reads its own method history through this index.

        The private loop asks :func:`workflows.reflection.experiments.context` for
        its recent episodes. A mind that thinks in the graph could not notice its
        own recent thinking, nor judge how it has been working, if its operations
        were absent from that view. The rendered material is what the step reads.
        """
        step = self._first_step()
        material = reflection_context(self.store.snapshot(capture=False))
        found = next((item for item in material.episodes if item.episode_id == step.operation_id), None)
        self.assertIsNotNone(found)
        assert found is not None
        # ``ReflectionEpisode.metadata`` is the typed reflection projection, so the
        # step reads named fields rather than the record's JSON envelope.
        self.assertEqual(found.metadata.reality, "inferred")
        self.assertEqual([item.disposition for item in found.metadata.alternatives], ["open"])
        self.assertIn("Both explanations remain possible.", str(private_material(material)))

    def test_a_graph_step_renders_once_with_its_reason(self) -> None:
        """The companion reads one step once, with the step's own reasons.

        One step commits an operation row and an inner frame row. The frame row
        has the later tick and repeats the same text, so a naive recent list
        showed one thought three times and hid why the step stopped.
        """
        step = self._first_step()
        material = reflection_context(self.store.snapshot(capture=False))
        text = str(private_material(material))
        # The step appears once as an episode. Its own record still appears in
        # the cited evidence, which is provenance, not a second episode row.
        self.assertEqual([item.summary for item in material.episodes].count(
            "Both explanations remain possible."), 1)
        self.assertIn("Compare the assumptions before concluding.", text)
        self.assertIn("Unknown motive.", text)
        frames = [item for item in material.episodes
                  if item.kind == "gwt_turn" and step.operation_id in item.outputs]
        self.assertEqual(frames, [])

    def test_the_newest_companion_episode_answers_with_the_step_evidence(self) -> None:
        """The host answers the newest companion row, which is the step itself.

        The frame row has the later tick and no evidence links. Reading
        ``recent[0]`` therefore answered "none retained" for a step that cites
        its original source.
        """
        step = self._first_step()
        index = EpisodeIndex(self.store)
        newest = recent_without_duplicate_inner_frames(index, limit=5)[0]
        self.assertEqual(newest.episode_id, step.operation_id)
        answers = answer_questions(self.store, newest)
        cited = [row["record_id"] for row in answers["information_available"]["evidence"]
                 if "record_id" in row]
        self.assertEqual(cited, [self.source.id])

    def test_owner_queries_keep_both_rows_while_the_companion_projection_drops_the_frame(self) -> None:
        step = self._first_step()
        index = EpisodeIndex(self.store)
        frames = [item for item in index.all()
                  if item.kind == "gwt_turn" and step.operation_id in item.outputs]
        self.assertEqual(len(frames), 1)
        companion = recent_without_duplicate_inner_frames(index, limit=6)
        self.assertNotIn(frames[0].episode_id, [item.episode_id for item in companion])
        self.assertEqual([item.episode_id for item in companion if item.loop == "inquiry"],
                         [step.operation_id])

    def test_a_second_graph_step_names_the_idea_it_answered(self) -> None:
        first = self._first_step()
        operation_id = first.operation_id
        executed = self.owner.accept_step(first.executed,
            self._step("The unfinished sketch itself is the better sign.", operation="feasibility_check",
                next_operation="future_projection", alternatives=(),
                relations=(ThoughtRelation(target_record_id=operation_id, kind="extends",
                    reason="It answers which explanation has evidence."),),),
            grant=self.owner.execution_grant(first.executed), commit_id="second", active_seconds=1.5,
            now_utc=self.now, mental_state=FullMentalState(), control_generation=0)
        second_id = executed.receipt.operation_record_id
        assert second_id is not None
        later = EpisodeIndex(self.store).by_id(second_id)
        assert later is not None
        self.assertEqual(later.links["cause"], (operation_id,))
        evidence = answer_questions(self.store, later)["cause"]["evidence"]
        # A cause answer cites record rows; an alternative row has no record id.
        cited = [row for row in evidence if "record_id" in row]
        self.assertEqual([row["record_id"] for row in cited], [operation_id])

    def test_a_graph_step_shows_its_operation_and_its_workspace_frame(self) -> None:
        """One accepted step contributes its decision row and its frame row.

        The graph commits the operation record and the inner workspace turn that
        holds it. That turn records no decision identity, so the index keeps both
        rows: the frame reports what the private mind was holding, and the decision
        row cites the evidence, the outcome, and the earlier node it answered.
        """
        operation_id = self._first_step().operation_id
        index = EpisodeIndex(self.store)
        self.assertEqual([item.episode_id for item in index.recent(loop="inquiry")], [operation_id])
        frames = [item for item in index.all()
                  if item.kind == "gwt_turn" and item.loop == "inner" and operation_id in item.outputs]
        self.assertEqual(len(frames), 1)

    def test_a_main_turn_steered_by_graph_work_shows_the_influence(self) -> None:
        step = self._first_step()
        operation_id, branches = step.operation_id, step.branch_ids
        batch = self.store.begin_batch("main", embed=False, new_tick=True)
        batch.set_gwt(GwtTurnRecord(tick_id=batch.tick_id, slot_id="main", stimulus_ids=[branches[0]],
            knoxel_ids=[branches[0]], summary="answer about the sketch"), CycleState(tick_id=batch.tick_id,
            stimulus_groups=[StimulusGroup(stimulus_ids=[branches[0]])]))
        self.store.commit(batch)
        pairs = EpisodeIndex(self.store).cross_loop_links()
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["inner_episode_id"], operation_id)
        self.assertEqual(pairs[0]["via_record_id"], branches[0])


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
