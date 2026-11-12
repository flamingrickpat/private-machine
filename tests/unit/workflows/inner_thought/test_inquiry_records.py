"""Exercise inquiry storage boundaries with actual commits and checkpoint reloads.

Explicit boundary proposals are test inputs. No stage, model, kernel, or
collaborator is replaced. These checks do not certify generated thought.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pydantic import ValidationError

from pm.memory.checkpoint import JSON_SERIALIZER
from pm.memory.snapshot import GhostConflict
from pm.memory.store import MemoryStore
from pm.memory.write_batch import WriteBatch
from pm.records.base import Provenance, RecordScope
from pm.records.cycle_state import PersonaConfig
from pm.records.enums import ClusterType, EntityClass, FeatureType, InterlocusType, RealityKind, SourceKind
from pm.records.lineage import RecordLineage
from pm.records.memory_records import Entity, Feature, MemoryClusterRecord, RealityRecord
from workflows.cognitive_control.schemas import ControlConfig
from workflows.inner_thought.inquiry import (
    AttentionRecord, InquiryRecord, inquiry_history, read_legacy_inquiry, stage_attention, stage_inquiry_step,
)
from workflows.inner_thought.nominations import (
    capture_inquiry_grant, extend_inquiry_grant, read_nomination, source_roots, stage_nominations,
)
from workflows.inner_thought.schemas import (
    InquiryAlternative, InquiryGrant, InquiryProgress, InquiryStep, LegacyEpisodeState,
    ThoughtAttentionDecision, ThoughtJudgments, ThoughtNomination, ThoughtRelation,
)


class InquiryStoreTests(unittest.TestCase):
    """Check graph causality, original evidence, bounded state, and recovery."""

    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path: Path = Path(directory.name) / "inquiry.pickle"
        self.store: MemoryStore = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"), checkpoint=self.path)
        batch = self.store.begin_batch("anchors", embed=False)
        self.observer: Entity = batch.create(Entity(content="Spamuel", aliases=[], entity_class=EntityClass.Self))
        self.partner: Entity = batch.create(Entity(content="Jesse", aliases=[], entity_class=EntityClass.Human))
        self.source: Feature = batch.create(Feature(content="I left the sketch unfinished.",
            feature_type=FeatureType.Dialogue, entity_id=self.partner.id,
            provenance=Provenance(source_kind=SourceKind.User)))
        batch.preserve_on_recovery()
        self.store.commit(batch)
        self.now: datetime = datetime(2026, 11, 12, 16, 0)
        self.limits: ControlConfig = ControlConfig()
        self.judgments: ThoughtJudgments = ThoughtJudgments(interestingness=0.8, character_relevance=0.7,
            plausibility=0.3, consequence=0.4, desirability=0.1, action_suitability=0.2,
            reason="The sketch matters to this character; its unfinished state has several possible explanations.")
        self.grant: InquiryGrant = capture_inquiry_grant(self.store.snapshot(capture=False),
            observer_entity_id=self.observer.id, scope=RecordScope(), source_ids=[self.source.id], critical_epoch=3)

    def batch(self, name: str) -> WriteBatch:
        """Open the actual append-only store boundary without embedding calls."""
        return self.store.begin_batch(name, new_tick=True, embed=False)

    def commit(self, batch: WriteBatch) -> None:
        """Retain these append-only records through the production recovery path."""
        batch.preserve_on_recovery()
        self.store.commit(batch)

    def nominate(self, *, batch_id: str = "nominate", grant: InquiryGrant | None = None) -> int:
        """Commit an explicit proposal, without executing a model or a thought."""
        batch = self.batch(batch_id)
        nomination = ThoughtNomination(text="An unfinished sketch can mean hesitation or lost interest.",
            question="What evidence distinguishes those explanations?", evidence_ids=(self.source.id,),
            reality="inferred", judgments=self.judgments, suggested_operation="future_projection")
        records = stage_nominations(batch, [nomination], grant=grant if grant is not None else self.grant, batch_id=batch_id)
        self.commit(batch)
        return records[0].record.id

    def select(self, nomination_id: int, *, episode_id: str = "inquiry") -> AttentionRecord:
        """Commit attention input through the real staging and persistence boundary."""
        batch = self.batch("select-" + episode_id)
        result = stage_attention(batch, [nomination_id], ThoughtAttentionDecision(nomination_id=nomination_id,
            disposition="select", reason="This question fits the present private attention slot."),
            episode_id=episode_id, now_utc=self.now)
        self.commit(batch)
        return result

    def begin(self) -> InquiryRecord:
        """Persist nomination and selection as distinct accepted records."""
        result = self.select(self.nominate())
        assert result.inquiry is not None
        return result.inquiry

    def generation(self, *, thought: str = "Two possible continuations remain uncertain.") -> InquiryStep:
        """Supply a generative boundary proposal with two independently stored branches."""
        alternatives = tuple(InquiryAlternative(text=text, evidence_ids=(self.source.id,), assumptions=(assumption,),
            reality="imagined", judgments=self.judgments, disposition="open", reason="A possibility, not an observed event.")
            for text, assumption in [("Jesse returns to the sketch.", "Interest remains."),
                                     ("Jesse starts another sketch.", "A different subject matters more.")])
        return InquiryStep(operation="future_projection", thought=thought, evidence_ids=(self.source.id,),
            assumptions=("The sketch is still unfinished.",), reality="inferred", judgments=self.judgments,
            relations=(), alternatives=alternatives,
            progress=InquiryProgress(kind="none", inspected_source_ids=(), compared_node_ids=(),
                meaningful_difference=False, reason="These are generated possibilities, not acquired evidence."),
            outcome="continue", reason="Compare the assumptions before forming a finding.",
            information_gap="Jesse's reason for stopping is unknown.", next_operation="feasibility_check")

    def discrimination(self, prior: InquiryRecord, *, meaningful: bool = True) -> InquiryStep:
        """Supply a comparison of the real accepted graph identities."""
        ids = prior.receipt.state.node_ids
        relations = tuple(ThoughtRelation(target_record_id=node_id, kind="merges",
            reason="Compare both interpretations without changing the original report.") for node_id in ids)
        return InquiryStep(operation="feasibility_check", thought="The report establishes neither motive.",
            evidence_ids=(self.source.id,), assumptions=(), reality="inferred", judgments=self.judgments,
            relations=relations, alternatives=(), progress=InquiryProgress(kind="discrimination",
                inspected_source_ids=(), compared_node_ids=ids, meaningful_difference=meaningful,
                reason="Different predicted choices depend on an unobserved motive." if meaningful else "The same uncertainty remains."),
            outcome="information_gap", reason="The evidence does not distinguish motives.",
            information_gap="Jesse's reason for stopping is unknown.", next_operation="")

    def stage_step(self, batch: WriteBatch, prior: InquiryRecord, step: InquiryStep, *,
                   active_seconds: float = 1.25, limits: ControlConfig | None = None) -> InquiryRecord:
        """Inspect accepted graph nodes, retain old fingerprints, and stage a step."""
        grant = extend_inquiry_grant(self.store.snapshot(capture=False), prior.receipt.state.grant, prior.receipt.state.node_ids)
        return stage_inquiry_step(batch, parent_receipt_id=prior.record.id, step=step, grant=grant,
            active_seconds=active_seconds, now_utc=self.now, limits=limits if limits is not None else self.limits)

    def advance(self, prior: InquiryRecord, step: InquiryStep, *, name: str = "step") -> InquiryRecord:
        """Use an actual durable commit; no generated response is substituted."""
        batch = self.batch(name)
        result = self.stage_step(batch, prior, step)
        self.commit(batch)
        return result

    def test_nomination_and_selection_survive_clean_reload_without_conscious_nodes(self) -> None:
        nomination_id = self.nominate()
        selected = self.select(nomination_id)
        restored = MemoryStore.load(self.path)
        nomination = read_nomination(restored.snapshot(capture=False), nomination_id)
        self.assertEqual(nomination.receipt.nomination.judgments, self.judgments)
        self.assertEqual(nomination.receipt.root_source_ids, (self.source.id,))
        history = inquiry_history(restored.snapshot(capture=False), "inquiry")
        self.assertEqual(history[0].record.id, selected.record.id)
        self.assertEqual(history[0].receipt.state.sequence, 0)
        self.assertEqual(history[0].receipt.state.node_ids, ())
        self.assertEqual(restored.gwt_turns, {})
        self.assertFalse(any(record.feature_type == FeatureType.InnerMonologue for record in restored.all_features))
        encoded = JSON_SERIALIZER.dumps(restored.checkpoint_data())
        decoded = JSON_SERIALIZER.loads(encoded)
        self.assertEqual(decoded.current_knoxel_id, restored.current_knoxel_id)
        self.assertIn(b"thought_nomination", encoded)

    def test_generation_discrimination_and_independent_branch_parents_survive_restart(self) -> None:
        origin = self.begin()
        generated = self.advance(origin, self.generation(), name="generation")
        self.assertEqual(len(generated.receipt.state.node_ids), 3)
        self.assertEqual(generated.receipt.state.pending_generation_ids, generated.receipt.state.node_ids)
        view = self.store.snapshot(capture=False)
        operation_id = generated.receipt.operation_record_id
        assert operation_id is not None
        operation = view.get(operation_id)
        assert operation.lineage is not None
        self.assertEqual(operation.lineage.parent_record_ids, [origin.receipt.state.origin_record_id])
        for branch_id in generated.receipt.alternative_record_ids:
            branch = view.get(branch_id)
            assert branch.lineage is not None
            self.assertEqual(branch.lineage.parent_record_ids, [operation_id])
            self.assertEqual(branch.lineage.reality, "imagined")
        final = self.advance(generated, self.discrimination(generated), name="comparison")
        restored = MemoryStore.load(self.path)
        chain = inquiry_history(restored.snapshot(capture=False), "inquiry")
        self.assertEqual(len(chain), 3)
        self.assertEqual(chain[-1].receipt, final.receipt)
        self.assertEqual(final.receipt.state.root_source_ids, (self.source.id,))
        self.assertEqual(final.receipt.state.pending_generation_ids, ())
        self.assertEqual(final.receipt.state.status, "completed")
        self.assertEqual(final.receipt.state.active_elapsed_seconds, 2.5)
        self.assertEqual(final.receipt.state.stalled_steps, 0)

    def test_empty_batch_and_deferred_attention_create_no_executed_thought(self) -> None:
        batch = self.batch("empty")
        self.assertEqual(stage_nominations(batch, [], grant=self.grant, batch_id="empty"), ())
        self.assertEqual(batch.created_ids, frozenset())
        nomination_id = self.nominate()
        batch = self.batch("defer")
        deferred = stage_attention(batch, [nomination_id], ThoughtAttentionDecision(nomination_id=None,
            disposition="defer", reason="Current conversation takes priority."), episode_id="", now_utc=self.now)
        self.commit(batch)
        self.assertIsNone(deferred.inquiry)
        self.assertEqual(inquiry_history(self.store.snapshot(capture=False), "inquiry"), ())
        self.select(nomination_id)

    def test_selected_and_dismissed_nominations_cannot_allocate_again(self) -> None:
        nomination_id = self.nominate()
        self.select(nomination_id)
        with self.assertRaisesRegex(ValueError, "cannot be selected again"):
            self.select(nomination_id, episode_id="duplicate")
        other_id = self.nominate(batch_id="other")
        batch = self.batch("dismiss")
        stage_attention(batch, [other_id], ThoughtAttentionDecision(nomination_id=other_id,
            disposition="dismiss", reason="This question has no current value."), episode_id="", now_utc=self.now)
        self.commit(batch)
        with self.assertRaisesRegex(ValueError, "cannot be selected again"):
            self.select(other_id, episode_id="dismissed")

    def test_competing_continuations_cannot_commit_the_same_sequence(self) -> None:
        prior = self.begin()
        first_batch = self.batch("first")
        second_batch = self.batch("second")
        first = self.stage_step(first_batch, prior, self.generation())
        self.stage_step(second_batch, prior, self.generation(thought="A competing generation."))
        self.commit(first_batch)
        with self.assertRaises(GhostConflict):
            self.commit(second_batch)
        chain = inquiry_history(MemoryStore.load(self.path).snapshot(capture=False), "inquiry")
        self.assertEqual(chain[-1].record.id, first.record.id)
        self.assertEqual(chain[-1].receipt.state.sequence, 1)

    def test_one_private_slot_rejects_concurrent_selections(self) -> None:
        first_id = self.nominate(batch_id="first-nomination")
        second_id = self.nominate(batch_id="second-nomination")
        first_batch = self.batch("first-selection")
        second_batch = self.batch("second-selection")
        for batch, record_id, episode_id in [(first_batch, first_id, "first"), (second_batch, second_id, "second")]:
            stage_attention(batch, [record_id], ThoughtAttentionDecision(nomination_id=record_id, disposition="select",
                reason="Consider this question."), episode_id=episode_id, now_utc=self.now)
        self.commit(first_batch)
        with self.assertRaises(GhostConflict):
            self.commit(second_batch)
        with self.assertRaisesRegex(GhostConflict, "already holds an active episode"):
            self.select(second_id, episode_id="third")

    def test_uncommitted_step_is_absent_after_reload_and_committed_step_is_not_repeated(self) -> None:
        prior = self.begin()
        batch = self.batch("uncommitted")
        staged = self.stage_step(batch, prior, self.generation())
        restored = MemoryStore.load(self.path)
        self.assertNotIn(staged.record.id, restored.revisions)
        self.assertEqual(len(inquiry_history(restored.snapshot(capture=False), "inquiry")), 1)
        self.commit(batch)
        receipt = self.store.commit(batch)
        self.assertIn(staged.record.id, receipt.added_ids)
        self.assertEqual(len(inquiry_history(MemoryStore.load(self.path).snapshot(capture=False), "inquiry")), 2)

    def test_operational_revision_and_unrelated_input_do_not_refresh_original_fingerprints(self) -> None:
        prior = self.begin()
        update = self.batch("operational")
        update.set_lifecycle(self.source.id, exchange_status="resolved")
        update.create(Feature(content="Unrelated: the kettle is ready.", feature_type=FeatureType.Dialogue,
                              entity_id=self.partner.id, provenance=Provenance(source_kind=SourceKind.User)))
        self.store.commit(update)
        retained = extend_inquiry_grant(self.store.snapshot(capture=False), prior.receipt.state.grant, [self.source.id])
        self.assertEqual(retained.sources, prior.receipt.state.grant.sources)
        self.assertNotEqual(self.store.revisions[self.source.id], self.grant.sources[-1].revision)
        generated = self.advance(prior, self.generation())
        self.assertEqual(generated.receipt.state.grant.sources, retained.sources)

    def test_source_meaning_change_rejects_staged_step_instead_of_rebasing(self) -> None:
        prior = self.begin()
        batch = self.batch("stale-source")
        self.stage_step(batch, prior, self.generation())
        changed = self.batch("changed-meaning")
        changed.set_lifecycle(self.source.id, semantic_correction="The sketch was actually finished.")
        self.store.commit(changed)
        with self.assertRaises(GhostConflict):
            self.commit(batch)
        self.assertEqual(len(inquiry_history(self.store.snapshot(capture=False), "inquiry")), 1)

    def test_historical_sources_survive_successors_but_governing_sources_require_current_authority(self) -> None:
        governing = capture_inquiry_grant(self.store.snapshot(capture=False), observer_entity_id=self.observer.id,
            scope=RecordScope(), source_ids=[self.source.id], governing_ids=[self.source.id], critical_epoch=3)
        historic_id = self.nominate()
        governed_id = self.nominate(batch_id="governed", grant=governing)
        replacement = self.batch("replacement")
        record = replacement.view().get(self.source.id).copy_for_edit()
        record.content = "The sketch is now finished."
        replacement.replace(record)
        self.store.commit(replacement)
        with self.assertRaises(GhostConflict):
            self.select(governed_id, episode_id="governed")
        self.select(historic_id)

    def test_unknown_and_cross_reality_sources_fail_before_any_record_is_staged(self) -> None:
        nomination_id = self.nominate()
        proposal = read_nomination(self.store.snapshot(capture=False), nomination_id).receipt.nomination
        invalid = proposal.model_copy(update={"evidence_ids": (987654,)})
        batch = self.batch("unknown-source")
        with self.assertRaisesRegex(ValueError, "outside its captured grant"):
            stage_nominations(batch, [invalid], grant=self.grant, batch_id="unknown")
        self.assertFalse(batch.created_ids)
        other = self.batch("other-reality")
        reality = other.create(RealityRecord(content="A hypothetical sketch world.", kind=RealityKind.Fantasy))
        foreign = other.create(Feature(content="An imagined external event.", feature_type=FeatureType.Dialogue,
            entity_id=self.partner.id, scope=RecordScope(reality_id=reality.id), provenance=Provenance(source_kind=SourceKind.User)))
        self.commit(other)
        with self.assertRaisesRegex(ValueError, "outside the captured reality"):
            capture_inquiry_grant(self.store.snapshot(capture=False), observer_entity_id=self.observer.id,
                scope=RecordScope(), source_ids=[foreign.id], critical_epoch=3)

    def test_new_nodes_retain_one_original_root_instead_of_self_corroboration(self) -> None:
        generated = self.advance(self.begin(), self.generation())
        view = self.store.snapshot(capture=False)
        grant = extend_inquiry_grant(view, generated.receipt.state.grant, generated.receipt.state.node_ids)
        roots = source_roots(view, generated.receipt.state.node_ids, scope=grant.scope,
                             permitted_ids=frozenset(source.record_id for source in grant.sources))
        self.assertEqual(roots, (self.source.id,))
        missing = self.store.snapshot([generated.receipt.state.node_ids[0]], capture=False)
        with self.assertRaises(KeyError):
            capture_inquiry_grant(missing, observer_entity_id=self.observer.id, scope=RecordScope(),
                                 source_ids=generated.receipt.state.node_ids, critical_epoch=3)

    def test_unknown_graph_relations_and_uninspected_nodes_are_rejected(self) -> None:
        generated = self.advance(self.begin(), self.generation())
        step = self.discrimination(generated)
        unknown = step.model_copy(update={"relations": (ThoughtRelation(target_record_id=999999,
            kind="merges", reason="Unknown relation."),)})
        with self.assertRaisesRegex(ValueError, "already accepted nodes"):
            self.stage_step(self.batch("unknown-relation"), generated, unknown)
        with self.assertRaisesRegex(ValueError, "captured grant of inspected graph nodes"):
            stage_inquiry_step(self.batch("uninspected"), parent_receipt_id=generated.record.id,
                step=step, grant=generated.receipt.state.grant, active_seconds=1, now_utc=self.now, limits=self.limits)

    def test_generation_cannot_become_a_conclusion_before_discrimination(self) -> None:
        prior = self.begin()
        conclusion = self.generation().model_copy(update={"outcome": "finding", "next_operation": ""})
        batch = self.batch("premature-finding")
        with self.assertRaisesRegex(ValueError, "requires discrimination"):
            self.stage_step(batch, prior, conclusion)
        self.assertFalse(batch.created_ids)

    def test_internal_nodes_cannot_count_as_acquired_original_evidence(self) -> None:
        generated = self.advance(self.begin(), self.generation())
        node_id = generated.receipt.state.node_ids[0]
        acquisition = self.discrimination(generated).model_copy(update={"evidence_ids": (node_id,),
            "progress": InquiryProgress(kind="new_evidence", inspected_source_ids=(node_id,), compared_node_ids=(),
                meaningful_difference=True, reason="This is still self-generated content.")})
        with self.assertRaisesRegex(ValueError, "cannot count as acquired original evidence"):
            self.stage_step(self.batch("self-corroboration"), generated, acquisition)

    def test_new_original_source_can_establish_evidence_progress(self) -> None:
        generated = self.advance(self.begin(), self.generation())
        acquired = self.batch("acquired-source")
        source = acquired.create(Feature(content="I still like the subject; I ran out of time.",
            feature_type=FeatureType.Dialogue, entity_id=self.partner.id, provenance=Provenance(source_kind=SourceKind.User)))
        self.commit(acquired)
        step = self.discrimination(generated).model_copy(update={"evidence_ids": (self.source.id, source.id),
            "progress": InquiryProgress(kind="new_evidence", inspected_source_ids=(source.id,), compared_node_ids=generated.receipt.state.node_ids,
                meaningful_difference=True, reason="A newly supplied direct report distinguishes the motives.")})
        grant = extend_inquiry_grant(self.store.snapshot(capture=False), generated.receipt.state.grant,
                                     [*generated.receipt.state.node_ids, source.id])
        batch = self.batch("new-evidence")
        final = stage_inquiry_step(batch, parent_receipt_id=generated.record.id, step=step, grant=grant,
                                  active_seconds=1, now_utc=self.now, limits=self.limits)
        self.commit(batch)
        self.assertEqual(final.receipt.state.root_source_ids, (self.source.id, source.id))
        self.assertEqual(final.receipt.state.stalled_steps, 0)

    def test_already_supplied_uncited_source_is_not_newly_acquired_evidence(self) -> None:
        source_batch = self.batch("already-known")
        other = source_batch.create(Feature(content="A previously supplied concern.", feature_type=FeatureType.Dialogue,
            entity_id=self.partner.id, provenance=Provenance(source_kind=SourceKind.User)))
        self.commit(source_batch)
        self.grant = extend_inquiry_grant(self.store.snapshot(capture=False), self.grant, [other.id])
        prior = self.begin()
        step = self.generation().model_copy(update={"evidence_ids": (other.id,),
            "progress": InquiryProgress(kind="new_evidence", inspected_source_ids=(other.id,), compared_node_ids=(),
                meaningful_difference=True, reason="The source was already supplied, despite not being cited by the nomination.")})
        result = self.advance(prior, step)
        self.assertEqual(result.receipt.state.stalled_steps, 1)

    def test_merge_and_dismissal_preserve_all_original_branch_records(self) -> None:
        generated = self.advance(self.begin(), self.generation())
        dismissed_id = generated.receipt.alternative_record_ids[0]
        step = self.discrimination(generated)
        relations = tuple(ThoughtRelation(target_record_id=relation.target_record_id,
            kind="dismisses" if relation.target_record_id == dismissed_id else "merges",
            reason="This provisional comparison leaves the original branch inspectable.") for relation in step.relations)
        final = self.advance(generated, step.model_copy(update={"relations": relations}), name="dismiss-and-merge")
        view = MemoryStore.load(self.path).snapshot(capture=False)
        self.assertIn(dismissed_id, final.receipt.state.node_ids)
        self.assertNotIn(dismissed_id, final.receipt.state.frontier_ids)
        self.assertEqual(view.get(dismissed_id).content, self.generation().alternatives[0].text)
        assert final.receipt.operation_record_id is not None
        merged = view.get(final.receipt.operation_record_id)
        assert merged.lineage is not None
        self.assertEqual(merged.lineage.parent_record_ids, sorted(generated.receipt.state.node_ids))

    def test_repeated_summary_sources_resolve_to_one_original_report(self) -> None:
        batch = self.batch("summary-association")
        summaries = [batch.create(MemoryClusterRecord(content=text, level=100, cluster_type=ClusterType.Topical,
            included_event_ids=str(self.source.id))) for text in ["The sketch is unfinished.", "The sketch remains unfinished."]]
        self.commit(batch)
        view = self.store.snapshot(capture=False)
        ids = [record.id for record in summaries]
        grant = capture_inquiry_grant(view, observer_entity_id=self.observer.id, scope=RecordScope(),
                                      source_ids=ids, critical_epoch=3)
        roots = source_roots(view, ids, scope=grant.scope, permitted_ids=frozenset(source.record_id for source in grant.sources))
        self.assertEqual(roots, (self.source.id,))

    def test_cyclic_source_lineage_fails_before_generation(self) -> None:
        batch = self.batch("cyclic-thought")
        record = batch.create(Feature(content="A corrupt cyclic thought.", feature_type=FeatureType.Thought,
            entity_id=self.observer.id, provenance=Provenance(source_kind=SourceKind.Self)))
        batch.annotate_lineage(record.id, RecordLineage(episode_id="corrupt", operation="nomination", parent_record_ids=[record.id]))
        self.commit(batch)
        with self.assertRaisesRegex(ValueError, "lineage contains a cycle"):
            capture_inquiry_grant(self.store.snapshot(capture=False), observer_entity_id=self.observer.id,
                scope=RecordScope(), source_ids=[record.id], critical_epoch=3)

    def test_missing_legacy_episode_metadata_fails_without_invented_state(self) -> None:
        batch = self.batch("invalid-legacy")
        record = batch.create(Feature(content="An incomplete old thought.", feature_type=FeatureType.InnerMonologue,
            entity_id=self.observer.id, provenance=Provenance(source_kind=SourceKind.Self),
            lineage=RecordLineage(episode_id="incomplete", operation="memory_gap_probe"),
            metadata={"loop": "inner", "episode": {"episode_id": "incomplete"}}))
        self.commit(batch)
        with self.assertRaises(ValidationError):
            read_legacy_inquiry(self.store.snapshot(capture=False), record.id)

    def test_different_wording_does_not_reset_stalls(self) -> None:
        generated = self.advance(self.begin(), self.generation())
        step = self.discrimination(generated, meaningful=False).model_copy(update={"outcome": "continue",
            "next_operation": "feasibility_check", "thought": "A new sentence about the same uncertainty."})
        repeated = self.advance(generated, step, name="repetition")
        self.assertEqual(repeated.receipt.state.stalled_steps, 2)
        self.assertEqual(repeated.receipt.state.status, "paused")
        self.assertEqual(repeated.receipt.state.stop_reason, "stalled progress")

    def test_step_time_and_stall_bounds_park_without_counting_world_or_offline_time(self) -> None:
        prior = self.begin()
        limits = self.limits.model_copy(update={"episode_steps": 1})
        batch = self.batch("bounded")
        result = self.stage_step(batch, prior, self.generation(), limits=limits)
        self.commit(batch)
        self.assertEqual(result.receipt.state.status, "paused")
        self.assertEqual(result.receipt.state.stop_reason, "step limit")
        self.assertEqual(result.receipt.state.resume_after_utc, self.now + timedelta(seconds=limits.parked_seconds))
        with self.assertRaisesRegex(ValueError, "Only an admitted active inquiry"):
            self.stage_step(self.batch("late"), result, self.generation())
        self.now += timedelta(days=40)
        self.assertEqual(MemoryStore.load(self.path).snapshot(capture=False).get(result.record.id).id, result.record.id)
        self.assertEqual(inquiry_history(MemoryStore.load(self.path).snapshot(capture=False), "inquiry")[-1]
                         .receipt.state.active_elapsed_seconds, 1.25)

    def test_active_computation_time_limit_is_independent_of_world_time(self) -> None:
        prior = self.begin()
        self.now += timedelta(days=40)
        limits = self.limits.model_copy(update={"episode_seconds": 1})
        batch = self.batch("time-bound")
        result = self.stage_step(batch, prior, self.generation(), active_seconds=2, limits=limits)
        self.commit(batch)
        self.assertEqual(result.receipt.state.active_elapsed_seconds, 2)
        self.assertEqual(result.receipt.state.stop_reason, "active computation time limit")
        self.assertEqual(result.receipt.state.started_at_utc, prior.receipt.state.started_at_utc)

    def test_invalid_duration_and_changed_epoch_cannot_change_an_admitted_episode(self) -> None:
        prior = self.begin()
        for value in [-1.0, float("inf"), float("nan")]:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite active seconds"):
                self.stage_step(self.batch(f"invalid-{value}"), prior, self.generation(), active_seconds=value)
        changed = self.grant.model_copy(update={"critical_epoch": 4})
        with self.assertRaisesRegex(ValueError, "cannot replace its observer"):
            stage_inquiry_step(self.batch("epoch-change"), parent_receipt_id=prior.record.id,
                step=self.generation(), grant=changed, active_seconds=1, now_utc=self.now, limits=self.limits)

    def test_limits_and_invalid_judgments_fail_at_the_typed_boundary(self) -> None:
        values = self.judgments.model_dump()
        values["plausibility"] = 1.5
        with self.assertRaises(ValidationError):
            ThoughtJudgments.model_validate(values)
        step = self.generation()
        values_step = step.model_dump()
        values_step["alternatives"] = [item.model_dump() for item in step.alternatives] * 2
        with self.assertRaises(ValidationError):
            InquiryStep.model_validate(values_step)
        nomination_id = self.nominate()
        nomination = read_nomination(self.store.snapshot(capture=False), nomination_id).receipt.nomination
        with self.assertRaisesRegex(ValueError, "at most three"):
            stage_nominations(self.batch("too-many"), [nomination] * 4, grant=self.grant, batch_id="too-many")

    def test_legacy_reader_retains_actual_linear_nodes_and_world_duration(self) -> None:
        parent_id: int | None = None
        for sequence in range(2):
            batch = self.batch(f"legacy-{sequence}")
            state = LegacyEpisodeState(episode_id="old-linear", sequence=sequence, started_at=self.now,
                elapsed_seconds=86400 * sequence, closed=False, closed_reason=None, operations=("memory_gap_probe",),
                evidence_ids=(self.source.id,), alternatives=("An old textual possibility.",), thoughts=("Prior concern.",),
                progress=("none",), stalled=sequence, continuation="continue", next_operation="memory_gap_probe",
                parked_until=None, parked_sources=(), parent_thought_id=parent_id)
            record = batch.create(Feature(content="A retained old thought.", feature_type=FeatureType.InnerMonologue,
                entity_id=self.observer.id, provenance=Provenance(source_kind=SourceKind.Self),
                interlocus=InterlocusType.PrivateInternal,
                lineage=RecordLineage(episode_id="old-linear", operation="memory_gap_probe", sequence=sequence,
                    parent_record_ids=[parent_id] if parent_id is not None else [], source_event_ids=[self.source.id]),
                metadata={"loop": "inner", "episode": state.model_dump(mode="json")}))
            self.commit(batch)
            parent_id = record.id
        assert parent_id is not None
        before = self.path.read_bytes()
        result = read_legacy_inquiry(MemoryStore.load(self.path).snapshot(capture=False), parent_id)
        self.assertEqual(len(result.records), 2)
        self.assertEqual(result.state.alternatives, ("An old textual possibility.",))
        self.assertEqual(result.state.elapsed_seconds, 86400)
        self.assertNotIn("active_elapsed_seconds", result.state.model_dump())
        self.assertNotIn("judgments", result.state.model_dump())
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(inquiry_history(self.store.snapshot(capture=False), "old-linear"), ())


if __name__ == "__main__":
    unittest.main()
