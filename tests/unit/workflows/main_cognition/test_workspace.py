"""Workspace mechanics of A19F step 6: structures, attention into shards, and causality.

The selection tests use the pure functions of
``workflows/main_cognition/workspace_selection.py`` with small records and
fixed token counts. The cycle tests run the real manifest and the real host
with the scripted model of ``cycle_support``.
"""
from __future__ import annotations

import random
import unittest
from datetime import datetime, timedelta

from pm.constants import (
    FOCUS_ACTIVATION_WEIGHT,
    FOCUS_MAX_SHARE,
    WORKSPACE_HOLD_BONUS,
    WORKSPACE_SHARD_TOKENS,
)
from pm.records.situational_model import SituationalItem, SituationalModel
from pm.records.workspace import (
    SELF_SHARD,
    RecordSelector,
    ShardKey,
    ShardRequirementSet,
    WorkspaceItem,
    WorkspaceRequirement,
    WorkspaceShard,
)

from workflows.main_cognition.phases.structures import TopicalCoalitionBuilder, average_linkage
from workflows.main_cognition.schemas import RecordView
from workflows.main_cognition.workspace_selection import (
    Candidate,
    MatchContext,
    SelectionInput,
    protected_ids,
    select_shards,
    select_tail,
    selector_matches,
)

START = datetime(2026, 10, 18, 12, 0)
SELF_ID = 2
USER_ID = 1
USER_SHARD = ShardKey(kind="entity", ref=str(USER_ID))


def view(record_id: int, record_class: str = "MemoryClusterRecord", *, entity_ids=(), category: str = "",
         feature_type: str = "", content: str = "") -> RecordView:
    """Return a record view with a world time that follows its id."""
    return RecordView(record_id=record_id, kind=record_class, record_class=record_class, entity_ids=list(entity_ids),
                      category=category, feature_type=feature_type, content=content or f"record {record_id}",
                      timestamp=START + timedelta(minutes=record_id))


CONTEXT = MatchContext(self_entity_id=SELF_ID, group_entity_ids=[USER_ID], stimulus_ids=[100])


def matcher(records: dict[int, RecordView]):
    return lambda requirement, record_id, key: selector_matches(requirement.select, records[record_id], key, CONTEXT)


def requirement(name: str, *, record=(), entity="any", min_items=0, max_items=None,
                max_token_share=None) -> WorkspaceRequirement:
    return WorkspaceRequirement(name=name, select=RecordSelector(record=list(record), entity=entity),
                                min_items=min_items, max_items=max_items, max_token_share=max_token_share)


def selection(records, newcomers, *, shard=None, requirements=(), tokens=100, activation=None, blocked=(),
              focus_ids=()) -> SelectionInput:
    """Return a one-shard selection over ``records`` with equal token counts."""
    shard = shard or WorkspaceShard(key=SELF_SHARD, requirement_set="self")
    return SelectionInput(
        tick=5, shards=[shard], requirement_sets={"self": ShardRequirementSet(requirements=list(requirements))},
        newcomers=newcomers, activation=dict(activation or {}),
        tokens={record_id: tokens for record_id in records}, blocked=list(blocked), focus_ids=list(focus_ids))


def offer(record_id: int, priority: float, *, structure_id: int = -50, bonus: float = 0.0) -> Candidate:
    return Candidate(record_id=record_id, key=SELF_SHARD, priority=priority, focus_bonus=bonus,
                     structure_id=structure_id, reason=f"structure {structure_id}")


class BudgetAndTailTests(unittest.TestCase):
    def test_the_budget_stops_the_lower_rated_records(self):
        records = {key: view(key) for key in (1, 2, 3)}
        tokens = WORKSPACE_SHARD_TOKENS // 2
        chosen = selection(records, [offer(1, 0.9), offer(2, 0.8), offer(3, 0.7)], tokens=tokens)
        change = select_shards(chosen, matcher(records))[0]
        self.assertEqual([item.record_id for item in chosen.shards[0].items], [1, 2])
        self.assertEqual([item.record_id for item in change.entered], [1, 2])
        self.assertEqual(sum(item.tokens for item in chosen.shards[0].items), WORKSPACE_SHARD_TOKENS)

    def test_a_tail_is_the_newest_stretch_within_its_budget_without_stimuli(self):
        sizes = {1: 40, 2: 40, 3: 40, 4: 40, 5: 10}
        tail = select_tail([1, 2, 3, 4, 5], sizes.__getitem__, budget=100, exclude={5})
        self.assertEqual(tail, [3, 4])

    def test_a_tail_stops_at_the_first_record_that_does_not_fit(self):
        sizes = {1: 5, 2: 500, 3: 30}
        self.assertEqual(select_tail([1, 2, 3], sizes.__getitem__, budget=100, exclude=set()), [3])


class RequirementTests(unittest.TestCase):
    def test_a_reserved_place_takes_the_best_matching_csm_item_when_no_structure_has_one(self):
        records = {1: view(1), 2: view(2), 7: view(7, "Narrative", entity_ids=[SELF_ID], category="SelfImage"),
                   8: view(8, "Narrative", entity_ids=[SELF_ID], category="Relations")}
        rule = requirement("self_narrative", record=["Narrative"], entity="self", min_items=1)
        chosen = selection(records, [offer(1, 0.9), offer(2, 0.8)], requirements=[rule],
                           activation={7: 0.3, 8: 0.6, 1: 0.5, 2: 0.5})
        select_shards(chosen, matcher(records))
        items = {item.record_id: item for item in chosen.shards[0].items}
        self.assertEqual(set(items), {1, 2, 8})
        self.assertEqual(items[8].required_by, "self_narrative")
        self.assertIn("best matching CSM item", items[8].reason)

    def test_a_reserved_place_prefers_a_rated_structure_record_even_below_the_minimum_rating(self):
        records = {7: view(7, "Narrative", entity_ids=[SELF_ID]), 8: view(8, "Narrative", entity_ids=[SELF_ID])}
        rule = requirement("self_narrative", record=["Narrative"], entity="self", min_items=1)
        chosen = selection(records, [offer(7, 0.2)], requirements=[rule], activation={8: 0.9})
        select_shards(chosen, matcher(records))
        self.assertEqual([(item.record_id, item.required_by) for item in chosen.shards[0].items],
                         [(7, "self_narrative")])

    def test_an_entity_narrative_does_not_fill_a_self_reservation(self):
        records = {7: view(7, "Narrative", entity_ids=[USER_ID])}
        rule = requirement("self_narrative", record=["Narrative"], entity="self", min_items=1)
        chosen = selection(records, [], requirements=[rule], activation={7: 0.9})
        select_shards(chosen, matcher(records))
        self.assertEqual(chosen.shards[0].items, [])

    def test_max_items_caps_the_matching_records(self):
        records = {key: view(key, "CodeletPercept") for key in (1, 2, 3)}
        rule = requirement("percepts", record=["CodeletPercept"], max_items=2)
        chosen = selection(records, [offer(1, 0.9), offer(2, 0.8), offer(3, 0.7)], requirements=[rule])
        change = select_shards(chosen, matcher(records))[0]
        self.assertEqual([item.record_id for item in chosen.shards[0].items], [1, 2])
        self.assertEqual(change.left, [])

    def test_max_token_share_caps_the_matching_records_but_not_others(self):
        records = {1: view(1, "CodeletPercept"), 2: view(2, "CodeletPercept"), 3: view(3)}
        rule = requirement("percepts", record=["CodeletPercept"], max_token_share=0.1)
        tokens = int(WORKSPACE_SHARD_TOKENS * 0.06)
        chosen = selection(records, [offer(1, 0.9), offer(2, 0.8), offer(3, 0.7)], requirements=[rule], tokens=tokens)
        select_shards(chosen, matcher(records))
        self.assertEqual([item.record_id for item in chosen.shards[0].items], [1, 3])

    def test_the_csm_requirement_protects_narratives_and_stimuli_from_prune(self):
        records = {100: view(100, "Feature", feature_type="Dialogue"),
                   7: view(7, "Narrative", entity_ids=[SELF_ID]), 8: view(8, "Narrative", entity_ids=[USER_ID]),
                   9: view(9, "Narrative", entity_ids=[5]), 10: view(10)}
        csm = SituationalModel(items={key: SituationalItem(record_id=key, activation=0.01) for key in records})
        rules = ShardRequirementSet(requirements=[
            requirement("current_stimuli").model_copy(update={"select": RecordSelector(scope="current_stimuli")}),
            requirement("self_narratives", record=["Narrative"], entity="self"),
            requirement("group_narratives", record=["Narrative"], entity="group")])
        protected = protected_ids(csm, rules, records, CONTEXT)
        self.assertEqual(protected, {100, 7, 8})
        removed = csm.prune(protected)
        self.assertEqual(sorted(removed), [9, 10])
        self.assertEqual(sorted(csm.items), [7, 8, 100])


class IncumbentTests(unittest.TestCase):
    def test_an_incumbent_with_the_hold_bonus_beats_a_slightly_better_newcomer(self):
        records = {1: view(1), 2: view(2)}
        shard = WorkspaceShard(key=SELF_SHARD, requirement_set="self", items=[
            WorkspaceItem(record_id=1, entered_tick=3, reason="earlier", tokens=10)])
        tokens = WORKSPACE_SHARD_TOKENS
        newcomer = 0.6
        chosen = selection(records, [offer(2, newcomer)], shard=shard, tokens=tokens,
                           activation={1: newcomer - WORKSPACE_HOLD_BONUS / 2})
        change = select_shards(chosen, matcher(records))[0]
        self.assertEqual([item.record_id for item in chosen.shards[0].items], [1])
        self.assertEqual(chosen.shards[0].items[0].entered_tick, 3)
        self.assertEqual(chosen.shards[0].items[0].reason, "earlier")
        self.assertEqual(change.kept_ids, [1])
        self.assertEqual(change.entered, [])

    def test_a_much_better_newcomer_displaces_the_incumbent_and_both_changes_are_recorded(self):
        records = {1: view(1), 2: view(2)}
        shard = WorkspaceShard(key=SELF_SHARD, requirement_set="self", items=[
            WorkspaceItem(record_id=1, entered_tick=3, reason="earlier", tokens=10)])
        chosen = selection(records, [offer(2, 0.95)], shard=shard, tokens=WORKSPACE_SHARD_TOKENS,
                           activation={1: 0.5})
        change = select_shards(chosen, matcher(records))[0]
        self.assertEqual([item.record_id for item in change.entered], [2])
        self.assertEqual(change.entered[0].entered_tick, 5)
        self.assertEqual(change.entered[0].structure_id, -50)
        self.assertEqual([item.record_id for item in change.left], [1])
        self.assertIn("displaced", change.left[0].reason)

    def test_a_faded_incumbent_leaves_with_its_reason(self):
        records = {1: view(1)}
        shard = WorkspaceShard(key=SELF_SHARD, requirement_set="self", items=[WorkspaceItem(record_id=1, tokens=10)])
        change = select_shards(selection(records, [], shard=shard, activation={1: 0.1}), matcher(records))[0]
        self.assertIn("faded", change.left[0].reason)

    def test_a_record_enters_only_the_shard_where_it_rates_higher(self):
        records = {1: view(1)}
        shards = [WorkspaceShard(key=SELF_SHARD, requirement_set="self"),
                  WorkspaceShard(key=USER_SHARD, requirement_set="entity")]
        chosen = SelectionInput(tick=5, shards=shards, requirement_sets={}, tokens={1: 10}, newcomers=[
            offer(1, 0.6), Candidate(record_id=1, key=USER_SHARD, priority=0.9, reason="user")])
        changes = select_shards(chosen, matcher(records))
        self.assertEqual(shards[0].items, [])
        self.assertEqual([item.record_id for item in shards[1].items], [1])
        self.assertEqual([len(change.entered) for change in changes], [0, 1])


class HomeShardTests(unittest.TestCase):
    def test_a_tied_rating_puts_each_narrative_into_its_home_shard(self):
        from pm.records.cycle_records import StimulusGroup
        from workflows.main_cognition.cycle_work import CycleRequest, CycleWork
        from workflows.main_cognition.phases.attention import newcomers
        from workflows.main_cognition.schemas import RatedStructure, ShardRatings, StructureDraft

        records = {7: view(7, "Narrative", entity_ids=[SELF_ID]), 8: view(8, "Narrative", entity_ids=[USER_ID])}
        work = CycleWork(request=CycleRequest(self_entity_id=SELF_ID, user_entity_id=USER_ID), records=records)
        work.state.stimulus_groups = [StimulusGroup(key=USER_SHARD)]
        work.structures.structures = [StructureDraft(record_id=-9, record_ids=[7, 8], target_shard=SELF_SHARD)]
        ratings = [ShardRatings(key=key, ratings=[RatedStructure(structure_id=-9, label="S1", rating=0.9, score=0.9)])
                   for key in (SELF_SHARD, USER_SHARD)]
        shards = [WorkspaceShard(key=SELF_SHARD, requirement_set="self"),
                  WorkspaceShard(key=USER_SHARD, requirement_set="entity")]
        chosen = SelectionInput(tick=5, shards=shards, tokens={7: 10, 8: 10}, newcomers=newcomers(work, ratings, {}))
        select_shards(chosen, matcher(records))
        self.assertEqual(shards[0].item_ids(), [7])
        self.assertEqual(shards[1].item_ids(), [8])


class FocusTests(unittest.TestCase):
    def test_deliberate_focus_fills_at_most_its_share_of_the_budget(self):
        records = {key: view(key) for key in range(1, 11)}
        tokens = int(WORKSPACE_SHARD_TOKENS * 0.1)
        # Every record is promoted by focus and rated above the others.
        chosen = selection(records, [offer(key, 0.5, bonus=FOCUS_ACTIVATION_WEIGHT) for key in records],
                           tokens=tokens, focus_ids=list(records))
        select_shards(chosen, matcher(records))
        used = sum(item.tokens for item in chosen.shards[0].items)
        self.assertLessEqual(used, FOCUS_MAX_SHARE * WORKSPACE_SHARD_TOKENS)
        self.assertEqual(len(chosen.shards[0].items), 3)

    def test_focus_cannot_remove_a_reserved_item(self):
        records = {7: view(7, "Narrative", entity_ids=[SELF_ID]), 8: view(8)}
        rule = requirement("self_narrative", record=["Narrative"], entity="self", min_items=1)
        chosen = selection(records, [offer(7, 0.6, bonus=-FOCUS_ACTIVATION_WEIGHT), offer(8, 0.7)],
                           requirements=[rule])
        select_shards(chosen, matcher(records))
        self.assertIn(7, chosen.shards[0].item_ids())

    def test_focus_endorsement_is_bounded_and_named(self):
        from types import SimpleNamespace

        from workflows.cognitive_control.schemas import FocusState
        from workflows.main_cognition.cycle_work import CycleWork
        from workflows.main_cognition.phases.attention import DELIBERATE_FOCUS, apply_focus

        work = CycleWork(records={1: view(1), 2: view(2)})
        work.state.csm.items = {1: SituationalItem(record_id=1, activation=0.2),
                                2: SituationalItem(record_id=2, activation=0.6)}
        focus = FocusState(accept=True, strength=0.9, promote_ids=[1], reduce_ids=[2])
        embedder = SimpleNamespace(record=lambda record_id: [1.0, float(record_id)])
        bonus = apply_focus(work, focus, embedder)
        self.assertEqual(bonus, {1: FOCUS_ACTIVATION_WEIGHT, 2: -FOCUS_ACTIVATION_WEIGHT})
        self.assertAlmostEqual(work.state.csm.items[1].activation, 0.2 + FOCUS_ACTIVATION_WEIGHT)
        self.assertIn(DELIBERATE_FOCUS, work.state.csm.items[1].endorsers)
        self.assertAlmostEqual(work.state.csm.items[2].activation, 0.6 - FOCUS_ACTIVATION_WEIGHT)
        # In danger, focus cannot reduce anything.
        work.state.csm.items[2].activation = 0.6
        apply_focus(work, focus, embedder, danger=True)
        self.assertAlmostEqual(work.state.csm.items[2].activation, 0.6)


class CoalitionTests(unittest.TestCase):
    def test_average_linkage_separates_two_topics(self):
        vectors = [[1.0, 0.0, 0.1], [0.9, 0.1, 0.0], [1.0, 0.05, 0.05], [0.0, 1.0, 0.1], [0.1, 0.9, 0.0]]
        self.assertEqual(average_linkage(vectors, 2), [0, 0, 0, 1, 1])
        self.assertEqual(average_linkage(vectors, 1), [0] * 5)
        self.assertEqual(average_linkage(vectors, 5), [0, 1, 2, 3, 4])

    def test_hard_coalitions_follow_clusters_and_balanced_ones_mix_types(self):
        # Sixteen records give two clusters. Topic A is near the x axis, topic B near the y axis.
        embeddings, records = {}, {}
        for index in range(16):
            topic_a = index < 8
            embeddings[index + 1] = [1.0, 0.05 * (index % 4)] if topic_a else [0.05 * (index % 4), 1.0]
            kind = "CodeletPercept" if index % 2 else "MemoryClusterRecord"
            records[index + 1] = view(index + 1, kind)
        builder = TopicalCoalitionBuilder(records, embeddings.__getitem__, random.Random(3))
        drafts, clusters = builder.build(list(records), SituationalModel())
        self.assertEqual(clusters, 2)
        hard = [draft for draft in drafts if draft.kind == "hard"]
        balanced = [draft for draft in drafts if draft.kind == "balanced"]
        self.assertEqual(sorted(sorted(draft.record_ids) for draft in hard), [list(range(1, 9)), list(range(9, 17))])
        self.assertEqual(len(balanced), 2)
        for draft in balanced:
            self.assertEqual({records[key].record_class for key in draft.record_ids},
                             {"CodeletPercept", "MemoryClusterRecord"})
        # Every coalition is in world-time order and names its target shard.
        for draft in drafts:
            times = [records[key].timestamp for key in draft.record_ids]
            self.assertEqual(times, sorted(times))
            self.assertEqual(draft.target_shard, SELF_SHARD)


class CycleTests(unittest.TestCase):
    """The real manifest and host with the scripted model."""

    def test_a_percept_that_enters_a_shard_becomes_causal_and_others_stay_unconscious(self):
        from tempfile import TemporaryDirectory

        from pm.bus.envelopes import Envelope, new_id
        from pm.records.codelet_records import CodeletPercept, Structure
        from pm.records.mental_state import features_to_history
        from tests.unit.workflows.main_cognition.test_cognition_state import _module, _reports, _small_ghost, _store

        ghost = _small_ghost()
        stimulus_id = _store(ghost, "you dodged it again")
        with TemporaryDirectory() as tmp:
            module, bus, errors = _module(ghost, _reports(stimulus_id), tmp)
            bus.put(Envelope(id=new_id(), source="user_input", kind="data", topic="user.input",
                             correlation_id="corr-causal", payload={"knoxel_id": stimulus_id}))
            self.assertTrue(module.step(block=False))
        self.assertEqual(errors, [])
        state = ghost.states[-1]
        conscious = set(state.workspace.item_ids())
        percepts = [item for item in ghost.all_knoxels.values() if isinstance(item, CodeletPercept)]
        self.assertTrue(conscious & {item.id for item in percepts})
        for percept in percepts:
            self.assertEqual(percept.causal, percept.id in conscious, percept.content)
        # Causal percepts enter the EMA history; unconscious ones do not.
        without = features_to_history(ghost.all_features)
        with_percepts = features_to_history(ghost.all_features, ghost.all_codelet_percepts)
        self.assertEqual(len(with_percepts) - len(without), sum(1 for item in percepts if item.causal))
        # Structures are records, and the cycle state names them.
        structures = [item for item in ghost.all_knoxels.values() if isinstance(item, Structure)]
        self.assertEqual(sorted(state.structure_ids), sorted(item.id for item in structures))
        self.assertTrue(all(item.primitive_metadata["reason"] for item in structures))
        # The turn record lists the entered items with record ids and the real tick.
        turn = ghost.gwt_turns[state.tick_id]
        entered = [item for change in turn.shard_changes for item in change.entered]
        self.assertTrue(entered)
        self.assertTrue(all(item.record_id > 0 and item.entered_tick == state.tick_id for item in entered))
        self.assertTrue(conscious <= set(turn.knoxel_ids))
        # A codelet whose percept entered gets a higher base-level activation.
        names = {item.codelet_name for item in percepts if item.causal}
        self.assertTrue(all(state.codelets.codelets[name].base_activation > 0.5 for name in names))


if __name__ == "__main__":
    unittest.main()
