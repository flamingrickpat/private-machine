"""Real-store checks for source projection, target isolation, and publication guards.

These deterministic checks establish context contracts, not model behavior.
The separate live gate runs the real decision phase through the workflow kernel.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import unittest

from pm.context_render import render_context
from pm.memory.snapshot import GhostConflict
from pm.memory.store import MemoryStore
from pm.records.base import Provenance, RecordScope
from pm.records.cognitive_records import BeliefRecord, Intention
from pm.records.cycle_records import StimulusGroup
from pm.records.cycle_state import PersonaConfig
from pm.records.enums import ActionType, EntityClass, FeatureType, FrameEnum, InterlocusType, IntentionType, NarrativeTypes, SourceKind
from pm.records.memory_records import ActionSchemaRecord, Entity, Feature, Narrative, RealityRecord
from pm.records.workspace import SELF_SHARD, ShardKey
from workflows.dialogue_story.schemas import StoryRequest
from workflows.dialogue_story.steps import build_story_messages, load_prompts
from workflows.main_cognition.cycle_work import CycleRequest, CycleWork
from workflows.main_cognition.schemas import ActionRequest
from workflows.main_cognition.story_prefix import StoryPrefix
from workflows.public_speech.public_context import build_public_context
from workflows.public_speech.publication import acknowledge_reply, commit_public_reply, validate_public_context
from workflows.public_speech.situational_guidance import (
    SituationalGuidance, build_situational_guidance, render_situational_guidance,
)


NOW = datetime(2026, 6, 1, 22, 10)


def make_store() -> MemoryStore:
    """Create actual record ownership for a character and two different partners."""
    store = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"))
    for name, kind in (("Spamuel", EntityClass.Self), ("Jesse", EntityClass.Human), ("Mara", EntityClass.AI)):
        store.store_knoxel(Entity(content=name, aliases=[], entity_class=kind))
    return store


def say(store: MemoryStore, speaker: str, text: str, *, addressee: int = 0,
        timestamp: datetime = NOW, visibility: InterlocusType = InterlocusType.Public,
        scope: RecordScope = RecordScope(), causal: bool = True) -> int:
    """Commit attributed source material without substituting for a collaborator."""
    kind = SourceKind.Self if speaker == "Spamuel" else SourceKind.User if speaker == "Jesse" else SourceKind.Agent
    return store.store_knoxel(Feature(content=text, feature_type=FeatureType.Dialogue,
        causal=causal, override_source=speaker, interlocus=visibility, addressee_entity_id=addressee,
        timestamp_world_begin=timestamp, timestamp_world_end=timestamp, scope=scope,
        provenance=Provenance(source_kind=kind, source_ref="scenario"))).id


def guidance(store: MemoryStore, partner: int | None = None) -> SituationalGuidance:
    """Project one frozen snapshot at an explicit runtime world time."""
    return build_situational_guidance(store.snapshot(), as_of_utc=NOW,
        companion_name="Spamuel", user_name="Jesse", partner_entity_id=partner)


class SituationalGuidanceTests(unittest.TestCase):
    def test_other_reality_cannot_supply_partner_state_or_action_purpose(self) -> None:
        store = make_store()
        partner_id = store.get_entity_id("Jesse")
        assert partner_id is not None
        other_world = RecordScope(reality_id=store.store_knoxel(RealityRecord(content="Other world")).id)
        say(store, "Jesse", "Other-world secret dialogue.", scope=other_world)
        store.store_knoxel(Narrative(content="Other-world secret relationship.", entity_id=partner_id,
                                    narrative_type=NarrativeTypes.Relations, scope=other_world))
        store.store_knoxel(BeliefRecord(content="Other-world secret belief.", about_entity_ids=[partner_id], scope=other_world))
        store.store_knoxel(Intention(content="Other-world secret purpose.", entity_id=0, affective_valence=0.4,
                                    intention_type=IntentionType.SelfSituation, scope=other_world))
        store.store_knoxel(ActionSchemaRecord(content="Other-world secret action.", action_type=ActionType.Reply,
                                             scope=other_world))
        self.assertNotIn("Other-world secret", render_situational_guidance(guidance(store)))
        scoped = build_situational_guidance(store.snapshot(), as_of_utc=NOW,
            companion_name="Spamuel", user_name="Jesse", scopes=[other_world])
        self.assertIn("Other-world secret", render_situational_guidance(scoped))
        self.assertIn(f"reality {other_world.reality_id}/in_world", render_situational_guidance(scoped))

    def test_current_head_guard_rejects_superseded_guidance(self) -> None:
        store = make_store()
        partner_id = store.get_entity_id("Jesse")
        assert partner_id is not None
        prior = store.store_knoxel(Narrative(content="The partner is away.", entity_id=partner_id,
                                             narrative_type=NarrativeTypes.Relations))
        frozen = guidance(store)
        correction = store.begin_batch("correct-situation")
        changed = correction.view().get(prior.id).copy_for_edit()
        changed.content = "The partner returned."
        correction.replace(changed)
        store.commit(correction)
        stale = store.begin_batch("stale-choice")
        for source_id, revision in frozen.source_revisions().items():
            stale.expect_head(source_id, revision)
        stale.create(ActionSchemaRecord(content="A stale action must not commit.", action_type=ActionType.Reply))
        with self.assertRaises(GhostConflict):
            store.commit(stale)
        self.assertFalse(any(record.content == "A stale action must not commit." for record in store.all_knoxels.values()))

    def test_self_wake_has_partner_evidence_but_entity_choice_excludes_other_partner(self) -> None:
        store = make_store()
        jesse_id = store.get_entity_id("Jesse")
        mara_id = store.get_entity_id("Mara")
        self.assertIsNotNone(jesse_id)
        self.assertIsNotNone(mara_id)
        say(store, "Jesse", "I cannot answer until I return.")
        say(store, "Mara", "Private to this conversation: the cobalt key.")
        jesse = guidance(store)
        mara = guidance(store, mara_id)
        request = CycleRequest(companion_name="Spamuel", user_name="Jesse", user_entity_id=jesse_id,
            cycle_time=NOW, situational_guidance=[jesse, mara])
        work = CycleWork(request=request)
        work.state.stimulus_groups = [StimulusGroup(key=SELF_SHARD)]
        wake_text = StoryPrefix(work, group=SELF_SHARD).story_text
        self.assertIn("I cannot answer until I return.", wake_text)
        self.assertNotIn("cobalt key", wake_text)
        self.assertIsNotNone(mara_id)
        mara_key = ShardKey(kind="entity", ref=str(mara_id))
        work.state.workspace.ensure_shard(mara_key)
        work.state.stimulus_groups = [StimulusGroup(key=mara_key)]
        mara_text = StoryPrefix(work, group=mara_key).story_text
        self.assertIn("cobalt key", mara_text)
        self.assertNotIn("cannot answer until", mara_text)
        self.assertIn(NOW.isoformat() + " UTC", mara_text)

    def test_private_noncausal_and_imagined_dialogue_are_not_actual_interaction(self) -> None:
        store = make_store()
        actual = say(store, "Jesse", "I am here now.")
        say(store, "Jesse", "private dialogue", visibility=InterlocusType.PrivateInternal)
        say(store, "Jesse", "imagined dialogue", scope=RecordScope(frame=FrameEnum.Meta))
        say(store, "Jesse", "virtual dialogue", causal=False)
        projected = guidance(store)
        self.assertIsNotNone(projected.partner)
        assert projected.partner is not None
        self.assertEqual([source.record_id for source in projected.partner.recent_dialogue], [actual])
        self.assertNotIn("is an AI companion", render_situational_guidance(projected))
        self.assertIn("not supplied in this projection", render_situational_guidance(projected))

    def test_purpose_delivery_and_intention_are_read_from_actual_commits(self) -> None:
        store = make_store()
        intention = store.store_knoxel(Intention(content="Keep the promise to explain tomorrow.", entity_id=0,
            affective_valence=0.4, intention_type=IntentionType.SelfSituation))
        action = store.store_knoxel(ActionSchemaRecord(content="Close this exchange warmly and wait.",
            action_type=ActionType.Reply, intention_ids=[intention.id],
            metadata={"dispatch_status": "pending", "group": {"kind": "self", "ref": ""}}))
        before = guidance(store)
        self.assertEqual(before.recent_moves[0].status, "selected")
        reply_id = commit_public_reply(store, "Take care. Talk later.", "", commit_id="actual-reply", action_id=action.id)
        self.assertIsNotNone(reply_id)
        self.assertEqual(guidance(store).recent_moves[0].status, "committed")
        assert reply_id is not None
        acknowledge_reply(store, reply_id)
        after = guidance(store)
        self.assertEqual(after.recent_moves[0].status, "delivered")
        self.assertEqual(after.intentions[0].record_id, intention.id)
        self.assertEqual(before.recent_moves[0].status, "selected")
        self.assertIn("Close this exchange warmly", render_situational_guidance(after))
        self.assertIn("Take care. Talk later.", render_situational_guidance(after))

    def test_new_contact_refreshes_projection_without_mutating_old_snapshot(self) -> None:
        store = make_store()
        say(store, "Jesse", "I will be away.", timestamp=NOW - timedelta(minutes=6))
        old = guidance(store)
        new_id = say(store, "Jesse", "I came back earlier than expected.")
        current = guidance(store)
        assert current.partner is not None
        self.assertEqual(current.partner.recent_dialogue[-1].record_id, new_id)
        self.assertNotIn("came back", render_situational_guidance(old))
        self.assertIn("keeping uncertainty explicit", render_situational_guidance(current))

    def test_public_and_story_prompts_receive_same_sources_with_revision_guard(self) -> None:
        store = make_store()
        source_id = say(store, "Jesse", "I am disconnected from the game, but reading this chat.")
        public = build_public_context(store, ActionRequest())
        self.assertIn(source_id, public.revisions)
        self.assertIn("disconnected from the game", render_context(public).text)
        request = StoryRequest(companion_name="Spamuel", partner_name="Jesse", character_card_story="",
                               current_user_message="", public_context=public)
        texts = "\n".join(turn.content.text for turn in build_story_messages(request, load_prompts()))
        self.assertIn("disconnected from the game", texts)
        edit = store.begin_batch("correct-source")
        changed = edit.view().get(source_id).copy_for_edit()
        changed.content = "The connection returned."
        edit.replace(changed)
        store.commit(edit)
        with self.assertRaises(GhostConflict):
            validate_public_context(store, public)


if __name__ == "__main__":
    unittest.main()
