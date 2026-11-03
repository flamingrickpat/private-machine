"""Deterministic checks of public grounding, identity, and saved continuity.

These use real memory stores and files. They do not certify model quality;
the real model check lives in ``dev.work_items.k002.actor_immersion``.
"""
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pm.host.durable_files import atomic_json
from pm.memory.store import MemoryStore
from pm.records.cycle_state import PersonaConfig
from pm.records.enums import FeatureType, InterlocusType
from pm.records.global_workspace import GwtTurnRecord
from pm.records.memory_records import Feature
from simulation.actors import load_profile, suggested_start
from simulation.personas import write_home
from simulation.plans import ActorDraftReview
from pm import vclock


class ActorProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.addCleanup(vclock.install, vclock.active())
        vclock.install(vclock.VirtualClock(datetime(2026, 5, 30, 21, 3, tzinfo=timezone.utc)))

    def test_public_profile_preserves_exact_words_and_saved_clock(self) -> None:
        with TemporaryDirectory() as directory:
            store = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"))
            batch = store.begin_batch("public grounding", new_tick=True)
            public = batch.create(Feature(content="spamuel i found a cat lol", override_source="Jesse",
                feature_type=FeatureType.Dialogue, interlocus=InterlocusType.Public,
                timestamp_world_begin=datetime(2026, 5, 30, 21, 3)))
            private = batch.create(Feature(content="PRIVATE THOUGHT", override_source="Spamuel",
                feature_type=FeatureType.Thought, interlocus=InterlocusType.Public))
            hidden = batch.create(Feature(content="HIDDEN DIALOGUE", override_source="Jesse",
                feature_type=FeatureType.Dialogue, interlocus=InterlocusType.Undefined))
            batch.set_gwt(GwtTurnRecord(tick_id=batch.tick_id, knoxel_ids=[public.id, private.id, hidden.id]))
            store.commit(batch)
            home = write_home(store, Path(directory) / "source")
            # Saved receipt includes arrays; reading time must not reject them.
            atomic_json(home.root / "runtime/clock.json", {"clock": {"now": "2026-12-30T23:59:00Z"},
                "timers": {}, "environment": [], "carry": []})
            profile = load_profile(home.root)
            self.assertEqual(profile.speaker_name, "Jesse")
            self.assertEqual(profile.companion_name, "Spamuel")
            self.assertEqual([row.text for row in profile.dialogue], [public.content])
            self.assertEqual(profile.dialogue[0].record_id, public.id)
            self.assertEqual(profile.dialogue[0].at.tzinfo, timezone.utc)
            self.assertEqual(suggested_start(profile).isoformat(), "2026-12-31T23:58:00+00:00")
            with home.ghost_pickle.open("rb") as source:
                self.assertEqual(profile.source_checkpoint_sha256, hashlib.file_digest(source, "sha256").hexdigest())

    def test_missing_user_history_fails_instead_of_inventing_a_generic_actor(self) -> None:
        with TemporaryDirectory() as directory:
            store = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"))
            home = write_home(store, Path(directory) / "source")
            with self.assertRaisesRegex(ValueError, "No public Jesse utterances"):
                load_profile(home.root)

    def test_acceptance_cannot_override_a_failed_immersion_check(self) -> None:
        with self.assertRaises(ValueError):
            ActorDraftReview(accepted=True, voice_faithful=False, identity_correct=True,
                             natural_dialogue=True, scenario_consistent=True, reason="Contradictory verdict")
