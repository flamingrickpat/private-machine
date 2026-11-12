"""Check named private request boundaries without replacing runtime collaborators.

The retained episode is explicit storage input. These checks certify boundary
validation and serialization, not generated thought or model quality.
"""
from __future__ import annotations

from datetime import datetime
import unittest

from pydantic import ValidationError

from workflows.inner_thought.schemas import InnerReport, InnerRequest, LegacyEpisodeContext, LegacyEpisodeState
from workflows.inner_thought.request_views import ExpectationView, GoalView, MainFrameView, RecentThoughtView
from workflows.inner_thought.pipeline import citable_ids
from workflows.inner_thought.rendering import sources


class NamedPrivateRequestTests(unittest.TestCase):
    """Preserve exact evidence identities and explicit legacy provenance."""

    def test_request_roundtrip_keeps_named_evidence_and_original_ids(self) -> None:
        request = InnerRequest(main_frame=MainFrameView(frame_id=17),
            active_goal_records=[GoalView(knoxel_id=19, text="Finish the story", revision=3)],
            pending_expectations=[ExpectationView(knoxel_id=23, text="A reply", revision=2)],
            recent_thoughts=[RecentThoughtView(knoxel_id=29, text="An unresolved reading", source_ids=[19])])
        restored = InnerRequest.model_validate_json(request.model_dump_json())
        self.assertEqual(citable_ids(restored), {17, 19, 23, 29})
        self.assertEqual(restored.active_goal_records[0].revision, 3)
        self.assertEqual(restored.recent_thoughts[0].source_ids, [19])
        self.assertIn("Finish the story", str(sources(restored.active_goal_records, "goal")))

    def test_legacy_duration_remains_world_history_without_graph_judgments(self) -> None:
        state = LegacyEpisodeState(episode_id="retained", sequence=1,
            started_at=datetime(2026, 11, 12, 8), elapsed_seconds=3600,
            closed=False, closed_reason=None, operations=("memory_search",), evidence_ids=(31,),
            alternatives=("An unresolved possibility",), thoughts=("A retained thought",),
            progress=("none",), stalled=1, continuation="continue", next_operation="perspective_shift",
            parked_until=None, parked_sources=(), parent_thought_id=37)
        context = LegacyEpisodeContext(provenance="legacy_linear", state=state,
            awaiting_acknowledgment=True, suppressed_source_ids=(41,))
        restored = InnerRequest.model_validate_json(InnerRequest(episode=context).model_dump_json())
        self.assertEqual(restored.episode.state, state)
        self.assertEqual(restored.episode.provenance, "legacy_linear")
        self.assertTrue(restored.episode.awaiting_acknowledgment)
        self.assertNotIn("active_elapsed_seconds", restored.episode.model_dump_json())
        self.assertNotIn("judgments", restored.episode.model_dump_json())
        with self.assertRaises(ValidationError):
            LegacyEpisodeContext(state=state)

    def test_incomplete_retained_episode_fails_at_the_boundary(self) -> None:
        with self.assertRaises(ValidationError):
            InnerRequest.model_validate({"episode": {"provenance": "legacy_linear",
                "state": {"episode_id": "broken", "sequence": 0}}})

    def test_report_recall_retains_scope_provenance_and_source_links(self) -> None:
        report = InnerReport.model_validate({"retrieved_memories": [{"knoxel_id": 43,
            "text": "A remembered report", "revision": 4, "source_ids": [47],
            "scope": {"reality_id": 0, "frame": "in_world"}, "historical": True,
            "evidence_role": "user_report"}]})
        restored = InnerReport.model_validate_json(report.model_dump_json())
        self.assertEqual(restored.retrieved_memories[0].source_ids, [47])
        self.assertEqual(restored.retrieved_memories[0].revision, 4)
        self.assertTrue(restored.retrieved_memories[0].historical)
