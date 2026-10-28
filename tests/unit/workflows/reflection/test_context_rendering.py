"""Keep original evidence while excluding runtime bookkeeping from model context."""
import unittest

from pm.context_render import Role, Turn, render_context
from pm.context_render.budget import ContextBudget, RequiredContextTooLarge, select_context
from pm.context_render.records import ArchitecturalEvidence
from pm.records.enums import FeatureType
from pm.records.memory_records import new_feature
from tests.unit.test_design_contracts import small_ghost
from workflows.reflection.experiments import source_evidence
from workflows.reflection.rendering import context_groups
from workflows.reflection.schemas import ReflectionContext, ReflectionEpisode


class ReflectionRenderingTests(unittest.TestCase):
    def test_projection_keeps_the_original_quote_and_leaves_bookkeeping_on_disk(self) -> None:
        store = small_ghost()
        source = new_feature(store, "The original quoted observation stays intact.", FeatureType.StateUpdate,
            source="Spamuel", metadata={"generation_source_revisions": {str(key): 1 for key in range(1000)},
                "dispatch_payload": {"workspace": "private runtime dispatch bookkeeping"},
                "reflection_versions": {"77": 88}})
        evidence = source_evidence(store.snapshot(), source)
        rendered = render_context(ArchitecturalEvidence((evidence, evidence))).text
        self.assertEqual(rendered.count(source.content), 1)
        self.assertIn('"record_ref"', rendered)
        self.assertIn('"reflection_versions"', rendered)
        self.assertNotIn("generation_source_revisions", rendered)
        self.assertNotIn("dispatch_payload", rendered)
        self.assertEqual(len(store.get_knoxel_by_id(source.id).metadata["generation_source_revisions"]), 1000)

    def test_pinned_episode_cannot_lose_its_source_to_optional_selection(self) -> None:
        store = small_ghost()
        source = new_feature(store, "Original required evidence. " * 1500, FeatureType.Thought, source="Spamuel")
        evidence = source_evidence(store.snapshot(), source)
        episode = ReflectionEpisode(episode_id=900, kind="thought", loop="inner", tick_id=5,
            operation="inspect", owner="inner_thought", summary="Inspect the source.", outcome="unresolved", inputs=[source.id])
        frozen = ReflectionContext(records=[evidence], episodes=[episode])
        groups = context_groups(frozen, {}, required_episode_ids=frozenset({900}))
        with self.assertRaises(RequiredContextTooLarge):
            select_context([Turn(Role.System, "Keep the required evidence.")], groups, ContextBudget(context_window=1000))
