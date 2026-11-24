"""Exercise self, partner and workflow evidence through the shared renderer at 32k."""
from dataclasses import replace
import unittest

from pm.context_render.prompt import RenderSettings
from pm.context_render.shards import ShardPromptInput, render_shard_prompt, select_shards
from pm.context_render.story import RecordView, ShardLabels
from pm.records.workspace import GlobalWorkspace, ShardKey, WorkspaceItem


class ShardRendererTests(unittest.TestCase):
    def test_specialized_shards_share_self_and_fit_to_the_actual_window(self) -> None:
        workspace = GlobalWorkspace()
        workspace.self_shard.items.append(WorkspaceItem(record_id=1, activation_at_entry=0.8))
        workspace.self_shard.tail_ids = [2]
        partner = ShardKey(kind="entity", ref="7")
        job = ShardKey(kind="workflow", ref="craft")
        workspace.ensure_shard(partner).tail_ids = [3, 4]
        workspace.ensure_shard(job).tail_ids = [5]
        names: dict[int, str] = {7: "Jesse"}  # pm-lint: allow PM004 -- names indexed by entity ID
        records: dict[int, RecordView] = {key: RecordView(record_id=key, content=content) for key, content in enumerate(  # pm-lint: allow PM004 -- evidence indexed by record ID
            ("a cognitive feature", "a private thought", "first turn", "latest turn", "job progress"), start=1)}
        settings = RenderSettings(context_window=32000, input_fraction=0.8, reserved_output_tokens=2048)
        partner_shards = select_shards(workspace, [partner], hidden_tail=set(), user_id=7,
                                      user_name="Jesse", entity_names=names)  # pm-lint: allow PM005 -- names indexed by entity ID
        frame = ShardPromptInput("choose", "Spamuel", "Choose as yourself.", "{shards}", "Understood.",
            "{items}\n{tail}", ShardLabels(), "", "", "", "", frozenset(), records, partner_shards)  # pm-lint: allow PM005 -- evidence indexed by record ID
        partner_prompt = render_shard_prompt(frame, settings)
        self.assertEqual(partner_prompt.visible_source_ids, (1, 2, 3, 4))
        job_shards = select_shards(workspace, [job], hidden_tail=set(), user_id=7,
                                  user_name="Jesse", entity_names=names)  # pm-lint: allow PM005 -- names indexed by entity ID
        self.assertEqual(render_shard_prompt(replace(frame, shards=job_shards), settings).visible_source_ids,
                         (1, 2, 5))
        records[3].content = "oversized dialogue " * 40_000
        fitted = render_shard_prompt(frame, settings)
        self.assertLessEqual(fitted.estimated_tokens, fitted.input_ceiling)
        self.assertIn(3, fitted.visible_source_ids)
        self.assertIn("source characters omitted", fitted.raw_text)
        self.assertIn(4, fitted.required_source_ids)
        self.assertNotIn(3, fitted.required_source_ids)
        self.assertEqual(fitted, render_shard_prompt(frame, settings))
