"""Check the one story renderer that cognition and the memory inspector share.

``pm.context_render.story`` owns the words that show one record and one
workspace shard. The main cognition cycle, the public dialogue workflows, and
the memory inspector all call these functions, so this file guards the shared
parts: the token unit, the shard names, the shard selection, and the block.
"""
from __future__ import annotations

import unittest
from datetime import datetime

from pm.context_render.story import (RecordView, ShardLabels, fill_story_names, render_story, shard_block_text,
                                     shard_name, story_entity_names, story_line, story_shards, story_tokens)
from pm.context_render.tokens import text_tokens
from pm.records.workspace import GlobalWorkspace, ShardKey, WorkspaceItem

MORNING = datetime(2026, 1, 2, 3, 4)
LATER = datetime(2026, 1, 2, 9, 30)

LABELS = ShardLabels(self_title="SELF {companion_name}", self_tail="the self tail",
                     entity_title="ENTITY {entity_name}", entity_tail="the tail of {entity_name}",
                     workflow_title="WORKFLOW {workflow_name}", workflow_tail="the workflow tail",
                     empty_items="nothing is here", empty_tail="nothing happened yet",
                     verbatim_tail="the lines are shown above", self_focus="about {companion_name}",
                     entity_focus="about {entity_name}", workflow_focus="about {workflow_name}")
TEMPLATE = "### BEGIN {title}\nitems:\n{items}\n{tail_label}:\n{tail}\n### END {title}"


def view(record_id: int, content: str, *, when=None, kind: str = "Feature", source: str = "") -> RecordView:
    return RecordView(record_id=record_id, kind=kind, source=source, content=content, timestamp=when)


def workspace() -> GlobalWorkspace:
    """Return a workspace with a self shard, one entity shard, and one workflow shard."""
    space = GlobalWorkspace()
    space.self_shard.items.append(WorkspaceItem(record_id=1))
    space.ensure_shard(ShardKey(kind="entity", ref="7"))
    space.ensure_shard(ShardKey(kind="workflow", ref="minecraft"))
    return space


class StoryTokenTests(unittest.TestCase):
    def test_story_tokens_is_the_plain_text_count(self):
        text = "Emmy says: the cave mouth breathes cold air.\nRick: let's go."
        self.assertEqual(story_tokens(text), text_tokens(text))

    def test_empty_text_costs_nothing(self):
        self.assertEqual(story_tokens(""), 0)


class StoryLineTests(unittest.TestCase):
    def test_time_and_id_wrap_the_same_line(self):
        line = story_line(view(11, "hello there", when=MORNING, kind="Feature"), with_id=False)
        self.assertEqual(line, "(2026-01-02 03:04) Feature: hello there")
        self.assertEqual(story_line(view(11, "hello there", when=MORNING, kind="Feature"), with_id=True),
                         f"[id 11] {line}")

    def test_untimed_line_starts_with_the_speaker(self):
        self.assertEqual(story_line(view(12, "hi", source="Rick"), with_id=False), "Rick: hi")

    def test_render_story_orders_by_world_time_and_keeps_one_copy(self):
        records = {1: view(1, "later", when=LATER), 2: view(2, "no time"), 3: view(3, "morning", when=MORNING)}
        self.assertEqual(render_story(records, [1, 2, 3, 1], with_id=False).splitlines(),
                         ["(2026-01-02 03:04) Feature: morning",
                          "(2026-01-02 09:30) Feature: later",
                          "Feature: no time"])


class NameTests(unittest.TestCase):
    def test_fill_names_keeps_other_placeholders(self):
        self.assertEqual(fill_story_names("{companion_name} meets {entity_name} at {place}", "Emmy", "Rick"),
                         "Emmy meets Rick at {place}")

    def test_shard_name_uses_the_user_and_the_known_entities(self):
        names = {7: "Notch"}
        self.assertEqual(shard_name(ShardKey(kind="self"), user_entity_id=1, user_name="Rick", entity_names=names),
                         "self")
        self.assertEqual(shard_name(ShardKey(kind="entity", ref="1"), user_entity_id=1, user_name="Rick",
                                    entity_names=names), "Rick")
        self.assertEqual(shard_name(ShardKey(kind="entity", ref="7"), user_entity_id=1, user_name="Rick",
                                    entity_names=names), "Notch")
        self.assertEqual(shard_name(ShardKey(kind="entity", ref="9"), user_entity_id=1, user_name="Rick",
                                    entity_names=names), "entity 9")
        self.assertEqual(shard_name(ShardKey(kind="workflow", ref="minecraft"), user_entity_id=1, user_name="Rick",
                                    entity_names=names), "minecraft")

    def test_entity_names_follow_group_order_without_duplicates(self):
        keys = [ShardKey(kind="workflow", ref="minecraft"), ShardKey(kind="entity", ref="7"),
                ShardKey(kind="entity", ref="1"), ShardKey(kind="entity", ref="7")]
        self.assertEqual(story_entity_names(keys, user_entity_id=1, user_name="Rick", entity_names={7: "Notch"}),
                         "Notch and Rick")

    def test_the_user_is_the_other_character_without_an_entity_group(self):
        keys = [ShardKey(kind="self"), ShardKey(kind="workflow", ref="minecraft")]
        self.assertEqual(story_entity_names(keys, user_entity_id=1, user_name="Rick", entity_names={}), "Rick")


class ShardSelectionTests(unittest.TestCase):
    def test_self_shard_comes_first_then_the_groups_in_order(self):
        space = workspace()
        keys = [ShardKey(kind="self"), ShardKey(kind="workflow", ref="minecraft"),
                ShardKey(kind="entity", ref="7")]
        self.assertEqual([shard.key for shard in story_shards(space, keys)],
                         [ShardKey(kind="self"), ShardKey(kind="workflow", ref="minecraft"),
                          ShardKey(kind="entity", ref="7")])

    def test_one_group_keeps_only_its_own_shard(self):
        space = workspace()
        keys = [ShardKey(kind="entity", ref="7"), ShardKey(kind="workflow", ref="minecraft")]
        chosen = story_shards(space, keys, group=ShardKey(kind="workflow", ref="minecraft"))
        self.assertEqual([shard.key for shard in chosen],
                         [ShardKey(kind="self"), ShardKey(kind="workflow", ref="minecraft")])

    def test_a_group_without_a_shard_raises_instead_of_hiding_it(self):
        with self.assertRaises(KeyError):
            story_shards(workspace(), [ShardKey(kind="entity", ref="42")])


class ShardBlockTests(unittest.TestCase):
    def test_block_fills_its_title_items_and_tail(self):
        records = {1: view(1, "I want to see the sea", when=MORNING, source="Emmy"),
                   2: view(2, "hello there", when=LATER, source="Rick")}
        block = shard_block_text(ShardKey(kind="entity", ref="7"), "Rick", "Emmy", records, [1], [2],
                                 template=TEMPLATE, labels=LABELS)
        self.assertEqual(block, "### BEGIN ENTITY Rick\nitems:\n(2026-01-02 03:04) Emmy: I want to see the sea\n"
                                "the tail of Rick:\n(2026-01-02 09:30) Rick: hello there\n### END ENTITY Rick")

    def test_empty_shard_says_so_with_the_prompt_words(self):
        block = shard_block_text(ShardKey(kind="self"), "self", "Emmy", {}, [], [],
                                 template=TEMPLATE, labels=LABELS)
        self.assertEqual(block, "### BEGIN SELF Emmy\nitems:\nnothing is here\nthe self tail:\n"
                                "nothing happened yet\n### END SELF Emmy")

    def test_a_verbatim_tail_labels_an_empty_tail_as_shown_above(self):
        block = shard_block_text(ShardKey(kind="self"), "self", "Emmy", {}, [], [],
                                 template=TEMPLATE, labels=LABELS, verbatim_tail=True)
        self.assertIn("the lines are shown above", block)

    def test_a_missing_view_raises(self):
        with self.assertRaises(KeyError):
            shard_block_text(ShardKey(kind="self"), "self", "Emmy", {}, [5], [], template=TEMPLATE, labels=LABELS)


if __name__ == "__main__":
    unittest.main()
