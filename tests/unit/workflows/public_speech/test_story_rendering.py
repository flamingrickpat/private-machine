"""Interleaved Story rendering with synthetic data.

The test fixes the message-role contract: dialogue lines (both speakers)
land in assistant turns, artifact lines land in user turns, blocks alternate
strictly, and the directive closes the last user turn. The literal expected
output is pinned so a layout regression fails immediately.
"""
from __future__ import annotations

import unittest

from pm.context_render.turns import Role
from workflows.public_speech.public_context import DialogueTurn
from workflows.public_speech.story_rendering import (
    ArtifactLine,
    DialogueLine,
    dialogue_lines,
    render_interleaved_story,
)


PROMPTS = {
    "speaker_line": '{speaker} says: "{text}"',
    "thought_line": "{speaker} thinks: *{text}*",
    "day_marker": "*({date})*",
}


class DialogueLinesTests(unittest.TestCase):
    def test_day_marker_attaches_to_first_line_of_day(self):
        turns = [
            DialogueTurn(speaker="Jesse", text="hello", knoxel_id=1, timestamp="2026-05-21T20:00:00"),
            DialogueTurn(speaker="Spamuel", text="hi", knoxel_id=2, timestamp="2026-05-21T20:01:00"),
            DialogueTurn(speaker="Jesse", text="next day", knoxel_id=3, timestamp="2026-05-22T09:00:00"),
        ]
        lines = dialogue_lines(turns, PROMPTS)
        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[0].text, "*(21 May 2026)*\nJesse says: \"hello\"")
        self.assertEqual(lines[1].text, 'Spamuel says: "hi"')
        self.assertEqual(lines[2].text, "*(22 May 2026)*\nJesse says: \"next day\"")

    def test_thought_uses_thought_template(self):
        turns = [DialogueTurn(speaker="Spamuel", text="I like cats", knoxel_id=5,
                              timestamp="2026-05-21T20:00:00", kind="thought")]
        lines = dialogue_lines(turns, PROMPTS)
        self.assertEqual(lines[0].text, "*(21 May 2026)*\nSpamuel thinks: *I like cats*")


class InterleavedRenderingTests(unittest.TestCase):
    """Full layout with literal expected output."""

    def _render(self):
        from datetime import datetime
        dialogue = [
            DialogueLine(timestamp=datetime(2026, 5, 21, 20, 0), record_id=1,
                         text='*(21 May 2026)*\nJesse says: "my sink is judging me"'),
            DialogueLine(timestamp=datetime(2026, 5, 21, 20, 1), record_id=2,
                         text='Spamuel says: "then you owe it a dish"'),
        ]
        artifacts = [
            ArtifactLine(timestamp=datetime(2026, 5, 20, 9, 0), record_id=9, lane="history",
                         text="## Earlier times with Jesse\n- Jesse started the garden."),
            ArtifactLine(timestamp=datetime(2026, 5, 21, 20, 30), record_id=10, lane="facts",
                         text="## What Spamuel knows\n- Jesse has a sink."),
        ]
        return render_interleaved_story(
            system_text="You are an expert story writer.",
            opening_text="Background material for Spamuel.\n\n## Character card\nSpamuel is brash.",
            dialogue=dialogue,
            artifacts=artifacts,
            directive_text="Inside Spamuel at this moment\nFeelings: clear pride.\n\n"
                           "Continue directly with Spamuel's next spoken line.",
        )

    def test_roles_alternate(self):
        turns = self._render()
        roles = [t.role.value for t in turns]
        # system, user (opening + leading artifact), assistant, user (artifact + directive)
        self.assertEqual(roles, ["system", "user", "assistant", "user"])

    def test_literal_output(self):
        turns = self._render()
        self.assertEqual(turns[0].role, Role.System)
        self.assertEqual(turns[0].content.text, "You are an expert story writer.")

        self.assertEqual(turns[1].role, Role.User)
        self.assertEqual(turns[1].content.text, (
            "Background material for Spamuel.\n\n"
            "## Character card\nSpamuel is brash.\n\n"
            "## Earlier times with Jesse\n- Jesse started the garden."
        ))

        self.assertEqual(turns[2].role, Role.Assistant)
        self.assertEqual(turns[2].content.text, (
            '*(21 May 2026)*\nJesse says: "my sink is judging me"\n'
            'Spamuel says: "then you owe it a dish"'
        ))

        self.assertEqual(turns[3].role, Role.User)
        self.assertEqual(turns[3].content.text, (
            "## What Spamuel knows\n- Jesse has a sink.\n\n"
            "Inside Spamuel at this moment\nFeelings: clear pride.\n\n"
            "Continue directly with Spamuel's next spoken line."
        ))

    def test_no_turn_ends_with_prefill(self):
        turns = self._render()
        for turn in turns:
            self.assertFalse(turn.content.text.endswith('Spamuel says: "'))

    def test_dialogue_only_no_artifacts(self):
        from datetime import datetime
        turns = render_interleaved_story(
            system_text="sys",
            opening_text="open",
            dialogue=[DialogueLine(timestamp=datetime(2026, 1, 1), record_id=1,
                                   text='Jesse says: "hi"')],
            artifacts=[],
            directive_text="Continue.",
        )
        # system, user (opening), assistant (dialogue), user (directive)
        self.assertEqual([t.role.value for t in turns], ["system", "user", "assistant", "user"])
        self.assertEqual(turns[1].content.text, "open")
        self.assertEqual(turns[3].content.text, "Continue.")

    def test_artifacts_before_dialogue_merge_into_opening(self):
        from datetime import datetime
        turns = render_interleaved_story(
            system_text="sys",
            opening_text="open",
            dialogue=[DialogueLine(timestamp=datetime(2026, 1, 2), record_id=2,
                                   text='Jesse says: "hi"')],
            artifacts=[ArtifactLine(timestamp=datetime(2026, 1, 1), record_id=1, lane="history",
                                    text="old memory")],
            directive_text="Continue.",
        )
        # artifact sorts before dialogue → merges into opening user turn
        self.assertEqual([t.role.value for t in turns], ["system", "user", "assistant", "user"])
        self.assertIn("old memory", turns[1].content.text)
        self.assertIn("open", turns[1].content.text)

    def test_trailing_artifact_merges_with_directive(self):
        from datetime import datetime
        turns = render_interleaved_story(
            system_text="sys",
            opening_text="open",
            dialogue=[DialogueLine(timestamp=datetime(2026, 1, 1), record_id=1,
                                   text='Jesse says: "hi"')],
            artifacts=[ArtifactLine(timestamp=datetime(2026, 1, 2), record_id=2, lane="facts",
                                    text="a fact")],
            directive_text="Continue.",
        )
        # artifact sorts after dialogue → user block, directive merges
        self.assertEqual([t.role.value for t in turns], ["system", "user", "assistant", "user"])
        self.assertIn("a fact", turns[3].content.text)
        self.assertIn("Continue.", turns[3].content.text)

    def test_empty_directive_raises(self):
        with self.assertRaises(ValueError):
            render_interleaved_story(
                system_text="sys", opening_text="open", dialogue=[], artifacts=[],
                directive_text="  ",
            )

    def test_no_dialogue_all_artifacts(self):
        from datetime import datetime
        turns = render_interleaved_story(
            system_text="sys",
            opening_text="open",
            dialogue=[],
            artifacts=[ArtifactLine(timestamp=datetime(2026, 1, 1), record_id=1, lane="facts",
                                    text="fact one")],
            directive_text="Continue.",
        )
        # system, user (opening + artifact + directive)
        self.assertEqual([t.role.value for t in turns], ["system", "user"])
        self.assertIn("fact one", turns[1].content.text)
        self.assertIn("open", turns[1].content.text)
        self.assertIn("Continue.", turns[1].content.text)


if __name__ == "__main__":
    unittest.main()
