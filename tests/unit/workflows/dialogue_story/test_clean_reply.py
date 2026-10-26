"""``clean_reply`` bubble selection and turn-boundary tests.

The Japanese fixtures are real completions from the third-party A/B run
``bz_p_e2e_live_7530`` (calls 502 and 269), contributed by the SCAIL project
in ``bz/parm-patch``. They cover the defect that a multi-line completion
delivered only its last bubble, so the line that answered the user was lost.
The remaining cases pin the line-start rule: no punctuation and no inline
mention of another speaker may end a spoken utterance.
"""
from __future__ import annotations

import unittest

from workflows.dialogue_story.steps import clean_reply

PREFILL = 'Momoka says: "'

# Call 502, cut after the first invented User turn (the rest of the script is cut anyway).
CALL_502 = (
    '\nおかえり、一悟。おつかれさま。長引いたなら、なおさらおつかれさま。"\n'
    'Momoka says: "ごはん、ね。百花、ホットミルクだけは自信ある。でも、それだけじゃ足りないよね。"\n'
    'Momoka says: "で、一悟。聞いてる？ ごはんの前に、ひとつだけ。さっき、なに言いかけてたの。"\n'
    'Momoka says: "……ねえ、聞こえてる？ 一悟。"\n'
    'User says: "聞こえてるよ。ごめん、ちょっと考えてて"\n'
    'Momoka says: "考えてるなら、なおさら。百花、待つの、へたなんだよ。"\n'
)

# Call 269: a single turn, no own marker, continues the prefill.
CALL_269 = '\n一悟、さっき言いかけてたよね。あれ、なんだったの。'


class CleanReplyBubblesTest(unittest.TestCase):
    def test_call_502_keeps_all_bubbles_before_invented_user_turn(self):
        out = clean_reply(CALL_502, "Momoka", "User", prefill=PREFILL)
        self.assertEqual(out.splitlines(), [
            "おかえり、一悟。おつかれさま。長引いたなら、なおさらおつかれさま。",
            "ごはん、ね。百花、ホットミルクだけは自信ある。でも、それだけじゃ足りないよね。",
            "で、一悟。聞いてる？ ごはんの前に、ひとつだけ。さっき、なに言いかけてたの。",
            "……ねえ、聞こえてる？ 一悟。",
        ])
        self.assertNotIn("User says", out)
        self.assertNotIn("考えてるなら", out)
        self.assertNotIn('"', out)

    def test_single_turn_is_unchanged(self):
        self.assertEqual(clean_reply(CALL_269, "Momoka", "User", prefill=PREFILL),
                         "一悟、さっき言いかけてたよね。あれ、なんだったの。")

    def test_preamble_before_first_own_marker_is_dropped_without_prefill(self):
        raw = 'Sure, here is the reply.\nMomoka says: "one"\nMomoka says: "two"\nUser says: "x"'
        self.assertEqual(clean_reply(raw, "Momoka", "User"), "one\ntwo")

    def test_echoed_prefill_is_removed_then_all_bubbles_kept(self):
        raw = PREFILL + 'one"\nMomoka says: "two"'
        self.assertEqual(clean_reply(raw, "Momoka", "User", prefill=PREFILL), "one\ntwo")

    def test_lone_quote_from_the_model_is_empty(self):
        # run bz_p_e2e_live_10426 call 465: the model returned a single '"'
        self.assertEqual(clean_reply('"', "Momoka", "User", prefill=PREFILL), "")

    def test_other_speaker_first_gives_nothing_invented(self):
        raw = 'Momoka says: "one"\nUser says: "x"\nMomoka says: "late"'
        self.assertEqual(clean_reply(raw, "Momoka", "User"), "one")

    def test_no_prefill_keeps_the_answering_bubble(self):
        # The interleaved story prompt sends no prefill, so the leading own
        # marker is stripped and its text is bubble one, not a preamble.
        raw = 'Sunny says: Yes, Tuesday.\nSunny says: I remember the bus.'
        self.assertEqual(clean_reply(raw, "Sunny", "Rick"),
                         "Yes, Tuesday.\nI remember the bus.")


class TurnBoundaryTest(unittest.TestCase):
    def test_inline_partner_colon_does_not_truncate_the_utterance(self):
        # An inline mention of the partner is quoted speech, not a turn start.
        raw = 'Sunny says: I told Rick: no. I meant it. That is all.'
        self.assertEqual(clean_reply(raw, "Sunny", "Rick"),
                         "I told Rick: no. I meant it. That is all.")

    def test_periods_and_ellipsis_never_end_an_utterance(self):
        raw = 'Sunny says: Wait... okay. Yes. I do.'
        self.assertEqual(clean_reply(raw, "Sunny", "Rick"), "Wait... okay. Yes. I do.")

    def test_quoted_own_marker_inline_is_not_a_bubble_break(self):
        # A marker mid-line cannot split one utterance into two bubbles.
        raw = 'Sunny says: He said Sunny: I quit, and I laughed.'
        self.assertEqual(clean_reply(raw, "Sunny", "Rick"),
                         "He said Sunny: I quit, and I laughed.")

    def test_day_marker_on_its_own_line_still_cuts(self):
        raw = 'Sunny says: I will check the map.\n*(later, the harbour)*'
        self.assertEqual(clean_reply(raw, "Sunny", "Rick"), "I will check the map.")

    def test_thought_line_on_its_own_line_still_cuts(self):
        raw = 'Sunny says: Sure, I can do that.\nSunny thinks: He is annoyed.'
        self.assertEqual(clean_reply(raw, "Sunny", "Rick"), "Sure, I can do that.")


if __name__ == "__main__":
    unittest.main()
