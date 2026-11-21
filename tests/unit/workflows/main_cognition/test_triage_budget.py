"""Exercise deterministic triage fitting without a model."""
from __future__ import annotations

import unittest

from pm.context_render.budget import ContextBudget, RequiredContextTooLarge
from pm.records.workspace import ShardKey
from workflows.main_cognition.cycle_work import CycleRequest
from workflows.main_cognition.phases.intake import start
from workflows.main_cognition.phases.model import phase_prompts
from workflows.main_cognition.phases.triage import fit_prompt
from workflows.main_cognition.schemas import RecordView, Stimulus


class TriageBudgetTests(unittest.TestCase):
    def test_recent_dialogue_pins_yield_to_a_relevant_small_line(self) -> None:
        current = RecordView(record_id=11, record_class="Feature", kind="Feature",
                             feature_type="Dialogue", source="Jesse", content="What about oak?")
        long_line = current.model_copy(update={"record_id": 9, "content": "Earlier oak discussion. " * 6400})
        short_line = current.model_copy(update={"record_id": 10, "content": "Oak grows on the other bank."})
        work = start(CycleRequest(companion_name="Spamuel", user_name="Jesse", user_entity_id=7,
            stimuli=[Stimulus(knoxel_id=11, source="user_input", text=current.content)],
            records={9: long_line, 10: short_line, 11: current}))
        work.state.workspace.ensure_shard(ShardKey(kind="entity", ref="7")).tail_ids = [9, 10]
        budget = ContextBudget(context_window=32000, input_fraction=0.712)

        result = fit_prompt(work, budget, prompts=phase_prompts("triage"), story_prompts=phase_prompts("story"))

        self.assertIn("triage dialogue pins relaxed", result.selection.omitted)
        self.assertIn(10, result.selection.source_ids)
        self.assertNotIn(9, result.selection.source_ids)
        self.assertLessEqual(budget.estimate(result.turns), budget.input_ceiling)

    def test_compact_triage_preserves_current_input_and_drops_large_story(self) -> None:
        utterance = "Which side of the river has oak trees?"
        input_record = RecordView(record_id=11, record_class="Feature", kind="Feature",
                                  feature_type="Dialogue", source="Jesse", content=utterance)
        work = start(CycleRequest(companion_name="Spamuel", user_name="Jesse",
            character_card="The old character story keeps going. " * 4500,
            stimuli=[Stimulus(knoxel_id=11, source="user_input", text=utterance)],
            records={11: input_record}))
        budget = ContextBudget(context_window=32000, input_fraction=0.712)
        prompt = fit_prompt(work, budget, prompts=phase_prompts("triage"), story_prompts=phase_prompts("story"))

        text = "\n".join(turn.content.text for turn in prompt.turns)
        self.assertEqual(text.count(utterance), 1)
        self.assertIn("id=11", text)
        self.assertIn("story and workspace context", prompt.selection.omitted)
        self.assertNotIn("The old character story keeps going.", text)
        self.assertLessEqual(budget.estimate(prompt.turns), budget.input_ceiling)

    def test_a_single_irreducible_input_is_rejected_before_model_io(self) -> None:
        utterance = "An indivisible oversized event. " * 7000
        input_record = RecordView(record_id=11, record_class="Feature", kind="Feature",
                                  feature_type="Dialogue", source="Jesse", content=utterance)
        work = start(CycleRequest(companion_name="Spamuel", user_name="Jesse",
            stimuli=[Stimulus(knoxel_id=11, source="user_input", text=utterance)],
            records={11: input_record}))
        with self.assertRaisesRegex(RequiredContextTooLarge, "compact triage"):
            fit_prompt(work, ContextBudget(context_window=32000), prompts=phase_prompts("triage"),
                       story_prompts=phase_prompts("story"))
