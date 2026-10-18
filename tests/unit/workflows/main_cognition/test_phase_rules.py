"""Retain active-goal focus, proposal consumers, and action ids from captured real cycles."""
import json
from pathlib import Path
import unittest

from workflows.main_cognition.cycle_work import CycleRequest, CycleWork
from workflows.main_cognition.dispatch import resolve_same_cycle
from workflows.main_cognition.phases.experience import allowed_focus_ids, candidate_texts, ensure_valid_result
from workflows.main_cognition.phases.goals import ACTION_KINDS, GOAL_KINDS, inner_proposals, proposal_consumers
from workflows.main_cognition.phases.route import stable_action_id
from workflows.main_cognition.schemas import ActionDispatch, ActionIntent, AttentionResult, WorkspaceResult


class WorkspaceGoalTests(unittest.TestCase):
    def test_an_attention_selected_goal_reaches_workspace_candidates(self):
        capture = json.loads(Path("test_data/captures/alpha/workspace-active-goal.json").read_text(encoding="utf-8"))
        request = capture["request"]
        # The capture predates step 5 of A19F. These attention policy fields
        # belonged to the removed codelet catalogue scheduler.
        for name in ("activation_threshold", "max_llm_codelets", "danger_llm_codelets"):
            del request["attention"]["competition"]["policy"][name]
        work = CycleWork(request=CycleRequest.model_validate(request),
                         attention=AttentionResult.model_validate(request["attention"]))
        candidates = candidate_texts(work)
        allowed = allowed_focus_ids(candidates, work.attention)
        result = WorkspaceResult.model_validate(capture["result"])
        ensure_valid_result(result, allowed)
        self.assertIn(11205, allowed)
        self.assertEqual(len(candidates), len({item.knoxel_id for item in candidates}))


class InnerProposalTests(unittest.TestCase):
    @staticmethod
    def _attention() -> AttentionResult:
        return AttentionResult.model_validate({"nominations": [
            {"nomination_id": "n1", "status": "selected", "attention_kind": "action_tendency",
             "reality": "inferred", "text": "I want to set a limit.", "target_ids": [11]},
            {"nomination_id": "n2", "status": "deferred", "attention_kind": "action_tendency",
             "reality": "inferred", "text": "I want to hide.", "target_ids": []},
            {"nomination_id": "n3", "status": "selected", "attention_kind": "belief",
             "reality": "inferred", "text": "I keep letting the user down.", "target_ids": [4]},
            {"nomination_id": "n4", "status": "selected", "attention_kind": "regulation",
             "reality": "internal", "text": "I will take my time.", "target_ids": []},
            {"nomination_id": "n5", "status": "selected", "attention_kind": "attention",
             "reality": "observed", "text": "The user said something.", "target_ids": [11]},
            {"nomination_id": "n6", "status": "selected", "attention_kind": "simulation",
             "reality": "imagined", "text": "Maybe he pulls away.", "target_ids": [4]},
        ]})

    def test_goals_reads_only_wants_and_beliefs(self):
        proposals = inner_proposals(self._attention(), GOAL_KINDS)
        self.assertEqual([item.kind for item in proposals], ["action_tendency", "belief"])
        self.assertTrue(all(item.text for item in proposals))

    def test_decide_reads_wants_imagination_and_steadiness(self):
        proposals = inner_proposals(self._attention(), ACTION_KINDS)
        self.assertEqual([item.kind for item in proposals],
                         ["action_tendency", "regulation", "simulation"])

    def test_every_kind_names_its_consumer(self):
        self.assertEqual(proposal_consumers("attention"), ["workspace"])
        self.assertEqual(proposal_consumers("action_tendency"),
                         ["workspace", "goals", "action_selection"])
        self.assertEqual(proposal_consumers("belief"), ["workspace", "goals", "belief_owner"])
        self.assertEqual(proposal_consumers("regulation"), ["workspace", "action_selection"])
        self.assertEqual(proposal_consumers("memory_access"), ["workspace"])


class ActionDependencyTests(unittest.TestCase):
    def test_real_model_workflow_labels_resolve_to_parent_action_ids(self):
        capture = json.loads(Path("test_data/captures/alpha/action-workflow-dependencies.json").read_text(encoding="utf-8"))
        dispatches = [ActionDispatch.model_validate({**item, "action_id": stable_action_id(ActionIntent.model_validate(item), index)})
                      for index, item in enumerate(capture["actions"])]
        resolved = resolve_same_cycle(dispatches)
        self.assertEqual([item.status for item in resolved], ["dispatched", "dispatched"])
        self.assertEqual(resolved[1].depends_on, [resolved[0].action_id])

    def test_repeated_workflow_labels_require_an_explicit_action_id(self):
        first = ActionDispatch(action_id="first", workflow_name="note_task", action_type="ToolCall", content="first note")
        second = first.model_copy(update={"action_id": "second", "content": "second note"})
        reply = ActionDispatch(action_id="reply", workflow_name="dialogue_assistant", action_type="Reply",
                               content="report", depends_on=["note_task"])
        self.assertEqual(resolve_same_cycle([first, second, reply])[-1].status, "rejected")
        explicit = reply.model_copy(update={"depends_on": ["second"]})
        self.assertEqual(resolve_same_cycle([first, second, explicit])[-1].status, "dispatched")
