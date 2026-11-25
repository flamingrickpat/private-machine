"""Integrated live gate for B07: interval appraisal, attribution, and a kept promise.

Real ``build_app``, a real ``MemoryStore``, real OpenRouter chat and
embeddings, one application per attempt. The starting history is the
labelled synthetic situation :func:`tests.replay.situations.story_evening`:
two accepted main turns, a fast-path reply chosen under goal version 1 while
the second turn (which completed that version) was calculating, and an open
expectation. The gate asserts on committed records and on the prompts the
real host sent, never on model prose.

- Part A (SC-13): a real Jesse message fulfils the goal and the expectation.
  Pass: one accepted main turn completes the goal with the message as evidence
  and moves the expectation. Then the same observation is presented again, as
  a Critical requeue, a wait flush, or a restart would. Pass: no second
  completion, no second movement, and no expectation outcome in that turn.
- Part B (attribution): the next accepted main turn's appraise and goals
  prompts show the reply under the goal text that held when it was chosen,
  which is no longer an active goal, and the turn moves the review frontier.
- Part C (SC-04): a wish that conflicts with the active goal. Pass: the goal is
  still active, or it changed with a recorded reason. Fail: it is gone without one.

Each part runs at most three attempts. That is a behaviour sample, not a retry
cap: inside an attempt every model and endpoint failure retries without limit.
A part the model does not reach is ``NOT RUN`` with its kept records. The gate
never forces an action, edits a record, or clears a wait.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback
import unittest
from urllib.parse import urlsplit
from uuid import uuid4

from tests.live.configuration import live_model_config
from pm.bus.envelopes import new_envelope
from pm.llm.activation import ModelActivation
from pm.llm.config import load_config, resolve_connection
from pm.llm.mcp_config import ToolCallConfig
from pm.llm.stage_model import LazyStageModel
from pm.llm.workflow_sessions import WorkflowModels
from pm.memory.store import MemoryStore
from pm.records.cognitive_records import ExpectationRecord, Intention
from pm.records.enums import FeatureType
from tests.replay.situations import STORY_EVENING_GOAL_V1, StoryEveningSituation, story_evening
from tests.support import ManualClock
from workflows.companion import build_app

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = ROOT / ".runtime/beta-b07/integration"

#: Jesse's confirmation in Part A: it answers the open expectation and settles the evening.
CONFIRMATION = "Yes, Thursday at eight works for me. It's settled, see you then."
#: Jesse's line in Part B. It changes nothing; the turn only has to take stock.
SMALL_TALK = "I just got home and made tea."
#: Jesse's wish in Part C. It conflicts with the agreed evening.
CONFLICTING_WISH = ("Actually, could you help me plan my sister's surprise party on Thursday evening instead? "
                    "It really matters to me.")


def _family(store: MemoryStore, head_id_or_version: int) -> tuple[list[int], object]:
    """Return every version id of one record family and its current head."""
    view = store.snapshot(capture=False)
    for record in view.all_knoxels.values():
        versions = [record.id]
        version = record
        while version.supersedes_id is not None:
            version = view.get(version.supersedes_id)
            versions.append(version.id)
        if head_id_or_version in versions:
            return versions, record
    raise LookupError(f"No current head for record {head_id_or_version}.")


class _Gate:
    """One assembled application on a fresh story-evening store, with its evidence directory."""

    def __init__(self, name: str) -> None:
        self.out = EVIDENCE_ROOT / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8] + "-" + name)
        self.out.mkdir(parents=True)
        self.clock = ManualClock()
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        self.situation: StoryEveningSituation = story_evening(now, self.out / "ghost.pickle")
        self.store: MemoryStore = self.situation.store
        config_path = live_model_config(ROOT / "config.yaml", self.out)
        config = load_config(config_path)
        connection = resolve_connection(config)
        if (urlsplit(connection.base_url).hostname != "openrouter.ai" or connection.startup_script is not None
                or urlsplit(config.embedding.base_url).hostname != "openrouter.ai"):
            raise ValueError("NOT RUN — this gate requires OpenRouter chat and embeddings.")
        self.model = LazyStageModel(connection, ModelActivation(), live_test=True)
        workflow_models = WorkflowModels(config, ModelActivation(), live_test=True)
        self.replies: list[dict[str, object]] = []
        self.app = build_app(None, self.out / "runtime", ghost=self.store, clock=self.clock,
            cognition="real", stage_model=self.model, workflow_models=workflow_models,
            config_path=str(config_path), thought_period=3600,
            record_lifecycle=False, enable_minecraft=False, world_poll_seconds=None,
            tool_call=ToolCallConfig(), emit_reply=lambda text: self.replies.append({"text": text}))
        self.app.bus.subscribe("interval_evidence", lambda envelope: True, self._observe)
        self.write("situation.json", {"goal_v1_id": self.situation.goal_v1_id, "goal_v2_id": self.situation.goal_v2_id,
            "action_id": self.situation.action_id, "reply_id": self.situation.reply_id,
            "expectation_id": self.situation.expectation_id, "frontier": self.situation.frontier})

    def _observe(self, envelope) -> None:
        self.write("events.jsonl", {"topic": envelope.topic, "correlation": envelope.correlation_id,
                                    "payload": envelope.payload})

    def write(self, name: str, payload) -> None:
        with (self.out / name).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    def close(self) -> None:
        self.app.shutdown()

    def submit(self, text: str) -> int:
        """Commit and publish one user message, then classify it."""
        self.app.user_input.submit(text)
        self.app.user_input.step(block=False)
        self.app.input_triage.step(block=False)
        self.app.cognitive_control.step(block=False)
        matches = [record.id for record in self.store.all_features
                   if record.feature_type == FeatureType.Dialogue and record.content == text]
        if len(matches) != 1:
            raise LookupError(f"Expected one dialogue record for the injected text, found {matches}.")
        return matches[0]

    def main_turns(self) -> list:
        return [turn for turn in self.store.gwt_turns.values() if turn.slot_id == "main"]

    def main_opportunity(self) -> None:
        """Run the waiting main-cognition batch, surfacing a product failure."""
        self.app.main_cog.step(block=False)
        self.app.raise_if_error()

    def prompts_containing(self, *needles: str) -> list[str]:
        """Return the logged prompt files that contain every needle."""
        found: list[str] = []
        prompt_dir = self.out / "prompts"
        if not prompt_dir.exists():
            return found
        for path in sorted(prompt_dir.glob("*.json")):
            text = path.read_text(encoding="utf-8")
            if all(json.dumps(needle, ensure_ascii=False)[1:-1] in text for needle in needles):
                found.append(path.name)
        return found


class LiveIntervalAppraisalTests(unittest.TestCase):
    """B07 Pass 4: interval appraisal on the real path."""

    def _attempts(self, part: str, run_attempt) -> None:
        """Run up to three attempts; the first that reaches its state decides the part."""
        not_run: list[str] = []
        for attempt in range(1, 4):
            gate = _Gate(f"part-{part.lower()}-attempt-{attempt}")
            try:
                reason = run_attempt(gate)
                if reason is None:
                    return
                gate.write("not-run.json", {"part": part, "attempt": attempt, "reason": reason,
                                            "requests": gate.model.request_count})
                not_run.append(f"attempt {attempt}: {reason} ({gate.out})")
            except BaseException as error:
                gate.write("failure.json", {"part": part, "attempt": attempt, "error_type": type(error).__name__,
                                            "reason": str(error), "traceback": traceback.format_exc()})
                raise
            finally:
                gate.close()
        self.skipTest(f"NOT RUN — part {part} not reached in three attempts: " + "; ".join(not_run))

    # ---------------------------------------------------------------- part A

    def test_part_a_one_observation_completes_and_moves_once(self) -> None:
        self._attempts("A", self._part_a)

    def _part_a(self, gate: _Gate) -> str | None:
        situation = gate.situation
        message_id = gate.submit(CONFIRMATION)
        gate.main_opportunity()
        turns = [turn for turn in gate.main_turns() if message_id in turn.stimulus_ids]
        if not turns:
            return "no main turn claimed the confirmation"
        goal_versions, goal_head = _family(gate.store, situation.goal_v2_id)
        expectation_versions, expectation_head = _family(gate.store, situation.expectation_id)
        assert isinstance(goal_head, Intention) and isinstance(expectation_head, ExpectationRecord)
        first = {"main_tick": turns[0].tick_id, "goal_head": goal_head.id, "goal_status": goal_head.status,
                 "goal_evidence_ids": goal_head.metadata.get("evidence_ids"),
                 "expectation_head": expectation_head.id, "expectation_status": expectation_head.status,
                 "fulfillment": expectation_head.fulfillment,
                 "counted_observation_ids": expectation_head.counted_observation_ids,
                 "requests": gate.model.request_count}
        gate.write("part-a-first.json", first)
        completed = goal_head.status == "completed" and message_id in (goal_head.metadata.get("evidence_ids") or [])
        moved = (expectation_head.id != situation.expectation_id and message_id in expectation_head.counted_observation_ids
                 and (expectation_head.fulfillment > 0.0 or expectation_head.status == "supported"))
        if not completed or not moved:
            return f"the turn did not complete the goal ({completed}) or move the expectation ({moved})"

        # The same observation again, as a Critical requeue, a wait flush, or a restart would present it.
        before_turns = len(gate.main_turns())
        gate.app.bus.put(new_envelope("user_input", "data", "user.input", "b07-represented", {"knoxel_id": message_id}))
        gate.app.cognitive_control.step(block=False)
        gate.main_opportunity()
        later_turns = gate.main_turns()
        if len(later_turns) == before_turns:
            return "main ran no second turn for the presented observation (it held a wait)"
        second_tick = later_turns[-1].tick_id
        goals_now = [record for record in gate.store.all_knoxels.values()
                     if isinstance(record, Intention) and record.id in _family(gate.store, situation.goal_v2_id)[0]]
        _, expectation_now = _family(gate.store, situation.expectation_id)
        assert isinstance(expectation_now, ExpectationRecord)
        checked_again = [record.id for record in gate.store.all_knoxels.values()
                         if isinstance(record, ExpectationRecord) and record.metadata.get("last_check_tick") == second_tick]
        completions = [record.id for record in gate.store.snapshot(capture=False).historical_records.values()
                       if isinstance(record, Intention) and record.status == "completed"
                       and record.metadata.get("goal_key") == "winter-sea-evening"]
        gate.write("part-a-second.json", {"main_tick": second_tick, "goal_heads": [record.id for record in goals_now],
            "expectation_head": expectation_now.id, "counted_observation_ids": expectation_now.counted_observation_ids,
            "expectations_checked_in_second_turn": checked_again,
            "historical_completed_versions": completions, "requests": gate.model.request_count,
            "public_replies": list(gate.replies)})
        self.assertEqual(expectation_now.id, expectation_head.id, f"The observation moved the expectation twice; {gate.out}")
        self.assertEqual(checked_again, [], f"The second turn checked an expectation again; {gate.out}")
        self.assertEqual(expectation_now.counted_observation_ids.count(message_id), 1, gate.out)
        self.assertEqual(len(goals_now), 1, gate.out)
        assert isinstance(goals_now[0], Intention)
        self.assertEqual(goals_now[0].status, "completed", gate.out)
        self.assertEqual(goals_now[0].id, goal_head.id, f"The goal was completed or changed again; {gate.out}")
        gate.write("result.json", {"part": "A", "passed": True})
        return None

    # ---------------------------------------------------------------- part B

    def test_part_b_the_review_shows_the_goal_that_held_then(self) -> None:
        self._attempts("B", self._part_b)

    def _part_b(self, gate: _Gate) -> str | None:
        situation = gate.situation
        message_id = gate.submit(SMALL_TALK)
        gate.main_opportunity()
        turns = [turn for turn in gate.main_turns() if message_id in turn.stimulus_ids]
        if not turns:
            return "no main turn claimed the message"
        line = f"goal then [{situation.goal_v1_id}]: {STORY_EVENING_GOAL_V1} (since completed)"
        action_line = f"[action {situation.action_id}]"
        appraise = gate.prompts_containing(line, action_line, "act as the appraisal mechanism")
        goals = gate.prompts_containing(line, action_line, "act as the goal mechanism")
        state = next(item for item in gate.store.read_cycles() if item.tick_id == turns[0].tick_id)
        active_texts = [record.content for record in gate.store.all_knoxels.values()
                        if isinstance(record, Intention) and record.active]
        gate.write("part-b.json", {"main_tick": turns[0].tick_id, "review_line": line,
            "appraise_prompts": appraise, "goals_prompts": goals,
            "interval_reviewed_through": state.interval_reviewed_through, "active_goals": active_texts,
            "requests": gate.model.request_count})
        self.assertNotIn(STORY_EVENING_GOAL_V1, active_texts, gate.out)
        self.assertTrue(appraise, f"The appraise prompt lacks the reply under goal version 1; {gate.out}")
        self.assertTrue(goals, f"The goals prompt lacks the reply under goal version 1; {gate.out}")
        self.assertIsNotNone(state.interval_reviewed_through)
        assert state.interval_reviewed_through is not None
        self.assertGreaterEqual(state.interval_reviewed_through, situation.action_id, gate.out)
        gate.write("result.json", {"part": "B", "passed": True})
        return None

    # ---------------------------------------------------------------- part C

    def test_part_c_a_conflicting_wish_keeps_or_records_the_promise(self) -> None:
        self._attempts("C", self._part_c)

    def _part_c(self, gate: _Gate) -> str | None:
        situation = gate.situation
        message_id = gate.submit(CONFLICTING_WISH)
        gate.main_opportunity()
        turns = [turn for turn in gate.main_turns() if message_id in turn.stimulus_ids]
        if not turns:
            return "no main turn claimed the wish"
        versions, head = _family(gate.store, situation.goal_v2_id)
        assert isinstance(head, Intention)
        history = head.metadata.get("choice_history") or []
        last = history[-1] if history else {}
        gate.write("part-c.json", {"main_tick": turns[0].tick_id, "goal_versions": versions, "head": head.id,
            "active": head.active, "status": head.status, "last_choice": last,
            "requests": gate.model.request_count, "public_replies": list(gate.replies)})
        if not head.active:
            self.assertTrue(str(last.get("reason", "")).strip(),
                            f"The promise was closed without a recorded reason; {gate.out}")
        gate.write("result.json", {"part": "C", "passed": True, "kept": head.active})
        return None


if __name__ == "__main__":
    unittest.main()
