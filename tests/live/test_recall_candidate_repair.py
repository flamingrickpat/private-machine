"""Real-endpoint gate for the recall candidate-id rule of main cognition.

The retrieve phase offers memory candidates and validates the model's answer
against exactly those ids (``RecallValidationContext``). A selected id outside
them is a rejected answer, so ``pm.llm.structured.complete_json`` asks the
model again with the problem text instead of ending the cognitive cycle.

The first sample of this gate is the real answer that killed Emmy's live run
on 2026-11-23: the model copied id 15368, a dialogue record id shown in the
same prompt as the candidates but never offered as one. That captured answer
is the initial sample, so the rejection and the repair turn are real, and the
repair answer comes from the real configured model. A captured answer is a
regression input, not a substitute endpoint.

Prerequisite: a reachable model endpoint. ``PM_MODEL_CONFIG`` names the model
config file, which is how Emmy runs locally; without it the gate uses the repo
``config.local.yaml`` or ``config.example.yaml``. With no reachable endpoint
the activation fails and the gate fails. It does not pass.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Event
import unittest
from uuid import uuid4

from pm.constants import Sampler
from pm.context_render.turns import Role, Turn, format_content
from pm.llm.activation import ModelActivation
from pm.llm.config import load_config, resolve_connection
from pm.llm.generation import GenerationSettings
from pm.llm.prompt_log import log_directory
from pm.llm.session import ModelSession
from pm.llm.stage_model import LazyStageModel
from pm.llm.structured import InitialResponse, complete_json
from pm.memory.evidence import EvidenceItem as MemoryCandidate
from pm.utils.operating_system import OperatingSystem
from workflows.main_cognition.codelet_registry import fill
from workflows.main_cognition.phases.model import phase_prompts
from workflows.main_cognition.phases.retrieve import render_memories
from workflows.main_cognition.prompt_values import RetrievalTemplateValues
from workflows.main_cognition.schemas import ContextResult, RecallValidationContext

ROOT = Path(__file__).resolve().parents[2]

#: The id of Emmy's captured wrong answer. It is a dialogue record id of the
#: story, so the prompt shows it, and the retrieve phase never offered it.
UNOFFERED_STORY_ID = 15368


def _candidate(knoxel_id: int, text: str) -> MemoryCandidate:
    """One memory candidate in the shape the retrieve phase renders."""
    return MemoryCandidate(knoxel_id=knoxel_id, text=text, revision=1, memory_kind="Dialogue",
                           evidence_role="user_report", historical=False, score=1.0)


def _recall_instruction(candidates: list[MemoryCandidate]) -> str:
    """Fill the real retrieve instruction turn with these candidates.

    The phase prompt is the thing that has to be unambiguous, so the gate uses
    it byte for byte. Only the story prefix and the affect summary are
    simulated, because the prefix comes from the frozen snapshot of a cycle.
    """
    template = fill(phase_prompts("retrieve").user_template, "Emmy", "Rick")
    values = RetrievalTemplateValues(
        state_summary="Rick restarted the machine, then moved the talk to the Minecraft server.",
        appraisal_summary="Judgment: interested. Feeling: warm and curious.",
        active_goals="- answer Rick about the Minecraft body",
        memory_candidates=render_memories(candidates),
        guidance="")
    return format_content(template, values).text


def _logged_attempts(directory: Path) -> list[dict]:
    """Return every logged model call, oldest file first."""
    attempts: list[dict] = []
    for path in sorted(directory.glob("*.json")):
        attempts.append(json.loads(path.read_text(encoding="utf-8")))
    return attempts


def _last_user_text(attempt: dict) -> str:
    """Return the text of the last user turn of one logged call."""
    messages = (attempt.get("payload") or {}).get("messages") or []
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content")
            return content if isinstance(content, str) else json.dumps(content)
    return ""


class LiveRecallCandidateRepairTests(unittest.TestCase):
    """Ask the real model to repair an id it copied out of the story."""

    @classmethod
    def setUpClass(cls) -> None:
        default = "config.local.yaml" if Path("config.local.yaml").is_file() else "config.example.yaml"
        cls.config_path = OperatingSystem.environment.get("PM_MODEL_CONFIG", default)
        cls.connection = resolve_connection(load_config(cls.config_path))
        cls.owner = ModelActivation()
        asyncio.run(cls.owner.activate(cls.connection, live_test=True))
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8]
        cls.out = ROOT / ".runtime/recall-candidate-repair" / stamp
        cls.out.mkdir(parents=True)
        cls.previous_log = OperatingSystem.environment.get("PM_PROMPT_LOG_DIR")
        OperatingSystem.environment["PM_PROMPT_LOG_DIR"] = str(cls.out / "llm-logs")

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.previous_log is None:
            OperatingSystem.environment.pop("PM_PROMPT_LOG_DIR", None)
        else:
            OperatingSystem.environment["PM_PROMPT_LOG_DIR"] = cls.previous_log

    def test_a_copied_story_id_is_rejected_and_repaired_into_the_candidates(self) -> None:
        # A candidate body cites the story id, exactly as a real attributed
        # record cites the dialogue it was derived from.
        candidates = [
            _candidate(15377, "[attributed_record; source=Feature; evidence=15377] Anchor the next reply in the "
                              "dandelions in the meadow, the proof that she survived the dark."),
            _candidate(15342, "Evidence:\n[15368] user_input: looks like you already found the minecraft body! "
                              "i told you i would give you a virtual body :3"),
        ]
        offered_ids = frozenset(candidate.knoxel_id for candidate in candidates)
        offered = RecallValidationContext(candidate_ids=offered_ids)
        instruction = _recall_instruction(candidates)
        turns = (Turn(Role.System, "You are the recall mechanism of one character."),
                 Turn(Role.User, instruction))
        wrong_answer = json.dumps({"memories": [], "active_goals": [],
                                   "candidate_ids": [15377, UNOFFERED_STORY_ID], "belief_updates": []},
                                  separators=(",", ":"))

        cancel = Event()
        model = LazyStageModel(self.connection, self.owner, live_test=True)
        result = complete_json(ModelSession(model, cancel), turns, ContextResult, validation_context=offered,
            initial=InitialResponse(wrong_answer), cancel=cancel,
            settings=GenerationSettings(temperature=Sampler.TempInstruct, enable_thinking=False))

        self.assertTrue(result.candidate_ids, "The repair answer selected no memory at all.")
        self.assertNotIn(UNOFFERED_STORY_ID, result.candidate_ids)
        self.assertLessEqual(set(result.candidate_ids), set(offered_ids))

        # The rejected sample is replayed from the capture, so only the repair
        # answer reaches the endpoint. The logged repair call proves the model,
        # not the test, produced the accepted ids.
        attempts = _logged_attempts(log_directory())
        self.assertGreaterEqual(len(attempts), 1, "The model was never asked to repair the rejected answer.")
        repair_turns = [attempt for attempt in attempts if "not a supplied memory candidate" in _last_user_text(attempt)]
        self.assertTrue(repair_turns, "The repair turn never told the model which rule it broke.")

        evidence = {
            "config": self.config_path,
            "offered_ids": sorted(offered_ids),
            "rejected_answer": wrong_answer,
            "repaired_candidate_ids": result.candidate_ids,
            "repaired_active_goals": result.active_goals,
            "logged_calls": len(attempts),
        }
        (self.out / "recall-repair.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        print(f"recall repair evidence: {self.out}")


if __name__ == "__main__":
    unittest.main()
