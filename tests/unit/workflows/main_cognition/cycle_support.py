"""A scripted model for main cognition tests without a model endpoint.

The real manifest, the real phases, and the real host run. Only the
``llm_complete`` tool is replaced. The script finds the phase of each call
from a marker phrase in the phase's system prompt and answers with the JSON
object that the test supplied for that phase. A phase without a supplied
answer gets a neutral default.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from unittest import mock

from pm.llm.prompt_encoding import normalize_turns

#: A phrase of each system prompt that only that phase uses.
PHASE_MARKERS = (
    ("triage", '"Triage" codelet'),
    ("appraise", '"Appraisal" codelet'),
    ("retrieve", '"Context" codelet'),
    ("goals", '"Goals" codelet'),
    ("experience", '"Workspace" codelet'),
    ("decide", '"Action Selection" codelet'),
    ("attention", "decide what to pay attention to"),
    ("codelets", "one small part of the private mind"),
    ("focus", "deliberate focus proposal"),
)

DEFAULT_ANSWERS = {
    "triage": {"items": []},
    "appraise": {"feeling": "calm"},
    "retrieve": {"candidate_ids": []},
    "codelets": {"fires": False},
    "attention": {"scores": []},
    "goals": {"changes": []},
    "experience": {"frame_reason": "reply", "focus_ids": [], "control_mode": "reactive"},
    "decide": {"actions": []},
    "focus": {"accept": False, "reason": "not now"},
}


def phase_of(messages) -> str:
    """Return the phase name of one model call."""
    system = normalize_turns(messages)[0].content.text
    return next(name for name, marker in PHASE_MARKERS if marker in system)


class ScriptedModel:
    """Answer each model call with the scripted JSON of its phase.

    An answer is a mapping, or a callable that receives the messages and
    returns a mapping. ``reports`` is an old stage report fixture. The model
    reads it at each call, so a test can change it between cycles.
    ``calls`` lists the phase of every call in order.
    """

    def __init__(self, answers: dict[str, object] | None = None, *, reports: dict | None = None):
        self.answers = dict(answers or {})
        self.reports = reports
        self.calls: list[str] = []
        self.messages: list[list] = []

    def answer_for(self, phase: str) -> object:
        mapped = answers_from_reports(self.reports) if self.reports is not None else {}
        return {**DEFAULT_ANSWERS, **mapped, **self.answers}[phase]

    def __call__(self, messages, **generation) -> str:
        phase = phase_of(messages)
        self.calls.append(phase)
        self.messages.append(normalize_turns(messages))
        answer = self.answer_for(phase)
        if isinstance(answer, Callable):
            answer = answer(messages)
        return json.dumps(answer)


def fixed_attention(reports: dict):
    """Patch the attention phase to use ``reports["attention"]`` as its result.

    The patch reads the fixture at each cycle. Use it as a context manager
    around the cycles of a test that needs fixed nominations.
    """
    from workflows.main_cognition.schemas import AttentionResult

    def run(context, work):
        work.attention = AttentionResult.model_validate(
            {key: value for key, value in reports["attention"].items() if key != "summary"})
        work.receipt("attention", "fixed attention result")
        return work

    return mock.patch("workflows.main_cognition.phases.attention.run", run)


def scripted_tools(model: ScriptedModel):
    """Return a ``tools_factory`` that grants ``model`` as ``llm_complete``."""
    return lambda environment, batch: {"llm_complete": model}


def answers_from_reports(reports: dict) -> dict:
    """Map old stage report fixtures to phase answers.

    Codelet and attention fixtures are not mapped: the real codelets phase
    runs, and attention scores nothing unless a test scripts it.
    """
    names = {"triage": "triage", "appraisal": "appraise", "context": "retrieve", "goals": "goals",
             "workspace": "experience", "action_selection": "decide"}
    answers = {}
    for stage, phase in names.items():
        if stage in reports:
            answers[phase] = {key: value for key, value in reports[stage].items() if key != "summary"}
    return answers


__all__ = ["DEFAULT_ANSWERS", "fixed_attention", "PHASE_MARKERS", "ScriptedModel", "answers_from_reports", "phase_of", "scripted_tools"]
