"""A scripted model for main cognition tests without a model endpoint.

The real manifest, the real phases, and the real host run. Only the tools
are replaced: ``llm_complete`` answers from a script, ``embed_texts`` hashes
words into small vectors, and ``codelet_agent`` returns a fixed answer. The
script finds the phase of each call from a marker phrase in the phase's
system prompt and answers with the JSON object that the test supplied for
that phase. The codelet calls share one system prompt, so their markers are
in the last turn. A phase without a supplied answer gets a neutral default.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable
from unittest import mock

from pm.llm.prompt_encoding import normalize_turns
from workflows.main_cognition.codelet_percepts import HEADER_FIELDS, MODEL_BY_TYPE, CodeletPerceptType
from workflows.main_cognition.schemas import AgentCodeletAnswer

#: A phrase of each system prompt that only that phase uses.
PHASE_MARKERS = (
    ("triage", '"Triage" codelet'),
    ("appraise", '"Appraisal" codelet'),
    ("retrieve", '"Context" codelet'),
    ("goals", '"Goals" codelet'),
    ("experience", '"Workspace" codelet'),
    ("decide", '"Action Selection" codelet'),
    ("attention", "decide what to pay attention to"),
    ("focus", "deliberate focus proposal"),
)

#: Markers in the last turn of the codelet calls. Their system prompt is shared.
CODELET_MARKERS = (
    ("codelet_select", "act as a selector"),
    ("codelet_percepts", "put the result of your analysis into JSON"),
    ("codelet_story", "act as one mental mechanism"),
)
STORY_SYSTEM_MARKER = "expert story writer"


def default_percepts(messages) -> dict:
    """Fill the first offered percept type with one text field."""
    text = normalize_turns(messages)[-1].content.text
    offered = re.findall(r'^- "(\w+)":', text, re.MULTILINE)
    model = MODEL_BY_TYPE[CodeletPerceptType(offered[0])]
    field = next(name for name in model.model_fields if name not in HEADER_FIELDS)
    value = ["scripted"] if model.model_fields[field].annotation == list[str] else "scripted percept"
    return {"target": "self", "about_ids": [], offered[0]: {"salience": 0.6, "valence": -0.2, field: value}}

DEFAULT_ANSWERS = {
    "triage": {"items": []},
    "appraise": {"feeling": "calm"},
    "retrieve": {"candidate_ids": []},
    "codelet_select": {"ratings": {}},
    "codelet_story": "She has had enough of waiting and wants to shake things up.",
    "codelet_percepts": default_percepts,
    "codelet_agent": {"recommendation": "She remembers this pattern and wants to name it.", "evidence_ids": []},
    "attention": {"scores": []},
    "goals": {"changes": []},
    "experience": {"frame_reason": "reply", "focus_ids": [], "control_mode": "reactive"},
    "decide": {"actions": []},
    "focus": {"accept": False, "reason": "not now"},
}


def phase_of(messages) -> str:
    """Return the phase name of one model call."""
    turns = normalize_turns(messages)
    system = turns[0].content.text
    if STORY_SYSTEM_MARKER in system:
        last = turns[-1].content.text
        return next(name for name, marker in CODELET_MARKERS if marker in last)
    return next(name for name, marker in PHASE_MARKERS if marker in system)


def fake_embedding(text: str) -> list[float]:
    """Hash the words of ``text`` into a 32-dimension unit vector. Index 0 is always set."""
    vector = [0.0] * 32
    vector[0] = 1.0
    for word in re.findall(r"\w+", text.lower()):
        vector[1 + int(hashlib.md5(word.encode()).hexdigest(), 16) % 31] += 1.0
    norm = math.hypot(*vector)
    return [value / norm for value in vector]


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
        return answer if isinstance(answer, str) else json.dumps(answer)

    def agent(self, context, job) -> AgentCodeletAnswer:
        """Answer one agent codelet session from the script."""
        self.calls.append("codelet_agent")
        answer = self.answer_for("codelet_agent")
        if isinstance(answer, Callable):
            answer = answer(job)
        return AgentCodeletAnswer.model_validate(answer)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return a hashed embedding for each text."""
        return [fake_embedding(text) for text in texts]

    def tools(self) -> dict:
        """Return the scripted tools of one run."""
        return {"llm_complete": self, "embed_texts": self.embed, "codelet_agent": self.agent}


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
    """Return a ``tools_factory`` that grants the scripted tools of ``model``."""
    return lambda environment, batch: model.tools()


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


__all__ = ["CODELET_MARKERS", "DEFAULT_ANSWERS", "fake_embedding", "fixed_attention", "PHASE_MARKERS", "ScriptedModel", "answers_from_reports", "phase_of", "scripted_tools"]
