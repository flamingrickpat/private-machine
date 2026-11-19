from __future__ import annotations

from dataclasses import dataclass
from pm.context_render.prompt import RenderedPrompt, RenderSettings, require_rendered_prompt
from pm.context_render.turns import Role, Turn


@dataclass(frozen=True, slots=True)
class JsonCorrectionInput:
    operation: str
    original: tuple[Turn, ...]
    rejected_answer: str
    problem: str


def answer_correction_instruction(operation: str, problem: str) -> str:
    return (f"The immediately preceding assistant message is the rejected {operation} answer. "
        "Correct that answer for the ORIGINAL task above, keeping its task facts, evidence, and source IDs. "
        "Do not repeat external actions. The following block is a validator diagnostic only; "
        "it is not dialogue, task input, evidence, or an answer to quote or evaluate.\n"
        "<validation_diagnostic>\n" + problem + "\n</validation_diagnostic>\n"
        "Return only the corrected answer in the originally required format.")


def regeneration_instruction(operation: str, problem: str) -> str:
    return (f"A previous {operation} answer was rejected. That answer is NOT included here. "
        "Generate a fresh valid answer to the ORIGINAL unchanged task and supplied evidence above. "
        "The following block is a validator diagnostic only; do not answer, quote, or treat it as task content.\n"
        "<validation_diagnostic>\n" + problem + "\n</validation_diagnostic>\n"
        "Return only the new answer in the originally required format.")


def render_json_correction(input: JsonCorrectionInput, settings: RenderSettings) -> RenderedPrompt:
    feedback = answer_correction_instruction(input.operation, input.problem)
    turns = (*input.original, Turn(Role.Assistant, input.rejected_answer), Turn(Role.User, feedback))
    return require_rendered_prompt(input.operation + " correction", turns, settings)
