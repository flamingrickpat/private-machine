from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pm.context_render.budget import RequiredContextTooLarge, TokenEstimator
from pm.context_render.prompt import RenderedPrompt, RenderSettings, require_rendered_prompt
from pm.context_render.turns import Role, Turn


@dataclass(frozen=True, slots=True)
class JsonCorrectionInput:
    operation: str
    original: tuple[Turn, ...]
    rejected_answer: str
    problem: str


@dataclass(frozen=True, slots=True)
class AgentReportCorrectionInput:
    operation: str
    instruction: str
    response_schema: str
    completed_report: str
    validation_problem: str


@dataclass(slots=True)
class CorrectionHistory:
    problems: list[str] = field(default_factory=list[str])
    rejected_answers: int = 0

    def record_rejection(self, operation: str, problem: str) -> str:
        """Record one rejected answer and say so in the log while the repair runs.

        Return the correction text for the next answer: the newest problems first,
        at most eight. The caller decides when a loop has repaired enough, because
        only the caller knows what one more answer costs. An hour of silent repairs
        on 2026-11-22 looked like a hang.
        """
        self.rejected_answers += 1
        if problem in self.problems:
            self.problems.remove(problem)
        self.problems.append(problem)
        if len(self.problems) > 8:
            self.problems.pop(0)
        logging.getLogger(__name__).warning(
            "Answer rejected operation=%s attempt=%s problem=%s",
            operation, self.rejected_answers, problem[:300])
        return "\n\n".join(reversed(self.problems))


@dataclass(frozen=True, slots=True)
class CorrectionFeedback:
    text: str
    estimated_tokens: int
    omitted_characters: int


def render_regeneration_feedback(operation: str, problem: str, settings: RenderSettings) -> CorrectionFeedback:
    estimator = TokenEstimator(estimate_factor=settings.estimate_factor)
    empty_tokens = estimator.estimate(())
    maximum_diagnostic_characters = settings.correction_tokens * 16
    if len(problem) <= maximum_diagnostic_characters:
        text = regeneration_instruction(operation, problem)
        tokens = estimator.estimate((Turn(Role.User, text),)) - empty_tokens + 1
        if tokens <= settings.correction_tokens:
            return CorrectionFeedback(text, tokens, 0)

    minimum = regeneration_instruction(operation,
        "[Diagnostic omitted from prompt; the complete diagnostic is retained in the renderer input capture.]")
    minimum_tokens = estimator.estimate((Turn(Role.User, minimum),)) - empty_tokens + 1
    if minimum_tokens > settings.correction_tokens:
        raise RequiredContextTooLarge(
            f"{operation}: correction instructions require {minimum_tokens} tokens, "
            f"but the correction reserve is {settings.correction_tokens}.")
    selected = CorrectionFeedback(minimum, minimum_tokens, len(problem))
    low = 0
    high = min(len(problem), maximum_diagnostic_characters)
    while low <= high:
        kept_characters = (low + high) // 2
        omitted = len(problem) - kept_characters
        diagnostic = problem[:kept_characters] + f"\n[{omitted} diagnostic characters omitted; complete diagnostic retained in capture.]"
        text = regeneration_instruction(operation, diagnostic)
        tokens = estimator.estimate((Turn(Role.User, text),)) - empty_tokens + 1
        if tokens <= settings.correction_tokens:
            selected = CorrectionFeedback(text, tokens, omitted)
            low = kept_characters + 1
        else:
            high = kept_characters - 1
    return selected


def regeneration_instruction(operation: str, problem: str) -> str:
    return (f"A previous {operation} answer was rejected. "
        "Generate a fresh valid answer to the ORIGINAL unchanged task and supplied evidence above. "
        "Do not repeat external actions. "
        "The following block is a validator diagnostic only; do not answer, quote, or treat it as task content.\n"
        "<validation_diagnostic>\n" + problem + "\n</validation_diagnostic>\n"
        "Return only the new answer in the originally required format.")


def render_json_correction(input: JsonCorrectionInput, settings: RenderSettings) -> RenderedPrompt:
    require_rendered_prompt(input.operation + " original", input.original, settings)
    feedback = render_regeneration_feedback(input.operation, input.problem, settings)
    turns = (*input.original, Turn(Role.User, feedback.text))
    return require_rendered_prompt(input.operation + " correction", turns, settings.without_correction_reserve(),
        selected_groups=("original task and evidence", "bounded validation diagnostic"))


def render_agent_report_correction(input: AgentReportCorrectionInput, settings: RenderSettings) -> RenderedPrompt:
    turns = (Turn(Role.System,
        "Correct the final report of a completed agent session. No tools are available. "
        "Keep recorded facts and source IDs. Do not invent tool results or repeat actions. "
        "The required JSON schema is: " + input.response_schema),
        Turn(Role.User, input.instruction),
        Turn(Role.User, "Completed agent report, retained as evidence of work already performed:\n"
            "<completed_agent_report>\n" + input.completed_report + "\n</completed_agent_report>"))
    return require_rendered_prompt(input.operation, turns, settings,
        selected_groups=("original instruction", "completed report evidence"))
