from __future__ import annotations

from dataclasses import dataclass
from pm.context_render.prompt import RenderSettings, RenderedPrompt, EvidenceDecision, require_rendered_prompt
from pm.context_render.budget import TokenEstimator
from pm.context_render.turns import Turn, Role
from workflows.minecraft_project.planning.context_contracts import DecisionContext, PlanningContext, ReviewContext, RetentionContext, VerificationContext


@dataclass(frozen=True, slots=True)
class TechnicalSection:
    name: str
    text: str
    required: bool


@dataclass(frozen=True, slots=True)
class ProjectDecisionRenderInput:
    operation: str
    system_prompt: str
    sections: tuple[TechnicalSection, ...]
    rejected_answer: str = ""
    problem: str = ""
    context: DecisionContext | None = None
    native: bool = False


def render_project_decision(input: ProjectDecisionRenderInput, settings: RenderSettings) -> RenderedPrompt:
    estimator = TokenEstimator(estimate_factor=settings.estimate_factor)
    budget = settings
    sections = input.sections
    if input.context is not None:
        from workflows.minecraft_project.planning.context_selection import decision_context
        from workflows.minecraft_project.planning.context_text import planning_sections, review_sections, retention_sections, verification_sections
        context = decision_context(input.context)
        match context:
            case PlanningContext():
                sections = planning_sections(context, input.native)
            case ReviewContext():
                sections = review_sections(context)
            case RetentionContext():
                sections = retention_sections(context)
            case VerificationContext():
                sections = verification_sections(context)
    selected = [f"{section.name}:\n{section.text}" for section in sections if section.required]
    correction: tuple[Turn, ...] = ()
    if input.problem:
        correction = (Turn(Role.Assistant, input.rejected_answer),
            Turn(Role.User, f"The answer failed validation: {input.problem}. Return one corrected object; keep task facts and source identities."))
    turns = (Turn(Role.System, input.system_prompt), Turn(Role.User, "\n\n".join(selected)), *correction)
    require_rendered_prompt(input.operation, turns, budget)
    decisions: list[EvidenceDecision] = []
    for index, section in enumerate(sections):
        if section.required:
            continue
        trial_text = "\n\n".join([*selected, f"{section.name}:\n{section.text}"])
        trial = (turns[0], Turn(Role.User, trial_text), *correction)
        included = estimator.estimate(trial) <= budget.input_ceiling
        if included:
            selected.append(f"{section.name}:\n{section.text}")
            turns = trial
        decisions.append(EvidenceDecision(index, 1, included,
            "historical technical context" if included else "available through original context archive"))
    return require_rendered_prompt(input.operation, turns, budget, tuple(decisions))


@dataclass(frozen=True, slots=True)
class NativePlannerRenderInput:
    instruction: str
    goal: str


def render_native_planner(input: NativePlannerRenderInput, settings: RenderSettings) -> RenderedPrompt:
    prompt = input.instruction + "\nCurrent goal: " + input.goal
    return require_rendered_prompt("Minecraft native planner", (Turn(Role.User, prompt),), settings)
