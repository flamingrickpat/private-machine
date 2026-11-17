"""Classify stimuli through the shared complete-prompt context fitter.

The shard renderer supplies record priorities. Both triage paths preserve new
stimuli, their instructions, and the newest partner lines. The configured
cognition window controls their input ceiling, as it does for other phases.
"""
from __future__ import annotations
from workflows.main_cognition.model_json import batch_internal_prompt, fit_internal_prompt
from workflows.main_cognition.context_reduction import evidence_text

from collections.abc import Sequence
from dataclasses import dataclass

from pm_workflows.python_role import RoleContext

from pm.constants import CODELET_BUDGET_DEFAULT, CODELET_BUDGET_MAX
from pm.context_render.budget import ContextBudget, ContextGroup, PromptSelection, RequiredContextTooLarge, context_recency, fit_turns
from pm.context_render.turns import Content
from pm.llm.concurrency import ordered_parallel_map
from workflows.inference_settings import DECISION_WORKERS
from pm.context_render.turns import Turn, format_content

from workflows.main_cognition.cycle_work import CycleWork
from workflows.main_cognition.model_json import PhasePrompts, model_context_budget
from workflows.main_cognition.phases.intake import group_key, group_stimuli
from workflows.main_cognition.phases.model import complete, grounding_turn, phase_prompts
from workflows.main_cognition.schemas import TRIAGE_CATEGORIES, TriageResult
from workflows.main_cognition.story_prefix import StoryPrefix

#: Triage categories from the least to the most important.
CATEGORIES = TRIAGE_CATEGORIES

@dataclass(frozen=True, slots=True)
class TriagePrompt:
    """One fitted triage prompt and the selection that produced it."""

    turns: list[Turn]
    selection: PromptSelection
    input_ceiling: int

    def summary(self) -> str:
        """Describe the fitting for the phase receipt."""
        optional_shown = len(self.selection.included)
        return (f"{optional_shown} optional records shown, {len(self.selection.omitted)} omitted, "
                f"estimated {self.selection.estimated_tokens} of {self.input_ceiling} input tokens")


def render_stimuli(work: CycleWork, prefix: StoryPrefix) -> str:
    """Render each stimulus with its id, source, kind, text, and the story part of its group."""
    lines: list[str] = []
    for stimulus in work.request.stimuli:
        part = prefix.label(group_key(stimulus, work.request))
        lines.append(f"- id={stimulus.knoxel_id} source={stimulus.source} "
                     f"kind={stimulus.kind} text={stimulus.text!r} part={part}")
    return "\n".join(lines) or "(none)"


def build_messages(work: CycleWork, prompts: PhasePrompts, prefix: StoryPrefix) -> list[Turn]:
    """Return the shared story prefix and then the triage instruction as the last turn."""
    request = work.request
    user = format_content(prefix.fill(prompts.user_template),
        state_summary=work.state_summary or "(no feeling recorded yet)",
        active_goals="; ".join(request.active_goals) or "(nothing in particular)",
        stimuli=evidence_text(render_stimuli(work, prefix), "triage-stimuli", [item.knoxel_id for item in request.stimuli]),
        budget_max=str(CODELET_BUDGET_MAX),
        budget_default=str(CODELET_BUDGET_DEFAULT),
    )
    return prefix.ask(user)


def rate_groups(work: CycleWork) -> None:
    """Copy the triage categories onto the stimulus groups of the state.

    A group gets the highest category of its stimuli. A group without a
    rated stimulus keeps an empty category and stays significant.
    """
    for group in work.state.stimulus_groups:
        categories = [item.category for item in work.triage.items if item.knoxel_id in group.stimulus_ids]
        if not categories:
            continue
        ranks = [CATEGORIES.index(name) for name in categories]
        group.category = CATEGORIES[max(ranks)]
        group.insignificant = max(ranks) == 0


def triage_budget(context: RoleContext) -> ContextBudget:
    """Resolve the configured internal window and serving limit."""
    return model_context_budget(context)


def fit_messages(context: RoleContext, work: CycleWork, *, appended: Sequence[Turn] = ()) -> TriagePrompt:
    """Fit both triage paths through the common renderer."""
    budget = triage_budget(context)
    try:
        return fit_prompt(work, budget, appended=appended)
    except RequiredContextTooLarge:
        prefix = StoryPrefix(work)
        turns = build_messages(work, phase_prompts("triage"), prefix)
        grounding = grounding_turn(work, emotion=True)
        if grounding is not None:
            turns.insert(len(prefix.turns), grounding)
        selection = fit_internal_prompt(context, [*turns, *appended], budget)
        return TriagePrompt(list(selection.turns), selection, budget.input_ceiling)


def fit_prompt(work: CycleWork, budget: ContextBudget, *, appended: Sequence[Turn] = ()) -> TriagePrompt:
    """Fit the whole triage attempt, including extraction instructions."""
    prefix = StoryPrefix(work)
    turns = build_messages(work, phase_prompts("triage"), prefix)
    grounding = grounding_turn(work, emotion=True)
    if grounding is not None:
        turns.insert(len(prefix.turns), grounding)
    selection = fit_turns([*turns, *appended], budget)
    return TriagePrompt(list(selection.turns), selection, budget.input_ceiling)


def classify(context: RoleContext, work: CycleWork) -> TriageResult:
    """Classify every new event, batching only when their required payload overflows."""
    budget = triage_budget(context)
    try:
        prompt = fit_prompt(work, budget)
    except RequiredContextTooLarge:
        # Events are task inputs, not optional recall. Move them into complete
        # per-batch instructions while keeping the old story context shared.
        empty_request = work.request.model_copy(update={"stimuli": []})
        prefix = StoryPrefix(work.model_copy(update={"request": empty_request}))
        grounding = grounding_turn(work, emotion=True)
        shared = [*prefix.turns, *([grounding] if grounding is not None else [])]
        by_id = {str(item.knoxel_id): item for item in work.request.stimuli}
        groups = [ContextGroup(key, Content(""), 0, source_ids=(item.knoxel_id,),
                               recency=context_recency(work.records[item.knoxel_id].timestamp))
                  for key, item in by_id.items()]

        def instruction(chosen: Sequence[ContextGroup]) -> Content:
            ids = {group.name for group in chosen}
            request = work.request.model_copy(update={"stimuli": [item for key, item in by_id.items() if key in ids]})
            return build_messages(work.model_copy(update={"request": request}), phase_prompts("triage"), prefix)[-1].content

        batches = batch_internal_prompt(context, shared, groups, budget, instruction)
        work.receipt("triage-context", f"{len(groups)} events in {len(batches)} batches; none omitted")

        def rate(index: int) -> TriageResult:
            batch = batches[index]
            return complete(context, work, batch.turns, TriageResult,
                            validation_context={"stimulus_ids": [int(name) for name in batch.included]})

        assert context.runtime is not None
        answers = ordered_parallel_map(rate, list(range(len(batches))), workers=DECISION_WORKERS,
                                       cancel=context.runtime.stop_event)
        items = [item for answer in answers for item in answer.items]
        if sorted(item.knoxel_id for item in items) != sorted(item.knoxel_id for item in work.request.stimuli):
            raise ValueError("Triage batches must classify every new event exactly once.")
        indexed = {item.knoxel_id: item for item in items}
        return TriageResult(items=[indexed[item.knoxel_id] for item in work.request.stimuli],
                            codelet_budget=max(answer.codelet_budget for answer in answers))
    work.receipt("triage-context", prompt.summary())
    return complete(context, work, prompt.turns, TriageResult,
                    validation_context={"stimulus_ids": [item.knoxel_id for item in work.request.stimuli]})


def run(context: RoleContext, work: CycleWork) -> CycleWork:
    """Reuse accepted input triage and classify only the remaining stimuli."""
    early = work.request.early_triage
    classified_ids: set[int] = {item.knoxel_id for item in early.items} if early is not None else set()
    remaining = [item for item in work.request.stimuli if item.knoxel_id not in classified_ids]
    if remaining:
        request = work.request.model_copy(update={"stimuli": remaining})
        state = work.state.model_copy(update={"stimulus_groups": group_stimuli(request)})
        pending = work.model_copy(update={"request": request, "state": state, "phases": list(work.phases)})
        result = classify(context, pending)
        work.phases.extend(pending.phases[len(work.phases):])
        if early is not None:
            result = TriageResult(items=[*early.items, *result.items],
                                  codelet_budget=max(early.codelet_budget, result.codelet_budget))
    else:
        if early is None:
            raise ValueError("Triage requires stimuli or an accepted early assessment.")
        result = early
    work.triage = result
    rate_groups(work)
    work.receipt("triage", f"triaged {len(result.items)} stimuli")
    return work


__all__ = ["CATEGORIES", "TriagePrompt", "build_messages", "fit_messages", "fit_prompt", "rate_groups", "render_stimuli", "run", "triage_budget"]
