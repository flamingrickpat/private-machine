"""Retrieve phase: select the memories that this cycle uses.

This phase ports the old context stage. The host already retrieved the
memory candidates by embedding before the run. The call is story-framed
(A19F step 7): it starts with the shared story prefix and ends with the
recall instruction. The model selects the relevant candidates and reports
user beliefs. Every selected id must come from the supplied candidates. That
rule lives in the answer schema itself: the call validates
:class:`~workflows.main_cognition.schemas.ContextResult` with a
:class:`~workflows.main_cognition.schemas.RecallValidationContext` that holds
the offered ids, so a wrong id is a rejected answer and
:func:`pm.llm.structured.complete_json` asks the model again with the problem
text until an answer fits or the run is cancelled. The phase runs no second
check after the model answers, so one copied id cannot end the cycle.
The prompt shows many ids that are not selectable, which is why the marker of
a candidate is written differently from every other id (see
:func:`workflows.main_cognition.rendering.memory_candidate_text`).

The selected memory text always comes from the supplied snapshot. A model
paraphrase cannot replace the source text.

At the end, the phase adds the stimuli, the last dialogue turns, the memory
candidates, the narratives, and the goals to the CSM with the endorser
``retrieval``, and the open expectations with the endorser ``expectation``
(:func:`~workflows.main_cognition.csm_view.enter_retrieval`).
"""
from __future__ import annotations

from workflows.main_cognition.prompt_values import RetrievalTemplateValues

from pm.llm.role_model import role_model

from pm_workflows.python_role import RoleContext

from pm.context_render.turns import Turn, format_content

from workflows.main_cognition.csm_view import Embedder, enter_retrieval
from workflows.main_cognition.rendering import render_appraisal, render_reflection_guidance, memory_candidate_text, select_recall_candidates
from workflows.main_cognition.cycle_work import CycleWork
from workflows.main_cognition.expectations import csm_expectation_ids
from workflows.main_cognition.prompt_config import PhasePrompts
from workflows.main_cognition.phases.model import complete, phase_prompts
from workflows.main_cognition.schemas import ContextResult, MemoryCandidate, RecallValidationContext
from workflows.main_cognition.story_prefix import StoryPrefix


def render_memories(candidates: list[MemoryCandidate]) -> str:
    return "\n".join(memory_candidate_text(candidate) for candidate in candidates) or "(none)"


def build_messages(work: CycleWork, prompts: PhasePrompts, prefix: StoryPrefix) -> list[Turn]:
    """Return the shared story prefix and then the recall instruction as the last turn.

    Scoped private experiments of reflection join the instruction as data.
    """
    request = work.request
    guidance = ""
    if request.reflection_guidance:
        guidance = ("\nScoped private experiments in effect. Use a strategy only within its scope. Its prediction "
                    "stays unverified until later evidence:\n"
                    + render_reflection_guidance(request.reflection_guidance) + "\n")
    user = format_content(prefix.fill(prompts.user_template), RetrievalTemplateValues(
        state_summary=work.state_summary or "(no feeling recorded yet)",
        appraisal_summary=render_appraisal(work.appraisal_view()),
        active_goals="\n".join(f"- {goal}" for goal in request.active_goals) or "(nothing in particular)",
        memory_candidates=render_memories(request.memory_candidates),
        guidance=guidance))
    return prefix.ask(user.text)


def select(context: RoleContext, work: CycleWork, prompts: PhasePrompts) -> ContextResult:
    """Ask the model for one selection of the candidates that fit the budget.

    Only the candidates that the rendered prompt can hold are offered, and
    their ids become the validation context of the answer. An id outside them
    is repaired inside the model call, which logs each rejected attempt
    (``Answer rejected operation=ContextResult``) and keeps the rejected
    attempts in the prompt log.
    """
    assert context.runtime is not None
    prefix = StoryPrefix(work, settings=role_model(context.runtime).render_settings(),
            capture_settings=role_model(context.runtime).capture_settings("main-cognition"))
    selected_candidates = select_recall_candidates(work.request.memory_candidates, prefix.settings)
    request = work.request.model_copy()
    request.memory_candidates = selected_candidates
    shown = work.model_copy()
    shown.request = request
    messages = build_messages(shown, prompts, prefix)
    offered = RecallValidationContext(candidate_ids=frozenset(item.knoxel_id for item in selected_candidates))
    return complete(context, messages, ContextResult, validation_context=offered)


def run(context: RoleContext, work: CycleWork) -> CycleWork:
    """Select memories and bind them to their original candidate records."""
    result = select(context, work, phase_prompts("retrieve"))
    # The snapshot, not the model, supplies the text of a selected memory.
    by_id = {item.knoxel_id: item for item in work.request.memory_candidates}
    selected = list({key: by_id[key] for key in result.candidate_ids}.values())
    result.memories = [item.text for item in selected]
    work.context = result
    work.selected_memories = selected
    removed = enter_retrieval(work, Embedder(context, work.records), csm_expectation_ids(work))
    work.receipt("retrieve", f"selected {len(result.candidate_ids)} memories and "
                             f"{len(result.active_goals)} goals; CSM holds {len(work.state.csm.items)} items, "
                             f"pruned {len(removed)}")
    return work


__all__ = ["build_messages", "render_memories", "run", "select"]
