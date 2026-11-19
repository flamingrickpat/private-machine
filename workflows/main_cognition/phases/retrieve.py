"""Retrieve phase: select the memories that this cycle uses.

This phase ports the old context stage. The host already retrieved the
memory candidates by embedding before the run. The call is story-framed
(A19F step 7): it starts with the shared story prefix and ends with the
recall instruction. The model selects the relevant candidates and reports
user beliefs. Every selected id must come
from the supplied candidates. An invalid id gets one repair turn. A failed
repair keeps only the supplied sources and records the dropped ids, so the
conversation can continue without false memory references.

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

import json

from pm_workflows.python_role import RoleContext

from pm.context_render.turns import Role, Turn, format_content

from workflows.main_cognition.csm_view import Embedder, enter_retrieval
from workflows.main_cognition.rendering import render_appraisal, render_reflection_guidance, memory_candidate_text, select_recall_candidates
from workflows.main_cognition.cycle_work import CycleWork
from workflows.main_cognition.expectations import csm_expectation_ids
from workflows.main_cognition.prompt_config import PhasePrompts
from pm.llm.structured import JsonOutputError as StageOutputError
from workflows.main_cognition.phases.model import complete, phase_prompts
from workflows.main_cognition.schemas import ContextResult, MemoryCandidate
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


def validate_selection(result: ContextResult, candidates: list[MemoryCandidate]) -> None:
    """Raise if the model selected an id outside the supplied candidate set."""
    allowed = {candidate.knoxel_id for candidate in candidates}
    for candidate_id in result.candidate_ids:
        if candidate_id not in allowed:
            raise StageOutputError(
                f"Context selected knoxel_id {candidate_id} that is not a "
                "supplied memory candidate."
            )


def retain_valid_selection(result: ContextResult, candidates: list[MemoryCandidate]) -> tuple[ContextResult, list[int]]:
    """Drop unavailable memory IDs after a failed repair, retaining real sources.

    Rebuild memory text from the supplied snapshot. This avoids promoting an
    unsupported model summary when its reference cannot be resolved. Current
    stimuli remain in the cycle, so failed recall does not block conversation.
    Return the repaired result and dropped IDs for the diagnostic receipt.
    """
    sources = {item.knoxel_id: item.text for item in candidates}
    retained = list(dict.fromkeys(key for key in result.candidate_ids if key in sources))
    dropped = [key for key in result.candidate_ids if key not in sources]
    selected = result.model_copy()
    selected.candidate_ids = retained
    selected.memories = [str(sources[key]) for key in retained]
    return selected, dropped


def select(context: RoleContext, work: CycleWork, prompts: PhasePrompts) -> ContextResult:
    """Ask the model for a selection and repair invalid ids once.

    Diagnostic files go to ``outbox/`` of the working target.
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
    candidates = selected_candidates
    result = complete(context, messages, ContextResult)
    try:
        validate_selection(result, candidates)
    except StageOutputError as error:
        # The public domain context also contains IDs. They are not a grant
        # for memory selection. Retain the invalid response and repair once.
        allowed = [item.knoxel_id for item in candidates]
        context.runtime.environment.write_text("outbox/selection-rejection.json", json.dumps({
            "error": str(error), "rejected_result": result.model_dump(mode="json"),
            "allowed_ids": allowed}, indent=2))
        recovery = [*messages, Turn(Role.User, str(error) + " Select only candidate IDs: " + str(allowed)
                     + ". IDs in other fields are not granted. Return an empty selection if none fits.")]
        result = complete(context, recovery, ContextResult)
        validate_selection(result, candidates)
    return result


def run(context: RoleContext, work: CycleWork) -> CycleWork:
    """Select memories and bind them to their original candidate records."""
    result = select(context, work, phase_prompts("retrieve"))
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


__all__ = ["build_messages", "render_memories", "retain_valid_selection", "run", "select", "validate_selection"]
