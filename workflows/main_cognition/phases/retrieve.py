"""Retrieve phase: select the memories that this cycle uses.

This phase ports the old context stage. The host already retrieved the
memory candidates by embedding before the run. The model selects the
relevant candidates and reports user beliefs. Every selected id must come
from the supplied candidates. An invalid id gets one repair turn. A failed
repair keeps only the supplied sources and records the dropped ids, so the
conversation can continue without false memory references.

The selected memory text always comes from the supplied snapshot. A model
paraphrase cannot replace the source text.
"""
from __future__ import annotations

import json

from pm.context_render import render_context
from pm.context_render.turns import Role, Turn, format_content

from workflows.main_cognition.cycle_work import CycleWork
from workflows.main_cognition.model_json import PhasePrompts, StageOutputError
from workflows.main_cognition.phases.model import complete, phase_prompts
from workflows.main_cognition.schemas import ContextResult, MemoryCandidate


def render_memories(candidates: list[MemoryCandidate]) -> str:
    lines = []
    for candidate in candidates:
        lines.append(
            f"- knoxel_id={candidate.knoxel_id} "
            f"kind={candidate.memory_kind} "
            f"evidence_role={candidate.evidence_role} "
            f"scope={candidate.scope} "
            f"source_ids={candidate.source_ids} "
            f"historical={candidate.historical} "
            f"text={candidate.text!r}"
        )
    return "\n".join(lines) or "(none)"


def build_messages(work: CycleWork, prompts: PhasePrompts) -> list[Turn]:
    request = work.request
    system = prompts.system_prompt.format(companion_name=request.companion_name)
    user = format_content(prompts.user_template,
        companion_name=request.companion_name,
        user_name=request.user_name,
        state_summary=work.state_summary or "(no recorded state update)",
        triage_summary=render_context(work.triage),
        appraisal_summary=render_context(work.appraisal_view()),
        active_goals="\n".join(f"- {goal}" for goal in request.active_goals) or "(none)",
        capabilities="\n".join(f"- {capability}" for capability in request.capabilities) or "(none)",
        memory_candidates=render_memories(request.memory_candidates),
        stimuli=render_context(request.stimuli),
    )
    if request.reflection_guidance:
        user = user + "\nScoped private experiments in effect:\n" + json.dumps(
            [item.model_dump(mode="json") for item in request.reflection_guidance])
        user = user + "\nUse a strategy only within its scope. Its prediction remains unverified until later evidence."
    return [
        Turn(Role.System, system.strip()),
        Turn(Role.User, user.strip()),
    ]


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
    return result.model_copy(update={"candidate_ids": retained,
        "memories": [str(sources[key]) for key in retained]}), dropped


def select(context, work: CycleWork, prompts: PhasePrompts) -> ContextResult:
    """Ask the model for a selection and repair invalid ids once.

    Diagnostic files go to ``outbox/`` of the working target.
    """
    messages = build_messages(work, prompts)
    candidates = work.request.memory_candidates
    result = complete(context, work, messages, ContextResult, grounded=True, emotion=False)
    try:
        validate_selection(result, candidates)
    except StageOutputError as error:
        # The public domain context also contains IDs. They are not a grant
        # for memory selection. Retain the invalid response and repair once.
        allowed = [item.knoxel_id for item in candidates]
        context.runtime.environment.write_text("outbox/selection-rejection.json", json.dumps({
            "error": str(error), "rejected_result": result.model_dump(mode="json"),
            "allowed_ids": allowed}, indent=2))
        recovery = [*messages, {"role": "assistant", "content": result.model_dump_json()},
                    {"role": "user", "content": str(error) + " Select only from these candidate IDs: "
                     + str(allowed)
                     + ". IDs in other context fields are not granted. Return an empty selection if none fits."}]
        result = complete(context, work, recovery, ContextResult, grounded=True, emotion=False)
        try:
            validate_selection(result, candidates)
        except StageOutputError as repair_error:
            rejected = result.model_dump(mode="json")
            result, dropped = retain_valid_selection(result, candidates)
            context.runtime.environment.write_text("outbox/selection-recovery.json", json.dumps({
                "error": str(repair_error), "rejected_result": rejected,
                "dropped_ids": dropped, "retained_ids": result.candidate_ids,
                "outcome": "continued_with_supplied_memories"}, indent=2))
    return result


def run(context, work: CycleWork) -> CycleWork:
    """Select memories and bind them to their original candidate records."""
    result = select(context, work, phase_prompts("retrieve"))
    by_id = {item.knoxel_id: item for item in work.request.memory_candidates}
    selected = list({key: by_id[key] for key in result.candidate_ids}.values())
    result.memories = [item.text for item in selected]
    work.context = result
    work.selected_memories = selected
    work.receipt("retrieve", f"selected {len(result.candidate_ids)} memories and "
                             f"{len(result.active_goals)} goals")
    return work


__all__ = ["build_messages", "render_memories", "retain_valid_selection", "run", "select", "validate_selection"]
