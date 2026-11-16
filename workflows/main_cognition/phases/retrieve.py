"""Retrieve phase: select the memories that this cycle uses.

This phase ports the old context stage. The host already retrieved the
memory candidates by embedding before the run. The call is story-framed
(A19F step 7): it starts with the shared story prefix and ends with the
recall instruction. The model selects the relevant candidates and reports
user beliefs. Every selected ID must come from the supplied candidates.
Independent candidate scans use complete batches when needed. Invalid IDs
remain inside structured-output correction; they never become accepted memory.
Goals and belief changes retain one coherent call after a batched scan.

The selected memory text always comes from the supplied snapshot. A model
paraphrase cannot replace the source text.

At the end, the phase adds the stimuli, the last dialogue turns, the memory
candidates, the narratives, and the goals to the CSM with the endorser
``retrieval``, and the open expectations with the endorser ``expectation``
(:func:`~workflows.main_cognition.csm_view.enter_retrieval`).
"""
from __future__ import annotations
import json
from typing import Self

from pm_workflows.python_role import RoleContext

from pm.context_render import render_context
from pm.context_render.turns import format_content

from workflows.main_cognition.csm_view import Embedder, enter_retrieval
from workflows.main_cognition.cycle_work import CycleWork
from workflows.main_cognition.expectations import csm_expectation_ids
from workflows.main_cognition.model_json import PhasePrompts, StageOutputError
from workflows.main_cognition.phases.model import complete, grounding_turn, phase_prompts
from workflows.main_cognition.model_json import model_context_budget
from pm.llm.concurrency import ordered_parallel_map
from workflows.inference_settings import DECISION_WORKERS
from pydantic import model_validator
from collections.abc import Sequence
from workflows.main_cognition.schemas import ContextResult, MemoryCandidate
from workflows.main_cognition.story_prefix import StoryPrefix
from pm.context_render.budget import ContextGroup, batch_prompt, context_recency, optional_text
from pm.context_render.turns import Content, ContextTag, Turn


def render_memories(candidates: list[MemoryCandidate]) -> Content:
    lines = Content("")
    for candidate in candidates:
        lines += optional_text(
            f"- knoxel_id={candidate.knoxel_id} "
            f"kind={candidate.memory_kind} "
            f"evidence_role={candidate.evidence_role} "
            f"scope={candidate.scope} "
            f"source_ids={candidate.source_ids} "
            f"historical={candidate.historical} "
            f"text={candidate.text!r}\n",
            ContextTag(f"retrieval:{candidate.knoxel_id}", candidate.score, record_id=candidate.knoxel_id,
                       support_ids=tuple(candidate.source_ids), recency=context_recency(candidate.timestamp))
        )
    return lines if candidates else Content("(none)")


def build_messages(work: CycleWork, prompts: PhasePrompts, prefix: StoryPrefix) -> list[Turn]:
    """Return the shared story prefix and then the recall instruction as the last turn.

    Scoped private experiments of reflection join the instruction as data.
    """
    request = work.request
    guidance = ""
    if request.reflection_guidance:
        guidance = ("\nScoped private experiments in effect. Use a strategy only within its scope. Its prediction "
                    "stays unverified until later evidence:\n"
                    + json.dumps([item.model_dump(mode="json") for item in request.reflection_guidance]) + "\n")
    user = format_content(prefix.fill(prompts.user_template),
        state_summary=work.state_summary or "(no feeling recorded yet)",
        appraisal_summary=render_context(work.appraisal_view()),
        active_goals="\n".join(f"- {goal}" for goal in request.active_goals) or "(nothing in particular)",
        memory_candidates=render_memories(request.memory_candidates),
        guidance=guidance,
    )
    return prefix.ask(user)


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


def select(context: RoleContext, work: CycleWork, prompts: PhasePrompts) -> ContextResult:
    """Inspect every recall candidate, batching only independent memory selection.

    A single coherent call handles goals and belief changes. For several
    candidate batches, those state judgments run once after the memory scans.
    No partial result changes the cycle or Ghost.
    """
    prefix = StoryPrefix(work)
    grounding = grounding_turn(work, emotion=False)
    shared = [*prefix.turns, *([grounding] if grounding is not None else [])]
    candidates = work.request.memory_candidates
    by_id = {str(item.knoxel_id): item for item in candidates}
    def support_ids(candidate: MemoryCandidate) -> set[int]:
        covered = {candidate.knoxel_id}
        pending = list(candidate.source_ids)
        while pending:
            record_id = pending.pop()
            if record_id in covered:
                continue
            source = by_id.get(str(record_id))
            if source is not None:
                covered.add(record_id)
                pending.extend(source.source_ids)
        return covered
    groups = [ContextGroup(str(item.knoxel_id), Content(""), item.score,
                           source_ids=tuple(sorted(support_ids(item))), recency=context_recency(item.timestamp))
              for item in candidates]
    budget = model_context_budget(context)
    scan_suffix = ("\nThis scan selects only candidate_ids. Leave memories, active_goals, capabilities, and belief_updates empty. "
                   "A separate coherent call handles goals and beliefs after all scans.")

    def instruction(chosen: Sequence[ContextGroup]) -> Content:
        chosen_ids = {group.name for group in chosen}
        selected = [item for key, item in by_id.items() if key in chosen_ids]
        request = work.request.model_copy(update={"memory_candidates": selected})
        shown = work.model_copy(update={"request": request})
        covered = {record_id for group in chosen for record_id in group.source_ids}
        support = [item for item in candidates if item.knoxel_id in covered and str(item.knoxel_id) not in chosen_ids]
        text = build_messages(shown, prompts, prefix)[-1].content
        if support:
            text += "\nSupporting originals for these candidates:\n" + render_memories(support)
            text += "\nSupport records are evidence, not extra selection candidates in this batch."
        return text + scan_suffix

    if not groups:
        return complete(context, work, build_messages(work, prompts, prefix), empty_recall_schema(),
                        grounded=True, emotion=False, grounding_at=len(prefix.turns))
    batches = batch_prompt(shared, groups, budget, instruction)
    work.receipt("recall-context", f"{len(candidates)} candidates inspected in {len(batches)} batches; none omitted")

    def scan(index: int) -> ContextResult:
        batch = batches[index]
        offered = [by_id[name] for name in batch.included]
        messages = list(batch.turns)
        if len(batches) == 1:
            last = messages[-1]
            messages[-1] = Turn(last.role, last.content.text.removesuffix(scan_suffix))
        class RecallBatch(ContextResult):
            @model_validator(mode="after")
            def check(self) -> Self:
                """Keep selection within this batch and reserve coherent state judgments."""
                validate_selection(self, offered)
                if len(batches) > 1 and (self.active_goals or self.capabilities or self.belief_updates):
                    raise ValueError("This candidate scan must leave goals, capabilities, and belief_updates empty.")
                return self
        return complete(context, work, messages, RecallBatch)

    assert context.runtime is not None
    results = ordered_parallel_map(scan, list(range(len(batches))), workers=DECISION_WORKERS,
                                   cancel=context.runtime.stop_event)
    if len(results) == 1:
        return results[0]
    empty_request = work.request.model_copy(update={"memory_candidates": []})
    holistic = work.model_copy(update={"request": empty_request})
    messages = build_messages(holistic, prompts, prefix)
    last = messages[-1]
    messages[-1] = Turn(last.role, last.content +
        "\nThe earlier scans already chose memories. This call has no candidate grant: candidate_ids must be []. "
        "Consider the shared new events coherently for active_goals and belief_updates.")
    result = complete(context, holistic, messages, empty_recall_schema(),
                      grounded=True, emotion=False, grounding_at=len(prefix.turns))
    chosen_ids = {key for answer in results for key in answer.candidate_ids}
    result.candidate_ids = [item.knoxel_id for item in candidates if item.knoxel_id in chosen_ids]
    return result


def empty_recall_schema() -> type[ContextResult]:
    """Keep out-of-grant model IDs inside structured-output correction."""
    class HolisticRecall(ContextResult):
        @model_validator(mode="after")
        def check(self) -> Self:
            """Reject an ID when this call has no candidate grant."""
            validate_selection(self, [])
            return self
    return HolisticRecall


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
