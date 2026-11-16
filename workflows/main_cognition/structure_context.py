"""Review oversized structure evidence before the final attention comparison.

The attention phase owns this prompt-only projection. Every original member
enters a real model review. Source-linked digests let attention compare the
whole structure without clipping its records or changing the stored structure.
Digests are derived evidence, not new observations or Ghost records.
"""
from collections.abc import Sequence
from pydantic import BaseModel, Field, model_validator
from typing import Self
from pm_workflows.python_role import RoleContext

from pm.context_render.budget import ContextBudget, ContextGroup, batch_prompt, context_recency, fit_turns
from pm.context_render.story import story_line
from pm.context_render.turns import Content, Turn
from pm.llm.concurrency import ordered_parallel_map
from workflows.inference_settings import ATTENTION_WORKERS
from workflows.main_cognition.cycle_work import CycleWork
from workflows.main_cognition.phases.model import complete
from workflows.main_cognition.schemas import StructureDraft


class StructureDigest(BaseModel):
    """A compact, attributed review of a covered set of original members."""

    summary: str = Field(min_length=1)
    cited_ids: list[int] = Field(description="Exact top-level member record IDs, not other IDs quoted inside a member.")


class StructureReviewCoverage(BaseModel):
    """Proof that the prompt-only review covered every stored member."""

    structure_id: int
    level: int
    member_ids: tuple[int, ...]
    batch_source_ids: tuple[tuple[int, ...], ...]


def review_structure(context: RoleContext, work: CycleWork, structure: StructureDraft,
                     story: Sequence[Turn], budget: ContextBudget, maximum_tokens: int) -> str:
    """Review all members, reducing only derived digests if another level is needed."""
    required_story = fit_turns(story, budget, include_optional=False).turns
    groups = [ContextGroup(str(key), Content(story_line(work.records[key], with_id=True)), structure.salience,
                           source_ids=(key,), recency=context_recency(work.records[key].timestamp))
              for key in dict.fromkeys(structure.record_ids)]
    original_ids = set(structure.record_ids)
    level = 0
    while True:
        def instruction(chosen: Sequence[ContextGroup]) -> Content:
            evidence = "\n\n".join(group.content.text for group in chosen)
            allowed_ids = sorted({key for group in chosen for key in group.source_ids})
            return Content(
                f"Review evidence of one structure before attention compares it: {structure.theme}.\n"
                f"Why its records belong together: {structure.reason}.\n{evidence}\n\n"
                "Write a source-linked digest, not an attention rating. Preserve distinct events, corrections, dates, "
                "intentions, competing interpretations, emotional stakes, and uncertainty. Distinguish speech, private "
                "thought, inference, and original observation. Repetition is not corroboration. Retain contradictions. "
                "A previous digest is derived evidence. Never invent a source or turn an assumption into a fact. "
                f"Allowed cited_ids are these top-level member headers: {allowed_ids}. "
                "IDs quoted or linked inside a member are not additional inspected records. Cite its enclosing member "
                "and preserve whether that member is a summary, observation, or private interpretation. "
                "Use at most DIGEST_TOKEN_ALLOWANCE estimated tokens and DIGEST_WORD_ALLOWANCE words. "
                "Return JSON: {\"summary\": \"...\", \"cited_ids\": [1]}.")

        batches = batch_prompt(required_story, groups, budget, instruction)
        headers = "\n\n".join(f"Derived evidence digest; original members {list(batch.source_ids)}:\n" for batch in batches)
        metadata_tokens = budget.estimate([Turn("user", headers)])
        digest_allowance = (maximum_tokens - metadata_tokens) // len(batches)
        if digest_allowance <= 0:
            raise ValueError("Original structure source IDs exceed the remaining attention window.")

        def review(index: int) -> StructureDigest:
            batch = batches[index]
            allowed = set(batch.source_ids)
            turns = list(batch.turns)
            last = turns[-1]
            text = last.content.text.replace("DIGEST_TOKEN_ALLOWANCE", str(digest_allowance))
            text = text.replace("DIGEST_WORD_ALLOWANCE", str(max(1, digest_allowance // 3)))
            turns[-1] = Turn(last.role, text)
            budget.require_fit(turns)
            class CheckedDigest(StructureDigest):
                @model_validator(mode="after")
                def covered_sources(self) -> Self:
                    """Reject invented citations and an oversized derived view."""
                    invalid = set(self.cited_ids) - allowed
                    if invalid or not self.cited_ids:
                        raise ValueError(f"cited_ids must use top-level member IDs only. Invalid IDs: {sorted(invalid)}. "
                                         f"Choose at least one from: {sorted(allowed)}. Linked IDs inside a member are not citation grants.")
                    if budget.estimate([Turn("user", self.summary)]) > digest_allowance:
                        raise ValueError(f"Compress the digest below {digest_allowance} estimated tokens, retaining corrections and source attribution.")
                    return self
            return complete(context, work, turns, CheckedDigest)

        assert context.runtime is not None
        answers = ordered_parallel_map(review, list(range(len(batches))), workers=ATTENTION_WORKERS,
                                       cancel=context.runtime.stop_event)
        reduced: list[ContextGroup] = []
        for index, (batch, answer) in enumerate(zip(batches, answers, strict=True)):
            text = f"Derived evidence digest; original members {list(batch.source_ids)}:\n{answer.summary}"
            reduced.append(ContextGroup(f"digest:{level}:{index}", Content(text), structure.salience,
                                        source_ids=batch.source_ids))
        covered = {key for group in reduced for key in group.source_ids}
        if covered != original_ids:
            raise AssertionError("Structure review lost original evidence coverage.")
        coverage = StructureReviewCoverage(structure_id=structure.record_id, level=level,
            member_ids=tuple(sorted(original_ids)), batch_source_ids=tuple(batch.source_ids for batch in batches))
        context.runtime.environment.write_text(f"scratch/structure-{abs(structure.record_id)}-coverage-{level}.json",
                                               coverage.model_dump_json())
        result = "\n\n".join(group.content.text for group in reduced)
        if budget.estimate([Turn("user", result)]) <= maximum_tokens:
            return result
        before = budget.estimate([Turn("user", "\n\n".join(group.content.text for group in groups))])
        after = budget.estimate([Turn("user", result)])
        if after >= before:
            raise ValueError("Structure evidence digests did not reduce the prompt; increase the cognition window or correct the digest instruction.")
        groups = reduced
        level += 1
