"""Pure orchestration of the private thought cycle.

``run_cycle`` owns no prompts, no model calls, and no memory writes. It calls
the injected ``run_stage`` for each stage, validates every output into its
typed schema, and enforces the evidence rules:

- a cycle with no deterministic trigger commits nothing and calls no model;
- the chosen operation must be a delivered operation;
- every cited id must be one of the ids the host supplied with a revision;
- a thought keeps its evidence summary separate from its interpretation, and
  carries a reality label that never turns imagination into observation.

The parent report is data for the host. The host owns the commit boundary.
"""
from __future__ import annotations

from workflows.inner_thought.operations import OPERATIONS, validate_operation
from workflows.inner_thought.schemas import (
    InnerReport,
    InnerRequest,
    IntegrationResult,
    REALITY_LABELS,
    ThoughtResult,
    TriggerChoice,
)

STAGE_ORDER = ("trigger", "thought", "integrate")

#: The continuation choices a thought may carry. Only ``continue`` keeps an
#: episode open; it must name the next operation.
CONTINUATIONS = ("continue", "summarize", "suspend")

#: The progress kinds an integration may report. ``none`` is always valid.
PROGRESS_KINDS = ("new_evidence", "new_option", "artifact", "decision", "none")


class ModelProposalError(ValueError):
    """A model proposal violates an explicit semantic evidence rule.

    The host discards this cycle without a commit. Schema and Python errors
    remain separate and propagate to the application.
    """


class IntegrationError(ModelProposalError):
    """A model proposal violates the expectation evidence contract."""


def citable_ids(request: InnerRequest) -> set[int]:
    """Return every id the model may cite, all read with a revision."""
    known: set[int] = set()
    known.update(int(item.knoxel_id) for item in request.candidates)
    known.update(int(item["knoxel_id"]) for item in request.active_goal_records)
    known.update(int(item["knoxel_id"]) for item in request.pending_expectations)
    known.update(int(item["knoxel_id"]) for item in request.recent_thoughts)
    if request.main_frame.get("frame_id") is not None:
        known.add(int(request.main_frame["frame_id"]))
    return known


def run_cycle(request: InnerRequest, run_stage, retrieve_memory=None) -> InnerReport:
    """Run the three private stages and return the typed parent report."""
    # The trigger policy already decided this cycle has nothing to do. The
    # loop must not generate a thought to satisfy a quota.
    if not request.trigger_candidates:
        return InnerReport(
            run_id=request.run_id, operation="none",
            summary="no trigger fired; the private cycle stays idle",
        )

    base = request.model_dump(mode="json")
    trigger = TriggerChoice.model_validate(run_stage("trigger", dict(base)))
    known = citable_ids(request)
    if trigger.operation == "none" or not trigger.operation:
        return InnerReport(
            run_id=request.run_id, operation="none", trigger=trigger.trigger,
            trigger_reason=trigger.reason,
            summary="the trigger stage chose no operation",
        )
    operation = validate_operation(trigger.operation)
    retrieved = []
    if trigger.retrieval_query.strip():
        if retrieve_memory is None:
            raise ModelProposalError("Private recall requested an unavailable retrieval tool.")
        from workflows.inner_thought.schemas import ThoughtCandidate
        retrieved = retrieve_memory(trigger.retrieval_query)["matches"]
        existing = {item.knoxel_id for item in request.candidates}
        for item in retrieved:
            if int(item["knoxel_id"]) not in existing:
                request.candidates.append(ThoughtCandidate(knoxel_id=item["knoxel_id"],
                    text=item["text"], revision=item["revision"], kind="recalled_memory",
                    reason="deliberate private recall"))
                existing.add(int(item["knoxel_id"]))
        known = citable_ids(request)
        base = request.model_dump(mode="json")
    narrowed = [int(item) for item in trigger.candidate_ids if int(item) in known] or [
        int(item.knoxel_id) for item in request.candidates]
    trigger = trigger.model_copy(update={"candidate_ids": narrowed})

    thought = ThoughtResult.model_validate(run_stage("thought", {
        **base, "trigger": trigger.model_dump(mode="json"),
        "operation": {"name": operation.name, "group": operation.group, "mode": operation.mode,
                      "temporal_focus": operation.temporal_focus, "description": operation.description},
    }))
    if thought.reality not in REALITY_LABELS:
        raise ModelProposalError(f"the thought stage returned an unknown reality label: {thought.reality}")
    invalid = [int(item) for item in thought.evidence_ids if int(item) not in known]
    if invalid:
        raise ModelProposalError(f"the thought stage cited ids outside the frozen view: {invalid}")
    if thought.continuation not in CONTINUATIONS:
        raise ModelProposalError(f"the thought stage returned an unknown continuation: {thought.continuation}")
    if thought.next_operation.strip():
        try:
            validate_operation(thought.next_operation.strip())
        except KeyError:
            raise ModelProposalError(
                f"the thought stage named an unknown next operation: {thought.next_operation}")
    if thought.continuation == "continue" and not thought.next_operation.strip():
        raise ModelProposalError("a continuing thought must name its next operation")
    focus = operation.temporal_focus if trigger.temporal_focus not in {"past", "present", "future"} else trigger.temporal_focus

    integration = IntegrationResult.model_validate(run_stage("integrate", {
        **base, "trigger": trigger.model_dump(mode="json"), "thought": thought.model_dump(mode="json"),
        "operation": {"name": operation.name, "group": operation.group},
    }))
    validate_integration(integration, known, request)
    progress, progress_reason = concrete_progress(integration, thought, request)

    return InnerReport(
        run_id=request.run_id,
        operation=operation.name,
        trigger=trigger.trigger,
        trigger_reason=trigger.reason,
        temporal_focus=focus,
        candidate_ids=trigger.candidate_ids,
        thought=thought.thought,
        evidence_ids=[int(item) for item in thought.evidence_ids],
        evidence_summary=thought.evidence_summary,
        interpretation=thought.interpretation,
        uncertainty=thought.uncertainty,
        reality=thought.reality,
        alternatives=thought.alternatives,
        continuation=thought.continuation,
        continuation_reason=thought.continuation_reason,
        goal_changes=integration.goal_changes,
        expectation=integration.expectation,
        noticing=integration.noticing.strip(),
        disclosure=integration.disclosure,
        channels=request.channels,
        summary=integration.summary or thought.thought[:120],
        next_operation=thought.next_operation,
        topic_hints=thought.topic_hints,
        progress=progress,
        progress_reason=progress_reason,
        retrieved_memories=retrieved,
    )


def concrete_progress(integration: IntegrationResult, thought: ThoughtResult,
                      request: InnerRequest) -> tuple[str, str]:
    """Return progress only when it names a concrete difference from the episode.

    A model's self-rating alone cannot establish work. ``new_evidence`` needs a
    cited id not already in the episode frontier, ``new_option`` needs an unseen
    alternative text, ``decision`` needs a goal change, and ``artifact`` needs a
    thought text distinct from the episode's earlier thoughts. Otherwise the
    kind downgrades to ``none``. Rephrasing is not progress.
    """
    kind = integration.progress
    reason = integration.progress_reason
    if kind == "none":
        return "none", reason
    episode = request.episode or {}
    if kind == "new_evidence":
        seen = {int(item) for item in episode.get("evidence_ids", [])}
        if any(int(item) not in seen for item in thought.evidence_ids):
            return kind, reason
    elif kind == "new_option":
        seen = set(episode.get("alternatives", []))
        if any(item.text and item.text not in seen for item in thought.alternatives):
            return kind, reason
    elif kind == "decision":
        if integration.goal_changes:
            return kind, reason
    elif kind == "artifact":
        seen = set(episode.get("thoughts", []))
        if thought.thought.strip() and thought.thought not in seen:
            return kind, reason
    return "none", (reason + " (no concrete difference from earlier episode work)").strip()


def validate_integration(integration: IntegrationResult, known: set[int], request: InnerRequest) -> None:
    """Reject unsupported expectation proposals with IntegrationError. Accept no state."""
    if integration.progress not in PROGRESS_KINDS:
        raise IntegrationError(f"unknown progress kind: {integration.progress}")
    update = integration.expectation
    if update.action == "open":
        if not (update.anticipation.strip() or update.expected_outcome.strip()):
            raise IntegrationError("an open expectation needs an anticipation or an expected outcome")
    elif update.action == "resolve":
        pending = {int(item["knoxel_id"]) for item in request.pending_expectations}
        if update.expectation_id is None or int(update.expectation_id) not in pending:
            raise IntegrationError("a resolution must name a pending expectation from the frozen view")
        invalid = [int(item) for item in update.observation_ids if int(item) not in known]
        if invalid:
            raise IntegrationError(f"a resolution cited ids outside the frozen view: {invalid}")
        if not update.observation_ids:
            raise IntegrationError("a resolution without observations would invent an outcome")
        if update.status not in {"supported", "contradicted", "unmet"}:
            raise IntegrationError(f"unknown resolution status: {update.status}")


__all__ = ["STAGE_ORDER", "CONTINUATIONS", "PROGRESS_KINDS", "ModelProposalError", "IntegrationError",
           "citable_ids", "concrete_progress", "run_cycle", "validate_integration"]
