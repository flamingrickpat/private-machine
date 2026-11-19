"""Project what the companion did since its last accepted review, under which goals.

The canonical overview is :mod:`workflows.main_cognition.host`. This module
owns one read-only projection that each main cycle builds from its frozen
request view and that the appraise and goals phases render (B07 decision D4,
milestone section 16). It writes no record and calls no model.

**What it models.** A person who takes stock looks back at what they did since
they last took stock, whether it arrived, and what they were trying to do when
they did it. Here that is: every action committed since the last accepted
review, its delivery status (the shared
:func:`~workflows.public_speech.situational_guidance.recent_move` projection),
the exact intention versions that held when it was chosen, how those goals
stand now, the expectations the action opened, and every ordinary exchange in
which the fast path set accepted guidance aside, with its reason.

**Frontier.** :attr:`pm.records.cycle_state.CycleState.interval_reviewed_through`
holds the highest record id the last accepted main cycle's review read. The
next review reads records with a higher id, up to the highest id of its own
request view. Record ids only grow, so an action committed while a cycle
calculates is read by the next cycle, and no action is reviewed twice. A cycle
that is not accepted leaves the frontier unchanged, so its interval is
reviewed again by the next accepted cycle. A state without a frontier (before
B07) reviews nothing and starts the frontier at its view's highest id; it does
not review the whole history.

**Attribution.** An action records ``intention_ids``: the exact intention
versions that were active when it was committed. A goal changes by appending
a successor with a new id, so the old version keeps its text. The review shows
each action under those versions, never under a later one, and states how the
goal's family head stands now (:func:`goal_family_heads`).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pm.memory.snapshot import Snapshot
from pm.records.cognitive_records import ExpectationRecord, Intention
from pm.records.memory_records import ActionSchemaRecord, Feature
from workflows.public_speech.situational_guidance import GuidanceDisposition, RecentMove, recent_move


class GoverningIntention(BaseModel):
    """One intention version that held when an action was chosen, and how its goal stands now."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    intention_id: int = Field(description="The exact historical intention version the action recorded.")
    text: str = Field(description="The goal as it read when the action was chosen.")
    later_state: Literal["current", "changed", "completed", "deferred", "refused"] = Field(
        description="current: this version is still the active head. changed: the goal is still active under a "
                    "newer version. completed, deferred, refused: the goal's head now has that status.")
    current_id: int | None = Field(description="The active head of the same goal now, or None when it is no longer active.")
    current_text: str | None = Field(
        description="The goal's current text when the active head reads differently; None otherwise.")


class IntervalMove(BaseModel):
    """One action committed since the last accepted review."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    origin: Literal["main", "interaction", "task_reply"] = Field(
        description="main: chosen by main cognition. interaction: chosen by the ordinary fast path. "
                    "task_reply: the answer to a finished task.")
    move: RecentMove = Field(description="The action, its utterance, and its delivery status at review time.")
    governing: list[GoverningIntention] = Field(description="The intention versions that held when it was chosen.")
    expectation_ids: list[int] = Field(description="Current records of the expectations this action opened.")


class GuidanceSuspension(BaseModel):
    """One ordinary exchange in which the fast path set accepted guidance aside."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    receipt_id: int = Field(description="The interaction receipt record.")
    disposition: GuidanceDisposition = Field(description="decline or suspend; never follow.")
    reason: str = Field(description="Why the exchange set the guidance aside, in the fast path's words.")
    source_ids: list[int] = Field(description="The inputs of that exchange.")


class InhibitedMove(BaseModel):
    """An undelivered local proposal; distinct from both silence and an executed action."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    receipt_id: int
    source_ids: list[int]
    proposals: list[str]
    reason: str


class IntervalReview(BaseModel):
    """What the companion did since its last accepted review. Read-only; built for each main cycle."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    after_record_id: int = Field(description="The frontier this review started from: it reads ids above it.")
    through_record_id: int = Field(description="The highest record id this review read; the next frontier.")
    first_review: bool = Field(description="True when no earlier accepted cycle recorded a frontier, so this review "
                                           "starts the frontier and reviews nothing.")
    moves: list[IntervalMove] = Field(description="Actions in the interval, oldest first.")
    suspensions: list[GuidanceSuspension] = Field(description="Set-aside guidance in the interval, oldest first.")
    inhibitions: list[InhibitedMove] = Field(default_factory=list[InhibitedMove],
        description="Proposed ordinary moves that were withheld, with their actual inhibition reasons.")


class _OriginPayload(BaseModel):
    """The origin field of an action's stored dispatch request."""

    model_config = ConfigDict(extra="ignore")
    action_origin: Literal["main", "interaction", "task_reply"]


class _ReviewedActionMetadata(BaseModel):
    """The metadata fields that tell who chose an action."""

    model_config = ConfigDict(extra="ignore")
    action_id: str
    dispatch_payload: _OriginPayload | None = None
    task_run_id: str | None = None


class _GuidanceDecision(BaseModel):
    model_config = ConfigDict(extra="ignore")
    guidance_disposition: GuidanceDisposition
    guidance_reason: str
    variants: list[str]


class _InhibitionJudgment(BaseModel):
    model_config = ConfigDict(extra="ignore")
    allow: bool
    reason: str


class _GuidanceReceipt(BaseModel):
    model_config = ConfigDict(extra="ignore")
    source_ids: list[int]
    decision: _GuidanceDecision
    inhibition: _InhibitionJudgment | None = None


class _GuidanceMetadata(BaseModel):
    """The part of an interaction receipt's metadata the review reads."""

    model_config = ConfigDict(extra="ignore")
    interaction_receipt: _GuidanceReceipt


def goal_family_heads(view: Snapshot) -> dict[int, Intention]:
    """Map every intention version to the current head of its goal.

    A goal changes by appending a successor with ``supersedes_id``. Walking
    that link back from each head intention reaches every earlier version.
    """
    heads: dict[int, Intention] = {}
    for record in view.all_knoxels.values():
        if not isinstance(record, Intention):
            continue
        version: Intention = record
        heads[version.id] = record
        while version.supersedes_id is not None:
            previous_id = version.supersedes_id
            previous = view.get(previous_id)
            if not isinstance(previous, Intention):
                raise TypeError(f"Intention {version.id} supersedes {previous_id}, which is not an intention.")
            version = previous
            heads[version.id] = record
    return heads


def _governing(view: Snapshot, action: ActionSchemaRecord, heads: dict[int, Intention]) -> list[GoverningIntention]:
    """Read the intention versions an action recorded, with how each goal stands now."""
    governing: list[GoverningIntention] = []
    for intention_id in action.intention_ids:
        if intention_id >= action.id:
            raise ValueError(f"Action {action.id} names intention {intention_id}, which did not exist when it was chosen.")
        version = view.get(intention_id)
        if not isinstance(version, Intention):
            raise TypeError(f"Action {action.id} names {intention_id}, which is not an intention.")
        head = heads[intention_id]
        later_state: Literal["current", "changed", "completed", "deferred", "refused"]
        current_id: int | None = None
        current_text: str | None = None
        if head.active:
            later_state = "current" if head.id == intention_id else "changed"
            current_id = head.id
            if head.content != version.content:
                current_text = head.content
        elif head.status == "completed":
            later_state = "completed"
        elif head.status == "deferred":
            later_state = "deferred"
        elif head.status == "refused":
            later_state = "refused"
        else:
            raise ValueError(f"Intention {head.id} is inactive with an unknown status {head.status!r}.")
        governing.append(GoverningIntention(intention_id=intention_id, text=version.content, later_state=later_state,
                                            current_id=current_id, current_text=current_text))
    return governing


def _origin(action: ActionSchemaRecord) -> Literal["main", "interaction", "task_reply"]:
    """Name who chose an action from its stored dispatch request."""
    metadata = _ReviewedActionMetadata.model_validate(action.metadata)
    if metadata.task_run_id is not None or metadata.action_id.startswith("task-rejection:"):
        return "task_reply"
    if metadata.dispatch_payload is None:
        raise ValueError(f"Action {action.id} has no dispatch request that names who chose it.")
    return metadata.dispatch_payload.action_origin


def build_interval_review(view: Snapshot, *, after_record_id: int | None) -> IntervalReview:
    """Build the review of every action and set-aside guidance with an id above ``after_record_id``.

    ``view`` is the main cycle's frozen request view; the review reads up to
    its highest record id. ``after_record_id`` is the previous accepted main
    cycle's ``interval_reviewed_through``; None starts the frontier and
    reviews nothing. Broken records raise: an action without a known origin, a
    governing id that is not an earlier intention, or a reply outside its
    interaction scope.
    """
    through_record_id = max(view.revisions)
    if after_record_id is None:
        return IntervalReview(after_record_id=through_record_id, through_record_id=through_record_id,
                              first_review=True, moves=[], suspensions=[])
    if after_record_id > through_record_id:
        raise ValueError(f"The review frontier {after_record_id} lies beyond this view's newest record.")
    heads = goal_family_heads(view)
    expectations: dict[int, list[int]] = {}
    for record in view.all_knoxels.values():
        if isinstance(record, ExpectationRecord) and record.created_from_action_id is not None:
            expectations.setdefault(record.created_from_action_id, []).append(record.id)
    moves: list[IntervalMove] = []
    suspensions: list[GuidanceSuspension] = []
    inhibitions: list[InhibitedMove] = []
    for record_id in sorted(view.all_knoxels):
        if record_id <= after_record_id:
            continue
        record = view.all_knoxels[record_id]
        if isinstance(record, ActionSchemaRecord):
            moves.append(IntervalMove(origin=_origin(record),
                move=recent_move(view, record, scopes=[record.scope]),
                governing=_governing(view, record, heads),
                expectation_ids=sorted(expectations[record.id]) if record.id in expectations else []))
        elif isinstance(record, Feature) and "interaction_receipt" in record.metadata:
            receipt = _GuidanceMetadata.model_validate(record.metadata).interaction_receipt
            decision = receipt.decision
            if receipt.inhibition is not None and not receipt.inhibition.allow:
                inhibitions.append(InhibitedMove(receipt_id=record.id, source_ids=list(receipt.source_ids),
                    proposals=list(decision.variants), reason=receipt.inhibition.reason))
            if decision.guidance_disposition == GuidanceDisposition.Follow:
                continue
            suspensions.append(GuidanceSuspension(receipt_id=record.id, disposition=decision.guidance_disposition,
                                                  reason=decision.guidance_reason, source_ids=list(receipt.source_ids)))
    return IntervalReview(after_record_id=after_record_id, through_record_id=through_record_id, first_review=False,
                          moves=moves, suspensions=suspensions, inhibitions=inhibitions)


_ORIGIN_WORDS: dict[str, str] = {
    "main": "after thinking it over",
    "interaction": "in the flow of the conversation",
    "task_reply": "to answer a finished task",
}
_STATUS_WORDS: dict[str, str] = {
    "unknown": "it is not known whether it went out",
    "selected": "it was chosen but has not gone out yet",
    "committed": "the words were written but not yet delivered",
    "delivered": "it was delivered",
    "failed": "it failed or was cancelled",
}
_LATER_WORDS: dict[str, str] = {
    "current": "still pursued as it was",
    "changed": "still pursued, since revised",
    "completed": "since completed",
    "deferred": "since put aside",
    "refused": "since refused",
}


def render_interval_review(review: IntervalReview, *, companion_name: str) -> str:
    """Render the review as story lines for the appraise and goals prompts.

    Every line names its record ids, so the model can cite them. The goal text
    is the version that held when the action was chosen.
    """
    if not review.moves and not review.suspensions and not review.inhibitions:
        return f"(nothing: {companion_name} has done nothing since last taking stock)"
    lines: list[str] = []
    for item in review.moves:
        move = item.move
        line = f"- [action {move.action.record_id}] {companion_name} chose {_ORIGIN_WORDS[item.origin]}: {move.action.text}"
        lines.append(line)
        if move.utterance is not None:
            lines.append(f"  said [{move.utterance.record_id}]: {move.utterance.text}")
        lines.append(f"  outcome so far: {_STATUS_WORDS[move.status]}")
        if item.governing:
            for goal in item.governing:
                goal_line = f"  goal then [{goal.intention_id}]: {goal.text} ({_LATER_WORDS[goal.later_state]}"
                if goal.current_text is not None and goal.current_id is not None:
                    goal_line += f"; now [{goal.current_id}]: {goal.current_text}"
                lines.append(goal_line + ")")
        else:
            lines.append("  goal then: none was active")
        if item.expectation_ids:
            lines.append(f"  expectations it opened: {item.expectation_ids}")
    for suspension in review.suspensions:
        lines.append(f"- [exchange {suspension.receipt_id}] in the flow of the conversation, {companion_name} "
                     f"set the accepted direction aside ({suspension.disposition.value}) for inputs "
                     f"{suspension.source_ids}: {suspension.reason}")
    for inhibition in review.inhibitions:
        lines.append(f"- [exchange {inhibition.receipt_id}] a proposed move was withheld, "
                     f"not executed: {'; '.join(inhibition.proposals)}. Reason: {inhibition.reason}. "
                     f"Inputs: {inhibition.source_ids}.")
    return "\n".join(lines)


__all__ = ["GoverningIntention", "GuidanceSuspension", "IntervalMove", "IntervalReview", "build_interval_review",
           "goal_family_heads", "render_interval_review"]
