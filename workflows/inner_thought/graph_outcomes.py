"""Apply retained inquiry integration at the derived inner-turn boundary.

See :mod:`workflows.inner_thought.host` for the private loop overview. The
graph operation stores its typed integration and original source guards before
the recovery-root commit. A separate GWT projection applies expectation effects
from that immutable payload. Repair does not run inference or refresh sources.
Main appraisal still owns acceptance of proposed goals and mental-state changes.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from pm.memory.dependencies import SourceDependency
from pm.memory.evidence import evidence_role
from pm.memory.snapshot import Snapshot
from pm.memory.write_batch import WriteBatch
from pm.records.cognitive_records import ExpectationRecord, Intention
from pm.records.memory_records import Feature
from workflows.inner_thought.expectations import open_expectation, resolve_expectation
from workflows.inner_thought.schemas import InquiryGrant, InquiryStep, IntegrationResult
from workflows.main_cognition.schemas import GoalChange


class GraphOutcomePayload(BaseModel):
    """Accepted model integration with the exact sources read before generation."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    integration: IntegrationResult
    sources: tuple[SourceDependency, ...]


class GraphMentalProposal(BaseModel):
    """The existing main-appraisal proposal, with named goal-change fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    instruction: str
    summary: str
    disposition: Literal["propose"] = "propose"
    evidence_ids: tuple[int, ...]
    goal_changes: tuple[GoalChange, ...]
    focus: None = None


class GraphOutcomeMetadata(BaseModel):
    """Read the graph's persisted integration from heterogeneous metadata."""

    model_config = ConfigDict(extra="ignore")
    graph_outcome: GraphOutcomePayload


def capture_graph_outcome(grant: InquiryGrant, step: InquiryStep,
                          integration: IntegrationResult) -> GraphOutcomePayload:
    """Copy source guards from the original grant; never capture current data."""
    if integration.disclosure.mode not in {"candid", "withhold"}:
        raise ValueError("Private outcome disclosure must be candid or withhold.")
    update = integration.expectation
    if update.action not in {"none", "open", "resolve"}:
        raise ValueError("An inquiry outcome returned an unknown expectation action.")
    supplied = {source.record_id: source for source in grant.sources}
    selected_ids = set(step.evidence_ids)
    selected_ids.update(update.goal_ids)
    selected_ids.update(update.observation_ids)
    if update.action == "open" and not (update.anticipation.strip() or update.expected_outcome.strip()):
        raise ValueError("An expectation needs its actual anticipation or expected outcome.")
    if update.action == "resolve":
        if update.expectation_id is None or not update.observation_ids:
            raise ValueError("An expectation resolution requires an exact target and cited observations.")
        if update.status not in {"supported", "contradicted", "unmet"}:
            raise ValueError("An expectation resolution returned an unsupported status.")
        selected_ids.add(update.expectation_id)
    invalid = selected_ids - supplied.keys()
    if invalid:
        raise ValueError(f"An inquiry outcome cites sources outside its original grant: {sorted(invalid)}")
    if update.action == "resolve":
        assert update.expectation_id is not None
        if not supplied[update.expectation_id].current_head:
            raise ValueError("An expectation resolution requires its captured current expectation authority.")
    return GraphOutcomePayload(integration=integration,
        sources=tuple(supplied[source_id] for source_id in sorted(selected_ids)))


def attach_graph_outcome(operation: Feature, step: InquiryStep, outcome: GraphOutcomePayload) -> None:
    """Attach integration to a newly staged operation before graph acceptance.

    The caller revises this still-uncommitted record in its graph batch. A
    projection never calls this function on an already accepted operation.
    """
    integration = outcome.integration
    proposal = GraphMentalProposal(instruction=integration.noticing or step.reason,
        summary=integration.summary or step.thought, evidence_ids=step.evidence_ids,
        goal_changes=tuple(integration.goal_changes))
    operation.metadata.update({"graph_outcome": outcome.model_dump(mode="json"),
        "summary": proposal.summary, "disclosure": integration.disclosure.model_dump(mode="json"),
        "mental_plan": proposal.model_dump(mode="json")})


def read_graph_outcome(operation: Feature) -> GraphOutcomePayload | None:
    """Read optional integration; intermediate graph nodes have no such payload."""
    if "graph_outcome" not in operation.metadata:
        return None
    return GraphOutcomeMetadata.model_validate(operation.metadata).graph_outcome


def stage_graph_outcome(batch: WriteBatch, view: Snapshot, operation: Feature, step: InquiryStep,
                        outcome: GraphOutcomePayload) -> None:
    """Stage expectation effects using immutable integration and original guards.

    The caller commits these effects with the derived GWT turn. A failed guard
    leaves the accepted graph intact. The helper mutates no accepted graph
    record and sends no finding or external action.
    """
    if read_graph_outcome(operation) != outcome:
        raise ValueError("Projection must use the operation's exact retained integration payload.")
    batch.depend_on([operation.id])
    sources = {source.record_id: source for source in outcome.sources}
    for source in outcome.sources:
        batch.expect_source(source)
    update = outcome.integration.expectation
    if update.action == "none":
        return
    for goal_id in update.goal_ids:
        if not isinstance(view.get(goal_id), Intention):
            raise ValueError("An expectation goal must identify an actual supplied intention.")
    if update.action == "open":
        expectation = open_expectation(batch, anticipation=update.anticipation,
            expected_outcome=update.expected_outcome, expected_valence=update.expected_valence,
            uncertainty=update.uncertainty, goal_ids=update.goal_ids,
            evidence_ids=list(step.evidence_ids) or [operation.id])
        expectation.scope = operation.scope
        batch.revise(expectation)
        return
    if update.action != "resolve" or update.expectation_id is None:
        raise ValueError("An accepted graph outcome has an invalid expectation action.")
    target = view.get(update.expectation_id)
    if not isinstance(target, ExpectationRecord) or target.scope != operation.scope or target.status != "pending":
        raise ValueError("An expectation resolution must use its original pending target in the same scope.")
    source = sources[target.id]
    if not source.current_head:
        raise ValueError("An expectation resolution requires original current-authority evidence.")
    for observation_id in update.observation_ids:
        observed = view.get(observation_id)
        if (observed.scope != operation.scope or evidence_role(view, observed) not in {
                "user_report", "companion_utterance", "world_observation", "runtime_event", "internal_state"}):
            raise ValueError("Generated or imagined content cannot resolve an expectation as an observed outcome.")
    resolve_expectation(batch, target, status=update.status, observation_ids=update.observation_ids,
        reason=update.reason, revision=source.revision)
