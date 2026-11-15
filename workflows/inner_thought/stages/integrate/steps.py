"""Private integration role.

Entry point: ``run(context)``. Reads the performed thought and proposes its
accepted consequences as a validated ``IntegrationResult``. The message
builder is pure and tested without a kernel, model, or MemoryStore.

A request that carries ``focus_advice`` ends deliberate mental work. Only then
does the prompt offer the optional ``focus`` field. A request that carries
``perspective_revision`` ends a perspective inquiry; only then does the prompt
offer the optional entity-model revision. Spontaneous integrations see neither.
See ``workflows/inner_thought/host.py`` for the loop overview.
"""
from __future__ import annotations

from pathlib import Path
from workflows.inner_thought.rendering import candidates as render_candidates, sources as render_sources
import json

from pydantic import BaseModel, ConfigDict, Field

from pm.constants import Sampler
from pm_workflows.python_role import RoleContext
from pm.context_render.turns import Role, Turn, format_content

from workflows.inner_thought.perspective import PerspectiveRevisionRequest
from workflows.inner_thought.schemas import (
    FocusAdviceRequest, GraphOutcomeRules, InnerRequest, IntegrationResult, ThoughtResult, TriggerChoice,
)
from workflows.inner_thought.pipeline import (
    citable_ids, validate_graph_outcome_rules, validate_integration,
)
from workflows.main_cognition.model_json import complete_json, load_yaml, read_request, write_report

WORKFLOW_DIR = Path(__file__).resolve().parent


class IntegrationPrompts(BaseModel):
    """The prompt texts of the integrate stage, read from ``prompts.yaml``."""

    model_config = ConfigDict(extra="forbid")
    system_prompt: str = Field(description="System turn. It states the rules and the JSON schema of every integration.")
    user_template: str = Field(description="User turn with the performed thought, its evidence, goals, and expectations.")
    focus_advice_template: str = Field(description="Appended to the user turn only when the thought ends deliberate mental work.")
    perspective_revision_template: str = Field(
        description="Appended to the user turn only at the outcome of a perspective inquiry.")


def load_prompts(path: Path | None = None) -> IntegrationPrompts:
    """Read and validate the stage prompts. A missing or unknown key raises."""
    return IntegrationPrompts.model_validate(load_yaml(path or WORKFLOW_DIR / "prompts.yaml"))


def build_messages(request: dict[str, object], prompts: IntegrationPrompts) -> list[Turn]:
    """Build typed turns for one integration decision."""
    parsed = InnerRequest.model_validate(request)
    trigger = TriggerChoice.model_validate(request.get("trigger") or {})
    thought = ThoughtResult.model_validate(request.get("thought") or {})
    operation = (request.get("operation") or {}).get("name", "")
    system = prompts.system_prompt.format(companion_name=parsed.companion_name)
    user = format_content(prompts.user_template,
        companion_name=parsed.companion_name,
        state_summary=parsed.state_summary or "(no recorded state update)",
        reality=thought.reality,
        operation=operation or trigger.operation,
        thought=thought.thought or "(empty)",
        evidence_summary=thought.evidence_summary or "(none)",
        interpretation=thought.interpretation or "(none)",
        goals=render_sources(parsed.active_goal_records, "goal"),
        expectations=render_sources(parsed.pending_expectations, "expectation"),
        evidence=render_candidates(parsed.candidates),
        evidence_ids=json.dumps(thought.evidence_ids),
    )
    user = user + "\nRecent private episode for progress comparison:\n" + parsed.episode.model_dump_json()
    user = user + "\nRecent private thoughts:\n" + render_sources(parsed.recent_thoughts[-6:], "private_thought")
    advice = request.get("focus_advice")
    if advice is not None:
        # Only deliberate mental work is asked for focus advice, so the prompt
        # of every spontaneous integration stays exactly as it was.
        focus_advice = FocusAdviceRequest.model_validate(advice)
        user = user + "\n\n" + prompts.focus_advice_template.format(
            companion_name=parsed.companion_name, instruction=focus_advice.instruction,
            citable_ids=json.dumps(list(focus_advice.citable_ids))).strip()
    revision = request.get("perspective_revision")
    if revision is not None:
        # Only a perspective inquiry is asked for a model revision, so no other
        # integration prompt changes.
        perspective = PerspectiveRevisionRequest.model_validate(revision)
        existing = json.dumps([{"id": item.claim_id, "key": item.claim_key,
                                "statement": item.statement, "source_ids": list(item.source_ids)}
                               for item in perspective.existing_claims], ensure_ascii=False)
        user = user + "\n\n" + prompts.perspective_revision_template.format(
            subject=perspective.subject_name,
            citable_source_ids=json.dumps(list(perspective.citable_source_ids)),
            existing_claims=existing).strip()
    return [Turn(Role.System, system.strip()), Turn(Role.User, user.strip())]


def run(context: RoleContext) -> dict[str, str]:
    """Integrate the chosen trigger and thought into the private report."""
    request = read_request(context.runtime.environment)
    prompts = load_prompts()
    messages = build_messages(request, prompts)
    parsed = InnerRequest.model_validate(request)
    known = citable_ids(parsed)
    focus_requested = request.get("focus_advice") is not None
    perspective: PerspectiveRevisionRequest | None = None
    revision_data = request.get("perspective_revision")
    if revision_data is not None:
        perspective = PerspectiveRevisionRequest.model_validate(revision_data)
    source_texts = {int(item.knoxel_id): item.text for item in parsed.candidates}
    # The graph path supplies the facts its store owner checks; the linear
    # pipeline has another store owner and sends none.
    rules_data = request.get("graph_outcome_rules")
    rules = GraphOutcomeRules.model_validate(rules_data) if rules_data is not None else None

    def validate(payload: dict[str, object], raw: str) -> IntegrationResult:
        """Apply the evidence rules inside the correction loop; a broken rule is asked again."""
        proposal = IntegrationResult.model_validate(payload)
        validate_integration(proposal, known, parsed, focus_requested=focus_requested,
                             perspective=perspective,
                             source_texts=source_texts if perspective is not None else None)
        if rules is not None:
            validate_graph_outcome_rules(proposal, rules, known)
        return proposal

    # IntegrationError is a ValueError, so complete_json keeps the rejected
    # answer, sends the rule's message, and asks again without an attempt cap.
    # Only RepeatedRejectionError (one identical problem through two complete
    # conversations) stops it, because then the message itself is defective.
    result = complete_json(context, messages, IntegrationResult, temperature=Sampler.TempInstruct,
                           enable_thinking=False, validate=validate)
    write_report(context.runtime.environment, result)
    return {
        "status": "done",
        "summary": result.summary or "integrated the private thought",
    }


__all__ = ["build_messages", "load_prompts", "run"]
