"""Private thought role.

Entry point: ``run(context)``. Performs the chosen operation over the frozen
private candidates and writes the validated ``ThoughtResult``. The message
builder is pure and tested without a kernel, model, or MemoryStore.
"""
from __future__ import annotations
from workflows.main_cognition.context_reduction import evidence_text

from pathlib import Path
import json
from workflows.inner_thought.rendering import candidates as render_candidates, inspection as render_inspection, sources as render_sources
from workflows.inner_thought.rendering import request_content

from pm.constants import Sampler
from pm_workflows.python_role import RoleContext
from pm.context_render.turns import Content, Role, Turn, format_content

from workflows.inner_thought.operations import validate_operation
from workflows.inner_thought.schemas import InnerRequest, ThoughtResult, TriggerChoice
from workflows.main_cognition.model_json import complete_json, load_yaml, read_request, write_report

WORKFLOW_DIR = Path(__file__).resolve().parent


def load_prompts(path: Path | None = None) -> dict:
    return load_yaml(path or WORKFLOW_DIR / "prompts.yaml")


def _render_candidates(request: InnerRequest, chosen: set[int]) -> Content:
    return (render_candidates(request.candidates, frozenset(chosen)) + "\n"
        + render_sources(request.active_goal_records, "goal") + "\n"
        + render_sources(request.pending_expectations, "expectation"))


def build_messages(request: dict, prompts: dict) -> list[Turn]:
    """Build typed turns for one thought operation."""
    parsed = InnerRequest.model_validate(request)
    trigger = TriggerChoice.model_validate(request.get("trigger") or {})
    operation = validate_operation(str(request.get("operation", {}).get("name", trigger.operation or "none")))
    system = prompts["system_prompt"].format(
        companion_name=parsed.companion_name,
        operation_name=operation.name, operation_group=operation.group,
        operation_mode=operation.mode, operation_focus=operation.temporal_focus,
        operation_description=operation.description,
    )
    chosen = {int(item) for item in trigger.candidate_ids}
    user = format_content(prompts["user_template"],
        companion_name=parsed.companion_name,
        state_summary=parsed.state_summary or "(no recorded state update)",
        trigger_reason=trigger.reason or parsed.trigger_candidates[0].cause if parsed.trigger_candidates else "(no trigger)",
        candidates=_render_candidates(parsed, chosen),
        goals=evidence_text("\n".join(f"- {item.text}" for item in parsed.active_goal_records) or "(none)", "thought-goals"),
        expectations=evidence_text("\n".join(f"- {item.text}" for item in parsed.pending_expectations) or "(none)", "thought-expectations"),
        thoughts=evidence_text("\n".join(f"- {item.text!r}" for item in parsed.recent_thoughts[-6:]) or "(none)", "thought-history"),
        inspection=evidence_text(render_inspection(parsed.inspection), "thought-inspection"),
    )
    user = user + "\nPrivate episode:\n" + evidence_text(parsed.episode.model_dump_json(), "private-episode")
    # The next operation must come from the invocation's catalogue. The
    # selected operation alone gives no names for a valid continuation.
    user = user + "\nAvailable next operations:\n" + "\n".join(item.name for item in parsed.operations)
    return [Turn(Role.System, system.strip()), Turn(Role.User, user.strip())]


def run(context: RoleContext) -> dict[str, str]:
    """Produce the private thought for the accepted trigger."""
    request = read_request(context.runtime.environment)
    if request.get("stage_kind") == "inquiry_operation":
        from workflows.inner_thought.graph_execution import GraphThoughtRequest, InquiryPromptGrant, graph_thought_instruction, validate_graph_step
        from workflows.inner_thought.schemas import InquiryStep
        parsed_graph = GraphThoughtRequest.model_validate(request)
        prompt_grant = InquiryPromptGrant(observer_entity_id=parsed_graph.grant.observer_entity_id,
            scope=parsed_graph.grant.scope,
            permitted_evidence_ids=tuple(source.record_id for source in parsed_graph.grant.sources))
        # Full grants remain in the host for validation. The model needs the
        # permitted citation IDs, not repeated persistence fingerprints.
        # Evidence rendered by the system instruction must not be echoed here.
        graph_messages = [Turn(Role.System, graph_thought_instruction(parsed_graph)),
                          Turn(Role.User, parsed_graph.model_dump_json(
                                   exclude={"request": True, "reflection_evidence": True,
                                            "inspection_evidence": True, "perspective_evidence": True,
                                            "activity_evidence": True, "original_source_ids": True,
                                            "grant": True, "state": {"grant"}}) +
                               "\nCitation authority:\n" + prompt_grant.model_dump_json() +
                               "\nFrozen request:\n" + request_content(parsed_graph.request,
                                   frozenset(parsed_graph.state.root_source_ids)) +
                               "\nReturn only an object conforming to this output schema. "
                               "Do not echo the input envelope or add run_id/stage_kind:\n" +
                               json.dumps(InquiryStep.model_json_schema(), ensure_ascii=False))]
        def validate(payload: dict[str, object], raw: str) -> InquiryStep:
            """Keep the original admitted graph during structured-output correction."""
            return validate_graph_step(parsed_graph, InquiryStep.model_validate(payload))
        graph_result = complete_json(context, graph_messages, InquiryStep, temperature=Sampler.TempReasoning, validate=validate)
        write_report(context.runtime.environment, graph_result)
        return {"status": "done", "summary": graph_result.reason}
    prompts = load_prompts()
    messages = build_messages(request, prompts)
    result = complete_json(context, messages, ThoughtResult, temperature=Sampler.TempReasoning)
    write_report(context.runtime.environment, result)
    return {
        "status": "done",
        "summary": f"performed thought with reality {result.reality}",
    }


__all__ = ["build_messages", "load_prompts", "run"]
