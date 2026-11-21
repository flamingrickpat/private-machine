"""Run the existing contextual triage mechanism without the full cycle.

The host owns durable assessment acceptance; this role only returns a proposal.
The same fitted call can also extract explicit workflow requests and entity claims.
"""
from __future__ import annotations

from pm_workflows.python_role import RoleContext
from pathlib import Path
from pydantic import BaseModel, ValidationError

from pm.utils import yaml_util
from pm.context_render.turns import Role, Turn
from pm.llm.structured import ModelFormatError
from workflows.input_triage.schemas import EarlyTriageAssessment, InputTriageAnswer, InputTriageRequest
from workflows.input_triage.entity_updates import hydrate_entity_extraction, render_extraction_context
from pm.runs.role_io import write_report
from workflows.main_cognition.phases import intake, triage
from workflows.main_cognition.phases.model import complete
from workflows.main_cognition.schemas import TriageResult, TriageCheck


class EarlyTriagePrompts(BaseModel):
    """Instructions specific to early triage."""

    workflow_requests: str
    extraction: str


def run(context: RoleContext) -> dict[str, str]:
    """Classify every supplied source with the shared triage prompt."""
    assert context.runtime is not None, "Early triage requires an active workflow runtime."
    environment = context.runtime.environment
    request = InputTriageRequest.model_validate(
        yaml_util.safe_load(environment.read_text("inbox/request.yaml")))
    work = intake.start(request.cycle)
    entity_proposals = None
    prompts = EarlyTriagePrompts.model_validate(yaml_util.safe_load(
        Path(__file__).with_name("prompts.yaml").read_text(encoding="utf-8")))
    appended: list[Turn] = []
    if request.direct_workflows:
        options = "\n".join(f"- {option.name}: {option.description}" for option in request.direct_workflows)
        appended.append(Turn(Role.User, prompts.workflow_requests + "\nAvailable workflows\n" + options))
    if request.entity_extraction_input is not None:
        extraction = render_extraction_context(request.entity_extraction_input) + "\n\n" + prompts.extraction
        appended.append(Turn(Role.User, extraction))
    prompt = triage.fit_messages(context, work, appended=appended)
    work.receipt("triage-context", prompt.summary())
    validation = TriageCheck(
        stimulus_ids=tuple(item.knoxel_id for item in request.cycle.stimuli),
        shown_evidence_ids=frozenset(prompt.selection.source_ids) | {item.knoxel_id for item in request.cycle.stimuli},
        workflow_task_names=frozenset(option.name for option in request.direct_workflows),
        original_inputs={item.knoxel_id: item.text for item in request.cycle.stimuli},  # pm-lint: allow PM005 -- original utterances indexed by source knoxel ID
    )
    if request.entity_extraction_input is None:
        result = complete(context, prompt.turns, TriageResult, validation_context=validation)
    else:
        answer = complete(context, prompt.turns, InputTriageAnswer, validation_context=validation)
        result = TriageResult(items=answer.items, codelet_budget=answer.codelet_budget)
        try:
            entity_proposals = hydrate_entity_extraction(request.entity_extraction_input, answer.entity_extraction,
                                                         producer="immediate")
        except ValidationError as error:
            raise ModelFormatError(f"Entity extraction has an invalid semantic contract: {error}") from error
    assessment = EarlyTriageAssessment(
        source_ids=[item.knoxel_id for item in request.cycle.stimuli],
        source_fingerprints=request.source_fingerprints,
        guidance_tick=request.guidance_tick,
        result=result,
        evidence=request.evidence,
        entity_proposals=entity_proposals,
    )
    write_report(environment, assessment)
    return {"status": "done", "summary": f"classified {len(assessment.source_ids)} partner inputs"}
