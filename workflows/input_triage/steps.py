"""Run the existing contextual triage mechanism without the full cycle.

The host owns durable assessment acceptance; this role only returns a proposal.
"""
from __future__ import annotations

from pm_workflows.python_role import RoleContext

from pm.utils import yaml_util
from workflows.input_triage.schemas import EarlyTriageAssessment, InputTriageRequest
from workflows.main_cognition.model_json import write_report
from workflows.main_cognition.phases import intake, triage


def run(context: RoleContext) -> dict[str, str]:
    """Classify every supplied source with the shared triage prompt."""
    assert context.runtime is not None, "Early triage requires an active workflow runtime."
    environment = context.runtime.environment
    request = InputTriageRequest.model_validate(
        yaml_util.safe_load(environment.read_text("inbox/request.yaml")))
    work = triage.run(context, intake.start(request.cycle))
    assessment = EarlyTriageAssessment(
        source_ids=[item.knoxel_id for item in request.cycle.stimuli],
        source_fingerprints=request.source_fingerprints,
        guidance_tick=request.guidance_tick,
        result=work.triage,
        evidence=request.evidence,
    )
    write_report(environment, assessment)
    return {"status": "done", "summary": f"classified {len(assessment.source_ids)} partner inputs"}
