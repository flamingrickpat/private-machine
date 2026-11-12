"""Run the existing contextual triage mechanism without the full cycle.

The host owns durable assessment acceptance; this role only returns a proposal.
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
from workflows.main_cognition.model_json import write_report
from workflows.main_cognition.phases import intake, triage
from workflows.main_cognition.phases.model import complete, phase_prompts
from workflows.main_cognition.schemas import TriageResult
from workflows.main_cognition.story_prefix import StoryPrefix


class ExtractionPrompts(BaseModel):
    """The model's semantic maintenance instructions, without host provenance fields."""

    extraction: str


def run(context: RoleContext) -> dict[str, str]:
    """Classify every supplied source with the shared triage prompt."""
    assert context.runtime is not None, "Early triage requires an active workflow runtime."
    environment = context.runtime.environment
    request = InputTriageRequest.model_validate(
        yaml_util.safe_load(environment.read_text("inbox/request.yaml")))
    work = intake.start(request.cycle)
    entity_proposals = None
    if request.entity_extraction_input is None:
        work = triage.run(context, work)
    else:
        prefix = StoryPrefix(work)
        prompts = ExtractionPrompts.model_validate(yaml_util.safe_load(
            Path(__file__).with_name("prompts.yaml").read_text(encoding="utf-8")))
        messages = triage.build_messages(work, phase_prompts("triage"), prefix)
        messages.append(Turn(Role.User, render_extraction_context(request.entity_extraction_input) + "\n\n" + prompts.extraction))
        answer = complete(context, work, messages, InputTriageAnswer,
            grounded=True, emotion=True, grounding_at=len(prefix.turns),
            validation_context={"stimulus_ids": [item.knoxel_id for item in request.cycle.stimuli]})
        work.triage = TriageResult(items=answer.items, codelet_budget=answer.codelet_budget)
        try:
            entity_proposals = hydrate_entity_extraction(request.entity_extraction_input, answer.entity_extraction,
                                                         producer="immediate")
        except ValidationError as error:
            raise ModelFormatError(f"Entity extraction has an invalid semantic contract: {error}") from error
    assessment = EarlyTriageAssessment(
        source_ids=[item.knoxel_id for item in request.cycle.stimuli],
        source_fingerprints=request.source_fingerprints,
        guidance_tick=request.guidance_tick,
        result=work.triage,
        evidence=request.evidence,
        entity_proposals=entity_proposals,
    )
    write_report(environment, assessment)
    return {"status": "done", "summary": f"classified {len(assessment.source_ids)} partner inputs"}
