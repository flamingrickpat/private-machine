"""Run the existing contextual triage mechanism without the full cycle.

The host owns durable assessment acceptance; this role only returns a proposal.
The same fitted call can also extract explicit workflow requests and entity claims.
"""
from __future__ import annotations

from pm.llm.role_model import role_model

from pm_workflows.python_role import RoleContext
from pathlib import Path
from pydantic import BaseModel, ValidationError

from pm.utils import yaml_util
from pm.context_render.budget import RequiredContextTooLarge
from pm.context_render.turns import Role, Turn
from pm.llm.structured import ModelFormatError, complete_json
from pm.context_render.correction import CorrectionHistory
from pm.llm.generation import GenerationSettings
from pm.constants import Sampler
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
    session = role_model(context.runtime)
    work.prepare_stimulus_summaries(session)
    entity_proposals = None
    prompts = EarlyTriagePrompts.model_validate(yaml_util.safe_load(
        Path(__file__).with_name("prompts.yaml").read_text(encoding="utf-8")))
    appended: list[Turn] = []
    if request.body_target is not None:
        target = request.body_target
        appended.append(Turn(Role.User, f"Minecraft run {target.run_id} is {target.state}. On the current input only, "
            "set body_operation to stop for an explicit request to stop or pause body work (including stop and report where); use yield to explicitly free the body for another task; "
            "status for a progress question about that work; resume for an explicit request to continue stopped work. "
            'Each element of items must contain its own "body_operation" field, even when null. '
            'Example item fragment: {"body_operation": "status", "workflow_task": null}. '
            "Quoted or hypothetical requests do not authorize control. Leave workflow_task null for these operations."))
    if request.direct_workflows:
        options = "\n".join(f"- {option.name}: {option.description}" for option in request.direct_workflows)
        appended.append(Turn(Role.User, prompts.workflow_requests + "\nAvailable workflows\n" + options))
    claim_limit = 0
    extraction_input = request.entity_extraction_input
    if extraction_input is not None:
        claim_limit = max((len(family.context.claims) for family in extraction_input.families), default=0)
        while True:
            extraction = render_extraction_context(extraction_input, claim_limit) + "\n\n" + prompts.extraction
            try:
                prompt = triage.fit_messages(context, work, appended=[*appended, Turn(Role.User, extraction)])
                break
            except RequiredContextTooLarge:
                if claim_limit == 0:
                    raise
                claim_limit //= 2
    else:
        prompt = triage.fit_messages(context, work, appended=appended)
    work.receipt("triage-context", prompt.summary())
    validation = TriageCheck(
        stimulus_ids=tuple(item.knoxel_id for item in request.cycle.stimuli),
        shown_evidence_ids=frozenset(prompt.selection.source_ids),
        shown_inputs=prompt.shown_stimulus_texts(work),
        classify_body_operation=request.body_target is not None,
        workflow_task_names=frozenset(option.name for option in request.direct_workflows),
        original_inputs={item.knoxel_id: item.text for item in request.cycle.stimuli},  # pm-lint: allow PM005 -- original utterances indexed by source knoxel ID
    )
    if extraction_input is None:
        result = complete(context, prompt.turns, TriageResult, validation_context=validation)
    else:
        visible_claims = {item.claim.claim_id for family in extraction_input.families
                          for item in family.context.claims[:claim_limit]}
        visible_sources = frozenset(prompt.selection.source_ids)
        corrections = CorrectionHistory()
        problem = ""
        generation = GenerationSettings(temperature=Sampler.TempInstruct, enable_thinking=False)
        while True:
            answer = complete_json(session, prompt.turns, InputTriageAnswer, validation_context=validation,
                                   initial_problem=problem, cancel=session.cancel, settings=generation)
            try:
                for claim in answer.entity_extraction.claims:
                    targets = set(claim.correction_target_ids) | set(claim.challenge_target_ids)
                    if targets - visible_claims or set(claim.source_ids) - visible_sources:
                        raise ModelFormatError("Entity claims may reference only sources and claim IDs shown in this attempt.")
                    if claim.exact_quote is not None:
                        # Historical sources were admitted as whole records. A new input may
                        # instead be represented by a summary or an explicitly marked excerpt.
                        quoted = False
                        for source in extraction_input.sources:
                            key = source.dependency.record_id
                            shown = validation.shown_inputs.get(key, source.content)
                            if key in claim.source_ids and claim.exact_quote in shown and claim.exact_quote in source.content:
                                quoted = True
                                break
                        if not quoted:
                            raise ModelFormatError("An entity quote must occur in both the visible and original source text.")
                entity_proposals = hydrate_entity_extraction(extraction_input, answer.entity_extraction,
                                                             producer="immediate")
            except (ModelFormatError, ValidationError) as error:
                problem = corrections.record_rejection("entity extraction", str(error))
                continue
            result = TriageResult(items=answer.items, codelet_budget=answer.codelet_budget)
            break
    assessment = EarlyTriageAssessment(
        source_ids=[item.knoxel_id for item in request.cycle.stimuli],
        source_fingerprints=request.source_fingerprints,
        guidance_tick=request.guidance_tick,
        result=result,
        evidence=request.evidence, body_target=request.body_target,
        entity_proposals=entity_proposals,
    )
    write_report(environment, assessment)
    return {"status": "done", "summary": f"classified {len(assessment.source_ids)} partner inputs"}
