"""Three focused inference calls for a versioned private experiment.

The monitor identifies evidence. The assessor compares explanations. The guide
chooses one permitted lifecycle step. Rejected proposals remain in artifacts.
No role writes accepted memory or performs an external intervention.
"""
import json

from workflows.main_cognition.stage_support import complete_json, read_request, write_report
from workflows.reflection.schemas import Assessment, MonitorResult, ReflectionDecision
from workflows.reflection.experiments import validate_decision

RULES = """You are the companion's private reflection process. Preserve its voice, mixed emotions,
privacy, and choices. A negative mood or disagreement is not a defect. Repeated thought alone
does not prove fixation. Separate implementation failure, missing evidence, interpretation,
personal choice, and success. Original user statements and observations cannot change.
Keep evidence, interpretation, and model confidence separate. Repeated explanations of one
source add no independent support. Cite only the supplied integer record IDs.
Return only JSON that matches the supplied schema. Do not report an unobserved outcome.
"""


def _call(context, task: str, schema, evidence: dict):
    """Run one focused inference call through the kernel's cancellation-aware model."""
    return complete_json(context, [
        {"role": "system", "content": RULES + "\n" + task + "\nSchema:\n" + json.dumps(schema.model_json_schema())},
        {"role": "user", "content": json.dumps(evidence, ensure_ascii=False)},
    ], schema, temperature=0.2)


def run(context) -> dict:
    """Write one validated decision after three real, separately scoped calls.

    A semantic rejection receives one model repair. A second rejection raises
    ValueError without changing accepted state. Cancellation belongs to the kernel.
    """
    request = read_request(context.runtime.environment)
    evidence = request["reflection"]
    monitor = _call(context, "Monitor the retained episodes and experiments. Identify a concrete mismatch or useful repair. "
        "For an existing experiment, look for a later event that can discriminate its recorded prediction. "
        "No new relevant evidence is a valid finding. Keep an unobserved prediction unresolved.",
        MonitorResult, evidence)
    assessment = _call(context, "Assess the monitor's finding against original evidence. Compare at least two plausible "
        "explanations, including counterevidence and limits. For a later outcome, distinguish support, contradiction, "
        "and unresolved evidence. A read version receipt establishes consumption, not improvement.",
        Assessment, {**evidence, "monitor": monitor.model_dump(mode="json")})
    task = """Guide one modest experiment step. Prefer finishing an existing relevant experiment over a duplicate.
Use action none when no useful change or new observation exists. Proposed predictions commit before application.
For propose, include a complete plan, the triggering episode ID, two distinct hypotheses, original support,
opposing evidence, uncertainty, the prediction, a discriminating observation, the future observation event,
validity scope, learning kind, and intervention. Set diagnosis from the actual finding.
Choose triggering_episode_id from the supplied episodes' episode_id values. A source ID or tick alone is not an episode.
If no relevant retained episode exists, choose none. The committed private thought can supply an episode for a later review.
Permitted owners: attention (temporary FocusProposal) or belief (an existing subjective BeliefRecord).
Attention expected_record_id must equal attention_head_id, including null. Focus IDs must be granted.
Belief expected_record_id must name an active subjective belief. A user report or known fact cannot be revised.
Use apply only for an existing proposed experiment whose sources still fit. No new plan accompanies it.
Use await for applied work with no relevant outcome. Use evaluate only for an existing applied or awaiting
experiment, or new evidence for an evaluated experiment. Observation IDs must be new accepted records after
application. Generated worry is not an observation. A supported or contradicted outcome also needs a later
decision receipt with reflection_versions naming that experiment's effective_record_id.
Each experiment's observation_grant lists eligible later IDs and consuming decisions. Use only those outcome IDs.
An external report needs an explicit reference to a consuming decision. Original evidence supports hypotheses,
but cannot become outcome evidence. Choose unresolved with no observations when no relevant eligible outcome exists.
Use retain only after at least two distinct later observations support the prediction. Retain its validity scope.
Use reverse to restore the previous effective state as a new version after a failed or misleading intervention.
Do not reverse across a newer owner change. Preserve failures and explain the corrected diagnosis in reason.
Use revise after evaluation, reversal, or interruption to propose a corrected experiment with a new complete plan.
Its replacement must cite original observations and new evidence. The host links its failed parent automatically.
Do not cite the parent ledger record as observation evidence. The old prediction stays in history.
Use interrupt to stop an obsolete experiment and retain its known partial state. Interrupt does not undo a change.
A timeout or missing reply stays unresolved. Source modification, grants, body controls, and emotional resets
are outside this contract. Keep experiment_id null for propose and none. Only propose and revise carry a plan.
"""
    guide_input = {**evidence, "monitor": monitor.model_dump(mode="json"),
                   "assessment": assessment.model_dump(mode="json")}
    for attempt in range(2):
        decision = _call(context, task, ReflectionDecision, guide_input)
        try:
            validate_decision(decision, evidence)
            break
        except ValueError as error:
            (context.kernel_data / f"reflection-rejection-{attempt}.json").write_text(
                json.dumps({"error": str(error), "decision": decision.model_dump(mode="json")}), encoding="utf-8")
            if attempt:
                raise
            guide_input["rejected_proposal"] = decision.model_dump(mode="json")
            guide_input["rejection_reason"] = str(error)
    assessment_json = json.dumps({"monitor": monitor.model_dump(mode="json"),
                                  "assessment": assessment.model_dump(mode="json")}, indent=2)
    (context.kernel_data / "assessment.json").write_text(assessment_json, encoding="utf-8")
    context.runtime.environment.write_text("outbox/assessment.json", assessment_json)
    write_report(context.runtime.environment, decision)
    return {"status": "done", "summary": decision.reason or "No experiment step is needed."}
