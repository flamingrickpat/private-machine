"""Three focused inference calls for a versioned private experiment.

The monitor identifies evidence. The assessor compares explanations. The guide
chooses one permitted lifecycle step. Rejected proposals remain in artifacts.
No role writes accepted memory or performs an external intervention.
"""
from pm.llm.generation import GenerationSettings
import json
from dataclasses import dataclass
from typing import TypeVar
from pydantic import BaseModel
from pm_workflows.python_role import RoleContext

from workflows.inner_thought.schemas import PrivateRoleResult
from workflows.inner_thought.schemas import InnerRequest
from workflows.reflection.schemas import ReflectionContext, ReflectionEpisode

from pm.constants import Sampler
from pm.runs.role_io import read_request, write_report
from pm.llm.role_model import role_model
from pm.llm.structured import parse_response, JsonOutputError
from pydantic import ValidationError
from pm.context_render.prompt import RenderSettings, RenderedPrompt, RendererCapture, RendererInput
from pm.context_render.records import render_record
from workflows.inner_thought.prompt_selection import PrivatePromptInput, PrivateEvidenceBlock, select_private_evidence
from workflows.reflection.schemas import Assessment, MonitorResult, ReflectionDecision
from workflows.reflection.experiments import validate_decision

RULES = """You are the companion's private reflection process. Preserve its voice, mixed emotions,
privacy, and choices. A negative mood or disagreement is not a defect. Repeated thought alone
does not prove fixation. Separate implementation failure, missing evidence, interpretation,
personal choice, and success. Original user statements and observations cannot change.
Keep evidence, interpretation, and model confidence separate. Repeated explanations of one
source add no independent support. Cite only the supplied integer record IDs.
Use each summary's stable experiment_id to identify an experiment; record_id names its current ledger knoxel.
Return only JSON that matches the supplied schema. Do not report an unobserved outcome.
"""


T = TypeVar("T", bound=BaseModel)


class ReflectionCall(BaseModel):
    """Typed stage inputs. The evidence owner retains the complete frozen grant."""

    context: ReflectionContext
    salience: dict[int, float]
    monitor: MonitorResult | None = None
    assessment: Assessment | None = None


@dataclass(frozen=True, slots=True)
class ReflectionRenderInput:
    call: ReflectionCall
    task: str
    schema: type[BaseModel]
    correction: str = ""


def render_reflection(call: ReflectionCall, task: str, schema: type[BaseModel], settings: RenderSettings,
                      correction: str = "") -> RenderedPrompt:
    pinned: set[int] = set()
    lines = [f"Review due {call.context.review_due}; attention head {call.context.attention_head_id}.",
        "Evidence limits: " + "; ".join(call.context.episode_limits)]
    if call.monitor is not None:
        monitor = call.monitor
        pinned.update(monitor.evidence_ids)
        lines.append(f"Monitor finding: {monitor.finding}; relevant {monitor.relevant}; episode {monitor.episode_id}; "
            f"evidence {monitor.evidence_ids}. Reason: {monitor.reason}")
    if call.assessment is not None:
        assessment = call.assessment
        pinned.update(assessment.observation_ids)
        lines.append(f"Assessment: {assessment.result}; observations {assessment.observation_ids}. {assessment.reason}")
        for hypothesis in assessment.hypotheses:
            pinned.update(hypothesis.supporting_ids)
            pinned.update(hypothesis.opposing_ids)
            lines.append(f"Explanation: {hypothesis.explanation}; supporting {hypothesis.supporting_ids}; "
                f"opposing {hypothesis.opposing_ids}; uncertainty {hypothesis.uncertainty}; confidence {hypothesis.confidence:.3f}")
    records = {record.knoxel_id: record for record in call.context.records}
    blocks: list[PrivateEvidenceBlock] = []
    grouped: set[int] = set()
    for experiment in call.context.experiments:
        metadata = experiment.metadata
        plan = metadata.plan
        sources = set(metadata.source_revisions) | set(experiment.observation_grant.observation_ids)
        sources.update(experiment.observation_grant.consumer_ids)
        sources.add(experiment.record_id)
        if metadata.effective_record_id is not None:
            sources.add(metadata.effective_record_id)
        text = (f"Experiment {experiment.experiment_id}; ledger record {experiment.record_id}; state {metadata.state}; "
            f"effective record {metadata.effective_record_id}; result {metadata.result}; stale sources {experiment.stale_source_ids}.\n"
            f"Finding: {plan.finding}\nDiagnosis: {plan.diagnosis}; triggering episode {plan.triggering_episode_id}.\n"
            f"Prediction: {plan.predicted_effect}\nDiscriminating observation: {plan.discriminating_observation}\n"
            f"Observation event: {plan.observation_event}\nScope: {plan.scope}; learning {plan.learning}.\n"
            f"Intervention owner {plan.intervention.owner}; target {plan.intervention.expected_record_id}; belief {plan.intervention.belief}.\n"
            f"Eligible later observations {experiment.observation_grant.observation_ids}; consuming decisions {experiment.observation_grant.consumer_ids}.")
        for hypothesis in plan.hypotheses:
            text += (f"\nHypothesis: {hypothesis.explanation}; support {hypothesis.supporting_ids}; "
                f"opposition {hypothesis.opposing_ids}; uncertainty {hypothesis.uncertainty}")
        if plan.intervention.focus is not None:
            focus = plan.intervention.focus
            text += f"\nFocus instruction: {focus.instruction}; promote {focus.promote_ids}; reduce {focus.reduce_ids}."
        for key in sorted(sources & records.keys()):
            text += "\n" + render_record(records[key]).text
        priority = 10 if metadata.state in {"proposed", "applied", "awaiting_evidence"} else 3
        blocks.append(PrivateEvidenceBlock(f"Experiment {experiment.record_id}", text, priority, tuple(sorted(sources)), bool(sources & pinned)))
        grouped.update(sources)
    for episode in reversed(call.context.episodes):
        sources = set(episode.inputs) | set(episode.outputs)
        for linked in episode.links.values():
            sources.update(linked)
        text = (f"Episode {episode.episode_id}; tick {episode.tick_id}; {episode.kind}; {episode.loop}; "
            f"operation {episode.operation}; owner {episode.owner}.\n{episode.summary}\nOutcome: {episode.outcome}")
        for key in sorted(sources & records.keys()):
            text += "\n" + render_record(records[key]).text
        required = call.monitor is not None and call.monitor.episode_id == episode.episode_id
        blocks.append(PrivateEvidenceBlock(f"Episode {episode.episode_id}", text, 6, tuple(sorted(sources)), required))
        grouped.update(sources)
    for record in call.context.records:
        if record.knoxel_id not in grouped or record.knoxel_id in pinned:
            blocks.append(PrivateEvidenceBlock("Original reflection source", render_record(record).text,
                call.salience.get(record.knoxel_id, 0), (record.knoxel_id,), record.knoxel_id in pinned))
    return select_private_evidence(PrivatePromptInput("private reflection " + schema.__name__,
        RULES + "\n" + task + "\nOutput schema:\n" + json.dumps(schema.model_json_schema()),
        "\n\n".join(lines), tuple(blocks), correction), settings)


def _call(context: RoleContext, task: str, schema: type[T], call: ReflectionCall) -> T:
    assert context.runtime is not None
    session = role_model(context.runtime)
    settings = session.render_settings()
    correction = ""
    attempt = 0
    while True:
        render_input = ReflectionRenderInput(call, task, schema, correction)
        with RendererCapture(session.capture_settings("private.reflection"),
                RendererInput("private.reflection", render_input, settings)) as capture:
            rendered = render_reflection(call, task, schema, settings, correction)
            capture.save_output(rendered)
        selected = {entry.record_id for entry in rendered.evidence if entry.included}
        episodes: list[ReflectionEpisode] = []
        for episode in call.context.episodes:
            ids = set(episode.inputs) | set(episode.outputs)
            for linked in episode.links.values():
                ids.update(linked)
            if f"Episode {episode.episode_id}" in rendered.selected_groups:
                episodes.append(episode)
        visible = ReflectionContext(
            experiments=[entry for entry in call.context.experiments if f'Experiment {entry.record_id}' in rendered.selected_groups],
            omitted_experiments=call.context.omitted_experiments,
            attention_head_id=call.context.attention_head_id,
            review_due=call.context.review_due,
            newest_observation_id=call.context.newest_observation_id,
            episodes=episodes,
            episode_limits=call.context.episode_limits,
            records=[record for record in call.context.records if record.knoxel_id in selected],
        )
        raw = session.complete(rendered.turns, settings=GenerationSettings(temperature=Sampler.TempReasoning))
        try:
            answer = parse_response(raw, schema)
            cited: set[int] = set()
            if isinstance(answer, MonitorResult):
                cited.update(answer.evidence_ids)
                if answer.episode_id is not None and answer.episode_id not in {episode.episode_id for episode in episodes}:
                    raise ValueError("The monitor cited an episode outside its selected context.")
            elif isinstance(answer, Assessment):
                cited.update(answer.observation_ids)
                for hypothesis in answer.hypotheses:
                    cited.update(hypothesis.supporting_ids)
                    cited.update(hypothesis.opposing_ids)
            elif isinstance(answer, ReflectionDecision):
                validate_decision(answer, visible)
            if cited - selected:
                raise ValueError("Reflection citations must name evidence selected for this prompt.")
        except (ValidationError, JsonOutputError, ValueError) as error:
            attempt += 1
            correction = str(error)
            context.runtime.environment.write_text(f"scratch/{schema.__name__}-rejected-{attempt:03}.txt", raw + "\n" + correction)
            continue
        return answer


def run(context: RoleContext) -> PrivateRoleResult:
    """Write one validated decision after three real, separately scoped calls.

    Semantic model rejections retain diagnostics and request another proposal.
    Cancellation belongs to the kernel. Internal and measurement errors propagate.
    """
    assert context.runtime is not None
    request = read_request(context.runtime.environment, InnerRequest)
    evidence = request.reflection
    call_input = ReflectionCall(context=evidence, salience={item.knoxel_id: item.urgency for item in request.candidates})
    monitor = _call(context, "Monitor the retained episodes and experiments. Identify a concrete mismatch or useful repair. "
        "For an existing experiment, look for a later event that can discriminate its recorded prediction. "
        "No new relevant evidence is a valid finding. Keep an unobserved prediction unresolved.",
        MonitorResult, call_input)
    assessment = _call(context, "Assess the monitor's finding against original evidence. Compare at least two plausible "
        "explanations, including counterevidence and limits. For a later outcome, distinguish support, contradiction, "
        "and unresolved evidence. A read version receipt establishes consumption, not improvement.",
        Assessment, ReflectionCall(context=evidence, salience=call_input.salience, monitor=monitor))
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
    guide_input = ReflectionCall(context=evidence, salience=call_input.salience, monitor=monitor, assessment=assessment)
    decision = _call(context, task, ReflectionDecision, guide_input)
    receipt_lines = [f"Monitor: {monitor.finding}; relevant={monitor.relevant}; episode={monitor.episode_id}.",
        f"Monitor evidence: {monitor.evidence_ids}. Reason: {monitor.reason}",
        f"Assessment: {assessment.result}; observations={assessment.observation_ids}. {assessment.reason}"]
    for hypothesis in assessment.hypotheses:
        receipt_lines.append(f"Explanation: {hypothesis.explanation}; confidence={hypothesis.confidence:.3f}; "
            f"supporting={hypothesis.supporting_ids}; opposing={hypothesis.opposing_ids}; "
            f"uncertainty: {hypothesis.uncertainty}")
    assessment_text = "\n".join(receipt_lines)
    (context.kernel_data / "assessment.txt").write_text(assessment_text, encoding="utf-8")
    context.runtime.environment.write_text("outbox/assessment.txt", assessment_text)
    write_report(context.runtime.environment, decision)
    return {"status": "done", "summary": decision.reason or "No experiment step is needed."}
