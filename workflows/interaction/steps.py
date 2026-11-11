"""Select an ordinary move through the existing character decision and route.

The host overview defines authority. This role writes proposals only.
It copies accepted mental state and never fabricates a new appraisal or GWT turn.
"""
from __future__ import annotations

from pm_workflows.python_role import RoleContext

from pm.context_render.turns import Role, Turn
from pm.records.copying import pickle_copy
from pm.records.cycle_records import ActionDecision
from pm.utils import yaml_util
from workflows.interaction.schemas import (
    GuidanceDisposition, InteractionDecision, InteractionOutcome, InteractionReceipt, InteractionRequest,
)
from workflows.main_cognition.affect import describe_mental_state, regulation_mode
from workflows.main_cognition.cycle_work import CycleWork
from workflows.main_cognition.model_json import write_report
from workflows.main_cognition.phases import decide, intake, route, triage
from workflows.main_cognition.phases.model import complete, phase_prompts
from workflows.main_cognition.schemas import AppraisalAffect, DecisionCheck
from workflows.main_cognition.story_prefix import StoryPrefix


def prepare_work(request: InteractionRequest) -> CycleWork:
    """Adapt accepted guidance to decision context without running cognitive phases."""
    previous = request.cycle.previous_state
    if previous is None:
        raise ValueError("Ordinary interaction requires accepted main guidance.")
    state = pickle_copy(previous)
    state.stimulus_groups = intake.group_stimuli(request.cycle)
    state.decisions = []
    state.dispatches = []
    for group in state.stimulus_groups:
        state.workspace.ensure_shard(group.key)
    work = CycleWork(request=request.cycle, state=state, records=dict(request.cycle.records),
                     state_summary=request.cycle.state_summary)
    intake.set_tails(work)
    if request.cycle.early_triage is None:
        raise ValueError("Ordinary interaction requires accepted early triage.")
    work.triage = request.cycle.early_triage
    triage.rate_groups(work)
    work.affect = AppraisalAffect(
        felt=describe_mental_state(state.mental_state, name=request.cycle.companion_name,
                                  partner_name=request.cycle.user_name),
        regulation=regulation_mode(state.mental_state, state.mental_state),
    )
    return work


def select_interaction(context: RoleContext, request: InteractionRequest) -> InteractionReceipt:
    """Choose one local reply, quiet outcome, or consequential reconsideration."""
    work = prepare_work(request)
    if len(work.state.stimulus_groups) != 1:
        raise ValueError("An ordinary selection must cover exactly one partner group.")
    group = work.state.stimulus_groups[0]
    prefix = StoryPrefix(work, group=group.key)
    messages = decide.build_messages(work, phase_prompts("decide"), prefix, group)
    messages.append(Turn(Role.User,
        "This is an ordinary exchange under the accepted direction above. "
        "Choose the character's local move using the same situation and motives. "
        "guidance_disposition is follow, decline, or suspend for this exchange. "
        "If the current situation makes prior guidance unsuitable, explain that mismatch in guidance_reason. "
        "This local choice does not rewrite durable intentions or establish a permanent wait. "
        "Keep consequential task choices available. Main cognition will reconsider them before execution."))
    check = DecisionCheck(workflow_names=[item.name for item in decide.available_workflows(work)])
    answer = complete(context, work, messages, InteractionDecision, validation_context=check.model_dump())
    source_ids = [item.knoxel_id for item in request.cycle.stimuli]
    if not answer.variants:
        outcome = InteractionOutcome.Quiet
        if answer.guidance_disposition != GuidanceDisposition.Follow:
            outcome = InteractionOutcome.Reconsider
        return InteractionReceipt(selection_id=request.selection_id, source_ids=source_ids,
            outcome=outcome, decision=answer, dependencies=request.dependencies)
    work.state.decisions = [ActionDecision(group=group.key, text=answer.variants[0],
        variants=list(answer.variants), stimulus_ids=source_ids)]
    work = route.select_routes(context, work)
    choice = work.routes[0].route
    local_reply = choice.action_type == "Reply" and choice.workflow_name == "dialogue_story"
    dispatch = None
    if local_reply:
        work = route.anticipate_dispatches(context, work)
        dispatch = work.dispatches[0]
        dispatch.action_id = request.selection_id + ":reply"
        outcome = InteractionOutcome.Reply
    elif choice.action_type in {"Idle", "Ignore"} and answer.guidance_disposition == GuidanceDisposition.Follow:
        outcome = InteractionOutcome.Quiet
    else:
        outcome = InteractionOutcome.Reconsider
    return InteractionReceipt(selection_id=request.selection_id, source_ids=source_ids,
        outcome=outcome, decision=answer, route=choice, dispatch=dispatch, dependencies=request.dependencies)


def run(context: RoleContext) -> dict[str, str]:
    """Run the real kernel role and write its typed selection proposal."""
    assert context.runtime is not None, "Ordinary selection requires an active workflow runtime."
    request = InteractionRequest.model_validate(yaml_util.safe_load(
        context.runtime.environment.read_text("inbox/request.yaml")))
    result = select_interaction(context, request)
    write_report(context.runtime.environment, result)
    return {"status": "done", "summary": result.outcome.value}
