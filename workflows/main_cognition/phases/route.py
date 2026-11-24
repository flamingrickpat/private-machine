"""Route phase: map each decision to an action, check conflicts, and record expectations.

A19F step 8 and decision D28. Steps:

1. Routing (model). One story-framed call for each decision, in parallel.
   The call reads the story prefix of the decision's group, the decision, the group's
   stimuli, the memories, the goal changes, the managed-run receipts, the
   capabilities, and the available workflows. It holds every capability,
   Minecraft, task, and presentation rule. It returns the action type, the
   workflow, the arguments, and the presentation (``RouteAnswer``). It never
   writes the decision text.
2. Dispatches (Python). A quiet route (Ignore, Idle, Sleep) makes no
   dispatch. Every other route makes one ``ActionDispatch`` whose
   ``content`` is the decision text, unchanged. So the target workflow
   reads the character's own decision as its intent. Each dispatch carries
   its group, the group's stimulus ids, and the group's correlation id, and
   gets a durable action id.
3. Conflict check (Python, :func:`resolve_conflicts`). One group gets at
   most one action, and two actions cannot claim the same body lease. The
   group with the higher triage category keeps its action.
4. Ordering and prerequisites (``workflows.main_cognition.dispatch``).
5. Expectations (model). For each released dispatch, one story-framed call
   writes the reaction that the companion now expects
   (:func:`workflows.main_cognition.expectations.anticipate`).

This phase has no external effect. The host stores the actions and the
expectations at commit and publishes the released dispatches after it.
"""
from __future__ import annotations

from workflows.main_cognition.prompt_values import RouteTemplateValues

from pm.llm.role_model import role_model

from pm_workflows.python_role import RoleContext

from pm.records.copying import pickle_copy

import hashlib
import json

from pm.constants import EXPECTATION_COUNT
from pm.context_render.shards import fit_shard_instruction
from pm.context_render.turns import Turn
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pm.llm.concurrency import wait_model_futures
from pm.records.cycle_records import ActionDecision, ActionDispatchRef, StimulusGroup
from pm.records.workspace import ShardKey

from workflows.inference_settings import DECISION_WORKERS
from workflows.main_cognition import expectations
from workflows.main_cognition.csm_view import Embedder
from workflows.main_cognition.rendering import render_managed_runs, render_goal_changes
from workflows.main_cognition.cycle_work import CycleWork
from workflows.main_cognition.dispatch import resolve_prerequisites, resolve_same_cycle
from workflows.main_cognition.prompt_config import PhasePrompts
from workflows.main_cognition.phases.decide import available_workflows, group_part
from workflows.main_cognition.phases.model import complete, phase_prompts
from workflows.main_cognition.phases.triage import CATEGORIES
from workflows.main_cognition.schemas import (
    QUIET_ACTIONS,
    ActionArguments,
    ActionDispatch,
    ExpectationDraft,
    RouteAnswer,
    RouteCheck,
    RouteChoice,
    RoutedDecision,
    WorkflowOption,
)
from workflows.main_cognition.story_prefix import StoryPrefix, shard_label

#: Workflows that hold a body lease for their whole run, by the body they claim.
BODY_WORKFLOWS = {"minecraft": "minecraft", "minecraft_project": "minecraft"}
#: Workflows whose goal is the decision text itself. Their dispatch drops a
#: routed ``instructions`` or ``instruction`` argument, so the start uses the
#: decision text unchanged. Other arguments (for example ``tool_goal``) stay.
DECISION_GOAL_WORKFLOWS = frozenset({"tool_call", "assistant"})


def stable_action_id(dispatch: ActionDispatch, index: int) -> str:
    """Return a durable identity for one proposed action.

    The same workflow, action type, decision text, and arguments get the same
    id across cycles, so a repeated start request can be recognised and
    suppressed. An action without content gets its list position, because
    it has no stable identity.
    """
    content = dispatch.content.strip()
    if not content:
        return f"action-{index}"
    material = json.dumps({
        "workflow_name": dispatch.workflow_name,
        "action_type": dispatch.action_type,
        "content": content,
        "arguments": dispatch.arguments.payload(),
    }, sort_keys=True, default=str)
    return "action-" + hashlib.sha1(material.encode("utf-8")).hexdigest()[:12]


def group_of(work: CycleWork, decision: ActionDecision) -> StimulusGroup:
    """Return the stimulus group of a decision. A decision without its group is a bug and raises."""
    return next(group for group in work.state.stimulus_groups if group.key == decision.group)


def render_group_stimuli(work: CycleWork, group: StimulusGroup) -> str:
    """Return the group's stimuli with their ids, source, and kind, so routing can cite them."""
    lines = [f"- id={item.knoxel_id} source={item.source} kind={item.kind} text={work.prompt_summaries.get(item.knoxel_id, item.text)!r}"
             for item in work.request.stimuli if item.knoxel_id in group.stimulus_ids]
    return "\n".join(lines) or "(none)"


def render_memories(work: CycleWork) -> str:
    """Return the selected memories with their ids and evidence roles."""
    lines = [f"- id={item.knoxel_id} kind={item.memory_kind} evidence_role={item.evidence_role} text={item.text!r}"
             for item in work.selected_memories]
    return "\n".join(lines) or "(none)"


def build_messages(work: CycleWork, prompts: PhasePrompts, prefix: StoryPrefix, decision: ActionDecision) -> list[Turn]:
    """Return the shared story prefix and then the routing instruction for one decision as the last turn."""
    request = work.request
    group = group_of(work, decision)
    values = RouteTemplateValues(
        part=group_part(work, prompts, prefix, group),
        decision=decision.text,
        stimuli=render_group_stimuli(work, group),
        memories=render_memories(work),
        goals=render_goal_changes(work.goals) if work.goals.changes else "(none)",
        managed_runs=render_managed_runs(request.managed_runs) if request.managed_runs else "(none)",
        capabilities="\n".join(f"- {item}" for item in request.capabilities) or "(none)",
        workflows="\n".join(f"- {item.name}: {item.description}" for item in available_workflows(work)),
        wait_default_seconds=request.wait_default_seconds)
    instruction = fit_shard_instruction(prefix.fill(prompts.user_template), values,
        frozenset({"decision", "stimuli", "memories", "goals", "managed_runs", "capabilities"}),
        request.images, prefix.settings)
    return prefix.ask(instruction)


def route(context: RoleContext, work: CycleWork, prompts: PhasePrompts, prefix: StoryPrefix,
          decision: ActionDecision) -> RouteAnswer:
    """Run the routing call of one decision. Runs in a worker thread and does not change ``work``."""
    runs = work.request.managed_runs
    changed_goal = any(item.change in {"add", "complete", "defer", "refuse", "reprioritize"}
                       for item in work.goals.changes)
    change_sources = {item.knoxel_id for item in work.request.stimuli
                      if item.knoxel_id in decision.stimulus_ids and
                      (item.kind in {"user.input", "minecraft.world", "workflow.question", "workflow.event"}
                       or changed_goal and item.kind == "thought.created")}
    check = RouteCheck(workflow_names=[item.name for item in available_workflows(work)],
        visible_run_ids=frozenset(run.run_id for run in runs if run.run_id),
        updatable_run_ids=frozenset(run.run_id for run in runs if run.run_id and run.updates_open
            and run.state in {"running", "accepted", "waiting_for_input"}),
        pending_revision_run_ids=frozenset(run.run_id for run in runs if run.run_id and run.revision > run.applied_revision),
        partner_sources=frozenset(item.knoxel_id for item in work.request.stimuli if item.kind == "user.input"),
        change_sources=frozenset(change_sources))
    return complete(context, build_messages(work, prompts, prefix, decision), RouteAnswer,
                    validation_context=check)


def routed_decisions(work: CycleWork, answers: list[RouteAnswer]) -> list[RoutedDecision]:
    """Pair each decision with its group facts and its route."""
    routed: list[RoutedDecision] = []
    for decision, answer in zip(work.state.decisions, answers, strict=True):
        group = group_of(work, decision)
        routed.append(RoutedDecision(group=group.key, stimulus_ids=list(decision.stimulus_ids),
                                     correlation_id=group.correlation_id, decision=decision.text,
                                     route=RouteChoice(action_type=answer.action_type, workflow_name=answer.workflow_name,
                                         arguments=answer.arguments, presentation=answer.presentation, reasoning=answer.reasoning)))
        if answer.parallel_inquiry:
            related = answer.arguments.related_run_id or answer.arguments.run_id
            routed.append(RoutedDecision(group=group.key, stimulus_ids=list(decision.stimulus_ids),
                correlation_id=group.correlation_id, decision=answer.parallel_inquiry,
                route=RouteChoice(action_type="ToolCall", workflow_name="assistant",
                                  arguments=ActionArguments(related_run_id=related),
                                  reasoning="Independent observation selected with the task.")))
    return routed


def action_dispatches(routes: list[RoutedDecision]) -> list[ActionDispatch]:
    """Return one dispatch proposal for each route that is not quiet, without external effects.

    ``content`` is the decision text, unchanged. Quiet routes stay in the
    decision record. They cannot invoke a body or public workflow, even when
    they carry a valid workflow name.
    """
    dispatches: list[ActionDispatch] = []
    for index, item in enumerate(routes):
        answer = item.route
        if answer.action_type in QUIET_ACTIONS:
            continue
        dispatch = ActionDispatch(
            workflow_name=answer.workflow_name, action_type=answer.action_type, content=item.decision,
            arguments=decision_goal_arguments(answer.workflow_name, answer.arguments),
            presentation=pickle_copy(answer.presentation),
            group=item.group, stimulus_ids=list(item.stimulus_ids), correlation_id=item.correlation_id)
        dispatch.action_id = stable_action_id(dispatch, index)
        dispatches.append(dispatch)
    return dispatches


def decision_goal_arguments(workflow_name: str, arguments: ActionArguments) -> ActionArguments:
    """Return a copy of the routed arguments, without instructions for a decision-goal workflow.

    A direct target in ``DECISION_GOAL_WORKFLOWS`` and a self-management start
    of one both lose ``instructions`` and ``instruction``. Self-management then
    starts the run with the decision text, unchanged (decision D28).
    """
    target = arguments.workflow_name if workflow_name == "self_management" and arguments.operation == "start" else workflow_name
    if target not in DECISION_GOAL_WORKFLOWS or arguments.run_id:
        return pickle_copy(arguments)
    return ActionArguments.model_validate({key: value for key, value in arguments.payload().items()
                                           if key not in {"instructions", "instruction"}})


def body_lease(dispatch: ActionDispatch, workflows: list[WorkflowOption] | None = None) -> str:
    """Return the body that a dispatch claims for its whole run, or empty text.

    A start of ``minecraft`` or ``minecraft_project``, directly or through
    ``self_management``, claims the ``minecraft`` body. An update or a stop
    of a running task claims nothing new. A read-only observation shares the
    body and claims nothing.
    """
    arguments = dispatch.arguments
    target = dispatch.workflow_name
    if target == "self_management" and arguments.operation == "start":
        target = arguments.workflow_name
    operation = arguments.operation or ("update" if arguments.run_id else "start")
    if operation != "start":
        return ""
    if workflows is None:
        return BODY_WORKFLOWS.get(target, "")
    return next((option.exclusive_resource for option in workflows if option.name == target), "")


def resolve_conflicts(dispatches: list[ActionDispatch], groups: list[StimulusGroup],
                      workflows: list[WorkflowOption] | None = None) -> list[ActionDispatch]:
    """Reject the dispatches that conflict with a dispatch of higher priority. Deterministic.

    Priority follows the triage category of the group (``Critical`` first),
    then the group order. A dispatch is rejected when its group already has
    a released action, or when its body lease is already claimed. A
    dispatch that ordering or prerequisites did not release keeps its
    status and claims nothing. The result keeps the input order.
    """
    order = {group.key: index for index, group in enumerate(groups)}
    rank = {group.key: CATEGORIES.index(group.category) if group.category else 0 for group in groups}
    by_priority = sorted(range(len(dispatches)), key=lambda index: (-rank[dispatches[index].group],
                                                                     order[dispatches[index].group], index))
    resolved = list(dispatches)
    acting: dict[tuple[ShardKey, bool], str] = {}
    leases: dict[str, str] = {}
    updates: set[str] = set()
    for index in by_priority:
        dispatch = dispatches[index]
        if dispatch.status != "dispatched":
            continue
        lease = body_lease(dispatch, workflows)
        reason = ""
        channel = (dispatch.group, dispatch.workflow_name == "assistant")
        if channel in acting:
            reason = f"group {dispatch.group.kind}:{dispatch.group.ref} already acts through {acting[channel]}"
        elif lease and lease in leases:
            reason = f"body lease {lease} already claimed by {leases[lease]}"
        updating = dispatch.arguments.operation == "update" or (
            bool(dispatch.arguments.run_id) and not dispatch.arguments.operation and dispatch.workflow_name != "self_management")
        if updating and dispatch.arguments.run_id in updates:
            reason = f"another group already revised run {dispatch.arguments.run_id} this cycle"
        if reason:
            rejected = dispatch.model_copy()
            rejected.status = "rejected"
            rejected.rejection_reason = reason
            resolved[index] = rejected
            continue
        acting[channel] = dispatch.action_id
        if updating:
            updates.add(dispatch.arguments.run_id)
        if lease:
            leases[lease] = dispatch.action_id
    return resolved


def anticipation_job(work: CycleWork, prompts: PhasePrompts, prefix: StoryPrefix,
                     dispatch: ActionDispatch) -> tuple[str, str, list[int]]:
    """Return the action words, the target name, and the expected entities of one dispatch."""
    request = work.request
    if dispatch.group.kind == "entity":
        target, entities = shard_label(dispatch.group, request), [int(dispatch.group.ref)]
    else:
        target = request.user_name
        entities = [request.user_entity_id] if request.user_entity_id is not None else []
    action = prefix.fill(prompts.action_words[dispatch.action_type]).format(
        target=target, workflow=dispatch.workflow_name)
    return action, target, entities


def anticipate(context: RoleContext, work: CycleWork, prompts: PhasePrompts, prefix: StoryPrefix,
               dispatch: ActionDispatch) -> list[ExpectationDraft]:
    """Return the expectations of one released dispatch. Runs in a worker thread and does not change ``work``."""
    action, target, entities = anticipation_job(work, prompts, prefix, dispatch)
    return expectations.anticipate(context, work, prefix, prompts, decision=dispatch.content, action=action,
                                   target=target, count=EXPECTATION_COUNT, entity_ids=entities,
                                   embedder=Embedder(context, work.records))


def select_routes(context: RoleContext, work: CycleWork) -> CycleWork:
    """Route character decisions without anticipating or executing their effects."""
    prompts = phase_prompts("route")
    # The route of a decision reads the prefix of the decision's group, like the decision itself.
    prefixes = {decision.group: StoryPrefix(work, group=decision.group, settings=role_model(context.runtime).render_settings(),
            capture_settings=role_model(context.runtime).capture_settings("main-cognition")) for decision in work.state.decisions}
    stop = role_model(context.runtime).cancel
    if role_model(context.runtime).model.connection.disable_threading:
        answers = [route(context, work, prompts, prefixes[decision.group], decision)
            for decision in work.state.decisions]
    else:
        with ThreadPoolExecutor(max_workers=DECISION_WORKERS) as executor:
            futures = [executor.submit(copy_context().run, route, context, work, prompts, prefixes[decision.group], decision)
                       for decision in work.state.decisions]
            answers = wait_model_futures(futures, stop)
    work.routes = routed_decisions(work, answers)
    dispatches = resolve_prerequisites(resolve_same_cycle(action_dispatches(work.routes)), {})
    dispatches = resolve_conflicts(dispatches, work.state.stimulus_groups, work.request.available_workflows or None)
    work.dispatches = dispatches
    return work


def anticipate_dispatches(context: RoleContext, work: CycleWork) -> CycleWork:
    """Add expectations only for the dispatches this owner can release."""
    prompts = phase_prompts("route")
    prefixes = {decision.group: StoryPrefix(work, group=decision.group, settings=role_model(context.runtime).render_settings(),
            capture_settings=role_model(context.runtime).capture_settings("main-cognition")) for decision in work.state.decisions}
    released = [item for item in work.dispatches if item.status == "dispatched"]
    stop = role_model(context.runtime).cancel
    if role_model(context.runtime).model.connection.disable_threading:
        drafts = [anticipate(context, work, prompts, prefixes[item.group], item) for item in released]
    else:
        with ThreadPoolExecutor(max_workers=DECISION_WORKERS) as executor:
            futures = [executor.submit(copy_context().run, anticipate, context, work, prompts, prefixes[item.group], item) for item in released]
            drafts = wait_model_futures(futures, stop)
    for dispatch, expected in zip(released, drafts, strict=True):
        dispatch.expectations = expected
    work.state.dispatches = [ActionDispatchRef(group=item.group, correlation_id=item.correlation_id,
                                               workflow=item.workflow_name, action_type=item.action_type)
                             for item in released]
    work.receipt("route", f"routed {len(work.routes)} decisions; released {len(released)} of "
                          f"{len(work.dispatches)} dispatches; {sum(len(item) for item in drafts)} expectations")
    return work


def run(context: RoleContext, work: CycleWork) -> CycleWork:
    """Route decisions, resolve dispatch conflicts, and anticipate released actions."""
    return anticipate_dispatches(context, select_routes(context, work))


__all__ = [
    "BODY_WORKFLOWS",
    "DECISION_GOAL_WORKFLOWS",
    "action_dispatches",
    "anticipate",
    "anticipation_job",
    "body_lease",
    "build_messages",
    "decision_goal_arguments",
    "group_of",
    "render_group_stimuli",
    "render_memories",
    "resolve_conflicts",
    "route",
    "routed_decisions",
    "run",
    "stable_action_id",
]
