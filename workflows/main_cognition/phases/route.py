"""Route phase: map each decision to an action, check conflicts, and record expectations.

A19F step 8 and decision D28. Steps:

1. Routing (model). One story-framed call for each decision, in parallel.
   The call reads the shared story prefix, the decision, the group's
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

import hashlib
import json

from pm.constants import EXPECTATION_COUNT
from pm.context_render import render_context
from pm.context_render.turns import Turn, format_content
from pm.llm.concurrency import ordered_parallel_map
from pm.records.cycle_records import ActionDecision, ActionDispatchRef, StimulusGroup
from pm.records.workspace import ShardKey

from workflows.inference_settings import DECISION_WORKERS
from workflows.main_cognition import expectations
from workflows.main_cognition.csm_view import Embedder
from workflows.main_cognition.cycle_work import CycleWork
from workflows.main_cognition.dispatch import resolve_prerequisites, resolve_same_cycle
from workflows.main_cognition.model_json import PhasePrompts
from workflows.main_cognition.phases.decide import available_workflows, group_part
from workflows.main_cognition.phases.model import complete, phase_prompts
from workflows.main_cognition.phases.triage import CATEGORIES
from workflows.main_cognition.schemas import (
    QUIET_ACTIONS,
    ActionDispatch,
    RouteAnswer,
    RouteCheck,
    RouteChoice,
    RoutedDecision,
)
from workflows.main_cognition.story_prefix import StoryPrefix, shard_label

#: Workflows that hold a body lease for their whole run, by the body they claim.
BODY_WORKFLOWS = {"minecraft": "minecraft", "minecraft_project": "minecraft"}


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
    lines = [f"- id={item.knoxel_id} source={item.source} kind={item.kind} text={item.text!r}"
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
    instruction = format_content(prefix.fill(prompts.user_template),
        part=group_part(work, prompts, prefix, group),
        decision=decision.text,
        stimuli=render_group_stimuli(work, group),
        memories=render_memories(work),
        goals=render_context(work.goals) if work.goals.changes else "(none)",
        managed_runs=render_context(request.managed_runs) if request.managed_runs else "(none)",
        capabilities="\n".join(f"- {item}" for item in request.capabilities) or "(none)",
        workflows="\n".join(f"- {item.name}: {item.description}" for item in available_workflows(work)),
        wait_default_seconds=request.wait_default_seconds,
    ).text
    return prefix.ask(instruction)


def route(context, work: CycleWork, prompts: PhasePrompts, prefix: StoryPrefix, decision: ActionDecision) -> RouteAnswer:
    """Run the routing call of one decision. Runs in a worker thread and does not change ``work``."""
    check = RouteCheck(workflow_names=[item.name for item in available_workflows(work)])
    return complete(context, work, build_messages(work, prompts, prefix, decision), RouteAnswer,
                    grounded=True, emotion=False, grounding_at=len(prefix.turns), capabilities=True,
                    validation_context=check.model_dump())


def routed_decisions(work: CycleWork, answers: list[RouteAnswer]) -> list[RoutedDecision]:
    """Pair each decision with its group facts and its route."""
    routed = []
    for decision, answer in zip(work.state.decisions, answers, strict=True):
        group = group_of(work, decision)
        routed.append(RoutedDecision(group=group.key, stimulus_ids=list(group.stimulus_ids),
                                     correlation_id=group.correlation_id, decision=decision.text,
                                     route=RouteChoice.model_validate(answer.model_dump())))
    return routed


def action_dispatches(routes: list[RoutedDecision]) -> list[ActionDispatch]:
    """Return one dispatch proposal for each route that is not quiet, without external effects.

    ``content`` is the decision text, unchanged. Quiet routes stay in the
    decision record. They cannot invoke a body or public workflow, even when
    they carry a valid workflow name.
    """
    dispatches = []
    for index, item in enumerate(routes):
        answer = item.route
        if answer.action_type in QUIET_ACTIONS:
            continue
        dispatch = ActionDispatch(
            workflow_name=answer.workflow_name, action_type=answer.action_type, content=item.decision,
            arguments=answer.arguments.model_copy(deep=True), presentation=answer.presentation.model_copy(deep=True),
            group=item.group, stimulus_ids=list(item.stimulus_ids), correlation_id=item.correlation_id)
        dispatch.action_id = stable_action_id(dispatch, index)
        dispatches.append(dispatch)
    return dispatches


def body_lease(dispatch: ActionDispatch) -> str:
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
    return BODY_WORKFLOWS[target] if target in BODY_WORKFLOWS and operation == "start" else ""


def resolve_conflicts(dispatches: list[ActionDispatch], groups: list[StimulusGroup]) -> list[ActionDispatch]:
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
    acting: dict[ShardKey, str] = {}
    leases: dict[str, str] = {}
    for index in by_priority:
        dispatch = dispatches[index]
        if dispatch.status != "dispatched":
            continue
        lease = body_lease(dispatch)
        reason = ""
        if dispatch.group in acting:
            reason = f"group {dispatch.group.kind}:{dispatch.group.ref} already acts through {acting[dispatch.group]}"
        elif lease and lease in leases:
            reason = f"body lease {lease} already claimed by {leases[lease]}"
        if reason:
            resolved[index] = dispatch.model_copy(update={"status": "rejected", "rejection_reason": reason})
            continue
        acting[dispatch.group] = dispatch.action_id
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


def anticipate(context, work: CycleWork, prompts: PhasePrompts, prefix: StoryPrefix, dispatch: ActionDispatch):
    """Return the expectations of one released dispatch. Runs in a worker thread and does not change ``work``."""
    action, target, entities = anticipation_job(work, prompts, prefix, dispatch)
    return expectations.anticipate(context, work, prefix, prompts, decision=dispatch.content, action=action,
                                   target=target, count=EXPECTATION_COUNT, entity_ids=entities,
                                   embedder=Embedder(context, work.records))


def run(context, work: CycleWork) -> CycleWork:
    """Route the decisions, check conflicts, and record the expected reactions.

    No receipts of earlier actions reach the cycle yet, so every success
    prerequisite stays queued.
    """
    prompts = phase_prompts("route")
    prefix = StoryPrefix(work)
    stop = context.runtime.stop_event
    answers = ordered_parallel_map(lambda decision: route(context, work, prompts, prefix, decision),
                                   work.state.decisions, workers=DECISION_WORKERS, cancel=stop)
    work.routes = routed_decisions(work, answers)
    dispatches = resolve_prerequisites(resolve_same_cycle(action_dispatches(work.routes)), {})
    dispatches = resolve_conflicts(dispatches, work.state.stimulus_groups)
    released = [item for item in dispatches if item.status == "dispatched"]
    drafts = ordered_parallel_map(lambda item: anticipate(context, work, prompts, prefix, item), released,
                                  workers=DECISION_WORKERS, cancel=stop)
    for dispatch, expected in zip(released, drafts, strict=True):
        dispatch.expectations = expected
    work.dispatches = dispatches
    work.state.dispatches = [ActionDispatchRef(group=item.group, correlation_id=item.correlation_id,
                                               workflow=item.workflow_name, action_type=item.action_type)
                             for item in released]
    work.receipt("route", f"routed {len(work.routes)} decisions; released {len(released)} of "
                          f"{len(dispatches)} dispatches; {sum(len(item) for item in drafts)} expectations")
    return work


__all__ = [
    "BODY_WORKFLOWS",
    "action_dispatches",
    "anticipate",
    "anticipation_job",
    "body_lease",
    "build_messages",
    "group_of",
    "render_group_stimuli",
    "render_memories",
    "resolve_conflicts",
    "route",
    "routed_decisions",
    "run",
    "stable_action_id",
]
