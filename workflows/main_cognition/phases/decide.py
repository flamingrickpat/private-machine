"""Decide phase: one in-character decision for each significant stimulus group.

A19F step 8 and decision D28. The phase runs one story-framed call for each
stimulus group that triage did not mark insignificant. The calls run in
parallel. Each call reads the story prefix of its own group (the self shard
and the group's shard after attention, the group's subjective experience in
the self tail, the group's new events) and then the group instruction: the group's stimuli, the feeling, the subjective experience,
the causal codelet percepts in the self shard and the group shard, and the
goals.

Each call returns zero or ``DECISION_VARIANTS`` decision texts. A decision
is a recommendation to the character in the story, in the character's voice
and attitude, for example "He has dodged choosing four times. Call him a
coward for it, then admit that you like the story unfinished too." It holds
no routing rules, no capability lists, and no workflow names. A text that
names a workflow or opens like a planner ticket fails validation and gets a
repair turn (``DecisionAnswer``). The route phase maps each decision to an
action.

An insignificant group gets no call and no decision. A group whose call
returns an empty list gets no decision: the character does nothing about it.
"""
from __future__ import annotations

from pm.constants import DECISION_VARIANTS
from pm.context_render.turns import Turn, format_content
from pm.llm.concurrency import ordered_parallel_map
from pm.records.cycle_records import ActionDecision, StimulusGroup
from pm.records.workspace import SELF_SHARD

from workflows.inference_settings import DECISION_WORKERS
from workflows.main_cognition.csm_view import render_story
from workflows.main_cognition.cycle_work import CycleWork
from workflows.main_cognition.model_json import PhasePrompts
from workflows.main_cognition.phases.model import complete, phase_prompts
from workflows.main_cognition.schemas import DecisionAnswer, DecisionCheck, WorkflowOption
from workflows.main_cognition.story_prefix import StoryPrefix, shard_label

#: The action space when the configuration names no workflows.
DEFAULT_WORKFLOWS = [
    WorkflowOption(name="dialogue_story", description="Speak in character, in the story voice."),
    WorkflowOption(name="self_management", description="Start, check, update, or stop a managed run."),
    WorkflowOption(name="local_task", description="Run one small real local task."),
    WorkflowOption(name="minecraft", description="Run a sustained Minecraft specialist with exclusive body ownership."),
]

#: Workflows that stay wired but are not offered as actions.
#: Assistant dialogue is disabled as a user-facing route. It still serves
#: task reply recovery, so remove a name here to offer it again.
DISABLED_WORKFLOWS = frozenset({"dialogue_assistant"})


def available_workflows(work: CycleWork) -> list[WorkflowOption]:
    """Return the configured workflows, or the default action space, without the disabled workflows."""
    workflows = work.request.available_workflows or DEFAULT_WORKFLOWS
    return [item for item in workflows if item.name not in DISABLED_WORKFLOWS]


def group_part(work: CycleWork, prompts: PhasePrompts, prefix: StoryPrefix, group: StimulusGroup) -> str:
    """Return the story words for the part of the story that a group is: an entity, a workflow, or the self."""
    if group.key.kind == "entity":
        return prompts.part_entity.format(name=shard_label(group.key, work.request))
    if group.key.kind == "workflow":
        return prompts.part_workflow.format(name=group.key.ref)
    return prefix.fill(prompts.part_self)


def group_stimuli(work: CycleWork, group: StimulusGroup) -> str:
    """Return the story lines of the group's stimuli, in world-time order."""
    return render_story(work.records, group.stimulus_ids, with_id=False) or "(nothing new)"


def group_percepts(work: CycleWork, group: StimulusGroup) -> str:
    """Return the conscious codelet percepts of the self shard and the group shard as story lines.

    A percept in a shard is causal: it entered the story. The decision draws
    on these urges.
    """
    workspace = work.state.workspace
    keys = list(dict.fromkeys([SELF_SHARD, group.key]))
    ids = [item.record_id for key in keys for item in workspace.shard(key).items
           if work.records[item.record_id].record_class == "CodeletPercept"]
    # One bullet for each percept, in world-time order. A percept text can
    # span several lines; they stay in its bullet.
    lines = [" ".join(render_story(work.records, [record_id], with_id=False).split())
             for record_id in sorted(ids, key=lambda key: (work.records[key].timestamp is None,
                                                           work.records[key].timestamp or work.request.cycle_time))]
    return "\n".join(f"- {line}" for line in lines) or "(none)"


def render_goals(work: CycleWork) -> str:
    """Return the active goals and the goals that this cycle adds, one per line."""
    added = [change.goal for change in work.goals.changes if change.change == "add"]
    goals = list(dict.fromkeys([*work.request.active_goals, *added]))
    return "\n".join(f"- {goal}" for goal in goals) or "(nothing in particular)"


def build_messages(work: CycleWork, prompts: PhasePrompts, prefix: StoryPrefix, group: StimulusGroup) -> list[Turn]:
    """Return the shared story prefix and then the decision instruction for one group as the last turn."""
    instruction = format_content(prefix.fill(prompts.user_template),
        part=group_part(work, prompts, prefix, group),
        stimuli=group_stimuli(work, group),
        affect=work.affect.felt or work.state_summary or "(no feeling recorded)",
        regulation_words=prompts.regulation_words[work.affect.regulation],
        experience=work.experience.text_for(group.key) or "(nothing)",
        impulses=group_percepts(work, group),
        goals=render_goals(work),
        variants=DECISION_VARIANTS,
    ).text
    return prefix.ask(instruction)


def decide(context, work: CycleWork, prompts: PhasePrompts, prefix: StoryPrefix,
           group: StimulusGroup) -> DecisionAnswer:
    """Run the decision call of one group. Runs in a worker thread and does not change ``work``."""
    check = DecisionCheck(workflow_names=[item.name for item in available_workflows(work)])
    return complete(context, work, build_messages(work, prompts, prefix, group), DecisionAnswer,
                    validation_context=check.model_dump())


def run(context, work: CycleWork) -> CycleWork:
    prompts = phase_prompts("decide")
    groups = [group for group in work.state.stimulus_groups if not group.insignificant]
    # Each group reads only the self shard and its own shard. The content of
    # another group must not leak into this group's decision.
    prefixes = {group.key: StoryPrefix(work, group=group.key) for group in groups}
    answers = ordered_parallel_map(lambda group: decide(context, work, prompts, prefixes[group.key], group), groups,
                                   workers=DECISION_WORKERS, cancel=context.runtime.stop_event)
    work.state.decisions = [
        ActionDecision(group=group.key, text=answer.variants[0], variants=list(answer.variants),
                       stimulus_ids=list(group.stimulus_ids))
        for group, answer in zip(groups, answers, strict=True) if answer.variants]
    skipped = len(work.state.stimulus_groups) - len(groups)
    work.receipt("decide", f"{len(work.state.decisions)} decisions for {len(groups)} significant groups; "
                           f"{skipped} insignificant groups skipped")
    return work


__all__ = [
    "DEFAULT_WORKFLOWS",
    "DISABLED_WORKFLOWS",
    "available_workflows",
    "build_messages",
    "decide",
    "group_part",
    "group_percepts",
    "group_stimuli",
    "render_goals",
    "run",
]
