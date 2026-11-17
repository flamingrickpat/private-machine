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

A nonempty proposal passes one focused inhibition judgment over the current
events and recorded situation. This compares purposes and interaction pressure,
not wording. A blocked proposal becomes quiet for this cycle; it is never
resampled into another version of the same move. The judgment and its reason
are retained with the phase receipt and in kernel scratch.

An insignificant group gets no call and no decision. A group whose call
returns an empty list gets no decision: the character does nothing about it.
"""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256

from pydantic import BaseModel

from workflows.main_cognition.context_reduction import evidence_text

from pm_workflows.python_role import RoleContext

from pm.constants import DECISION_VARIANTS
from pm.context_render import render_context
from pm.context_render.turns import Role, Turn, format_content
from pm.llm.concurrency import ordered_parallel_map
from pm.records.cycle_records import ActionDecision, StimulusGroup
from pm.records.workspace import SELF_SHARD, ShardKey

from workflows.inference_settings import DECISION_WORKERS
from workflows.main_cognition.csm_view import render_story
from workflows.main_cognition.cycle_work import CycleWork
from workflows.main_cognition.model_json import PhasePrompts
from workflows.main_cognition.phases.model import complete, phase_prompts
from workflows.main_cognition.schemas import DecisionAnswer, DecisionCheck, DecisionInhibition, WorkflowOption
from workflows.main_cognition.story_prefix import StoryPrefix, shard_label
from workflows.public_speech.situational_guidance import render_situational_guidance

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


class DecisionInhibitionReceipt(BaseModel):
    """Retained proposal and judgment for one group's decision boundary."""

    group: ShardKey
    stimulus_ids: list[int]
    proposal: DecisionAnswer
    judgment: DecisionInhibition


@dataclass(frozen=True, slots=True)
class DecisionSelection:
    """A proposed character move and its inhibition judgment, absent for an empty proposal."""

    proposal: DecisionAnswer
    inhibition: DecisionInhibition | None

    @property
    def accepted(self) -> DecisionAnswer:
        """Return the original proposal or quiet; inhibition never rewrites an action."""
        if self.inhibition is not None and not self.inhibition.allow:
            return DecisionAnswer()
        return self.proposal


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
        stimuli=evidence_text(group_stimuli(work, group), "decide-stimuli", group.stimulus_ids),
        affect=work.affect.felt or work.state_summary or "(no feeling recorded)",
        regulation_words=prompts.regulation_words[work.affect.regulation],
        experience=evidence_text(work.experience.text_for(group.key) or "(nothing)", "decide-experience"),
        impulses=evidence_text(group_percepts(work, group), "decide-impulses"),
        goals=evidence_text(render_goals(work), "decide-goals"),
        variants=DECISION_VARIANTS,
    )
    return prefix.ask(instruction)


def inhibit_decision(context: RoleContext, work: CycleWork, proposal: DecisionAnswer,
                     group: StimulusGroup, prompts: PhasePrompts | None = None) -> DecisionInhibition:
    """Judge one proposal against delivered moves, without a replacement or retry on inhibition.

    The compact prompt keeps the group's attributed situation and new events,
    rather than recopying its whole workspace. Failed inference or malformed
    JSON follows the existing model boundary's repair/cancellation policy. A
    valid ``allow=False`` is an accepted outcome, never a model-output failure.
    """
    if not proposal.variants:
        raise ValueError("Action inhibition requires a nonempty proposal.")
    selected_prompts = prompts or phase_prompts("decide")
    target_id = int(group.key.ref) if group.key.kind == "entity" else work.request.user_entity_id
    situations = [situation for situation in work.request.situational_guidance
                  if situation.partner is not None and situation.partner.entity_id == target_id]
    situation_text = "\n\n".join(render_situational_guidance(situation) for situation in situations)
    instruction = format_content(selected_prompts.check_template,
        companion_name=work.request.companion_name,
        character=evidence_text(work.request.character_card or "(character not supplied)", "inhibit-character"),
        situation=evidence_text(situation_text or "Partner availability and earlier delivery are unknown.",
                                "inhibit-situation"),
        stimuli=evidence_text(render_story(work.records, group.stimulus_ids, with_id=True) or "(nothing new)",
                              "inhibit-stimuli", group.stimulus_ids),
        proposals=evidence_text("\n".join(proposal.variants), "inhibit-proposals"),
        managed_runs=evidence_text(render_context(work.request.managed_runs) if work.request.managed_runs else "(none)",
                                  "inhibit-runs"),
    )
    messages = [Turn(Role.System, "Judge whether to inhibit a proposed action using attributed current evidence. "
                     "Do not write a new action, a reply, or a story continuation."),
                Turn(Role.User, instruction)]
    result = complete(context, work, messages, DecisionInhibition)
    assert context.runtime is not None
    group_identity = sha256(group.key.model_dump_json().encode("utf-8")).hexdigest()[:16]
    receipt_name = f"action-inhibition-{group_identity}.json"
    receipt = DecisionInhibitionReceipt(group=group.key, stimulus_ids=list(group.stimulus_ids),
                                        proposal=proposal, judgment=result)
    receipt_text = receipt.model_dump_json(indent=2)
    context.runtime.environment.write_text(f"scratch/{receipt_name}", receipt_text)
    receipt_path = context.kernel_data / receipt_name
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(receipt_text, encoding="utf-8")
    return result


def decide(context: RoleContext, work: CycleWork, prompts: PhasePrompts, prefix: StoryPrefix,
           group: StimulusGroup) -> DecisionSelection:
    """Propose and inhibit one group's move in a worker, without changing ``work``."""
    check = DecisionCheck(workflow_names=[item.name for item in available_workflows(work)])
    proposal = complete(context, work, build_messages(work, prompts, prefix, group), DecisionAnswer,
                        validation_context=check.model_dump())
    inhibition = inhibit_decision(context, work, proposal, group, prompts) if proposal.variants else None
    return DecisionSelection(proposal=proposal, inhibition=inhibition)


def run(context: RoleContext, work: CycleWork) -> CycleWork:
    prompts = phase_prompts("decide")
    interaction_sources = set(work.request.interaction_source_ids)
    groups: list[StimulusGroup] = []
    for group in work.state.stimulus_groups:
        remaining_ids = [source_id for source_id in group.stimulus_ids if source_id not in interaction_sources]
        if not group.insignificant and remaining_ids:
            groups.append(group.model_copy(update={"stimulus_ids": remaining_ids}))
    # Each group reads only the self shard and its own shard. The content of
    # another group must not leak into this group's decision.
    prefixes = {group.key: StoryPrefix(work, group=group.key) for group in groups}
    selections = ordered_parallel_map(lambda group: decide(context, work, prompts, prefixes[group.key], group), groups,
                                      workers=DECISION_WORKERS, cancel=context.runtime.stop_event)
    answers = [selection.accepted for selection in selections]
    work.state.decisions = [
        ActionDecision(group=group.key, text=answer.variants[0], variants=list(answer.variants),
                       stimulus_ids=list(group.stimulus_ids))
        for group, answer in zip(groups, answers, strict=True) if answer.variants]
    for group, selection in zip(groups, selections, strict=True):
        if selection.inhibition is not None:
            result = selection.inhibition
            work.receipt("action_inhibition", f"{group.key.kind}:{group.key.ref}: "
                         f"{'allowed' if result.allow else 'inhibited'}: {result.reason}")
    skipped = len(work.state.stimulus_groups) - len(groups)
    work.receipt("decide", f"{len(work.state.decisions)} decisions for {len(groups)} significant groups; "
                           f"{skipped} insignificant groups skipped")
    return work


__all__ = [
    "DEFAULT_WORKFLOWS",
    "DISABLED_WORKFLOWS",
    "DecisionInhibition",
    "DecisionSelection",
    "available_workflows",
    "build_messages",
    "decide",
    "group_part",
    "group_percepts",
    "group_stimuli",
    "inhibit_decision",
    "render_goals",
    "run",
]
