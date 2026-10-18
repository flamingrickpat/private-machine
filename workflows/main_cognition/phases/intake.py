"""Intake phase: start the cycle from the request and the previous state.

The phase changes the previous cycle state in place. It does not rebuild it.
It decays the CSM and the codelet activations by one cycle, groups the
stimuli, creates the workspace shards of new groups, and sets the shard
tails. It makes no model call.

The tails come first because they do not depend on attention: a tail is the
newest stream of its shard. So the codelets of this cycle already read the
latest dialogue in the shared story prefix.
"""
from __future__ import annotations

from pm.constants import WORKSPACE_TAIL_TOKENS
from pm.records.cycle_records import StimulusGroup
from pm.records.cycle_state import CycleState
from pm.records.workspace import SELF_SHARD, ShardKey

from workflows.main_cognition.codelet_registry import decay_registry
from workflows.main_cognition.csm_view import record_tokens
from workflows.main_cognition.cycle_work import CycleRequest, CycleWork
from workflows.main_cognition.schemas import Stimulus
from workflows.main_cognition.workspace_selection import select_tail


def group_key(stimulus: Stimulus, request: CycleRequest) -> ShardKey:
    """Return the key of the stimulus group that owns one stimulus.

    - A message from a known entity other than the companion belongs to the
      group of that entity (``stimulus.entity_id``). A second companion is
      an entity like the user.
    - A user message belongs to the entity group of the user. A persona
      without an entity record for the user has no entity shard, so its
      messages stay in the self group.
    - A Minecraft observation belongs to the ``minecraft`` workflow group.
    - A workflow event or question belongs to the workflow group of the
      module that reported it.
    - An own thought or a system event belongs to the self group.
    """
    if stimulus.entity_id is not None:
        return ShardKey(kind="entity", ref=str(stimulus.entity_id))
    if stimulus.source == "user_input" and request.user_entity_id is not None:
        return ShardKey(kind="entity", ref=str(request.user_entity_id))
    if stimulus.kind == "minecraft.world":
        return ShardKey(kind="workflow", ref="minecraft")
    if stimulus.kind.startswith("workflow."):
        return ShardKey(kind="workflow", ref=stimulus.source)
    return SELF_SHARD


def group_stimuli(request: CycleRequest) -> list[StimulusGroup]:
    """Put every stimulus into exactly one group, in order of first arrival.

    The correlation of a group is the correlation of its newest stimulus.
    """
    groups: dict[ShardKey, StimulusGroup] = {}
    for stimulus in request.stimuli:
        key = group_key(stimulus, request)
        group = groups.setdefault(key, StimulusGroup(key=key))
        group.stimulus_ids.append(stimulus.knoxel_id)
        group.correlation_id = stimulus.correlation_id
    return list(groups.values())


def start(request: CycleRequest) -> CycleWork:
    """Create the shared state of a new cycle.

    The previous state keeps its CSM, workspace, codelet state, and
    broadcasts. The per-cycle fields (groups, decisions, dispatches, stimuli,
    qualia channels) start empty for this cycle. The host sets the tick at
    commit.
    """
    # Copy, so the request keeps the previous state for the appraise phase.
    previous = request.previous_state
    state = previous.model_copy(deep=True) if previous is not None else CycleState(tick_id=0)
    state.csm.decay_step()
    decay_registry(state.codelets)
    state.stimulus_groups = group_stimuli(request)
    for group in state.stimulus_groups:
        state.workspace.ensure_shard(group.key)
    state.decisions = []
    state.dispatches = []
    state.stimuli_ids = [item.knoxel_id for item in request.stimuli]
    state.qualia_channels = {}
    work = CycleWork(request=request, state=state, state_summary=request.state_summary,
                     records=dict(request.records))
    set_tails(work)
    work.receipt("intake", f"{len(request.stimuli)} stimuli in {len(state.stimulus_groups)} groups")
    return work


def set_tails(work: CycleWork) -> None:
    """Set the tail of each shard from the streams of the request.

    Each tail is the newest stretch of its stream within
    ``WORKSPACE_TAIL_TOKENS``. The current stimuli are not part of a tail,
    because the prompt shows them as new. A stream for a shard that does not
    exist yet creates the shard.
    """
    exclude = {item.knoxel_id for item in work.request.stimuli}
    for stream in work.request.streams:
        shard = work.state.workspace.ensure_shard(stream.key)
        shard.tail_ids = select_tail(stream.record_ids, lambda record_id: record_tokens(work.records, record_id),
                                     WORKSPACE_TAIL_TOKENS, exclude)


__all__ = ["group_key", "group_stimuli", "set_tails", "start"]
