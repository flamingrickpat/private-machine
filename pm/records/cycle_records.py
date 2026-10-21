"""Per-cycle references: stimulus groups, decisions, dispatches, and broadcasts.

The cycle state owns these models and saves them with each cycle. They hold
ids and short text only. The memory store owns the records that they name.

The routed action contract itself is ``ActionDispatch`` in
``workflows/main_cognition/schemas.py``. The runtime cannot import
workflows, so ``ActionDispatchRef`` keeps only the fields that the cycle
state needs to connect a dispatch to its group.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from pm.records.workspace import ShardKey


class StimulusGroup(BaseModel):
    """The stimuli of one cycle that belong to one entity, workflow, or the self.

    ``key`` is the shard key of the group. ``correlation_id`` connects the
    group to its routed action and the later reply. Triage sets
    ``insignificant`` and ``category``. An insignificant group enters the CSM
    but gets no decision.
    """

    key: ShardKey = Field(default_factory=ShardKey)
    stimulus_ids: list[int] = Field(default_factory=list)
    correlation_id: str = ""
    insignificant: bool = False
    category: str = ""


class ActionDecision(BaseModel):
    """The in-character decision of one cycle for one stimulus group.

    ``text`` is the chosen recommendation for the character, written as
    story. ``variants`` holds all generated variants, with ``text`` among
    them. The decision holds no workflow names or routing rules.
    """

    group: ShardKey = Field(default_factory=ShardKey)
    text: str = ""
    variants: list[str] = Field(default_factory=list)
    stimulus_ids: list[int] = Field(default_factory=list)


class ActionDispatchRef(BaseModel):
    """A reference to one routed action of the cycle.

    ``action_id`` is the record id of the stored action, or None before the
    commit assigns it. ``group`` and ``correlation_id`` come from the
    stimulus group of the decision.
    """

    action_id: int | None = None
    group: ShardKey = Field(default_factory=ShardKey)
    correlation_id: str = ""
    workflow: str = ""
    action_type: str = ""


class BroadcastRef(BaseModel):
    """One entry of the conscious contents queue: a past workspace turn.

    ``tick_id`` names the GWT turn record. ``experience_id`` is the record id
    of the subjective experience of that turn, or None.
    """

    tick_id: int = 0
    summary: str = ""
    experience_id: int | None = None
