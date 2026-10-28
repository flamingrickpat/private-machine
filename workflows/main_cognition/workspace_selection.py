"""Workspace mechanics: requirement matching, CSM protection, tails, and shard selection.

This module is pure Python. It makes no model calls and does no I/O. The
attention phase supplies the ratings, the CSM activations, and the token
counts. The rules come from the A19F plan, section "Global workspace and
shards":

- ``min_items`` reserves places that other content cannot take. Attention
  decides which record fills a reserved place: the best rated matching
  record of the structures, or else the best matching CSM item.
- ``max_items`` and ``max_token_share`` cap the matching content of a shard.
- An incumbent item stays while its priority plus ``WORKSPACE_HOLD_BONUS``
  beats the newcomers for its place.
- Deliberate focus fills at most ``FOCUS_MAX_SHARE`` of a shard budget. It
  cannot fill a reserved place and cannot remove a reserved item.
- Every entered and left item is recorded with its tick and reason.

One record is in at most one shard. A record that a structure brings to two
shards goes to the shard where its priority is highest.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable

from pydantic import BaseModel, Field

from pm.constants import (
    FOCUS_MAX_SHARE,
    WORKSPACE_HOLD_BONUS,
    WORKSPACE_MIN_RATING,
    WORKSPACE_SHARD_TOKENS,
)
from pm.records.global_workspace import LeftItem, ShardChange
from pm.records.situational_model import SituationalModel
from pm.records.workspace import (
    RecordSelector,
    ShardKey,
    ShardRequirementSet,
    WorkspaceItem,
    WorkspaceRequirement,
    WorkspaceShard,
)

from workflows.main_cognition.schemas import RecordView

#: Returns the token count of the rendered story element of a record id.
TokensOf = Callable[[int], int]


class MatchContext(BaseModel):
    """The cycle facts that the record selectors need.

    ``self_entity_id`` is the entity record of the companion, or None.
    ``group_entity_ids`` are the entities of the current stimulus groups.
    ``stimulus_ids`` are the stimuli that the current cycle claimed.
    """

    self_entity_id: int | None = None
    group_entity_ids: list[int] = Field(default_factory=list)
    stimulus_ids: list[int] = Field(default_factory=list)


def shard_entity(key: ShardKey) -> int | None:
    """Return the entity id of an entity shard, or None for the self and workflow shards."""
    return int(key.ref) if key.kind == "entity" else None


def selector_matches(selector: RecordSelector, view: RecordView, key: ShardKey, context: MatchContext) -> bool:
    """Return True when the record of ``view`` matches every condition of ``selector``.

    ``key`` is the shard that the requirement belongs to. The ``shard``
    entity condition needs an entity shard. The CSM requirements pass the
    self shard key, so ``entity: shard`` never matches there.
    """
    if selector.record and view.record_class not in selector.record:
        return False
    if selector.feature_type and view.record_class == "Feature" and view.feature_type != selector.feature_type:
        return False
    if selector.category and view.category != selector.category:
        return False
    if selector.entity == "self" and context.self_entity_id not in view.entity_ids:
        return False
    if selector.entity == "shard" and shard_entity(key) not in view.entity_ids:
        return False
    if selector.entity == "group" and not set(view.entity_ids) & set(context.group_entity_ids):
        return False
    if selector.scope == "current_stimuli" and view.record_id not in context.stimulus_ids:
        return False
    return True


def protected_ids(csm: SituationalModel, requirements: ShardRequirementSet, records: dict[int, RecordView],
                  context: MatchContext) -> set[int]:
    """Return the CSM record ids that prune must keep.

    An item is protected when its record or one of its merged records
    matches a CSM requirement. A requirement with ``max_items`` protects
    only that number of matching items, the most active first. Merged
    records without a view are skipped: the host supplies views only for
    the records that the CSM holds as keys.
    """
    protected: set[int] = set()
    for requirement in requirements.requirements:
        matching = [item for item in csm.items.values()
                    if any(record_id in records
                           and selector_matches(requirement.select, records[record_id], ShardKey(), context)
                           for record_id in [item.record_id, *item.merged_ids])]
        matching.sort(key=lambda item: (-item.activation, item.record_id))
        if requirement.max_items is not None:
            matching = matching[:requirement.max_items]
        protected.update(item.record_id for item in matching)
    return protected


def select_tail(stream_ids: Iterable[int], tokens_of: TokensOf, budget: int, exclude: set[int]) -> list[int]:
    """Return the newest records of a stream that fit ``budget`` tokens, oldest first.

    ``stream_ids`` is oldest first. The selection walks from the newest
    record backward and stops at the first record that does not fit, so the
    tail is one unbroken stretch of the stream. Records in ``exclude`` (the
    current stimuli) are skipped: the prompt shows them as new.
    """
    selected: list[int] = []
    used = 0
    for record_id in reversed(list(dict.fromkeys(stream_ids))):
        if record_id in exclude:
            continue
        tokens = tokens_of(record_id)
        if used + tokens > budget:
            break
        selected.append(record_id)
        used += tokens
    return list(reversed(selected))


class Candidate(BaseModel):
    """One record that can fill a place in one shard.

    ``priority`` is the attention score of the best structure that brings
    the record to the shard. For an incumbent it is the larger of that score
    and the CSM activation, plus ``WORKSPACE_HOLD_BONUS``. ``focus_bonus``
    is the bounded change from deliberate focus. It is positive for a
    promoted record and negative for a reduced one. Reserved places ignore
    it. ``incumbent`` is True for an item that was in the shard before.
    """

    record_id: int = 0
    key: ShardKey = Field(default_factory=ShardKey)
    priority: float = 0.0
    focus_bonus: float = 0.0
    structure_id: int | None = None
    reason: str = ""
    incumbent: bool = False

    def total(self) -> float:
        """Return the priority with the focus bonus."""
        return self.priority + self.focus_bonus


class SelectionInput(BaseModel):
    """Everything that one workspace update reads.

    ``shards`` are the shards that attention updates in this cycle, with
    their previous items. ``newcomers`` hold one candidate for each record
    and shard that a rated structure offers. ``activation`` is the CSM
    activation by record id. ``tokens`` is the token count of each record
    that can enter. ``blocked`` holds records that cannot become items: the
    current stimuli, the tail records, and the items of the shards that this
    cycle does not update. ``focus_ids`` are the records with a deliberate
    focus endorsement.
    """

    tick: int = 0
    shards: list[WorkspaceShard] = Field(default_factory=list)
    requirement_sets: dict[str, ShardRequirementSet] = Field(default_factory=dict)
    newcomers: list[Candidate] = Field(default_factory=list)
    activation: dict[int, float] = Field(default_factory=dict)
    tokens: dict[int, int] = Field(default_factory=dict)
    blocked: list[int] = Field(default_factory=list)
    focus_ids: list[int] = Field(default_factory=list)


#: Returns True when one record matches one requirement of one shard.
Matcher = Callable[[WorkspaceRequirement, int, ShardKey], bool]
#: Returns True when a shard may hold a record (the home shard rule of the attention phase).
Admits = Callable[[ShardKey, int], bool]


class _ShardFill:
    """The content of one shard while the selection runs."""

    def __init__(self, shard: WorkspaceShard, requirements: list[WorkspaceRequirement]) -> None:
        self.shard = shard
        self.requirements = requirements
        self.items: list[WorkspaceItem] = []
        self.tokens = 0
        self.focus_tokens = 0
        self.previous = {item.record_id: item for item in shard.items}

    def refusal(self, record_id: int, tokens: int, focus: bool, matcher: Matcher) -> str:
        """Return why the record cannot enter, or an empty text when it can."""
        if self.tokens + tokens > WORKSPACE_SHARD_TOKENS:
            return f"budget: {self.tokens} of {WORKSPACE_SHARD_TOKENS} tokens used, the record needs {tokens}"
        if focus and self.focus_tokens + tokens > FOCUS_MAX_SHARE * WORKSPACE_SHARD_TOKENS:
            return f"focus share: deliberate focus already fills {self.focus_tokens} tokens"
        for requirement in self.requirements:
            if not matcher(requirement, record_id, self.shard.key):
                continue
            members = [item for item in self.items if matcher(requirement, item.record_id, self.shard.key)]
            if requirement.max_items is not None and len(members) >= requirement.max_items:
                return f"cap {requirement.name}: {requirement.max_items} items"
            if requirement.max_token_share is not None:
                limit = requirement.max_token_share * WORKSPACE_SHARD_TOKENS
                if sum(item.tokens for item in members) + tokens > limit:
                    return f"cap {requirement.name}: {requirement.max_token_share:g} of the budget"
        return ""

    def place(self, candidate: Candidate, *, tokens: int, activation: float, tick: int, required_by: str,
              focus: bool) -> None:
        """Add the record. An incumbent keeps its entry tick and its first reason."""
        if candidate.record_id in self.previous:
            item = self.previous[candidate.record_id].model_copy(update={"tokens": tokens, "required_by": required_by})
        else:
            item = WorkspaceItem(record_id=candidate.record_id, entered_tick=tick, activation_at_entry=activation,
                                 structure_id=candidate.structure_id, reason=candidate.reason, tokens=tokens,
                                 required_by=required_by)
        self.items.append(item)
        self.tokens += tokens
        if focus:
            self.focus_tokens += tokens


def select_shards(selection: SelectionInput, matcher: Matcher, admits: Admits | None = None) -> list[ShardChange]:
    """Update the shards of ``selection`` in place and return their changes.

    ``admits`` decides whether a shard may hold a record. A newcomer or a
    reserved CSM item that a shard does not admit cannot enter it. An
    incumbent that its shard no longer admits leaves it. Without
    ``admits``, every shard admits every record.

    Steps:

    1. Pool: the newcomers and the incumbents of each shard. An incumbent's
       priority is the larger of its CSM activation and its newcomer
       priority, plus ``WORKSPACE_HOLD_BONUS``.
    2. Reserved places: each ``min_items`` requirement takes the matching
       pool records with the highest priority, without the focus bonus. If
       the pool has none, it takes the matching CSM items with the highest
       activation.
    3. Fill: all remaining pool entries in priority order. An entry below
       ``WORKSPACE_MIN_RATING`` does not enter. The budget, the caps, and the
       focus share can refuse an entry.
    4. Changes: entered items, left items with reasons, kept ids. The caller
       sets the tail ids.

    Ties keep the shard order of ``selection.shards`` and then the record
    id order, so the result is repeatable.
    """
    blocked = set(selection.blocked)
    focus_ids = set(selection.focus_ids)
    activation = selection.activation
    fills = [_ShardFill(shard, selection.requirement_sets[shard.requirement_set].requirements
                        if shard.requirement_set in selection.requirement_sets else [])
             for shard in selection.shards]
    by_key = {fill.shard.key: fill for fill in fills}
    order = {fill.shard.key: index for index, fill in enumerate(fills)}

    def active(record_id: int) -> float:
        return activation[record_id] if record_id in activation else 0.0

    def allowed(key: ShardKey, record_id: int) -> bool:
        return admits is None or admits(key, record_id)

    pools: dict[ShardKey, dict[int, Candidate]] = {fill.shard.key: {} for fill in fills}
    for candidate in selection.newcomers:
        if candidate.record_id in blocked or candidate.key not in pools:
            continue
        pool = pools[candidate.key]
        if candidate.record_id not in pool or candidate.total() > pool[candidate.record_id].total():
            pool[candidate.record_id] = candidate
    left_reasons: dict[ShardKey, dict[int, str]] = {fill.shard.key: {} for fill in fills}
    for fill in fills:
        pool = pools[fill.shard.key]
        for item in fill.shard.items:
            if item.record_id in blocked:
                left_reasons[fill.shard.key][item.record_id] = "now in a tail or a current stimulus"
                continue
            if not allowed(fill.shard.key, item.record_id):
                left_reasons[fill.shard.key][item.record_id] = "belongs to another shard"
                continue
            offered = pool[item.record_id] if item.record_id in pool else None
            base = max(active(item.record_id), offered.priority if offered is not None else 0.0)
            pool[item.record_id] = Candidate(
                record_id=item.record_id, key=fill.shard.key, priority=base + WORKSPACE_HOLD_BONUS,
                focus_bonus=offered.focus_bonus if offered is not None else 0.0,
                structure_id=item.structure_id, reason=item.reason, incumbent=True)

    placed: dict[int, ShardKey] = {}

    def tokens_of(record_id: int) -> int:
        return selection.tokens[record_id]

    # Step 2: reserved places, before any other content.
    for fill in fills:
        for requirement in fill.requirements:
            if requirement.min_items <= 0:
                continue
            options = sorted((candidate for candidate in pools[fill.shard.key].values()
                              if candidate.record_id not in placed
                              and matcher(requirement, candidate.record_id, fill.shard.key)),
                             key=lambda candidate: (-candidate.priority, candidate.record_id))
            if not options:
                options = [Candidate(record_id=record_id, key=fill.shard.key, priority=active(record_id),
                                     reason=f"reserved for {requirement.name}: the best matching CSM item")
                           for record_id in sorted(activation, key=lambda record_id: (-activation[record_id], record_id))
                           if record_id not in placed and record_id not in blocked and record_id in selection.tokens
                           and allowed(fill.shard.key, record_id) and matcher(requirement, record_id, fill.shard.key)]
            taken = 0
            for candidate in options:
                if taken >= requirement.min_items:
                    break
                tokens = tokens_of(candidate.record_id)
                if fill.tokens + tokens > WORKSPACE_SHARD_TOKENS:
                    continue
                fill.place(candidate, tokens=tokens, activation=active(candidate.record_id), tick=selection.tick,
                           required_by=requirement.name, focus=False)
                placed[candidate.record_id] = fill.shard.key
                taken += 1

    # Step 3: everything else in priority order.
    entries = sorted((candidate for pool in pools.values() for candidate in pool.values()),
                     key=lambda candidate: (-candidate.total(), order[candidate.key], candidate.record_id))
    for candidate in entries:
        fill = by_key[candidate.key]
        reasons = left_reasons[candidate.key]
        if candidate.record_id in placed:
            if placed[candidate.record_id] != candidate.key:
                reasons.setdefault(candidate.record_id, "moved to another shard with a higher rating")
            continue
        if candidate.total() < WORKSPACE_MIN_RATING:
            reasons.setdefault(candidate.record_id,
                               f"faded: priority {candidate.total():.2f} is below {WORKSPACE_MIN_RATING:g}")
            continue
        tokens = tokens_of(candidate.record_id)
        focus = candidate.record_id in focus_ids
        refusal = fill.refusal(candidate.record_id, tokens, focus, matcher)
        if refusal:
            reasons.setdefault(candidate.record_id, f"displaced by stronger content ({refusal})")
            continue
        fill.place(candidate, tokens=tokens, activation=active(candidate.record_id), tick=selection.tick,
                   required_by="", focus=focus)
        placed[candidate.record_id] = candidate.key

    # Step 4: changes, then the new content of each shard.
    changes: list[ShardChange] = []
    for fill in fills:
        now = {item.record_id for item in fill.items}
        reasons = left_reasons[fill.shard.key]
        change = ShardChange(key=fill.shard.key)
        change.entered = [item for item in fill.items if item.record_id not in fill.previous]
        change.kept_ids = [item.record_id for item in fill.items if item.record_id in fill.previous]
        change.left = [LeftItem(record_id=record_id,
                                reason=reasons[record_id] if record_id in reasons else "left the situational model")
                       for record_id in fill.previous if record_id not in now]
        fill.shard.items = fill.items
        changes.append(change)
    return changes


__all__ = [
    "Admits",
    "Candidate",
    "MatchContext",
    "Matcher",
    "SelectionInput",
    "protected_ids",
    "select_shards",
    "select_tail",
    "selector_matches",
    "shard_entity",
]
