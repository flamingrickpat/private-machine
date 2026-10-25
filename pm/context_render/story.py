"""Render one memory record, and one workspace shard, as story text.

This module owns the words. Cognition shows the CSM and the workspace shards
to the model as story lines, and the memory inspector shows the same lines to
the maintainer. Both call the functions here, so the inspector shows exactly
what the model reads. Nothing in this module builds a prompt on its own.

The pieces:

- :func:`record_view` turns one memory store record into a :class:`RecordView`.
- :func:`story_line` renders one view, and :func:`render_story` renders a list
  of record ids in world-time order.
- :func:`story_tokens` and :func:`record_tokens` count the tokens of that text.
  These are the counts that the shard and tail budgets spend.
- :func:`shard_name`, :func:`shard_titles`, and :func:`shard_block_text`
  render one shard: its name, its title, its items, and its tail.

The layout lives here and the words live in the prompt file
``workflows/main_cognition/phases/prompts/story.yaml``. The cycle passes the
template and the labels of its loaded prompts into
:func:`shard_block_text`. The inspector reads the same YAML file. See
``pm/tools/inspector/rendering.py``.
"""
from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from pm.context_render.tokens import text_tokens
from pm.context_render.turns import format_content
from pm.records.base import persisted_type_name
from pm.records.enums import FeatureType
from pm.records.workspace import SELF_SHARD, GlobalWorkspace, ShardKey, WorkspaceShard


class RecordView(BaseModel):
    """One record that the CSM can hold, as the codelet prompts render it.

    The host builds a view from the memory store for every record that the
    CSM of the previous cycle holds and for every stimulus, dialogue turn, and
    memory candidate of this cycle. The codelets phase adds views for new
    percepts. ``kind`` is the persisted record type or the memory kind.
    ``source`` is the speaker of a dialogue turn or the codelet of a percept.
    ``timestamp`` is the world time (naive UTC), or None when unknown.

    The workspace requirements read the last four fields
    (``workflows/main_cognition/workspace.yaml``). ``record_class`` is the
    Python record class name, for example ``Narrative``. ``entity_ids`` are
    the entities that the record is about. ``category`` is the narrative
    type, the claim kind, or the belief kind. ``feature_type`` is the
    ``FeatureType`` value of a ``Feature`` record and empty otherwise.
    ``addressee_entity_id`` is the listener of a dialogue line (0 for the
    user or unknown). Attention uses it to find the shard of a companion
    line.
    """

    record_id: int = 0
    kind: str = ""
    source: str = ""
    content: str = ""
    timestamp: datetime | None = None
    record_class: str = ""
    entity_ids: list[int] = Field(default_factory=list)
    category: str = ""
    feature_type: str = ""
    addressee_entity_id: int = 0
    embedding: list[float] = Field(default_factory=list, repr=False)

    def story_line(self) -> str:
        """Return the record as one line of story text.

        The codelets phase also embeds this text, so it never contains the
        record id or the time. The same record therefore gets the same
        embedding in every cycle.
        """
        return f"{self.source or self.kind}: {self.content}"


class ShardLabels(BaseModel):
    """The words that the shared story prefix uses for the workspace shards.

    The texts can hold ``{companion_name}``, ``{entity_name}``, and
    ``{workflow_name}``.
    """

    model_config = ConfigDict(extra="forbid")

    self_title: str = ""
    self_tail: str = ""
    entity_title: str = ""
    entity_tail: str = ""
    workflow_title: str = ""
    workflow_tail: str = ""
    empty_items: str = ""
    empty_tail: str = ""
    # A public reply shows the verbatim lines of a tail in its transcript.
    verbatim_tail: str = ""
    # The attention phase: what the awareness of each shard kind is about.
    self_focus: str = ""
    entity_focus: str = ""
    workflow_focus: str = ""


def _entity_facts(record, record_class: str) -> tuple[list[int], str]:
    """Return the entity ids that a record is about and its category label.

    The record classes name their entity in different fields. A class that
    names no entity returns an empty list. Entity id 0 means "unknown" in
    every class, so it never counts.
    """
    if record_class == "Narrative":
        return [record.entity_id] if record.entity_id else [], record.narrative_type.value
    if record_class in {"SemanticRecord", "DeclarativeFactRecord"}:
        return [record.subject_entity_id] if record.subject_entity_id else [], record.record_type.value
    if record_class == "BeliefRecord":
        return list(record.about_entity_ids), record.belief_kind
    if record_class in {"Feature", "Stimulus", "Intention"}:
        return [record.entity_id] if record.entity_id else [], ""
    if record_class == "ExpectationRecord":
        return list(record.about_entity_ids), record.status
    return [], ""


def record_view(record) -> RecordView:
    """Return the view of one memory store record. The host and the agent tool call it.

    A dialogue turn names its speaker. Another feature names its source and
    its feature type, so a private thought never reads like speech. A
    codelet percept names its codelet. A narrative names its type and the
    entity it is about. An expectation names its status. Other records name
    their type. The view also holds
    the facts that the workspace requirements match (see ``RecordView``).

    ``record`` is one record of the memory store. The inspector calls this
    function with a record of a loaded checkpoint, so a rendered line is the
    line that the cycle of that checkpoint showed.
    """
    kind = persisted_type_name(type(record))
    record_class = type(record).__name__
    entity_ids, category = _entity_facts(record, record_class)
    source = ""
    feature_type = ""
    addressee = 0
    if record_class == "Feature":
        feature_type = record.feature_type.value
        addressee = record.addressee_entity_id
        source = record.source if record.feature_type == FeatureType.Dialogue else f"{record.source} ({feature_type})"
    elif record_class == "CodeletPercept":
        source = record.codelet_name
    elif record_class == "Narrative":
        source = f"narrative ({category}) about {record.target_name}"
    elif record_class == "ExpectationRecord":
        source = f"expectation ({category})"
    return RecordView(record_id=record.id, kind=kind, source=source, content=record.content,
                      timestamp=record.timestamp_world_begin, record_class=record_class,
                      entity_ids=entity_ids, category=category, feature_type=feature_type,
                      addressee_entity_id=addressee, embedding=record.embedding)


def story_tokens(text: str) -> int:
    """Return the estimated token count of one story text, without chat framing.

    This is the plain-text count of :func:`pm.context_render.tokens.text_tokens`.
    The old form measured a one-turn message list and took away the cost of an
    empty turn, which leaves the text alone.
    """
    return text_tokens(text)


def record_tokens(records: dict[int, RecordView], record_id: int) -> int:
    """Return the token count of the story line of one record, without its id.

    The workspace budgets and the tails count tokens with this function.
    """
    return story_tokens(story_line(records[record_id], with_id=False))


def story_line(view: RecordView, *, with_id: bool) -> str:
    """Render one record as a story line: time, speaker or kind, and content.

    ``with_id`` puts ``[id N]`` in front, so a model can cite the record.
    """
    when = f"({view.timestamp:%Y-%m-%d %H:%M}) " if view.timestamp is not None else ""
    prefix = f"[id {view.record_id}] " if with_id else ""
    return f"{prefix}{when}{view.story_line()}"


def render_story(records: dict[int, RecordView], record_ids: Iterable[int], *, with_id: bool) -> str:
    """Render records in world-time order. Records without a time come last, in input order."""
    views = [records[record_id] for record_id in dict.fromkeys(record_ids)]
    timed = sorted((view for view in views if view.timestamp is not None), key=lambda view: view.timestamp)
    untimed = [view for view in views if view.timestamp is None]
    return "\n".join(story_line(view, with_id=with_id) for view in [*timed, *untimed])


def fill_story_names(text: str, companion_name: str, entity_name: str) -> str:
    """Replace the two name placeholders of a story prompt. Other braces stay unchanged.

    The cycle fills its prompt texts with these two names before it fills the
    content slots, so a brace that belongs to a content slot survives this
    step. See ``workflows/main_cognition/codelet_registry.fill``.
    """
    return text.replace("{companion_name}", companion_name).replace("{entity_name}", entity_name)


def shard_name(key: ShardKey, *, user_entity_id: int, user_name: str,
               entity_names: dict[int, str]) -> str:
    """Return the story name of a shard: ``self``, the entity name, or the workflow name.

    The caller supplies the names of the group entities (the host reads them
    from the memory store). The user is named by ``user_name``. An entity
    without a known name is ``entity <id>``. The cycle and the inspector pass
    the same names, so both give the shard the same title.
    """
    if key.kind == "self":
        return "self"
    if key.kind == "entity":
        if key.ref == str(user_entity_id):
            return user_name
        entity_id = int(key.ref)
        return entity_names[entity_id] if entity_id in entity_names else f"entity {key.ref}"
    return key.ref


def story_entity_names(keys: Iterable[ShardKey], *, user_entity_id: int, user_name: str,
                       entity_names: dict[int, str]) -> str:
    """Return the names of the other characters of this moment, for ``{entity_name}``.

    ``keys`` are the shard keys of the current stimulus groups, in group
    order. Without an entity group, the other character is the user.
    """
    names = [shard_name(key, user_entity_id=user_entity_id, user_name=user_name, entity_names=entity_names)
             for key in keys if key.kind == "entity"]
    return " and ".join(dict.fromkeys(names)) or user_name


def story_shards(workspace: GlobalWorkspace, group_keys: Iterable[ShardKey],
                 *, group: ShardKey | None = None) -> list[WorkspaceShard]:
    """Return the shards that the story shows: the self shard, then the group shards.

    ``group_keys`` are the keys of the current stimulus groups, in group
    order. The group shards keep that order. A self group adds no second
    self shard. With ``group``, only the self shard and the shard of that
    group remain: the calls for one group read only their own part of the
    story. A key with no shard in ``workspace`` raises KeyError, so a cycle
    that lost a shard fails instead of hiding it.
    """
    keys = list(dict.fromkeys(key for key in group_keys
                              if key != SELF_SHARD and (group is None or key == group)))
    return [workspace.self_shard, *(workspace.shard(key) for key in keys)]


def shard_values(companion: str, name: str) -> dict[str, str]:
    """Return the name placeholders that the shard texts may use.

    A shard has one name. The story texts name it ``{entity_name}`` for an
    entity and ``{workflow_name}`` for a workflow, so both keys carry it.
    """
    return {"companion_name": companion, "entity_name": name, "workflow_name": name}


def shard_titles(key: ShardKey, name: str, companion: str, labels: ShardLabels) -> tuple[str, str]:
    """Return the block title and the tail label of one shard, with names filled in."""
    if key.kind == "self":
        title, tail = labels.self_title, labels.self_tail
    elif key.kind == "entity":
        title, tail = labels.entity_title, labels.entity_tail
    else:
        title, tail = labels.workflow_title, labels.workflow_tail
    values = shard_values(companion, name)
    return title.format(**values), tail.format(**values)


def shard_focus(key: ShardKey, name: str, companion: str, labels: ShardLabels) -> str:
    """Return the attention focus text of one shard, with names filled in.

    The attention phase asks what the awareness of this shard is about. The
    inspector shows the same words beside the shard it renders.
    """
    focus = {"self": labels.self_focus, "entity": labels.entity_focus, "workflow": labels.workflow_focus}[key.kind]
    return focus.format(**shard_values(companion, name))


def shard_block_text(key: ShardKey, name: str, companion: str,
                     records: dict[int, RecordView], item_ids: list[int], tail_ids: list[int],
                     *, template: str, labels: ShardLabels,
                     verbatim_tail: bool = False) -> str:
    """Render one shard as a story block: its title, its items, and its tail.

    ``name`` is the shard name (see :func:`shard_name`). ``records`` holds a
    view of each item and tail record. Items and tail render in world-time
    order, without record ids. ``template`` and ``labels`` are the shard
    template and the shard labels of ``phases/prompts/story.yaml``.

    The main cognition cycle, the public dialogue workflows, and the memory
    inspector call this one function, so the reply, the cycle, and the
    inspector read the shard in the same words. A reply shows the dialogue of
    a tail verbatim in its transcript. With ``verbatim_tail``, an empty
    ``tail_ids`` then says so, instead of saying that nothing happened yet.

    A record id with no view raises KeyError. That is a bug in the caller that
    collected the ids.
    """
    title, tail_label = shard_titles(key, name, companion, labels)
    return format_content(template,
        title=title, tail_label=tail_label,
        items=render_story(records, item_ids, with_id=False) or labels.empty_items,
        tail=(render_story(records, tail_ids, with_id=False)
              or (labels.verbatim_tail if verbatim_tail else labels.empty_tail)),
    ).text.strip()


__all__ = ["RecordView", "ShardLabels", "fill_story_names", "record_tokens", "record_view", "render_story",
           "shard_block_text", "shard_focus", "shard_name", "shard_titles", "shard_values", "story_entity_names",
           "story_line", "story_shards", "story_tokens"]
