"""Typed state of the sharded global workspace and its content requirements.

The workspace holds the conscious content of the cycle. It has one shard for
the self, one shard for each entity, and one shard for each workflow. Each
shard holds record references, not record copies. The memory store owns the
records. The cycle state owns the workspace and saves it with each cycle.

This module holds only data and simple lookups. The attention phase of a later
step fills the shards. The requirement models describe what the attention
phase must keep and what it must cap.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field

from pm.utils import yaml_util

ShardKind = Literal["self", "entity", "workflow"]


class ShardKey(BaseModel):
    """The name of one workspace shard.

    ``kind`` is ``self``, ``entity``, or ``workflow``. ``ref`` is empty for the
    self shard, the entity id as text for an entity shard, and the workflow
    name for a workflow shard. The key is frozen, so it can be a dict key or a
    set member.
    """

    model_config = ConfigDict(frozen=True)

    kind: ShardKind = "self"
    ref: str = ""


SELF_SHARD = ShardKey(kind="self")


class WorkspaceItem(BaseModel):
    """One record that is present in a shard.

    ``activation_at_entry`` is the CSM activation when the record entered.
    ``structure_id`` names the structure that brought the record, or None.
    ``tokens`` is the token count of the rendered story element of the record.
    ``required_by`` names the requirement that reserved the place. An empty
    value means that no requirement reserved it.
    """

    record_id: int = 0
    entered_tick: int = 0
    activation_at_entry: float = 0.0
    structure_id: int | None = None
    reason: str = ""
    tokens: int = 0
    required_by: str = ""


class WorkspaceShard(BaseModel):
    """The conscious content for one self, entity, or workflow.

    ``items`` is the content. ``tail_ids`` is the newest stream of the shard,
    for example the latest dialogue with an entity. The tail has its own token
    budget. ``requirement_set`` names the requirement set of the shard in the
    workspace requirement file. It is usually the shard kind.
    """

    key: ShardKey = Field(default_factory=ShardKey)
    items: list[WorkspaceItem] = Field(default_factory=list)
    tail_ids: list[int] = Field(default_factory=list)
    requirement_set: str = ""

    def item_ids(self) -> list[int]:
        """Return the record ids of the items in shard order."""
        return [item.record_id for item in self.items]

    def content_tokens(self) -> int:
        """Return the token sum of the items. The tail is not part of the sum."""
        return sum(item.tokens for item in self.items)


class GlobalWorkspace(BaseModel):
    """All workspace shards of one cycle.

    A shard for an entity or a workflow appears when its first stimulus
    arrives. It stays until a later item adds retirement. The keys of
    ``entity_shards`` are entity ids. JSON stores them as text, and
    ``model_validate`` converts them back to integers.
    """

    self_shard: WorkspaceShard = Field(default_factory=lambda: WorkspaceShard(key=SELF_SHARD, requirement_set="self"))
    entity_shards: dict[int, WorkspaceShard] = Field(default_factory=dict)
    workflow_shards: dict[str, WorkspaceShard] = Field(default_factory=dict)

    def shard(self, key: ShardKey) -> WorkspaceShard:
        """Return the shard for ``key``.

        Raises KeyError when the shard does not exist. Use ``ensure_shard``
        when a missing shard must be created.
        """
        if key.kind == "self":
            return self.self_shard
        if key.kind == "entity":
            return self.entity_shards[int(key.ref)]
        return self.workflow_shards[key.ref]

    def ensure_shard(self, key: ShardKey) -> WorkspaceShard:
        """Return the shard for ``key``. Create an empty shard when it does not exist.

        The new shard uses the shard kind as its requirement set name.
        """
        if key.kind == "self":
            return self.self_shard
        if key.kind == "entity":
            return self.entity_shards.setdefault(int(key.ref), WorkspaceShard(key=key, requirement_set=key.kind))
        return self.workflow_shards.setdefault(key.ref, WorkspaceShard(key=key, requirement_set=key.kind))

    def all_shards(self) -> Iterator[WorkspaceShard]:
        """Yield the self shard, then the entity shards, then the workflow shards.

        Entity shards come in entity id order and workflow shards in name
        order, so prompts that render the shards are deterministic.
        """
        yield self.self_shard
        for entity_id in sorted(self.entity_shards):
            yield self.entity_shards[entity_id]
        for name in sorted(self.workflow_shards):
            yield self.workflow_shards[name]

    def item_ids(self) -> list[int]:
        """Return the unique record ids of all shard items, in shard order. Tails are not included."""
        return list(dict.fromkeys(record_id for shard in self.all_shards() for record_id in shard.item_ids()))


class RecordSelector(BaseModel):
    """Selects the records that one requirement counts.

    All conditions must match. An empty condition matches everything.

    - ``record``: Python record class names, for example ``Narrative``.
    - ``entity``: ``self`` selects records about the companion. ``shard``
      selects records about the entity of the shard. ``group`` selects records
      about any entity of the current stimulus groups. ``any`` does not filter.
    - ``category``: a category label of the record, for example the belief
      kind ``self_model``.
    - ``feature_type``: a ``FeatureType`` value. It filters only ``Feature``
      records. Other record classes in ``record`` match without it.
    - ``scope``: ``current_stimuli`` selects only the stimuli that the current
      cycle claimed. ``any`` does not filter.
    """

    record: list[str] = Field(default_factory=list)
    entity: Literal["self", "shard", "group", "any"] = "any"
    category: str = ""
    feature_type: str = ""
    scope: Literal["current_stimuli", "any"] = "any"


class WorkspaceRequirement(BaseModel):
    """One content rule for a shard or for the CSM.

    In a shard, ``min_items`` reserves places that other content cannot take.
    ``max_items`` and ``max_token_share`` cap the matching content. The share
    is a part of the shard content budget, from 0 to 1.

    In the CSM, prune never removes an item that matches the selector. When
    ``max_items`` is set, only that number of the matching items has
    protection.
    """

    name: str = ""
    select: RecordSelector = Field(default_factory=RecordSelector)
    min_items: int = 0
    max_items: int | None = None
    max_token_share: float | None = None


class ShardRequirementSet(BaseModel):
    """The requirements of one shard kind or of the CSM."""

    requirements: list[WorkspaceRequirement] = Field(default_factory=list)


class WorkspaceRequirementConfig(BaseModel):
    """All content requirements of the cycle.

    ``shards`` has one requirement set for each shard kind. ``csm`` holds the
    rules that protect CSM items from prune.
    """

    model_config = ConfigDict(extra="forbid")

    shards: dict[ShardKind, ShardRequirementSet] = Field(default_factory=dict)
    csm: ShardRequirementSet = Field(default_factory=ShardRequirementSet)


def load_workspace_requirements(path: str | Path) -> WorkspaceRequirementConfig:
    """Read and validate a workspace requirement file.

    Raises ``yaml_util.YAMLError`` for invalid YAML and a pydantic
    ``ValidationError`` for content that does not match the models.
    """
    return WorkspaceRequirementConfig.model_validate(yaml_util.safe_load(Path(path).read_text(encoding="utf-8")))
