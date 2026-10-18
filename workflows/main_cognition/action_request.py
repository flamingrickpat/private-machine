"""Build the typed action request of one routed decision.

Main cognition publishes one :class:`~workflows.main_cognition.schemas.ActionRequest`
for each released dispatch. The public workflows answer the partner of the
request and read the workspace shards that it carries. Task replies build a
request without a cycle and use the same helpers on the newest cycle state.
"""
from __future__ import annotations

from collections.abc import Callable

from pm.records.workspace import SELF_SHARD, GlobalWorkspace, ShardKey

from workflows.main_cognition.schemas import RequestShard


def partner_key(group: ShardKey, user_entity_id: int | None) -> ShardKey | None:
    """Return the entity shard key of the partner that a reply to ``group`` addresses.

    A reply to an entity group addresses that entity. A reply to the self
    group or to a workflow group addresses the user. Without an entity
    record for the user, the partner has no shard and the result is None.
    """
    if group.kind == "entity":
        return group
    return ShardKey(kind="entity", ref=str(user_entity_id)) if user_entity_id is not None else None


def request_shards(workspace: GlobalWorkspace, group: ShardKey, partner: ShardKey | None,
                   label: Callable[[ShardKey], str]) -> list[RequestShard]:
    """Return the shards that a request carries: self, the partner, and a workflow group.

    The shards are copies, so a later cycle cannot change a stored request.
    A partner without a shard yet gets an empty shard: the companion has no
    conscious content about it and no dialogue tail. ``label`` returns the
    story name of a shard key.
    """
    keys = [SELF_SHARD, *([partner] if partner is not None else []),
            *([group] if group.kind == "workflow" else [])]
    shards = []
    for key in dict.fromkeys(keys):
        shard = (workspace.shard(key).model_copy(deep=True) if workspace.has_shard(key)
                 else GlobalWorkspace().ensure_shard(key))
        shards.append(RequestShard(name=label(key), shard=shard))
    return shards


__all__ = ["partner_key", "request_shards"]
