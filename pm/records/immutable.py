"""Protect committed collections while keeping edit copies ordinary and mutable."""
from __future__ import annotations

from pydantic import BaseModel


def _reject(*args, **kwargs):
    raise TypeError("Committed knoxels are immutable. Copy the record and append a successor.")


class FrozenDict(dict):
    """Read-only mapping. Pickle restores a mutable mapping for isolated edits."""
    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _reject

    def __reduce_ex__(self, protocol):
        return dict, (dict(self),)


class FrozenList(list):
    """Read-only sequence. Pickle restores a mutable sequence for isolated edits."""
    __setitem__ = __delitem__ = append = clear = extend = insert = pop = remove = reverse = sort = __iadd__ = __imul__ = _reject

    def __reduce_ex__(self, protocol):
        return list, (list(self),)


def freeze(value):
    """Protect nested collections without changing their serialized payloads."""
    if isinstance(value, dict):
        return FrozenDict({key: freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return FrozenList(freeze(item) for item in value)
    if isinstance(value, BaseModel):
        for name in type(value).model_fields:
            object.__setattr__(value, name, freeze(getattr(value, name)))
        # Nested models have no derived cache fields. Their field mapping must
        # also reject assignment, including vector helpers that write it directly.
        object.__setattr__(value, "__dict__", FrozenDict(value.__dict__))
    return value


def freeze_record(record) -> None:
    """Protect logical fields. Derived embeddings and access times remain writable."""
    for name in type(record).model_fields:
        if name not in {"embedding", "embedding_model", "last_accessed"}:
            object.__setattr__(record, name, freeze(getattr(record, name)))
