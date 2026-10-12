"""Serialized exchange values. Unsupported live objects fail on publication."""
from __future__ import annotations

import json
import uuid
from typing import Literal


def new_id() -> str:
    """Return a fresh envelope identifier."""
    return uuid.uuid4().hex


class Envelope:
    """One independently delivered bus message."""

    __slots__ = ("id", "source", "kind", "topic", "correlation_id", "payload", "target", "causation_id")

    def __init__(
        self,
        id: str,
        source: str,
        kind: Literal["data", "telemetry", "control"],
        topic: str,
        correlation_id: str,
        payload: dict,
        target: str | None = None,
        causation_id: str | None = None,
    ):
        self.id = id
        self.source = source
        self.kind = kind
        self.topic = topic
        self.correlation_id = correlation_id
        self.payload = payload
        self.target = target
        self.causation_id = causation_id

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Envelope) and other.id == self.id

    def __repr__(self) -> str:
        return f"Envelope({self.topic!r}, {self.id[:8]})"

    def to_json(self) -> str:
        return json.dumps({name: getattr(self, name) for name in self.__slots__}, allow_nan=False)

    @classmethod
    def from_json(cls, text: str) -> "Envelope":
        return cls(**json.loads(text))
