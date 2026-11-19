from __future__ import annotations

from dataclasses import dataclass
from threading import Event
from typing import Protocol
from pm.memory.store import MemoryStore
from pm.memory.read_scope import ReadScope
from pm.memory.agent_tools import MemoryToolConfig


class MemoryConfigOwner(Protocol):
    def create(self, store: MemoryStore, cancel: Event) -> MemoryToolConfig: ...


@dataclass(frozen=True, slots=True)
class ManagedMemoryContext:
    store: MemoryStore
    cancel: Event
    reads: ReadScope
    owner: MemoryConfigOwner

    def create(self) -> MemoryToolConfig:
        view = self.store.snapshot()
        self.reads.observe(self.store, view.revisions, view.commit_ids)
        return self.owner.create(self.store, self.cancel)
