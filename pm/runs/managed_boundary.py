from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from threading import Event
from typing import TYPE_CHECKING, Generator, TypedDict
from pm.memory.agent_tools import guard_revisions

if TYPE_CHECKING:
    from pm.runs.supervisor import ManagedRunSupervisor


class InstructionRevisionChanged(RuntimeError):
    pass


class BoundaryInstructions(TypedDict):
    instruction: str
    revision: int


@dataclass(frozen=True, slots=True)
class BoundaryReadSettings:
    wait: bool = False
    close: bool = False


@dataclass(frozen=True, slots=True)
class LeafBoundarySettings:
    instruction: str
    expected_revision: int


@dataclass(frozen=True, slots=True)
class ManagedRunBoundary:
    supervisor: ManagedRunSupervisor
    run_id: str
    epoch: int
    cancel: Event
    leaf: LeafBoundarySettings | None = None

    def read(self, settings: BoundaryReadSettings = BoundaryReadSettings()) -> BoundaryInstructions:
        current = self.supervisor.instructions_at_boundary(self.run_id, wait=settings.wait, close=settings.close, epoch=self.epoch)
        if self.leaf is not None:
            if current["revision"] != self.leaf.expected_revision:
                raise InstructionRevisionChanged("The project instruction changed. Replan from retained world effects.")
            return BoundaryInstructions(instruction=self.leaf.instruction, revision=current["revision"])
        return current

    def instructions(self, wait: bool = False, close: bool = False) -> BoundaryInstructions:
        return self.read(BoundaryReadSettings(wait=wait, close=close))

    def milestone(self, key: str, summary: str, evidence: str) -> None:
        self.supervisor.milestone(self.run_id, self.epoch, key, summary, evidence)

    def cancelled(self) -> bool:
        return self.supervisor.cancelled_at_boundary(self.run_id, self.epoch)

    @contextmanager
    def authorize(self, revision: int) -> Generator[bool, None, None]:
        with self.supervisor.dispatch_authorization(self.run_id, self.epoch, revision) as accepted:
            yield accepted

    @contextmanager
    def memory_guard(self, revisions: dict[int, int]) -> Generator[None, None, None]:
        with guard_revisions(self.supervisor.ghost, revisions, self.cancel):
            yield

    def for_leaf(self, settings: LeafBoundarySettings) -> ManagedRunBoundary:
        return ManagedRunBoundary(self.supervisor, self.run_id, self.epoch, self.cancel, settings)
