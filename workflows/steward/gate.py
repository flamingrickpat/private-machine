"""Dormant steward for managed runs.

The steward is an optional representative of a managed run, not a phase of
that run. It stays dormant until explicitly activated. While dormant it reads
and reports nothing and changes nothing. Its status report names its
observation time and the committed evidence it read. It submits data or control
requests through the same managed-run operations as SelfManagement.

It never edits a file, prompt, or workflow state. Termination is code-only: it
sets the run's cancellation signal. An instruction change is queued for the
next declared safe boundary and returns an accepted or rejected receipt with
the request ID and the applied revision.
"""
from __future__ import annotations

from typing import Callable

from pm.runs.run_receipt import ManagedRunError
from pm.runs.supervisor import ManagedRunSupervisor

DORMANT = "dormant"


class Steward:
    """One representative bound to one managed-run supervisor."""

    def __init__(
        self,
        supervisor: ManagedRunSupervisor,
        *,
        clock: Callable[[], float],
        active: bool = False,
    ) -> None:
        self.supervisor = supervisor
        self.clock = clock
        self._active = active

    @property
    def dormant(self) -> bool:
        return not self._active

    def activate(self) -> None:
        self._active = True

    def deactivate(self) -> None:
        self._active = False

    def _refuse_dormant(self) -> dict:
        return {
            "accepted": False,
            "reason": "The steward is dormant.",
            "state": DORMANT,
        }

    # ---------------------------------------------------------- operations

    def status(self, run_id: str) -> dict:
        """Read committed progress and return an evidence-backed report."""
        if self.dormant:
            return self._refuse_dormant()
        receipt = self.supervisor.status(run_id)
        return {
            "accepted": True,
            "run_id": run_id,
            "state": receipt["state"],
            "observed_at": self.clock(),
            "latest_event": receipt["latest_event"],
            "artifact_path": receipt["artifact_path"],
            "evidence": list(receipt.get("evidence", [])),
            "question_id": receipt.get("question_id", ""),
            "revision": receipt.get("revision", 0),
            "applied_revision": receipt.get("applied_revision", -1),
            "updates_open": receipt.get("updates_open", False),
            "instructions": receipt.get("instructions", ""),
        }

    def answer(self, run_id: str, question_id: str, answer: str) -> dict:
        """Deliver a correlated answer to an outstanding question ID."""
        if self.dormant:
            return self._refuse_dormant()
        receipt = self.supervisor.status(run_id)
        if receipt["state"] != "waiting_for_input":
            return {
                "accepted": False,
                "reason": f"Run {run_id!r} is {receipt['state']}, not waiting.",
                "run_id": run_id,
            }
        if question_id and question_id != receipt.get("question_id", ""):
            return {
                "accepted": False,
                "reason": f"Question {question_id!r} is not outstanding.",
                "run_id": run_id,
            }
        try:
            updated = self.supervisor.answer(run_id, question_id, answer)
        except ManagedRunError as error:
            return {"accepted": False, "reason": str(error), "run_id": run_id}
        return {"accepted": True, "run_id": run_id, "state": updated["state"]}

    def update(self, run_id: str, instructions: str, *, request_id: str = "") -> dict:
        """Queue one revision at the next safe boundary."""
        if self.dormant:
            return self._refuse_dormant()
        try:
            receipt = self.supervisor.update(run_id, instructions, request_id=request_id)
        except ManagedRunError as error:
            return {
                "accepted": False,
                "reason": str(error),
                "run_id": run_id,
                "request_id": request_id,
            }
        return {
            "accepted": True,
            "run_id": run_id,
            "request_id": request_id,
            "revision": receipt["revision"],
            "applied_revision": receipt["applied_revision"],
            "boundary": "next declared safe boundary",
        }

    def resume(self, run_id: str, *, request_id: str = "") -> dict:
        """Request explicit body recovery through the shared control domain."""
        if self.dormant:
            return self._refuse_dormant()
        try:
            return {"accepted": True, **self.supervisor.resume(run_id, request_id=request_id)}
        except ManagedRunError as error:
            return {"accepted": False, "run_id": run_id, "reason": str(error)}

    def terminate(self, run_id: str, *, request_id: str = "") -> dict:
        """Set the run's code-owned cancellation signal. Files are untouched."""
        if self.dormant:
            return self._refuse_dormant()
        receipt = self.supervisor.terminate(run_id, request_id=request_id)
        return {
            "accepted": True,
            "run_id": run_id,
            "state": receipt["state"],
            "artifact_path": receipt["artifact_path"],
            "reason": "cancellation signal set by code",
        }


def build_steward_tools(steward: Steward) -> dict[str, Callable[..., object]]:
    """Return the steward operations as plain functions for Python and MCP."""

    return {
        "steward_status": steward.status,
        "steward_answer": steward.answer,
        "steward_update": steward.update,
        "steward_terminate": steward.terminate,
        "steward_resume": steward.resume,
    }


__all__ = ["DORMANT", "Steward", "build_steward_tools"]
