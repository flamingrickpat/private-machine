"""Bus modules of the dummy cognition mode (``--cognition dummy``).

Their fixed outputs are labeled prototype behavior. ``TaskTable`` holds the
in-memory state of the dummy scheduler.
"""
from __future__ import annotations

from threading import RLock

from pm.records.cycle_records import StimulusGroup
from pm.records.cycle_state import CycleState
from pm.records.global_workspace import GwtTurnRecord
from pm.records.memory_records import ActionSchemaRecord
from pm.records.enums import ActionType, FeatureType, InterlocusType

from pm.bus.envelopes import Envelope, new_id
from pm.bus.envelopes import new_envelope
from pm.records.memory_records import new_feature
from pm.runs.workflow_module import CognitiveWorkflow


ACTION_TYPE = {
    "dialogue": ActionType.Reply,
    "tool_dialogue": ActionType.ToolCallAndReply,
    "schedule": ActionType.ToolCall,
    "status": ActionType.ToolCall,
}


class TaskTable:
    """One stateful table shared by the scheduler and status modules."""

    def __init__(self, clock):
        self.clock = clock
        self._lock = RLock()
        self._records: dict[str, dict] = {}

    def create(self, task_id: str, task_kind: str, correlation_id: str) -> dict | None:
        with self._lock:
            if task_id in self._records:
                return None
            record = {
                "task_id": task_id,
                "kind": task_kind,
                "correlation_id": correlation_id,
                "status": "accepted",
                "next_transition": None,
            }
            self._records[task_id] = record
            return record

    def get(self, task_id: str) -> dict | None:
        with self._lock:
            return self._records.get(task_id)

    def due(self, now: float) -> list[dict]:
        with self._lock:
            return [
                record for record in self._records.values()
                if record["status"] in {"accepted", "running"}
                and record.get("next_transition") is not None
                and record["next_transition"] <= now
            ]

    def advance(self, record: dict, period: float) -> str:
        with self._lock:
            now = self.clock()
            if record["status"] == "accepted":
                record["status"] = "running"
            elif record["status"] == "running":
                record["status"] = "completed"
            record["next_transition"] = now + period
            return record["status"]

    def active(self) -> list[dict]:
        with self._lock:
            return [r for r in self._records.values() if r["status"] in {"accepted", "running"}]

    def mark_terminated(self, task_id: str) -> None:
        with self._lock:
            record = self._records.get(task_id)
            if record is not None and record["status"] in {"accepted", "running"}:
                record["status"] = "terminated"


class DummyMainCogLoop(CognitiveWorkflow):
    """Labeled demonstration fixture: fixed keyword routing, no model calls."""

    DATA_TOPICS = frozenset({
        "user.input", "thought.created", "task.telemetry",
    })

    def publish_output(self, report: dict, batch: list[Envelope]) -> None:
        summary = report.get("summary", "")
        decisions = report.get("decisions", [])
        causes = [next(envelope.id for envelope in batch if envelope.correlation_id == decision["correlation_id"])
                  for decision in decisions]
        incoming_ids = [
            envelope.payload["knoxel_id"] for envelope in batch
            if "knoxel_id" in envelope.payload
        ]

        changes = self.ghost.begin_batch(self.invocation_id or new_id(), new_tick=True, cancel=self.stop_event)
        tick = changes.tick_id
        action_ids: list[int] = []
        for decision in decisions:
            action = changes.create(ActionSchemaRecord(
                content=decision.get("content", ""), action_type=ACTION_TYPE[decision["route"]],
                metadata={"route": decision["route"], "task_id": decision.get("task_id", ""),
                          "task_kind": decision.get("task_kind", "")},
            ))
            action_ids.append(action.id)
        changes.set_gwt(
            GwtTurnRecord(tick_id=tick, stimulus_ids=list(incoming_ids),
                          knoxel_ids=incoming_ids + action_ids, summary=summary),
            CycleState(tick_id=tick, stimulus_groups=[StimulusGroup(stimulus_ids=list(incoming_ids))]),
        )
        receipt = self.ghost.commit(changes)
        # A repeated commit returns the original IDs, not this attempt's proposals.
        tick = receipt.tick_id
        action_ids = list(receipt.added_ids)

        self.bus.put(new_envelope(
            self.module_id, "data", "gwt.updated", "", {"gwt_tick": tick},
        ))
        for index, decision in enumerate(decisions):
            self.bus.put(new_envelope(
                self.module_id, "data", f"action.{decision['route']}",
                decision["correlation_id"],
                {"knoxel_id": action_ids[index], "gwt_tick": tick},
                causation_id=causes[index],
            ))


class DummyDialogue(CognitiveWorkflow):
    """Labeled demonstration fixture: echoes the action content as a reply."""

    DATA_TOPICS = frozenset({"action.dialogue"})

    def publish_output(self, report: dict, batch: list[Envelope]) -> None:
        replies = report.get("replies", [])
        causes = [next(envelope.id for envelope in batch if envelope.correlation_id == reply["correlation_id"])
                  for reply in replies]
        changes = self.ghost.begin_batch(self.invocation_id or new_id(), cancel=self.stop_event)
        for reply in replies:
            new_feature(
                self.ghost, reply["content"], FeatureType.Dialogue,
                source=self.ghost.get_companion_name(), interlocus=InterlocusType.Public,
                changes=changes,
            )
        receipt = self.ghost.commit(changes)
        for reply, record_id, cause in zip(replies, receipt.added_ids, causes, strict=True):
            self.bus.put(new_envelope(
                self.module_id, "data", "dialogue.reply",
                reply["correlation_id"], {"knoxel_id": record_id}, causation_id=cause,
            ))


class ScheduleSubprocesses(CognitiveWorkflow):
    DATA_TOPICS = frozenset({"action.schedule"})

    def __init__(self, module_id, bus, on_error, clock, tick_seconds, *,
                 working_target, workflow, tools_factory, runtime_root, ghost, tasks):
        super().__init__(
            module_id, bus, on_error, clock, tick_seconds,
            working_target=working_target, workflow=workflow,
            tools_factory=tools_factory, runtime_root=runtime_root, ghost=ghost,
        )
        self.tasks = tasks

    def handle_batch(self, batch: list[Envelope]) -> None:
        if not batch:
            self._advance_tasks()
            return
        super().handle_batch(batch)

    def publish_output(self, report: dict, batch: list[Envelope]) -> None:
        for entry in report.get("tasks", []):
            task_id = entry.get("task_id", "")
            task_kind = entry.get("task_kind", "")
            correlation_id = entry.get("correlation_id", "")
            if task_kind == "minecraft":
                self._emit(task_id, "unsupported", correlation_id)
                continue
            created = self.tasks.create(task_id, task_kind, correlation_id)
            if created is None:
                continue
            created["next_transition"] = self.clock() + self.tick_seconds
            self._emit(task_id, "accepted", correlation_id)

    def _advance_tasks(self) -> None:
        for record in self.tasks.due(self.clock()):
            status = self.tasks.advance(record, self.tick_seconds)
            self._emit(record["task_id"], status, record["correlation_id"])

    def _emit(self, task_id: str, status: str, correlation_id: str) -> None:
        feature = new_feature(
            self.ghost, f"Task {task_id} {status}",
            FeatureType.WorldEvent, source=self.module_id,
        )
        self.bus.put(new_envelope(
            self.module_id, "telemetry", "task.telemetry", correlation_id,
            {"knoxel_id": feature.id, "task_id": task_id, "status": status},
        ))

    def on_terminate(self) -> None:
        for record in self.tasks.active():
            self.tasks.mark_terminated(record["task_id"])
            self._emit(record["task_id"], "terminated", record["correlation_id"])


class CheckSubprocess(CognitiveWorkflow):
    DATA_TOPICS = frozenset({"action.status"})

    def __init__(self, module_id, bus, on_error, clock, tick_seconds, *,
                 working_target, workflow, tools_factory, runtime_root, ghost, tasks):
        super().__init__(
            module_id, bus, on_error, clock, tick_seconds,
            working_target=working_target, workflow=workflow,
            tools_factory=tools_factory, runtime_root=runtime_root, ghost=ghost,
        )
        self.tasks = tasks

    def publish_output(self, report: dict, batch: list[Envelope]) -> None:
        for query in report.get("queries", []):
            task_id = query.get("task_id", "")
            record = self.tasks.get(task_id)
            status = record["status"] if record else "unknown"
            feature = new_feature(
                self.ghost, f"Task {task_id} {status}",
                FeatureType.WorldEvent, source=self.module_id,
            )
            self.bus.put(new_envelope(
                self.module_id, "telemetry", "task.telemetry",
                query.get("correlation_id", ""),
                {"knoxel_id": feature.id, "task_id": task_id, "status": status},
            ))


class ToolPlusDialogue(DummyDialogue):
    DATA_TOPICS = frozenset({"action.tool_dialogue"})

