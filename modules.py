"""Cognitive module subclasses and the shared task table.

Every module is a ``ShellModule`` or ``ShellModuleWorkflow``. Fixed routing and
fixed output are explicit prototype behavior here and in ``workflows/roles.py``,
never LLM output.
"""
from __future__ import annotations

from threading import RLock

from pm_core.model.ghost_state import CycleState
from pm_core.model.gwt import GwtTurnRecord
from pm_core.model.knoxel_common import ActionSchemaKnoxel, Feature
from pm_core.model.knoxel_enums import ActionType, FeatureType, InterlocusType
from pm_core.model.knoxel_state_change import AttentionFrameKnoxel

from shell import Envelope, ShellModule, _new_id
from workflow_adapter import ShellModuleWorkflow
from ghost_mcp import build_tools

ACTION_TYPE = {
    "dialogue": ActionType.Reply,
    "tool_dialogue": ActionType.ToolCallAndReply,
    "schedule": ActionType.ToolCall,
    "status": ActionType.ToolCall,
}


def _envelope(source, kind, topic, correlation_id, payload, target=None):
    return Envelope(
        id=_new_id(), source=source, kind=kind, topic=topic,
        correlation_id=correlation_id, payload=payload, target=target,
    )


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


def _feature(ghost, content, feature_type, *, source, interlocus=None, causal=True):
    kwargs = {
        "content": content,
        "feature_type": feature_type,
        "causal": causal,
        "override_source": source,
    }
    if interlocus is not None:
        kwargs["interlocus"] = interlocus
    return ghost.store_knoxel(Feature(**kwargs))


class UserInputProxy(ShellModule):
    """Wraps terminal input and prints public replies."""

    DATA_TOPICS = frozenset({"dialogue.reply"})

    def __init__(self, module_id, bus, on_error, clock, ghost, emit_reply):
        super().__init__(module_id, bus, on_error, clock, None)
        self.ghost = ghost
        self.emit_reply = emit_reply

    def submit(self, text: str) -> str:
        feature = _feature(
            self.ghost, text, FeatureType.Dialogue,
            source=self.ghost.get_user_name(), interlocus=InterlocusType.Public,
        )
        correlation_id = _new_id()
        self.bus.put(_envelope(
            self.module_id, "data", "user.input",
            correlation_id, {"knoxel_id": feature.id},
        ))
        return correlation_id

    def handle_batch(self, batch) -> None:
        for envelope in batch:
            feature = self.ghost.get_knoxel_by_id(envelope.payload["knoxel_id"])
            if feature is not None:
                self.emit_reply(feature.content)


class CognitiveWorkflow(ShellModuleWorkflow):
    """Knoxel routing shared by cognitive workflows, outside the generic runner."""

    def __init__(self, *args, ghost, **kwargs):
        super().__init__(*args, **kwargs)
        self.ghost = ghost

    def prepare_input(self, batch: list[Envelope]) -> dict:
        # Resolve input through the invocation scope, never the unrestricted Ghost.
        tools = build_tools(self.last_environment)
        messages = []
        for envelope in batch:
            message = {
                "id": envelope.id,
                "source": envelope.source,
                "kind": envelope.kind,
                "topic": envelope.topic,
                "correlation_id": envelope.correlation_id,
                "target": envelope.target,
                "payload": envelope.payload,
                "knoxel": tools["ghost_get"](envelope.payload["knoxel_id"]),
            }
            if "gwt_tick" in envelope.payload:
                message["gwt"] = {
                    "__type__": "GwtTurnRecord",
                    "data": tools["ghost_gwt"](envelope.payload["gwt_tick"]),
                }
            messages.append(message)
        return {
            "invocation_id": self.invocation_id,
            "module_id": self.module_id,
            "messages": messages,
        }


class MainCogLoop(CognitiveWorkflow):
    """Stores incoming knoxels, creates one GWT, and selects fixed actions."""

    DATA_TOPICS = frozenset({
        "user.input", "dialogue.reply", "thought.created", "task.telemetry",
    })

    def publish_output(self, report: dict, batch: list[Envelope]) -> None:
        summary = report.get("summary", "")
        decisions = report.get("decisions", [])
        incoming_ids = [
            envelope.payload["knoxel_id"] for envelope in batch
            if "knoxel_id" in envelope.payload
        ]

        with self.ghost.lock:
            tick = self.ghost.current_tick_id + 1
            self.ghost.current_tick_id = tick
            frame = self.ghost.add_knoxel(
                AttentionFrameKnoxel(
                    content=summary,
                    tick_id=tick,
                    focus_knoxel_ids=list(incoming_ids),
                    active_goal_ids=[],
                    winning_attention_ids=[],
                    control_mode="reactive",
                    frame_reason=summary,
                ),
                generate_embedding=False,
            )
            action_ids: list[int] = []
            for decision in decisions:
                action = self.ghost.add_knoxel(
                    ActionSchemaKnoxel(
                        content=decision.get("content", ""),
                        action_type=ACTION_TYPE[decision["route"]],
                        attention_frame_id=frame.id,
                        metadata={
                            "route": decision["route"],
                            "task_id": decision.get("task_id", ""),
                            "task_kind": decision.get("task_kind", ""),
                        },
                    ),
                    generate_embedding=False,
                )
                action_ids.append(action.id)
            ids = incoming_ids + [frame.id] + action_ids
            self.ghost.gwt_turns[tick] = GwtTurnRecord(
                tick_id=tick,
                attention_frame_id=frame.id,
                knoxel_ids=ids,
                summary=summary,
            )
            self.ghost.states.append(CycleState(
                tick_id=tick,
                attention_frame_id=frame.id,
                published_slots=["main"],
            ))

        self.bus.put(_envelope(
            self.module_id, "data", "gwt.updated", "", {"gwt_tick": tick},
        ))
        for index, decision in enumerate(decisions):
            self.bus.put(_envelope(
                self.module_id, "data", f"action.{decision['route']}",
                decision["correlation_id"],
                {"knoxel_id": action_ids[index], "gwt_tick": tick},
            ))


class Dialogue(CognitiveWorkflow):
    DATA_TOPICS = frozenset({"action.dialogue"})

    def publish_output(self, report: dict, batch: list[Envelope]) -> None:
        for reply in report.get("replies", []):
            feature = _feature(
                self.ghost, reply["content"], FeatureType.Dialogue,
                source=self.ghost.get_companion_name(), interlocus=InterlocusType.Public,
            )
            self.bus.put(_envelope(
                self.module_id, "data", "dialogue.reply",
                reply["correlation_id"], {"knoxel_id": feature.id},
            ))


class InnerThoughtLoop(CognitiveWorkflow):
    DATA_TOPICS = frozenset()

    def publish_output(self, report: dict, batch: list[Envelope]) -> None:
        feature = _feature(
            self.ghost, report["thought"], FeatureType.InnerMonologue,
            source=self.ghost.get_companion_name(),
        )
        self.bus.put(_envelope(
            self.module_id, "data", "thought.created", "", {"knoxel_id": feature.id},
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
        feature = _feature(
            self.ghost, f"Task {task_id} {status}",
            FeatureType.WorldEvent, source=self.module_id,
        )
        self.bus.put(_envelope(
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
            feature = _feature(
                self.ghost, f"Task {task_id} {status}",
                FeatureType.WorldEvent, source=self.module_id,
            )
            self.bus.put(_envelope(
                self.module_id, "telemetry", "task.telemetry",
                query.get("correlation_id", ""),
                {"knoxel_id": feature.id, "task_id": task_id, "status": status},
            ))


class ToolPlusDialogue(Dialogue):
    DATA_TOPICS = frozenset({"action.tool_dialogue"})

__all__ = [
    "TaskTable",
    "UserInputProxy",
    "MainCogLoop",
    "Dialogue",
    "InnerThoughtLoop",
    "ScheduleSubprocesses",
    "CheckSubprocess",
    "ToolPlusDialogue",
]
