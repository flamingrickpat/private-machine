"""Cognitive module subclasses and the shared task table.

Every module is a ``ShellModule`` or ``ShellModuleWorkflow``. Fixed routing and
fixed output are explicit prototype behavior here and in ``workflows/roles.py``,
never LLM output.
"""
from __future__ import annotations

from pathlib import Path
from threading import RLock

from pm_core.model.ghost_state import CycleState
from pm_core.model.gwt import GwtTurnRecord
from pm_core.model.knoxel_common import ActionSchemaKnoxel, Feature
from pm_core.model.knoxel_enums import ActionType, FeatureType, InterlocusType
from pm_core.model.knoxel_state_change import AttentionFrameKnoxel

from bus.envelopes import Envelope, _new_id
from runtime.shell import ShellModule
from workflow_adapter import PROJECT_ROOT, ShellModuleWorkflow
from ghost_mcp import build_tools
from workflows.dialogue_story.schemas import DialogueTurn, StoryReport, StoryRequest
from workflows.main_cognition.pipeline import DEFAULT_WORKFLOWS
from workflows.main_cognition.schemas import (
    CognitiveReport,
    CognitiveRequest,
    MemoryCandidate,
    Stimulus,
    WorkflowOption,
)

ACTION_BY_NAME = {member.value: member for member in ActionType}

ACTION_TYPE = {
    "dialogue": ActionType.Reply,
    "tool_dialogue": ActionType.ToolCallAndReply,
    "schedule": ActionType.ToolCall,
    "status": ActionType.ToolCall,
}


def _envelope(source, kind, topic, correlation_id, payload, target=None, causation_id=None):
    return Envelope(
        id=_new_id(), source=source, kind=kind, topic=topic,
        correlation_id=correlation_id, payload=payload, target=target, causation_id=causation_id,
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


def _feature(ghost, content, feature_type, *, source, interlocus=None, causal=True, changes=None):
    kwargs = {
        "content": content,
        "feature_type": feature_type,
        "causal": causal,
        "override_source": source,
    }
    if interlocus is not None:
        kwargs["interlocus"] = interlocus
    create = ghost.store_knoxel if changes is None else changes.create
    return create(Feature(**kwargs))


def _recency(knoxel) -> tuple:
    return (getattr(knoxel, "tick_id", 0), getattr(knoxel, "id", 0))


def _recent_public_dialogue(ghost, limit: int = 12) -> list[DialogueTurn]:
    """Return recent public dialogue as Story transcript turns."""
    features = [
        knoxel for knoxel in ghost.all_features
        if knoxel.feature_type == FeatureType.Dialogue
        and knoxel.interlocus == InterlocusType.Public
        and str(knoxel.content).strip()
    ]
    features.sort(key=_recency)
    return [
        DialogueTurn(speaker=str(knoxel.override_source or "unknown"), text=str(knoxel.content))
        for knoxel in features[-limit:]
    ]


def _recent_thoughts(ghost, limit: int = 4) -> list[str]:
    thoughts = [
        knoxel for knoxel in ghost.all_features
        if knoxel.feature_type in {FeatureType.Thought, FeatureType.InnerMonologue}
        and str(knoxel.content).strip()
    ]
    thoughts.sort(key=_recency)
    return [
        f"{ghost.get_companion_name()} thinks: *{knoxel.content}*"
        for knoxel in thoughts[-limit:]
    ]


def _clip(text: str, limit: int = 500) -> str:
    text = str(text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "\u2026"


def _memory_candidates(ghost, limit: int = 16) -> list[MemoryCandidate]:
    """Bounded candidates for the context stage to re-rank. No embeddings here."""
    items: dict[int, str] = {}
    dialogue = [
        knoxel for knoxel in ghost.all_features
        if knoxel.feature_type == FeatureType.Dialogue and str(knoxel.content).strip()
    ]
    dialogue.sort(key=_recency)
    for knoxel in dialogue[-limit:]:
        items[knoxel.id] = str(knoxel.content)
    for knoxel in sorted(ghost.all_declarative_facts, key=lambda k: (-k.importance, k.id))[:limit]:
        items.setdefault(knoxel.id, str(knoxel.content))
    for knoxel in sorted(ghost.all_narratives, key=_recency)[-8:]:
        items.setdefault(knoxel.id, str(knoxel.content))
    return [
        MemoryCandidate(knoxel_id=knoxel_id, text=_clip(text))
        for knoxel_id, text in items.items()
    ]


def _capabilities(ghost) -> list[str]:
    return [str(item) for item in ghost.ghost_config.supported_capabilities]


def _state_summary(ghost) -> str:
    turns = getattr(ghost, "gwt_turns", None) or {}
    if not turns:
        return ""
    return str(getattr(turns[max(turns)], "summary", ""))


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
                "causation_id": envelope.causation_id,
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

        changes = self.ghost.begin_batch(self.invocation_id or _new_id(), new_tick=True, cancel=self.stop_event)
        tick = changes.tick_id
        frame = changes.create(AttentionFrameKnoxel(
            content=summary, tick_id=tick, focus_knoxel_ids=list(incoming_ids),
            active_goal_ids=[], winning_attention_ids=[], control_mode="reactive", frame_reason=summary,
        ))
        action_ids: list[int] = []
        for decision in decisions:
            action = changes.create(ActionSchemaKnoxel(
                content=decision.get("content", ""), action_type=ACTION_TYPE[decision["route"]],
                attention_frame_id=frame.id,
                metadata={"route": decision["route"], "task_id": decision.get("task_id", ""),
                          "task_kind": decision.get("task_kind", "")},
            ))
            action_ids.append(action.id)
        changes.set_gwt(
            GwtTurnRecord(tick_id=tick, attention_frame_id=frame.id,
                          knoxel_ids=incoming_ids + [frame.id] + action_ids, summary=summary),
            CycleState(tick_id=tick, attention_frame_id=frame.id, published_slots=["main"]),
        )
        receipt = self.ghost.commit(changes)
        # A repeated commit returns the original IDs, not this attempt's proposals.
        tick = receipt.tick_id
        action_ids = list(receipt.added_ids[1:])

        self.bus.put(_envelope(
            self.module_id, "data", "gwt.updated", "", {"gwt_tick": tick},
        ))
        for index, decision in enumerate(decisions):
            self.bus.put(_envelope(
                self.module_id, "data", f"action.{decision['route']}",
                decision["correlation_id"],
                {"knoxel_id": action_ids[index], "gwt_tick": tick},
                causation_id=causes[index],
            ))


class MainCogLoop(CognitiveWorkflow):
    """Real cognitive parent: seven stages, committed GWT, typed dispatches.

    The parent workflow owns the stage order and the dispatch ids. This module
    owns the bus: it builds one typed request from the batch, commits one
    attention frame, GWT turn, and action schema per released dispatch, then
    publishes ``action.request`` targeted at each dispatch's workflow name.
    Rejected dispatches publish an explicit receipt and never an action.
    """

    DATA_TOPICS = frozenset({
        "user.input", "thought.created", "task.telemetry", "workflow.question",
    })

    def accepts(self, envelope: Envelope) -> bool:
        if envelope.topic == "workflow.question" and envelope.kind == "telemetry":
            return envelope.target in {None, self.module_id}
        return super().accepts(envelope)

    def prepare_input(self, batch: list[Envelope]) -> dict:
        stimuli: list[Stimulus] = []
        self._stimulus_ids: list[int] = []
        self._stimulus_records: list[dict] = []
        correlation = ""
        for envelope in batch:
            knoxel_id, text, kind = self._stimulus_from(envelope)
            if knoxel_id is None:
                continue
            stimuli.append(Stimulus(
                knoxel_id=knoxel_id, source=envelope.source, text=text,
                correlation_id=envelope.correlation_id, kind=kind,
            ))
            self._stimulus_ids.append(knoxel_id)
            self._stimulus_records.append({"knoxel_id": knoxel_id, "text": text})
            if not correlation:
                correlation = envelope.correlation_id
        self._batch_correlation = correlation
        request = CognitiveRequest(
            run_id=self.invocation_id or "",
            correlation_id=correlation,
            companion_name=self.ghost.get_companion_name(),
            user_name=self.ghost.get_user_name(),
            state_summary=_state_summary(self.ghost),
            active_goals=[str(knoxel.content) for knoxel in self.ghost.all_intentions][:8],
            stimuli=stimuli,
            memory_candidates=_memory_candidates(self.ghost),
            capabilities=_capabilities(self.ghost),
            available_workflows=[WorkflowOption(**workflow) for workflow in DEFAULT_WORKFLOWS],
        )
        return request.model_dump(mode="json")

    def _stimulus_from(self, envelope: Envelope):
        payload = envelope.payload
        if envelope.topic == "workflow.question" and envelope.kind == "telemetry":
            question = str(payload.get("question", "")).strip()
            if not question:
                return None, "", ""
            feature = _feature(
                self.ghost, f"Workflow question from {envelope.source}: {question}",
                FeatureType.WorldEvent, source=envelope.source or "workflow",
            )
            return feature.id, question, "workflow.question"
        if "knoxel_id" not in payload:
            return None, "", ""
        knoxel_id = int(payload["knoxel_id"])
        knoxel = self.ghost.get_knoxel_by_id(knoxel_id)
        return knoxel_id, str(knoxel.content) if knoxel is not None else "", "user.input"

    def publish_output(self, report: dict, batch: list[Envelope]) -> None:
        parsed = CognitiveReport.model_validate(report)
        correlation = self._batch_correlation or (batch[0].correlation_id if batch else "")
        cause = batch[0].id if batch else None
        changes = self.ghost.begin_batch(
            self.invocation_id or _new_id(), new_tick=True, cancel=self.stop_event,
        )
        tick = changes.tick_id
        frame = changes.create(AttentionFrameKnoxel(
            content=parsed.frame_reason or parsed.summary,
            tick_id=tick,
            focus_knoxel_ids=list(parsed.focus_ids),
            winning_attention_ids=list(parsed.winning_attention_ids),
            active_goal_ids=list(parsed.active_goal_ids),
            control_mode=parsed.control_mode,
            frame_reason=parsed.frame_reason or parsed.summary,
        ))
        published = []
        rejected = []
        action_ids: list[int] = []
        for dispatch in parsed.dispatches:
            if dispatch.status != "dispatched":
                rejected.append(dispatch)
                continue
            action = changes.create(ActionSchemaKnoxel(
                content=dispatch.content,
                action_type=ACTION_BY_NAME.get(dispatch.action_type, ActionType.Reply),
                attention_frame_id=frame.id,
                metadata={
                    "action_id": dispatch.action_id,
                    "route": dispatch.workflow_name,
                    "workflow_name": dispatch.workflow_name,
                    "depends_on": list(dispatch.depends_on),
                    "arguments": dict(dispatch.arguments),
                },
            ))
            action_ids.append(action.id)
            published.append(dispatch)
        changes.set_gwt(
            GwtTurnRecord(
                tick_id=tick, attention_frame_id=frame.id,
                knoxel_ids=list(self._stimulus_ids) + [frame.id] + action_ids,
                summary=parsed.summary,
            ),
            CycleState(tick_id=tick, attention_frame_id=frame.id, published_slots=["main"]),
        )
        receipt = self.ghost.commit(changes)
        tick = receipt.tick_id
        if receipt.added_ids:
            action_ids = list(receipt.added_ids[1:])

        memory_texts = [candidate.text for candidate in _memory_candidates(self.ghost, limit=8)]
        self.bus.put(_envelope(self.module_id, "data", "gwt.updated", "", {"gwt_tick": tick}))
        for dispatch, action_id in zip(published, action_ids):
            self.bus.put(_envelope(
                self.module_id, "data", "action.request", correlation,
                {
                    "knoxel_id": action_id,
                    "gwt_tick": tick,
                    "action_id": dispatch.action_id,
                    "workflow_name": dispatch.workflow_name,
                    "content": dispatch.content,
                    "arguments": dict(dispatch.arguments),
                    "stimuli": list(self._stimulus_records),
                    "memories": memory_texts,
                    "workspace": parsed.subjective_experience or parsed.frame_reason,
                },
                target=dispatch.workflow_name,
                causation_id=cause,
            ))
        for dispatch in rejected:
            self.bus.put(_envelope(
                self.module_id, "telemetry", "action.receipt", correlation,
                {
                    "action_id": dispatch.action_id,
                    "workflow_name": dispatch.workflow_name,
                    "status": "rejected",
                    "rejection_reason": dispatch.rejection_reason,
                },
                causation_id=cause,
            ))


class DummyDialogue(CognitiveWorkflow):
    """Labeled demonstration fixture: echoes the action content as a reply."""

    DATA_TOPICS = frozenset({"action.dialogue"})

    def publish_output(self, report: dict, batch: list[Envelope]) -> None:
        replies = report.get("replies", [])
        causes = [next(envelope.id for envelope in batch if envelope.correlation_id == reply["correlation_id"])
                  for reply in replies]
        changes = self.ghost.begin_batch(self.invocation_id or _new_id(), cancel=self.stop_event)
        for reply in replies:
            _feature(
                self.ghost, reply["content"], FeatureType.Dialogue,
                source=self.ghost.get_companion_name(), interlocus=InterlocusType.Public,
                changes=changes,
            )
        receipt = self.ghost.commit(changes)
        for reply, record_id, cause in zip(replies, receipt.added_ids, causes, strict=True):
            self.bus.put(_envelope(
                self.module_id, "data", "dialogue.reply",
                reply["correlation_id"], {"knoxel_id": record_id}, causation_id=cause,
            ))


class StoryDialogue(CognitiveWorkflow):
    """Real Story generation module. Accepts typed actions targeting it."""

    DATA_TOPICS = frozenset({"action.request"})
    WORKFLOW_NAME = "dialogue_story"

    def accepts(self, envelope: Envelope) -> bool:
        if envelope.topic == "action.request" and envelope.kind == "data":
            return envelope.target in {None, self.module_id, self.WORKFLOW_NAME}
        return super().accepts(envelope)

    def prepare_input(self, batch: list[Envelope]) -> dict:
        payload = batch[0].payload
        action = self.ghost.get_knoxel_by_id(payload["knoxel_id"]) if "knoxel_id" in payload else None
        intent = str(action.content) if action is not None else str(payload.get("content", ""))
        stimuli = [
            item for item in payload.get("stimuli", [])
            if str(item.get("text", "")).strip()
        ]
        current = str(stimuli[-1]["text"]) if stimuli else intent
        memories = [str(item) for item in payload.get("memories", []) if str(item).strip()]
        request = StoryRequest(
            companion_name=self.ghost.get_companion_name(),
            user_name=self.ghost.get_user_name(),
            character_card_story=self.ghost.get_simple_character_story_block(),
            supported_capabilities=[str(item) for item in self.ghost.ghost_config.supported_capabilities],
            unsupported_capabilities=[str(item) for item in self.ghost.ghost_config.unsupported_capabilities],
            capability_notes=[str(item) for item in self.ghost.ghost_config.capability_notes],
            temporal_context=self.ghost.build_body_and_situation_addendum(),
            prior_dialogue=_recent_public_dialogue(self.ghost),
            current_user_message=current or intent or "(no message)",
            feelings=_recent_thoughts(self.ghost),
            memories=memories,
            workspace=str(payload.get("workspace", "")),
        )
        return request.model_dump(mode="json")

    def publish_output(self, report: dict, batch: list[Envelope]) -> None:
        story = StoryReport.model_validate(report)
        correlation = batch[0].correlation_id if batch else ""
        cause = batch[0].id if batch else None
        if not story.accepted:
            reason = "; ".join(violation.reason for violation in story.ontology.violations)
            self.bus.put(_envelope(
                self.module_id, "telemetry", "action.receipt", correlation,
                {
                    "status": "rejected",
                    "workflow_name": self.WORKFLOW_NAME,
                    "rejection_reason": reason or "ontology boundary",
                },
                causation_id=cause,
            ))
            return
        changes = self.ghost.begin_batch(self.invocation_id or _new_id(), cancel=self.stop_event)
        _feature(
            self.ghost, story.candidate, FeatureType.Dialogue,
            source=self.ghost.get_companion_name(), interlocus=InterlocusType.Public,
            changes=changes,
        )
        receipt = self.ghost.commit(changes)
        self.bus.put(_envelope(
            self.module_id, "data", "dialogue.reply", correlation,
            {
                "knoxel_id": receipt.added_ids[0],
                "action_id": batch[0].payload.get("action_id", "") if batch else "",
            },
            causation_id=cause,
        ))


class InnerThoughtLoop(CognitiveWorkflow):
    DATA_TOPICS = frozenset()

    def publish_output(self, report: dict, batch: list[Envelope]) -> None:
        changes = self.ghost.begin_batch(self.invocation_id or _new_id(), cancel=self.stop_event)
        _feature(
            self.ghost, report["thought"], FeatureType.InnerMonologue,
            source=self.ghost.get_companion_name(), changes=changes,
        )
        receipt = self.ghost.commit(changes)
        self.bus.put(_envelope(
            self.module_id, "data", "thought.created", "", {"knoxel_id": receipt.added_ids[0]},
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


class ToolPlusDialogue(DummyDialogue):
    DATA_TOPICS = frozenset({"action.tool_dialogue"})

__all__ = [
    "TaskTable",
    "UserInputProxy",
    "MainCogLoop",
    "DummyMainCogLoop",
    "DummyDialogue",
    "StoryDialogue",
    "InnerThoughtLoop",
    "ScheduleSubprocesses",
    "CheckSubprocess",
    "ToolPlusDialogue",
]
