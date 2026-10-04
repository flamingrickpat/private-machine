"""Cognitive module subclasses and the shared task table.

Every module is a ``ShellModule`` or ``ShellModuleWorkflow``. Fixed routing and
fixed output are explicit prototype behavior here and in ``workflows/roles.py``,
never LLM output.
"""
from __future__ import annotations

from pathlib import Path
import asyncio
from datetime import datetime
from threading import RLock

from pm_core.model.affect_dynamics import AffectProjection
from pm_core.model.ghost_state import CycleState
from pm_core.model.gwt import GwtTurnRecord
from pm_core.model.knoxel_common import ActionSchemaKnoxel, Feature
from pm_core.model.knoxel_core import Provenance
from pm_core.model.knoxel_enums import ActionType, FeatureType, InterlocusType, SourceKind
from pm_core.model.knoxel_state_change import AttentionFrameKnoxel
from pm_core.model.mental_state_vectors import FullMentalState, create_empty_state
from runtime.experience import consolidate_cycle, update_beliefs, utterance_metadata

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
from workflows.main_cognition.state_changes import (
    active_goal_texts,
    active_intentions,
    apply_appraisal,
    apply_goal_changes,
)
from workflows.self_management.schemas import SelfManagementRequest
from workflows.steward.schemas import StewardRequest
from workflows.dialogue_assistant.schemas import AssistantMessage, AssistantReport, AssistantRequest
from workflows.dialogue_assistant.steps import unsupported_management_claims
from workflows.dialogue_assistant.tools import TOOL_DESCRIPTORS, TOOL_NAMES

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


def _feature(ghost, content, feature_type, *, source, interlocus=None, causal=True, changes=None,
             source_kind=None, metadata=None):
    kwargs = {
        "content": content,
        "feature_type": feature_type,
        "causal": causal,
        "override_source": source,
    }
    if interlocus is not None:
        kwargs["interlocus"] = interlocus
    if source_kind is not None:
        kwargs["provenance"] = Provenance(source_kind=source_kind, source_ref=str(source))
    elif feature_type == FeatureType.Dialogue:
        kind = SourceKind.User if source == ghost.get_user_name() else SourceKind.Self
        kwargs["provenance"] = Provenance(source_kind=kind, source_ref=str(source))
    kwargs["metadata"] = {
        "memory_kind": ("utterance" if feature_type == FeatureType.Dialogue else
                        "speculation" if feature_type in {FeatureType.Thought, FeatureType.InnerMonologue} else
                        "executed_event"),
        "executed_at": datetime.utcnow().isoformat(),
    }
    if metadata:
        kwargs["metadata"].update(metadata)
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


def _projection_from_report(report, previous_vec) -> AffectProjection | None:
    """Rebuild the parent affect projection from the committed report.

    The result lets the module commit the exact state that the stages already
    saw. A report without a latent vector returns None, and the caller then
    recomputes the step from the previous accepted state.
    """
    vector = list(getattr(report, "mental_state_latent_vec", None) or [])
    if not vector:
        return None
    updated = FullMentalState.init_from_list(vector)
    previous = FullMentalState.init_from_list(previous_vec) if previous_vec else create_empty_state()
    delta = FullMentalState.init_from_list(
        [new - old for new, old in zip(updated.to_list(), previous.to_list())]
    )
    trace = dict(getattr(report, "affect_trace", None) or {})
    behavior = dict(trace.get("behavior", {})) if isinstance(trace.get("behavior"), dict) else {}
    regulation = str(getattr(report, "regulation", "") or trace.get("regulation", "") or "")
    return AffectProjection(
        state=updated,
        delta=delta,
        trace=trace,
        summary=str(getattr(report, "affect_summary", "") or ""),
        attention_bias=(0.0, 0.0),
        regulation=regulation,
        behavior=behavior,
    )


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
        "user.input", "thought.created", "task.telemetry", "workflow.question", "workflow.event",
        "system.lifecycle",
    })

    def __init__(self, *args, workflow_catalogue=None, embedding_config_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Configuration supplies the action space. The fixed catalogue remains
        # the fallback so direct tests and older callers keep working.
        self.workflow_catalogue = list(workflow_catalogue) if workflow_catalogue else None
        self.embedding_config_path = embedding_config_path
        self._embeddings = None

    def accepts(self, envelope: Envelope) -> bool:
        if envelope.topic == "workflow.question" and envelope.kind == "telemetry":
            return envelope.target in {None, self.module_id}
        return super().accepts(envelope)

    def prepare_input(self, batch: list[Envelope]) -> dict:
        stimuli: list[Stimulus] = []
        self._stimulus_ids: list[int] = []
        self._stimulus_records: list[dict] = []
        self._stimulus_revisions: dict[int, int] = {}
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
            frozen = getattr(self.last_environment, "ghost_view", None)
            if frozen is not None and knoxel_id in frozen.revisions:
                self._stimulus_revisions[knoxel_id] = frozen.revisions[knoxel_id]
            else:
                with self.ghost.lock:
                    self._stimulus_revisions[knoxel_id] = self.ghost.revisions[knoxel_id]
            if not correlation:
                correlation = envelope.correlation_id
        self._batch_correlation = correlation
        # The working target owns the frozen permitted view. Materialized host
        # events remain stimuli, but cannot expose concurrent future memory.
        environment_view = getattr(self.last_environment, "ghost_view", None)
        memory_view = environment_view if environment_view is not None else self.ghost.snapshot()
        sources = [memory_view.get_knoxel_by_id(item.knoxel_id) or self.ghost.get_knoxel_by_id(item.knoxel_id)
                   for item in stimuli]
        sources = [item for item in sources if item is not None]
        frontier = max(((item.tick_id, item.sub_tick_id) for item in sources), default=None)
        scopes = {(item.scope.reality_id, item.scope.frame.value) for item in sources} or None
        if self.embedding_config_path and self._embeddings is None:
            from pm_core.embeddings import EmbeddingService
            from runtime.models import load_config
            self._embeddings = EmbeddingService(load_config(self.embedding_config_path).embedding)
        retrieval = asyncio.run(self.ghost.retrieve_experiences(
            "\n".join(item.text for item in stimuli), self._embeddings, view=memory_view,
            frontier=frontier, scopes=scopes, cancel=self.run_stop,
        ))
        self._memory_view = memory_view
        self._memory_candidates = retrieval["matches"]
        previous = self.ghost.states[-1] if self.ghost.states else None
        previous_vec = list(previous.mental_state_latent_vec) if previous is not None else []
        age_seconds = 0.0
        if previous is not None:
            age_seconds = max(0.0, (datetime.now() - previous.timestamp).total_seconds())
        request = CognitiveRequest(
            run_id=self.invocation_id or "",
            correlation_id=correlation,
            companion_name=self.ghost.get_companion_name(),
            user_name=self.ghost.get_user_name(),
            state_summary=_state_summary(self.ghost),
            mental_state_latent_vec=previous_vec,
            mental_state_age_seconds=age_seconds,
            personality_vec=list(self.ghost.ghost_config.personality_profile),
            active_goals=active_goal_texts(self.ghost)[:8],
            stimuli=stimuli,
            memory_candidates=[MemoryCandidate(**item) for item in retrieval["matches"]],
            memory_trace=retrieval["trace"],
            capabilities=_capabilities(self.ghost),
            available_workflows=[
                WorkflowOption(**workflow)
                for workflow in (self.workflow_catalogue or DEFAULT_WORKFLOWS)
            ],
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
                source_kind=SourceKind.Workflow,
                metadata={"origin": "workflow.question", "correlation_id": envelope.correlation_id},
            )
            return feature.id, question, "workflow.question"
        if envelope.topic == "workflow.event" and envelope.kind == "telemetry":
            run_id = str(payload.get("run_id", "")).strip()
            state = str(payload.get("event", "")).strip()
            if not run_id or not state:
                return None, "", ""
            # Keep progress detail and evidence so the cycle can judge the task.
            detail = str(payload.get("detail", "") or payload.get("message", "")).strip()
            artifact = str(payload.get("artifact_path", "")).strip()
            if artifact:
                from runtime.consolidation import Consolidator
                Consolidator(self.ghost, clock=self.clock).consolidate(run_id, artifact)
            text = f"Managed run {run_id} {state}"
            if detail:
                text += f": {detail}"
            feature = _feature(
                self.ghost, text, FeatureType.WorldEvent, source=envelope.source or "workflow",
                source_kind=SourceKind.Workflow,
                metadata={
                    "origin": "workflow.event",
                    "run_id": run_id,
                    "event": state,
                    "artifact_path": artifact,
                    "evidence": dict(payload),
                    "executed_at": payload.get("executed_at") or datetime.utcnow().isoformat(),
                },
            )
            return feature.id, text, "workflow.event"
        if envelope.topic == "system.lifecycle":
            knoxel_id = payload.get("knoxel_id")
            if knoxel_id is None:
                return None, "", ""
            knoxel = self._input_knoxel(int(knoxel_id))
            return int(knoxel_id), str(knoxel.content) if knoxel is not None else "", "system.lifecycle"
        if "knoxel_id" not in payload:
            return None, "", ""
        knoxel_id = int(payload["knoxel_id"])
        knoxel = self._input_knoxel(knoxel_id)
        kind = "user.input"
        if knoxel is not None:
            if knoxel.feature_type in {FeatureType.Thought, FeatureType.InnerMonologue}:
                kind = "thought.created"
            elif knoxel.feature_type == FeatureType.SystemMessage:
                kind = "system.lifecycle"
            elif knoxel.feature_type == FeatureType.WorldEvent:
                kind = "workflow.event"
        return knoxel_id, str(knoxel.content) if knoxel is not None else "", kind

    def _input_knoxel(self, knoxel_id: int):
        """Resolve a supplied ID through the frozen invocation grant."""
        view = getattr(self.last_environment, "ghost_view", None)
        if view is not None:
            if knoxel_id not in view.revisions:
                raise KeyError(f"Stimulus is outside the invocation view: {knoxel_id}")
            return view.get(knoxel_id)
        return self.ghost.get_knoxel_by_id(knoxel_id)

    def publish_output(self, report: dict, batch: list[Envelope]) -> None:
        parsed = CognitiveReport.model_validate(report)
        correlation = self._batch_correlation or (batch[0].correlation_id if batch else "")
        cause = batch[0].id if batch else None
        changes = self.ghost.begin_batch(
            self.invocation_id or _new_id(), new_tick=True, cancel=self.stop_event,
        )
        tick = changes.tick_id
        view = changes.view()
        for source_id, revision in self._stimulus_revisions.items():
            changes.expect_revision(source_id, revision)
        for memory in parsed.selected_memories:
            changes.expect_revision(memory.knoxel_id, memory.revision)
            for source_id in memory.source_ids:
                if source_id in self._memory_view.revisions:
                    changes.expect_revision(source_id, self._memory_view.revisions[source_id])

        # Apply the appraisal and goal proposals into this same batch before the
        # frame. A failure here aborts the whole batch, so stale proposals never
        # partly update accepted state. The parent pipeline already projected
        # the affect step; reuse it so accepted state matches the stage input.
        previous = self.ghost.states[-1] if self.ghost.states else None
        previous_vec = list(previous.mental_state_latent_vec) if previous is not None else None
        projection = _projection_from_report(parsed, previous_vec)
        appraisal_feature, mental_state = apply_appraisal(
            changes, view, parsed.appraisal,
            previous_vec=previous_vec,
            evidence_ids=[int(item) for item in self._stimulus_ids],
            companion_name=self.ghost.get_companion_name(),
            projection=projection,
            memory_trace=parsed.memory_trace,
        )
        goal_records = apply_goal_changes(
            changes, view, parsed.goal_changes,
            evidence_ids=[int(item) for item in self._stimulus_ids],
        )
        created_goal_ids = [
            int(record["intention_id"]) for record in goal_records
            if record.get("created") and record.get("intention_id") is not None
        ]
        goal_ids = [int(item.id) for item in active_intentions(changes.view())]
        belief_receipt = update_beliefs(changes, parsed.belief_updates, user_name=self.ghost.get_user_name(),
                                      permitted_source_ids=set(self._stimulus_ids))
        experience_ids = consolidate_cycle(changes, self._stimulus_ids, appraisal_feature, parsed.affect_summary)

        frame = changes.create(AttentionFrameKnoxel(
            content=parsed.frame_reason or parsed.summary,
            tick_id=tick,
            focus_knoxel_ids=list(parsed.focus_ids),
            winning_attention_ids=list(parsed.winning_attention_ids),
            active_goal_ids=goal_ids or list(parsed.active_goal_ids),
            control_mode=parsed.control_mode,
            frame_reason=parsed.frame_reason or parsed.summary,
        ))
        created_order: list[int] = [frame.id, appraisal_feature.id, *created_goal_ids, *experience_ids,
                                    *belief_receipt["accepted_ids"]]
        published: list[tuple] = []
        rejected = []
        queued = []
        duplicates = []
        for dispatch in parsed.dispatches:
            if dispatch.status == "queued":
                queued.append(dispatch)
                continue
            if dispatch.status != "dispatched":
                rejected.append(dispatch)
                continue
            existing = self._existing_action_identity(dispatch)
            if existing is not None:
                duplicates.append((dispatch, existing))
                continue
            action = changes.create(ActionSchemaKnoxel(
                content=dispatch.content,
                action_type=ACTION_BY_NAME.get(dispatch.action_type, ActionType.Reply),
                attention_frame_id=frame.id,
                intention_ids=list(goal_ids),
                metadata={
                    "action_id": dispatch.action_id,
                    "route": dispatch.workflow_name,
                    "workflow_name": dispatch.workflow_name,
                    "depends_on": list(dispatch.depends_on),
                    "requires_completed": list(dispatch.requires_completed),
                    "arguments": dict(dispatch.arguments),
                    "affect_summary": parsed.affect_summary,
                    "regulation": parsed.regulation,
                    "memory_evidence": [item.model_dump(mode="json") for item in parsed.selected_memories],
                    "memory_trace": parsed.memory_trace,
                    "belief_updates": belief_receipt,
                },
            ))
            created_order.append(action.id)
            published.append((dispatch, action.id))
        changes.set_gwt(
            GwtTurnRecord(
                tick_id=tick, attention_frame_id=frame.id,
                knoxel_ids=list(dict.fromkeys([*self._stimulus_ids, *created_order,
                                             *(item.knoxel_id for item in parsed.selected_memories)])),
                summary=(
                    parsed.subjective_experience.strip()
                    or parsed.affect_summary.strip()
                    or parsed.summary
                ),
            ),
            CycleState(
                tick_id=tick, attention_frame_id=frame.id, published_slots=["main"],
                stimuli_ids=list(self._stimulus_ids),
                mental_state_latent_vec=mental_state.to_list(),
            ),
        )
        receipt = self.ghost.commit(changes)
        tick = receipt.tick_id

        memory_texts = [item.text for item in parsed.selected_memories]
        self.bus.put(_envelope(self.module_id, "data", "gwt.updated", "", {"gwt_tick": tick}))
        for dispatch, created_id in published:
            # A repeated commit returns the original receipt and does not apply
            # the new batch. Resolve the accepted action by durable identity and
            # tick so the published id is always committed.
            committed_id = self._committed_action_id(dispatch, tick) or created_id
            self.bus.put(_envelope(
                self.module_id, "data", "action.request", correlation,
                {
                    "knoxel_id": committed_id,
                    "gwt_tick": tick,
                    "action_id": dispatch.action_id,
                    "workflow_name": dispatch.workflow_name,
                    "content": dispatch.content,
                    "arguments": dict(dispatch.arguments),
                    "stimuli": list(self._stimulus_records),
                    "memories": memory_texts,
                    "memory_evidence": [item.model_dump(mode="json") for item in parsed.selected_memories],
                    "workspace": parsed.subjective_experience or parsed.frame_reason,
                    "affect": parsed.affect_summary,
                    "regulation": parsed.regulation,
                },
                target=dispatch.workflow_name,
                causation_id=cause,
            ))
        for dispatch in queued:
            self._publish_receipt(correlation, dispatch, "queued", cause)
        for dispatch in rejected:
            self._publish_receipt(correlation, dispatch, "rejected", cause)
        for dispatch, existing_id in duplicates:
            self._publish_receipt(
                correlation, dispatch, "duplicate", cause,
                detail=f"active action {existing_id} has identity {dispatch.action_id}",
            )

    def _committed_action_id(self, dispatch, tick: int) -> int | None:
        """Return the accepted action id for this identity and tick, if any."""
        for knoxel in self.ghost.all_knoxels.values():
            if not isinstance(knoxel, ActionSchemaKnoxel):
                continue
            if int(knoxel.tick_id) != int(tick):
                continue
            if str(knoxel.metadata.get("action_id", "")) == dispatch.action_id:
                return int(knoxel.id)
        return None

    def _existing_action_identity(self, dispatch) -> int | None:
        """Return the existing non-reply action with this identity, if any.

        Replies may repeat their content. A tool or start action keeps its
        durable identity, so a repeated request does not start another task.
        """
        if ACTION_BY_NAME.get(dispatch.action_type, ActionType.Reply) == ActionType.Reply:
            return None
        for knoxel in self.ghost.all_knoxels.values():
            if not isinstance(knoxel, ActionSchemaKnoxel):
                continue
            if str(knoxel.metadata.get("action_id", "")) == dispatch.action_id:
                return int(knoxel.id)
        return None

    def _publish_receipt(self, correlation: str, dispatch, status: str, cause,
                         detail: str = "") -> None:
        payload = {
            "action_id": dispatch.action_id,
            "workflow_name": dispatch.workflow_name,
            "status": status,
            "rejection_reason": dispatch.rejection_reason,
        }
        if detail:
            payload["detail"] = detail
        self.bus.put(_envelope(
            self.module_id, "telemetry", "action.receipt", correlation, payload,
            causation_id=cause,
        ))

    def on_terminate(self) -> None:
        """Record a clean shutdown once the module stops normally.

        An abrupt process exit never reaches this hook, so its next startup
        records an uncertain end interval instead of a clean timestamp.
        """
        from runtime.lifecycle import record_shutdown
        record_shutdown(self.ghost)


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
            selected_action=intent,
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
            metadata=utterance_metadata(changes.view(), batch[0].payload.get("knoxel_id") if batch else None),
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


class SelfManagement(CognitiveWorkflow):
    """Deterministic managed-run operations as a workflow module.

    It accepts typed actions targeting ``self_management`` and maps the action
    arguments into one typed request. The workflow's tools write the durable
    run receipt; this module publishes an explicit action receipt and never
    invents a completed run.
    """

    DATA_TOPICS = frozenset({"action.request"})
    WORKFLOW_NAME = "self_management"
    managed_workflows = frozenset()

    def accepts(self, envelope: Envelope) -> bool:
        if envelope.topic == "action.request" and envelope.kind == "data":
            return envelope.target in {None, self.module_id, self.WORKFLOW_NAME} | self.managed_workflows
        return super().accepts(envelope)

    def prepare_input(self, batch: list[Envelope]) -> dict:
        payload = batch[0].payload
        action = self.ghost.get_knoxel_by_id(payload["knoxel_id"]) if "knoxel_id" in payload else None
        arguments = dict(payload.get("arguments") or {})
        if batch[0].target in self.managed_workflows:
            arguments = {
                "operation": "start", "workflow_name": batch[0].target,
                "instructions": str(payload.get("content", "")), "arguments": arguments,
            }
        request = SelfManagementRequest(
            operation=str(arguments.get("operation", "status")),
            action_id=str(payload.get("action_id", arguments.get("action_id", ""))),
            run_id=str(arguments.get("run_id", "")),
            workflow_name=str(arguments.get("workflow_name", "")),
            instructions=str(arguments.get("instructions", getattr(action, "content", ""))),
            parent_correlation=str(arguments.get("parent_correlation", batch[0].correlation_id)),
            request_id=str(arguments.get("request_id", "")),
            question_id=str(arguments.get("question_id", "")),
            answer=str(arguments.get("answer", "")),
            arguments=dict(arguments.get("arguments", {})),
        )
        return request.model_dump(mode="json")

    def publish_output(self, report: dict, batch: list[Envelope]) -> None:
        correlation = batch[0].correlation_id if batch else ""
        cause = batch[0].id if batch else None
        self.bus.put(_envelope(
            self.module_id, "telemetry", "action.receipt", correlation,
            {
                "workflow_name": self.WORKFLOW_NAME,
                "status": "accepted" if report.get("accepted") else "rejected",
                "operation": report.get("operation", ""),
                "summary": report.get("summary", ""),
                "receipt": report.get("receipt", {}),
            },
            causation_id=cause,
        ))


class AssistantDialogue(CognitiveWorkflow):
    """Real Assistant dialogue. Tool-capable session with a separate contract.

    It accepts typed actions targeting ``dialogue_assistant``. On an accepted
    reply it commits one public dialogue feature and publishes
    ``dialogue.reply``. A management claim without an executed tool receipt is
    rejected instead of committed.
    """

    DATA_TOPICS = frozenset({"action.request"})
    WORKFLOW_NAME = "dialogue_assistant"

    def accepts(self, envelope: Envelope) -> bool:
        if envelope.topic == "action.request" and envelope.kind == "data":
            return envelope.target in {None, self.module_id, self.WORKFLOW_NAME}
        return super().accepts(envelope)

    def prepare_input(self, batch: list[Envelope]) -> dict:
        payload = batch[0].payload
        action = self.ghost.get_knoxel_by_id(payload["knoxel_id"]) if "knoxel_id" in payload else None
        intent = str(action.content) if action is not None else str(payload.get("content", ""))
        current = str(payload.get("stimuli", [{}])[-1].get("text", "")) if payload.get("stimuli") else intent
        request = AssistantRequest(
            companion_name=self.ghost.get_companion_name(),
            user_name=self.ghost.get_user_name(),
            character_card=self.ghost.get_simple_character_story_block(),
            private_ontology=self._ontology(),
            temporal_context=self.ghost.build_body_and_situation_addendum(),
            prior_dialogue=self._prior_dialogue(current),
            current_user_message=current or intent or "(no message)",
            memories=[str(item) for item in payload.get("memories", []) if str(item).strip()],
            workspace=str(payload.get("workspace", "")),
            tools=list(TOOL_DESCRIPTORS),
            tool_grant=list(TOOL_NAMES),
        )
        return request.model_dump(mode="json")

    def _ontology(self) -> str:
        config = self.ghost.ghost_config
        parts = [str(item) for item in config.supported_capabilities]
        parts += [f"Cannot: {item}" for item in config.unsupported_capabilities]
        parts += [str(item) for item in config.capability_notes]
        return "\n".join(f"- {part}" for part in parts) or "(none declared)"

    def _prior_dialogue(self, current: str) -> list[AssistantMessage]:
        turns = _recent_public_dialogue(self.ghost)
        messages: list[AssistantMessage] = []
        for index, turn in enumerate(turns):
            is_last = index == len(turns) - 1
            if is_last and turn.speaker == self.ghost.get_user_name() and turn.text == current:
                continue
            role = "user" if turn.speaker == self.ghost.get_user_name() else "assistant"
            messages.append(AssistantMessage(role=role, content=_clip(turn.text, 1500)))
        return messages

    def publish_output(self, report: dict, batch: list[Envelope]) -> None:
        parsed = AssistantReport.model_validate(report)
        correlation = batch[0].correlation_id if batch else ""
        cause = batch[0].id if batch else None
        unsupported = unsupported_management_claims(parsed.final_text, parsed.tool_receipts)
        if not parsed.accepted or unsupported:
            reason = parsed.rejection_reason or (
                "unsupported management claim: " + ", ".join(unsupported)
            )
            self.bus.put(_envelope(
                self.module_id, "telemetry", "action.receipt", correlation,
                {
                    "status": "rejected",
                    "workflow_name": self.WORKFLOW_NAME,
                    "rejection_reason": reason,
                },
                causation_id=cause,
            ))
            return
        changes = self.ghost.begin_batch(self.invocation_id or _new_id(), cancel=self.stop_event)
        _feature(
            self.ghost, parsed.final_text, FeatureType.Dialogue,
            source=self.ghost.get_companion_name(), interlocus=InterlocusType.Public,
            changes=changes,
        )
        receipt = self.ghost.commit(changes)
        self.bus.put(_envelope(
            self.module_id, "data", "dialogue.reply", correlation,
            {
                "knoxel_id": receipt.added_ids[0],
                "action_id": batch[0].payload.get("action_id", "") if batch else "",
                "tool_receipts": [item.model_dump(mode="json") for item in parsed.tool_receipts],
            },
            causation_id=cause,
        ))


class StewardModule(CognitiveWorkflow):
    """Route typed actions to the dormant steward.

    The module never activates the steward. While dormant, every operation is
    refused and no run state changes. A caller activates ``app.steward`` when a
    representative is wanted.
    """

    DATA_TOPICS = frozenset({"action.request"})
    WORKFLOW_NAME = "steward"

    def accepts(self, envelope: Envelope) -> bool:
        if envelope.topic == "action.request" and envelope.kind == "data":
            return envelope.target in {None, self.module_id, self.WORKFLOW_NAME}
        return super().accepts(envelope)

    def prepare_input(self, batch: list[Envelope]) -> dict:
        payload = batch[0].payload
        arguments = dict(payload.get("arguments") or {})
        request = StewardRequest(
            operation=str(arguments.get("operation", "status")),
            run_id=str(arguments.get("run_id", "")),
            question_id=str(arguments.get("question_id", "")),
            answer=str(arguments.get("answer", "")),
            instructions=str(arguments.get("instructions", payload.get("content", ""))),
            request_id=str(arguments.get("request_id", "")),
        )
        return request.model_dump(mode="json")

    def publish_output(self, report: dict, batch: list[Envelope]) -> None:
        correlation = batch[0].correlation_id if batch else ""
        cause = batch[0].id if batch else None
        self.bus.put(_envelope(
            self.module_id, "telemetry", "action.receipt", correlation,
            {
                "workflow_name": self.WORKFLOW_NAME,
                "status": "accepted" if report.get("accepted") else "rejected",
                "operation": report.get("operation", ""),
                "summary": report.get("summary", ""),
                "receipt": report.get("result", {}),
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
            source_kind=SourceKind.Self,
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
    "AssistantDialogue",
    "SelfManagement",
    "StewardModule",
    "InnerThoughtLoop",
    "ScheduleSubprocesses",
    "CheckSubprocess",
    "ToolPlusDialogue",
]
