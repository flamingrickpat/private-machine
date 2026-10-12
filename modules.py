"""Cognitive module subclasses and the shared task table.

Every module is a ``BusModule`` or ``WorkflowModule``. Fixed routing and
fixed output are explicit prototype behavior here and in ``workflows/roles.py``,
never LLM output.
"""
from __future__ import annotations

from pathlib import Path
import asyncio
from datetime import datetime
from threading import RLock
import yaml

from pm_core.model.affect_dynamics import AffectProjection
from pm.records.cycle_state import CycleState
from pm.records.global_workspace import GwtTurnRecord
from pm.records.memory_records import ActionSchemaRecord, Feature
from pm.records.base import Provenance
from pm.records.enums import ActionType, FeatureType, InterlocusType, SourceKind
from pm.records.cognitive_records import AttentionFrameRecord
from pm.records.mental_state import FullMentalState, create_empty_state
from runtime.experience import consolidate_cycle, update_beliefs, utterance_metadata

from pm.bus.envelopes import Envelope, new_id
from pm.host.application import BusModule
from pm.host.workflow_module import PROJECT_ROOT, WorkflowModule
from pm.memory.store_tools import build_tools
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
from workflows.public_context import build_public_context
from runtime.publication import acknowledge_reply, commit_public_reply, exchange_record, publish_reply

ACTION_BY_NAME = {member.value: member for member in ActionType}

ACTION_TYPE = {
    "dialogue": ActionType.Reply,
    "tool_dialogue": ActionType.ToolCallAndReply,
    "schedule": ActionType.ToolCall,
    "status": ActionType.ToolCall,
}


def _envelope(source, kind, topic, correlation_id, payload, target=None, causation_id=None):
    return Envelope(
        id=new_id(), source=source, kind=kind, topic=topic,
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


def _recent_public_dialogue(ghost, limit: int = 4, *, exclude_ids=()) -> list[DialogueTurn]:
    """Bound prior turns and exclude the input rendered as the current message.

    Rendering a pending user input twice invents a repeated conversational turn.
    The original records remain complete in the canonical MemoryStore.
    """
    features = [
        knoxel for knoxel in ghost.all_features
        if knoxel.feature_type == FeatureType.Dialogue
        and knoxel.interlocus == InterlocusType.Public
        and str(knoxel.content).strip()
        and knoxel.id not in exclude_ids
    ]
    features.sort(key=_recency)
    return [
        DialogueTurn(speaker=str(knoxel.override_source or ghost.get_entity_name(knoxel.entity_id) or "unknown"),
                     text=_clip(str(knoxel.content), 240))
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


class UserInputProxy(BusModule):
    """Wraps terminal input and prints public replies."""

    DATA_TOPICS = frozenset({"dialogue.reply"})

    def __init__(self, module_id, bus, on_error, clock, ghost, emit_reply):
        super().__init__(module_id, bus, on_error, clock, None)
        self.ghost = ghost
        self.emit_reply = emit_reply

    def submit(self, text: str, *, session: str = "") -> str:
        correlation_id = new_id()
        feature = _feature(
            self.ghost, text, FeatureType.Dialogue,
            source=self.ghost.get_user_name(), interlocus=InterlocusType.Public,
            metadata={"exchange_correlation": correlation_id, "exchange_status": "pending", "session": session},
        )
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
                acknowledge_reply(self.ghost, feature.id)


class CognitiveWorkflow(WorkflowModule):
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

        changes = self.ghost.begin_batch(self.invocation_id or new_id(), new_tick=True, cancel=self.stop_event)
        tick = changes.tick_id
        frame = changes.create(AttentionFrameRecord(
            content=summary, tick_id=tick, focus_knoxel_ids=list(incoming_ids),
            active_goal_ids=[], winning_attention_ids=[], control_mode="reactive", frame_reason=summary,
        ))
        action_ids: list[int] = []
        for decision in decisions:
            action = changes.create(ActionSchemaRecord(
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

    WORKFLOW_NAME = "main_cognition"
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

    def handle_batch(self, batch: list[Envelope]) -> None:
        """Close a user exchange after a model protocol failure.

        Logic errors still reach application shutdown. Invalid model output
        retains its workflow artifacts and receives an explicit public receipt.
        """
        from pm.llm.http_client import ModelError
        from pm.memory.snapshot import GhostConflict
        from workflows.main_cognition.stage_support import StageOutputError
        try:
            super().handle_batch(batch)
        except (ModelError, StageOutputError, GhostConflict) as error:
            if self.stop_event.is_set() or self.run_stop.is_set():
                return
            outcome = "stale" if isinstance(error, GhostConflict) else "protocol_failed"
            if not any(envelope.source == "user_input" for envelope in batch):
                # Public resolution can change a source marker while a task
                # event cycle reads it. Reject that cycle without shutdown.
                self.bus.put(_envelope(self.module_id, "telemetry", "action.receipt",
                    batch[0].correlation_id if batch else "", {"status": outcome,
                    "workflow_name": self.WORKFLOW_NAME, "rejection_reason": str(error),
                    "recovery_outcome": "cycle_rejected"}))
            for envelope in batch:
                if envelope.source != "user_input":
                    continue
                reply_id = commit_public_reply(self.ghost,
                    "I could not complete the decision for that request. No task result was verified.",
                    envelope.correlation_id, commit_id="cognitive-recovery:" + envelope.id,
                    recovery=outcome, cancel=self.stop_event)
                self.bus.put(_envelope(self.module_id, "telemetry", "action.receipt", envelope.correlation_id,
                    {"status": "failed", "workflow_name": self.WORKFLOW_NAME,
                     "rejection_reason": f"{type(error).__name__}: {error}",
                     "recovery_outcome": "published" if reply_id else "already_resolved",
                     "public_reply_id": reply_id}, causation_id=envelope.id))
                if reply_id is not None:
                    publish_reply(self.bus, self.module_id, envelope.correlation_id, reply_id,
                                  cause=envelope.id, recovery_outcome=outcome)

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
        # A queued task event cannot take the cause of a newer user request.
        # Individual stimuli retain their own correlation for attribution.
        user_correlations = [item.correlation_id for item in stimuli if item.source == "user_input"]
        if user_correlations:
            correlation = user_correlations[-1]
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
            from pm.memory.embeddings import EmbeddingService
            from pm.llm.config import load_config
            self._embeddings = EmbeddingService(load_config(self.embedding_config_path).embedding)
        original_query = "\n".join(item.text for item in stimuli)
        rewrite = getattr(self, "retrieval_query_provider", None)
        rewrite_result = (rewrite(original_query) if rewrite is not None and any(item.source == "user_input" for item in stimuli)
                          else {"query": original_query, "evidence_use": "general", "status": "literal"})
        query = rewrite_result["query"]
        retrieval = asyncio.run(self.ghost.retrieve_experiences(
            query, self._embeddings, view=memory_view,
            frontier=frontier, scopes=scopes, cancel=self.run_stop,
        ))
        retrieval["trace"]["original_query"] = original_query
        retrieval["trace"]["query_rewrite"] = rewrite_result
        if rewrite_result["evidence_use"] == "event":
            from pm.memory.retrieval import evidence_role
            records = memory_view.serialize_knoxels()
            selected_sources = {key for item in retrieval["matches"] for key in item["source_ids"] if key in records}
            original_sources = {key for key in selected_sources
                                if evidence_role(memory_view, records[key]) in {"user_report", "runtime_event", "world_observation"}}
            # An answer to a recall question cannot become event evidence for
            # its repetition. Keep speech available for explicit speech recall.
            rejected = [item for item in retrieval["matches"]
                        if item["evidence_role"] in {"companion_utterance", "intent", "imagined", "interpretation"}
                        or item["evidence_role"] == "derived" and not original_sources.intersection(item["source_ids"])]
            rejected_ids = {item["knoxel_id"] for item in rejected}
            retrieval["matches"] = [item for item in retrieval["matches"] if item["knoxel_id"] not in rejected_ids]
            retrieval["trace"]["speech_only_excluded_ids"] = sorted(rejected_ids)
        self._memory_view = memory_view
        self._memory_candidates = retrieval["matches"]
        previous = self.ghost.states[-1] if self.ghost.states else None
        previous_vec = list(previous.mental_state_latent_vec) if previous is not None else []
        age_seconds = 0.0
        if previous is not None:
            age_seconds = max(0.0, (datetime.now() - previous.timestamp).total_seconds())
        provider = getattr(self, "capability_provider", None)
        public = build_public_context(self.ghost, {"stimuli": [item.model_dump(mode="json") for item in stimuli]},
                                      capabilities=provider() if provider else None)
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
            public_context=public.model_dump(mode="json"),
            managed_runs=[{key: run.get(key) for key in
                          ("run_id", "workflow_name", "state", "instructions", "parent_correlation")}
                          for run in list(self.ghost.managed_runs.values())[-16:]],
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
            self.invocation_id or new_id(), new_tick=True, cancel=self.stop_event,
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

        frame = changes.create(AttentionFrameRecord(
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
            from workflows.public_context import Presentation
            presentation = Presentation.model_validate(dispatch.presentation)
            if presentation.mode == "deceive" and (
                not presentation.purpose or not presentation.known_fact_ids
                or any(key not in view.revisions for key in presentation.known_fact_ids)
            ):
                raise ValueError("The proposed deception lacks retained factual context.")
            action = changes.create(ActionSchemaRecord(
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
                    "presentation": presentation.model_dump(mode="json"),
                    "dispatch_status": "pending",
                    "dispatch_target": dispatch.workflow_name,
                    "dispatch_correlation": correlation,
                    "dispatch_payload": {"action_id": dispatch.action_id, "workflow_name": dispatch.workflow_name,
                        "content": dispatch.content, "arguments": dict(dispatch.arguments),
                        "stimuli": list(self._stimulus_records), "memories": [item.text for item in parsed.selected_memories],
                        "memory_evidence": [item.model_dump(mode="json") for item in parsed.selected_memories],
                        "workspace": parsed.subjective_experience or parsed.frame_reason,
                        "affect": parsed.affect_summary, "regulation": parsed.regulation},
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
            if not isinstance(knoxel, ActionSchemaRecord):
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
            if not isinstance(knoxel, ActionSchemaRecord):
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
        from pm.app.lifecycle_records import record_shutdown
        record_shutdown(self.ghost)


class DummyDialogue(CognitiveWorkflow):
    """Labeled demonstration fixture: echoes the action content as a reply."""

    DATA_TOPICS = frozenset({"action.dialogue"})

    def publish_output(self, report: dict, batch: list[Envelope]) -> None:
        replies = report.get("replies", [])
        causes = [next(envelope.id for envelope in batch if envelope.correlation_id == reply["correlation_id"])
                  for reply in replies]
        changes = self.ghost.begin_batch(self.invocation_id or new_id(), cancel=self.stop_event)
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


class PublicDialogue(CognitiveWorkflow):
    """Shared stale guards and observable recovery for public workflows."""

    capability_provider = None
    grounding_recovery = None

    def public_context(self, payload):
        capabilities = self.capability_provider() if self.capability_provider else None
        with self.ghost.lock:
            self._public_context = build_public_context(self.ghost, payload, capabilities=capabilities)
        return self._public_context

    def handle_batch(self, batch: list[Envelope]) -> None:
        # Each public action owns one report and one exchange resolution.
        # A second envelope must not disappear inside a merged mailbox batch.
        for envelope in batch:
            exchange = exchange_record(self.ghost, envelope.correlation_id)
            if exchange is not None and exchange.metadata.get("exchange_status") == "resolved":
                continue
            # Preparation can fail before a new snapshot exists. A prior task
            # receipt must never become the recovery evidence for this action.
            self._public_context = None
            try:
                super().handle_batch([envelope])
            except Exception as error:
                if self.stop_event.is_set() or self.run_stop.is_set():
                    return
                self.reject_reply([envelope], f"{type(error).__name__}: {error}", status="failed")

    def reject_reply(self, batch, reason: str, *, status: str = "rejected") -> None:
        """Retain a failed receipt and close the exchange with explicit recovery."""
        envelope = batch[0]
        stale = status == "stale"
        text = ("That reply became stale before I could send it. Please ask again."
                if stale else "That reply failed before I could send it. Please ask again."
                if status == "failed" else "I could not ground that reply in the available evidence. Please ask again.")
        context = getattr(self, "_public_context", None)
        if status in {"rejected", "failed"} and context is not None:
            from pm.memory.snapshot import GhostConflict
            from runtime.publication import validate_public_context
            try:
                with self.ghost.lock:
                    validate_public_context(self.ghost, context)
            except GhostConflict as error:
                status, stale, reason = "stale", True, str(error)
                text = "That reply became stale before I could send it. Please ask again."
        recovery = None
        # A protocol error establishes a failed request, not a user report.
        # Keep that failure explicit instead of inventing its conversational cause.
        if status == "rejected" and context is not None and not context.task_receipt and self.grounding_recovery is not None:
            original = exchange_record(self.ghost, envelope.correlation_id)
            if original is not None and original.id == context.user_frontier:
                from pm.llm.http_client import ModelError
                from workflows.main_cognition.stage_support import StageOutputError
                try:
                    sources = [item for item in context.evidence if item.get("evidence_role") in
                               {"user_report", "runtime_event", "world_observation"}]
                    fresh = self.public_context({"stimuli": [{"knoxel_id": original.id, "text": original.content}],
                                                 "memory_evidence": sources})
                    fresh.presentation = context.presentation
                    fresh.capabilities["evidence_limit"]["source_use"] = context.capabilities.get(
                        "evidence_limit", {}).get("source_use", "event")
                    recovery = self.grounding_recovery(fresh)
                    # The kernel already finished. Its virtual outbox no longer
                    # exports new files, so retain this attempt in host artifacts.
                    artifact = self.runtime_root / self.module_id / self.invocation_id / "workspace" / "outbox" / "grounding-recovery.yaml"
                    artifact.parent.mkdir(parents=True, exist_ok=True)
                    artifact.write_text(yaml.safe_dump(recovery, allow_unicode=True), encoding="utf-8")
                    if recovery["accepted"]:
                        from workflows.public_context import PublicContext
                        self._public_context = PublicContext.model_validate(recovery["context"])
                        self.bus.put(_envelope(self.module_id, "telemetry", "action.receipt", envelope.correlation_id,
                            {"status": status, "workflow_name": self.WORKFLOW_NAME, "rejection_reason": reason,
                             "recovery_outcome": "grounding_recovered", "recovery": recovery}, causation_id=envelope.id))
                        self.accept_reply(recovery["candidate"], batch, recovery_outcome="grounding_recovered")
                        return
                except (ModelError, StageOutputError, ValueError, OSError) as error:
                    reason += f"; scoped recovery failed: {type(error).__name__}: {error}"
        if context is not None and context.task_receipt:
            task = context.task_receipt
            if task.get("task_verified"):
                spec = task["report"]["note_spec"]
                text = f"The note passed exact byte verification at {spec['path']}: {spec['text']!r}. My conversational reply failed validation."
            else:
                text = (f"The {task['workflow_name']} worker is {task['state']}. "
                        "Its receipt does not verify the requested note. My conversational reply failed validation.")
        reply_id = commit_public_reply(self.ghost, text, envelope.correlation_id,
                                      commit_id=f"public-recovery:{envelope.id}", recovery=status,
                                      cancel=self.stop_event)
        self.bus.put(_envelope(self.module_id, "telemetry", "action.receipt", envelope.correlation_id,
                               {"status": status, "workflow_name": self.WORKFLOW_NAME,
                                "rejection_reason": reason, "recovery_outcome": "published" if reply_id else "already_resolved",
                                "public_reply_id": reply_id, **({"recovery": recovery} if recovery is not None else {})}, causation_id=envelope.id))
        if reply_id is not None:
            publish_reply(self.bus, self.module_id, envelope.correlation_id, reply_id,
                          cause=envelope.id, recovery_outcome=status)

    def accept_reply(self, text: str, batch, *, tool_receipts=None, recovery_outcome="") -> None:
        """Commit an evidence-guarded reply, or expose its stale recovery result."""
        from pm.memory.snapshot import GhostConflict
        envelope = batch[0]
        try:
            if self.capability_provider is not None:
                current = self.capability_provider().get("minecraft", {})
                prior = self._public_context.capabilities.get("minecraft", {})
                if (current.get("status"), current.get("tools")) != (prior.get("status"), prior.get("tools")):
                    raise GhostConflict("Minecraft capability changed during public generation.")
            if self._public_context.task_receipt.get("task_verified"):
                from runtime.reply_recovery import task_evidence
                try:
                    current_task = task_evidence(self._public_context.task_receipt)
                except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
                    self._public_context.task_receipt = {**self._public_context.task_receipt,
                        "task_verified": False, "verification_error": type(error).__name__}
                    raise GhostConflict("Task evidence became unreadable during public generation.") from error
                if not current_task.get("task_verified"):
                    self._public_context.task_receipt = current_task
                    raise GhostConflict("Verified task bytes changed during public generation.")
            reply_id = commit_public_reply(
                self.ghost, text, envelope.correlation_id, commit_id=self.invocation_id or new_id(),
                action_id=None if recovery_outcome else envelope.payload.get("knoxel_id"), context=self._public_context,
                recovery=recovery_outcome, cancel=self.stop_event,
            )
        except GhostConflict as error:
            self.reject_reply(batch, str(error), status="stale")
            return
        if reply_id is not None:
            publish_reply(self.bus, self.module_id, envelope.correlation_id, reply_id, cause=envelope.id,
                          action_id=envelope.payload.get("action_id", ""), tool_receipts=tool_receipts or [],
                          recovery_outcome=recovery_outcome)


class StoryDialogue(PublicDialogue):
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
        public = self.public_context(payload)
        request = StoryRequest(
            companion_name=self.ghost.get_companion_name(),
            user_name=self.ghost.get_user_name(),
            character_card_story=self.ghost.get_simple_character_story_block(),
            supported_capabilities=[str(item) for item in self.ghost.ghost_config.supported_capabilities],
            unsupported_capabilities=[str(item) for item in self.ghost.ghost_config.unsupported_capabilities],
            capability_notes=[str(item) for item in self.ghost.ghost_config.capability_notes],
            temporal_context=public.temporal_context,
            prior_dialogue=_recent_public_dialogue(self.ghost, exclude_ids={item["knoxel_id"] for item in stimuli if "knoxel_id" in item}),
            current_user_message=current or intent or "(no message)",
            feelings=[public.affect] if public.affect else [],
            memories=public.memories,
            workspace=public.workspace,
            selected_action=intent,
            public_context=public,
        )
        return request.model_dump(mode="json")

    def publish_output(self, report: dict, batch: list[Envelope]) -> None:
        story = StoryReport.model_validate(report)
        correlation = batch[0].correlation_id if batch else ""
        cause = batch[0].id if batch else None
        if not story.accepted:
            reason = "; ".join(violation.reason for violation in story.ontology.violations)
            reason = reason or "; ".join(story.ontology.fact_conflicts) or "ontology boundary"
            self.reject_reply(batch, reason)
            return
        self.accept_reply(story.candidate, batch)


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
        original = exchange_record(self.ghost, batch[0].correlation_id)
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
            arguments={**dict(arguments.get("arguments", {})),
                       "_public_reply_requested": bool(action is not None and action.action_type == ActionType.ToolCallAndReply),
                       "_public_action_schema_id": getattr(action, "id", None),
                       "_public_original_request": original.content if original is not None else "",
                       "_public_original_request_id": original.id if original is not None else None},
        )
        return request.model_dump(mode="json")

    def publish_output(self, report: dict, batch: list[Envelope]) -> None:
        correlation = batch[0].correlation_id if batch else ""
        cause = batch[0].id if batch else None
        source = self.ghost.get_knoxel_by_id(batch[0].payload.get("knoxel_id"))
        if source is not None:
            from runtime.intents import acknowledge_intent
            acknowledge_intent(self.ghost, source.id)
        self.bus.put(_envelope(
            self.module_id, "telemetry", "action.receipt", correlation,
            {
                "workflow_name": self.WORKFLOW_NAME,
                "status": "accepted" if report.get("accepted") else "rejected",
                "operation": report.get("operation", ""),
                "summary": report.get("summary", ""),
                "receipt": report.get("receipt", {}),
                "action_schema_id": getattr(source, "id", None),
                "public_reply_requested": isinstance(source, ActionSchemaRecord)
                and source.action_type == ActionType.ToolCallAndReply,
            },
            causation_id=cause,
        ))


class AssistantDialogue(PublicDialogue):
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
        public = self.public_context(payload)
        request = AssistantRequest(
            companion_name=self.ghost.get_companion_name(),
            user_name=self.ghost.get_user_name(),
            character_card=self.ghost.get_simple_character_story_block(),
            private_ontology=self._ontology(),
            temporal_context=public.temporal_context,
            prior_dialogue=self._prior_dialogue(current, exclude_ids={item["knoxel_id"] for item in payload.get("stimuli", []) if "knoxel_id" in item}),
            current_user_message=current or intent or "(no message)",
            memories=public.memories,
            workspace=public.workspace,
            tools=list(TOOL_DESCRIPTORS),
            tool_grant=list(TOOL_NAMES),
            selected_action=intent,
            public_context=public,
            minecraft_read=payload.get("arguments", {}).get("minecraft_read", ""),
        )
        return request.model_dump(mode="json")

    def _ontology(self) -> str:
        config = self.ghost.ghost_config
        parts = [str(item) for item in config.supported_capabilities]
        parts += [f"Cannot: {item}" for item in config.unsupported_capabilities]
        parts += [str(item) for item in config.capability_notes]
        return "\n".join(f"- {part}" for part in parts) or "(none declared)"

    def _prior_dialogue(self, current: str, *, exclude_ids=()) -> list[AssistantMessage]:
        turns = _recent_public_dialogue(self.ghost, exclude_ids=exclude_ids)
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
        unsupported = (unsupported_management_claims(parsed.final_text, parsed.tool_receipts)
                       if not self._public_context.task_receipt else [])
        if not parsed.accepted or unsupported:
            reason = parsed.rejection_reason or (
                "unsupported management claim: " + ", ".join(unsupported)
            )
            self.reject_reply(batch, reason)
            return
        self.accept_reply(parsed.final_text, batch, tool_receipts=[item.model_dump(mode="json") for item in parsed.tool_receipts])


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
        changes = self.ghost.begin_batch(self.invocation_id or new_id(), cancel=self.stop_event)
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
