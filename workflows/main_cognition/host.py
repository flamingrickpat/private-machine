"""Module-level tool factory for the real main-cognition parent.

The application supplies one lazily-activated completion callable. This module
binds it, the MemoryStore, and the runtime root into the ``run_stage`` tool the
parent Python role names in its manifest. Each ``run_stage`` call builds a
fresh read-only MemoryStore projection and invokes the real child kernel exactly as
the child's standalone tests do.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from workflows.main_cognition.stage_runner import build_stage_runner
import asyncio
from datetime import datetime
from workflows.main_cognition.affect_dynamics import AffectProjection
from pm.records.cycle_state import CycleState
from pm.records.global_workspace import GwtTurnRecord
from pm.records.memory_records import ActionSchemaRecord
from pm.records.enums import ActionType, FeatureType, SourceKind
from pm.records.cognitive_records import AttentionFrameRecord, AttentionRecord
from pm.records.mental_state import FullMentalState, create_empty_state
from workflows.main_cognition.experience import consolidate_cycle, update_beliefs
from pm.bus.envelopes import Envelope, new_id
from workflows.main_cognition.pipeline import DEFAULT_WORKFLOWS
from workflows.main_cognition.schemas import (
    CognitiveReport,
    CognitiveRequest,
    MemoryCandidate,
    Stimulus,
    WorkflowOption,
)
from workflows.main_cognition.stage_commits import active_goal_texts, active_intentions, apply_appraisal, apply_goal_changes
from workflows.public_speech.public_context import build_public_context
from workflows.public_speech.publication import commit_public_reply, publish_reply
from pm.bus.envelopes import new_envelope
from pm.records.memory_records import new_feature
from workflows.public_speech.public_dialogue import _clip, _recency
from pm.runs.workflow_module import CognitiveWorkflow


def build_main_cognition_tools(
    ghost,
    runtime_root: Path,
    llm_complete: Callable[..., str],
) -> Callable[[object, object], dict]:
    """Return the ``tools_factory(environment, batch)`` for the parent module."""

    def factory(environment, batch) -> dict:
        return {"run_stage": build_stage_runner(ghost, llm_complete, Path(runtime_root),
                                               read_environment=environment)}

    return factory


__all__ = ["build_main_cognition_tools"]


ACTION_BY_NAME = {member.value: member for member in ActionType}


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


def codelet_history(ghost) -> dict[str, dict]:
    """Return per-codelet run history from committed attention frames.

    Each committed ``AttentionFrameRecord`` stores the codelets that ran in
    its cycle under ``metadata["codelet_runs"]``. Runs that stayed silent
    count too, so a quiet LLM codelet still waits out its cooldown and does
    not spend a model call every cycle. The history is derived, never stored
    separately, so it survives restart and has one owner: the MemoryStore.

    ``cycles_since`` is the number of committed cycles after the last run.
    ``seconds_since`` uses the frame's creation time.
    """
    frames = sorted(ghost.all_attention_frames, key=lambda item: (item.tick_id, item.id))
    if not frames:
        return {}
    latest_tick = max(int(item.tick_id) for item in frames)
    now = datetime.utcnow()
    history: dict[str, dict] = {}
    for frame in frames:
        for entry in frame.metadata.get("codelet_runs", []) or []:
            name = str(entry.get("codelet", ""))
            if not name or entry.get("status") not in {"ran", "silent", "failed"}:
                continue
            record = history.setdefault(name, {"runs": 0})
            record["runs"] += 1
            record["last_tick"] = int(frame.tick_id)
            record["cycles_since"] = latest_tick - int(frame.tick_id)
            record["seconds_since"] = max(0.0, (now - frame.timestamp_creation).total_seconds())
    return history


def seconds_since_user_input(ghost, exclude_ids) -> float | None:
    """Return seconds since the newest earlier user message, or None.

    The current batch's own messages are excluded; otherwise every message
    would measure an idle time of zero.
    """
    user = ghost.get_user_name()
    excluded = {int(item) for item in exclude_ids}
    times = [knoxel.timestamp_world_begin for knoxel in ghost.all_features
             if knoxel.feature_type == FeatureType.Dialogue and knoxel.source == user
             and int(knoxel.id) not in excluded]
    if not times:
        return None
    return max(0.0, (datetime.utcnow() - max(times)).total_seconds())


def stale_nominations(nominations, revisions) -> set[str]:
    """Return ids of nominations whose read sources changed before commit.

    ``revisions`` is the commit batch's current revision map. A source that is
    missing from it was removed, which also makes the proposal stale.
    """
    stale = set()
    for nomination in nominations:
        for key, seen in (nomination.source_revisions or {}).items():
            if revisions.get(int(key)) != int(seen):
                stale.add(nomination.nomination_id)
                break
    return stale


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


class MainCogLoop(CognitiveWorkflow):
    """Real cognitive parent: eight stages, committed GWT, typed dispatches.

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

    def __init__(self, *args, workflow_catalogue=None, embedding_config_path=None,
                 codelet_catalogue=None, **kwargs):
        super().__init__(*args, **kwargs)
        # The codelet catalogue is a workflow resource. ``None`` selects the
        # default file; a gate or persona can name another one.
        self.codelet_catalogue = str(codelet_catalogue) if codelet_catalogue else ""
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
                self.bus.put(new_envelope(self.module_id, "telemetry", "action.receipt",
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
                self.bus.put(new_envelope(self.module_id, "telemetry", "action.receipt", envelope.correlation_id,
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
            stimulus_revisions={str(key): int(value) for key, value in self._stimulus_revisions.items()},
            active_goal_records=[
                {"knoxel_id": int(item.id), "text": str(item.content),
                 "revision": int(memory_view.revisions.get(int(item.id), 1))}
                for item in active_intentions(memory_view)[:8]
            ],
            codelet_history=codelet_history(self.ghost),
            seconds_since_user_input=seconds_since_user_input(self.ghost, self._stimulus_ids),
            codelet_catalogue=self.codelet_catalogue,
        )
        return request.model_dump(mode="json")

    def _stimulus_from(self, envelope: Envelope):
        payload = envelope.payload
        if envelope.topic == "workflow.question" and envelope.kind == "telemetry":
            question = str(payload.get("question", "")).strip()
            if not question:
                return None, "", ""
            feature = new_feature(
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
                from workflows.memory_consolidation.artifact_consolidation import Consolidator
                Consolidator(self.ghost, clock=self.clock).consolidate(run_id, artifact)
            text = f"Managed run {run_id} {state}"
            if detail:
                text += f": {detail}"
            feature = new_feature(
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

        # Commit every nomination, winner or not, as an AttentionRecord. A
        # nomination whose read sources changed since the cycle began is
        # marked stale and cannot win. The cycle itself continues: one stale
        # proposal must not cost the user a reply.
        stale = stale_nominations(parsed.nominations, view.revisions)
        trace = dict(parsed.codelet_trace or {})
        attention_ids: list[int] = []
        winning_ids: list[int] = []
        for nomination in parsed.nominations:
            status, reason = nomination.status, nomination.reason
            if nomination.nomination_id in stale:
                status, reason = "stale", "a source changed before commit; was " + (nomination.status or "unscored")
            record = changes.create(AttentionRecord(
                content=nomination.text or nomination.reasoning or nomination.attention_kind,
                tick_id=tick,
                target_knoxel_ids=[int(key) for key in nomination.target_ids],
                attention_kind=nomination.attention_kind,
                score=max(0.0, min(1.0, float(nomination.score))),
                novelty=float(nomination.novelty), urgency=float(nomination.urgency),
                goal_relevance=float(nomination.goal_relevance), reportability=float(nomination.reportability),
                competition_group="main",
                selected=status == "selected",
                from_codelet=nomination.codelet,
                metadata={
                    "nomination_id": nomination.nomination_id, "status": status, "reason": reason,
                    "origin": nomination.origin, "reality": nomination.reality, "family": nomination.family,
                    "cause": list(nomination.cause), "activation": nomination.activation,
                    "source_revisions": dict(nomination.source_revisions), "why": nomination.reasoning,
                    "loop": trace.get("loop", "main"),
                    "catalogue_version": trace.get("catalogue_version", ""),
                    "catalogue_digest": trace.get("catalogue_digest", ""),
                },
            ))
            attention_ids.append(record.id)
            if status == "selected":
                winning_ids.append(record.id)
        frame = changes.create(AttentionFrameRecord(
            content=parsed.frame_reason or parsed.summary,
            tick_id=tick,
            focus_knoxel_ids=list(parsed.focus_ids),
            # Older reports carry no nominations; keep their selected ids.
            winning_attention_ids=winning_ids if parsed.nominations else list(parsed.winning_attention_ids),
            active_goal_ids=goal_ids or list(parsed.active_goal_ids),
            control_mode=parsed.control_mode,
            frame_reason=parsed.frame_reason or parsed.summary,
            metadata={
                "codelet_runs": [{key: run.get(key) for key in
                                  ("codelet", "kind", "status", "reason", "activation", "calls")}
                                 for run in trace.get("runs", [])],
                "codelet_cost": trace.get("cost", {}),
                "catalogue_version": trace.get("catalogue_version", ""),
                "catalogue_digest": trace.get("catalogue_digest", ""),
                "stale_nominations": sorted(stale),
                "attention_record_ids": attention_ids,
            },
        ))
        created_order: list[int] = [frame.id, appraisal_feature.id, *created_goal_ids, *experience_ids,
                                    *belief_receipt["accepted_ids"], *winning_ids]
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
            from workflows.public_speech.public_context import Presentation
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
        self.bus.put(new_envelope(self.module_id, "data", "gwt.updated", "", {"gwt_tick": tick}))
        for dispatch, created_id in published:
            # A repeated commit returns the original receipt and does not apply
            # the new batch. Resolve the accepted action by durable identity and
            # tick so the published id is always committed.
            committed_id = self._committed_action_id(dispatch, tick) or created_id
            self.bus.put(new_envelope(
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
        self.bus.put(new_envelope(
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


class InnerThoughtLoop(CognitiveWorkflow):
    DATA_TOPICS = frozenset()

    def publish_output(self, report: dict, batch: list[Envelope]) -> None:
        changes = self.ghost.begin_batch(self.invocation_id or new_id(), cancel=self.stop_event)
        new_feature(
            self.ghost, report["thought"], FeatureType.InnerMonologue,
            source=self.ghost.get_companion_name(), changes=changes,
            source_kind=SourceKind.Self,
        )
        receipt = self.ghost.commit(changes)
        self.bus.put(new_envelope(
            self.module_id, "data", "thought.created", "", {"knoxel_id": receipt.added_ids[0]},
        ))
