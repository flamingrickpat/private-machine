"""Module-level tool factory for the real main-cognition parent.

The application supplies one lazily-activated completion callable. This module
binds it, the MemoryStore, and the runtime root into the ``run_stage`` tool the
parent Python role names in its manifest. Each ``run_stage`` call builds a
fresh read-only MemoryStore projection and invokes the real child kernel exactly as
the child's standalone tests do.
"""
from __future__ import annotations

from pathlib import Path
import json
import traceback
from typing import Callable
from pm.llm.completion import TextCompletion
from collections.abc import Mapping

from workflows.main_cognition.stage_runner import build_stage_runner
from workflows.main_cognition.stage_support import ModelFormatError
import asyncio
from datetime import datetime, timedelta
from workflows.main_cognition.affect_dynamics import AffectProjection
from pm.records.cycle_state import CycleState
from pm.records.global_workspace import GwtTurnRecord
from pm.records.memory_records import ActionSchemaRecord
from pm.records.enums import ActionType, FeatureType, SourceKind
from pm.records.cognitive_records import AttentionFrameRecord, AttentionRecord
from pm.records.mental_state import FullMentalState, create_empty_state
from workflows.main_cognition.experience import consolidate_cycle, update_beliefs
from pm.bus.envelopes import Envelope, new_id
from workflows.main_cognition.codelet_schedule import load_catalogue
from workflows.main_cognition.pipeline import DEFAULT_WORKFLOWS, proposal_consumers
from workflows.main_cognition.schemas import (
    CognitiveReport,
    CognitiveRequest,
    MemoryCandidate,
    Stimulus,
    WorkflowOption,
)
from workflows.main_cognition.stage_commits import (
    active_goal_texts, active_intentions, apply_appraisal, apply_belief_proposals, apply_goal_changes,
)
from workflows.public_speech.public_context import build_public_context
from pm.bus.envelopes import new_envelope
from pm.records.memory_records import new_feature
from workflows.public_speech.public_dialogue import _clip, _recency
from pm.runs.workflow_module import CognitiveWorkflow
from pm.memory.snapshot import GhostCancelled


def build_main_cognition_tools(
    ghost,
    runtime_root: Path,
    llm_complete: TextCompletion,
    *, generation_overrides: Mapping[str, Mapping[str, object]] | None = None,
    cancel_provider: Callable | None = None,
) -> Callable[[object, object], dict]:
    """Return the ``tools_factory(environment, batch)`` for the parent module."""

    def factory(environment, batch) -> dict:
        return {"run_stage": build_stage_runner(ghost, llm_complete, Path(runtime_root),
                                               read_environment=environment, generation_overrides=generation_overrides,
                                               cancel=cancel_provider() if cancel_provider is not None else None)}

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


def pathway_precursors(ghost, loaded) -> tuple[list[dict], int]:
    """Derive fresh pathway precursor evidence from committed attention.

    A precursor is a committed ``AttentionRecord`` that attention accepted
    (``selected``) from a codelet that belongs to a pathway. The records are
    re-read from the MemoryStore on every cycle, so a restart keeps the
    pathway state and no cycle can reuse stale pathway state. Each precursor
    carries the record's revision at read time; the trace keeps it visible.

    Returns ``(precursors, latest_tick)`` where ``latest_tick`` is the newest
    committed attention tick. The codelets stage measures tick age from it,
    not from the newest precursor, so an old precursor keeps its true age.
    """
    pathways = loaded.catalogue.pathway_map()
    membership: dict[str, list[str]] = {}
    window = 1
    for name, pathway in pathways.items():
        window = max(window, int(pathway.window_ticks))
        for codelet in pathway.members:
            membership.setdefault(codelet, []).append(name)
    records = [record for record in ghost.all_attention_knoxels if record.selected]
    if not records or not membership:
        return [], 0
    latest_tick = max(int(record.tick_id) for record in records)
    precursors = []
    with ghost.lock:
        for record in records:
            codelet = str(record.from_codelet)
            if codelet not in membership:
                continue
            tick = int(record.tick_id)
            if latest_tick - tick >= window:
                continue
            precursors.append({
                "codelet": codelet,
                "record_id": int(record.id),
                "tick_id": tick,
                "attention_kind": str(record.attention_kind),
                "reality": str(record.metadata.get("reality", "")),
                "family": str(record.metadata.get("family", "")),
                "pathways": list(membership[codelet]),
                "selected": True,
                "revision": int(ghost.revisions.get(int(record.id), 1)),
            })
    return precursors, latest_tick


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
        "minecraft.world",
        "cognition.wake",
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
        self._cancelled_invocation_id: str | None = None

    def cancel_run(self) -> None:
        """Cancel this invocation and prevent a format retry for the same request."""
        self._cancelled_invocation_id = getattr(self, "invocation_id", None)
        super().cancel_run()

    def accepts(self, envelope: Envelope) -> bool:
        if envelope.topic == "workflow.question" and envelope.kind == "telemetry":
            return envelope.target in {None, self.module_id}
        return super().accepts(envelope)

    def deliver(self, envelope: Envelope) -> None:
        """Record external wake causes before queueing the next cognition cycle."""
        control = getattr(self, "cognitive_control", None)
        if control is not None:
            control.observe(envelope)
        super().deliver(envelope)

    def handle_batch(self, batch: list[Envelope]) -> None:
        """Honor a wait and reject obsolete private stimuli before inference."""
        control = getattr(self, "cognitive_control", None)
        original = batch
        if control is not None:
            batch, self._control_generation = control.admit(batch)
            if not batch:
                control.release_wakes(original)
                release = getattr(self, "release_private_results", None)
                if release is not None:
                    release(original)
                return
        self._private_only = bool(batch) and all(item.topic == "thought.created" for item in batch)
        try:
            self._handle_awake_batch(batch)
        except GhostCancelled:
            if not (control is not None and self._private_only
                    and self._control_generation != control.generation):
                raise
        finally:
            if control is not None:
                control.release_wakes(original)
            release = getattr(self, "release_private_results", None)
            if release is not None:
                release(original)

    def _handle_awake_batch(self, batch: list[Envelope]) -> None:
        """Retry rejected model formats and revision conflicts without speech.

        A concurrent commit can invalidate the cycle's proposal. Each retry
        opens a new view and keeps the original causes. Three failed attempts
        raise the conflict, so contention cannot silently discard user input.
        Exhausted format repairs retain a diagnostic and retry the same batch.
        Cancellation stops retries. Ordinary Python and provider errors propagate.
        """
        from pm.memory.snapshot import GhostConflict
        self._event_stimuli = {}
        attempt = 0
        while not self.stop_event.is_set():
            try:
                super().handle_batch(batch)
                return
            except ModelFormatError as error:
                diagnostic = {"invocation_id": self.invocation_id, "error_type": type(error).__name__,
                              "reason": str(error), "traceback": traceback.format_exc()}
                path = self.runtime_root / "kernel" / self.invocation_id / "model-output-rejection.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
                self.bus.put(new_envelope(self.module_id, "telemetry", "cycle.rejected",
                    batch[0].correlation_id if batch else "",
                    {key: value for key, value in diagnostic.items() if key != "traceback"}))
                # No report reached publication. Retain the original input and
                # its event records while the next invocation reads fresh state.
                # A failed parallel stage sets run_stop to join its siblings.
                # Only an explicit run cancellation or module stop ends retry.
                if (self.stop_event.wait(0.2)
                        or self._cancelled_invocation_id == self.invocation_id):
                    return
            except GhostConflict as error:
                attempt += 1
                self.bus.put(new_envelope(self.module_id, "telemetry", "cycle.conflict",
                    batch[0].correlation_id if batch else "",
                    {"attempt": attempt, "reason": str(error)}))
                if self.stop_event.is_set() or self.run_stop.is_set():
                    return
                if attempt == 3:
                    raise

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
            self._stimulus_records.append(stimuli[-1].model_dump(mode="json"))
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
        from workflows.cognitive_control.attention import active_focus, focus_versions
        control = getattr(self, "cognitive_control", None)
        maximum_focus_bias = control.config.maximum_focus_bias if control is not None else None
        focus = active_focus(self.ghost, maximum_strength=maximum_focus_bias)
        self._focus_family_revisions = focus_versions(memory_view)
        from workflows.reflection.experiments import effective_inputs
        self._reflection_inputs = effective_inputs(memory_view, focus)
        mental_proposals, focus_proposals = [], []
        for source in sources:
            plan = source.metadata.get("mental_plan")
            if plan and plan.get("disposition") == "propose":
                mental_proposals.append({"record_id": source.id, **plan})
                if plan.get("focus"):
                    advice = dict(plan["focus"])
                    if maximum_focus_bias is not None:
                        advice["strength"] = min(advice.get("strength", 0.3), maximum_focus_bias)
                    focus_proposals.append({"record_id": source.id, "focus": advice})
        # Deliberate recall can introduce sources that a literal stimulus
        # query missed. They remain original, revisioned memory candidates.
        focus_ids = set(focus.get("promote_ids", []) + focus.get("reduce_ids", []))
        focus_ids.update(self._reflection_inputs["source_ids"])
        for item in mental_proposals:
            focus_ids.update(item.get("evidence_ids", []))
        for item in focus_proposals:
            focus_ids.update(item["focus"].get("promote_ids", []) + item["focus"].get("reduce_ids", []))
        self._focus_source_revisions = {}
        present = {item["knoxel_id"] for item in retrieval["matches"]}
        from pm.memory.retrieval import evidence_role, memory_kind
        records = memory_view.serialize_knoxels() if focus_ids else {}
        for key in sorted(focus_ids):
            if key not in memory_view.revisions:
                continue
            record = memory_view.get(key)
            self._focus_source_revisions[key] = memory_view.revisions[key]
            if key not in present:
                raw = records[key]
                retrieval["matches"].append({"knoxel_id": key, "text": record.content,
                    "revision": memory_view.revisions[key], "source_ids": [key],
                    "memory_kind": memory_kind(raw), "evidence_role": evidence_role(memory_view, raw),
                    "reasons": ["deliberate mental recall"]})
        self._memory_candidates = retrieval["matches"]
        previous = self.ghost.states[-1] if self.ghost.states else None
        previous_vec = list(previous.mental_state_latent_vec) if previous is not None else []
        age_seconds = 0.0
        if previous is not None:
            from pm.memory.retrieval import utc_time
            age_seconds = max(0.0, (utc_time(datetime.utcnow()) - utc_time(previous.timestamp)).total_seconds())
        provider = getattr(self, "capability_provider", None)
        public = build_public_context(self.ghost, {"stimuli": [item.model_dump(mode="json") for item in stimuli]},
                                      capabilities=provider() if provider else None)
        request = CognitiveRequest(
            run_id=self.invocation_id or "",
            correlation_id=correlation,
            companion_name=self.ghost.get_companion_name(),
            user_name=self.ghost.get_user_name(),
            relationship_entity_id=self.ghost.get_entity_id(self.ghost.get_user_name())
                if any(item.source == "user_input" for item in stimuli) else None,
            state_summary=_state_summary(self.ghost),
            mental_state_latent_vec=previous_vec,
            mental_state_age_seconds=age_seconds,
            personality_vec=list(self.ghost.ghost_config.personality_profile),
            active_goals=active_goal_texts(self.ghost)[:8],
            stimuli=stimuli,
            memory_candidates=[MemoryCandidate(**item) for item in retrieval["matches"]],
            memory_trace=retrieval["trace"],
            active_focus=focus,
            focus_proposals=focus_proposals,
            mental_proposals=mental_proposals,
            reflection_guidance=self._reflection_inputs["guidance"],
            # Continuation cannot skip action selection until a bounded
            # episode executor owns the next step and its acknowledgment.
            deliberation_only=False,
            wait_default_seconds=getattr(getattr(self, "cognitive_control", None), "config", None).wait_seconds
                if getattr(self, "cognitive_control", None) is not None else 1800,
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
        # The pathway catalogue is the same file the codelets stage reads. A
        # broken catalogue stops the cycle here, exactly as it would in the
        # stage, rather than silently running without pathways.
        loaded = load_catalogue(self.codelet_catalogue or None)
        precursors, latest_tick = pathway_precursors(self.ghost, loaded)
        request.pathway_precursors = precursors
        request.pathway_latest_tick = latest_tick
        return request.model_dump(mode="json")

    def _stimulus_from(self, envelope: Envelope):
        # Retrying a rejected cycle must reuse its materialized event records.
        # The cache belongs to one handled batch and does not survive restart.
        cached = getattr(self, "_event_stimuli", {})
        if envelope.id in cached:
            return cached[envelope.id]
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
            result = feature.id, question, "workflow.question"
            self._event_stimuli = {**cached, envelope.id: result}
            return result
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
            result = feature.id, text, "workflow.event"
            self._event_stimuli = {**cached, envelope.id: result}
            return result
        if envelope.topic == "cognition.wake":
            knoxel_id = int(payload["knoxel_id"])
            knoxel = self._input_knoxel(knoxel_id)
            return knoxel_id, str(knoxel.content), "cognition.wake"
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
            if envelope.topic == "minecraft.world":
                kind = "minecraft.world"
            elif knoxel.feature_type in {FeatureType.Thought, FeatureType.InnerMonologue}:
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
        if self.stop_event.is_set() or self.run_stop.is_set():
            return
        parsed = CognitiveReport.model_validate(report)
        correlation = self._batch_correlation or (batch[0].correlation_id if batch else "")
        cause = batch[0].id if batch else None
        changes = self.ghost.begin_batch(
            self.invocation_id or new_id(), new_tick=True, cancel=self.run_stop,
        )
        tick = changes.tick_id
        view = changes.view()
        if hasattr(self, "_focus_family_revisions"):
            changes.expect_metadata_version("control_kind", "attention_focus", self._focus_family_revisions)
        for source_id, revision in self._stimulus_revisions.items():
            changes.expect_revision(source_id, revision)
        for source_id, revision in getattr(self, "_focus_source_revisions", {}).items():
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
            appraisal_id=appraisal_feature.id,
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
        # The belief owner consumes accepted belief proposals through the same
        # batch. A repeat of the same interpretation retains the prior record;
        # a changed one supersedes it and keeps its history. Nothing here
        # rewrites the observed records that the interpretation cites.
        codelet_beliefs = apply_belief_proposals(
            changes, parsed.nominations,
            companion_name=self.ghost.get_companion_name(),
            stale_ids=stale,
            catalogue_version=str((parsed.codelet_trace or {}).get("catalogue_version", "")),
        )
        # The winning nominations are the workspace focus that the public
        # workflows plan their memory context from. Stale ones never win.
        attention_focus = [{"text": item.text, "target_ids": list(item.target_ids), "reality": item.reality}
                           for item in parsed.nominations
                           if item.status == "selected" and item.nomination_id not in stale and item.text]
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
                    "topic": nomination.topic,
                    "consumers": proposal_consumers(str(nomination.attention_kind)) if status == "selected" else [],
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
                "belief_proposals": codelet_beliefs,
                "pathway_signals": dict(trace.get("pathway_signals") or {}),
            },
        ))
        created_order: list[int] = [frame.id, appraisal_feature.id, *created_goal_ids, *experience_ids,
                                    *belief_receipt["accepted_ids"], *codelet_beliefs["accepted_ids"],
                                    *winning_ids]
        from pm.records.lineage import RecordLineage
        parent = next((view.get(key) for key in self._stimulus_ids if view.get(key).lineage is not None), None)
        lineage = RecordLineage(
            episode_id=parent.lineage.episode_id if parent is not None else changes.commit_id,
            sequence=parent.lineage.sequence + 1 if parent is not None else 0,
            operation="main_appraisal", parent_record_ids=list(self._stimulus_ids),
            source_event_ids=list(self._stimulus_ids), intention_ids=goal_ids,
            topic_hints=list(parent.lineage.topic_hints) if parent is not None else [])
        for key in set([*created_order, *attention_ids]) & changes.created_ids:
            changes.annotate_lineage(key, lineage)
        reflection_versions = {}
        for item in getattr(self, "_reflection_inputs", {}).get("guidance", []):
            effective_id = item["effective_record_id"]
            if (item["owner"] == "attention" and parsed.focus_decision.get("control_record_id") == effective_id
                    or item["owner"] == "belief" and effective_id in {memory.knoxel_id for memory in parsed.selected_memories}):
                reflection_versions[str(item["experiment_id"])] = effective_id
        selection = new_feature(self.ghost, str(parsed.action_selection.get("reasoning") or parsed.summary),
            FeatureType.StateUpdate, source=self.ghost.get_companion_name(), source_kind=SourceKind.Self,
            changes=changes, metadata={"operation": "action_selection", "selection": parsed.action_selection,
                                       "reflection_versions": reflection_versions},
            lineage=lineage.model_copy(update={"operation": "action_selection", "parent_record_ids": [frame.id]}))
        created_order.append(selection.id)
        focus = parsed.focus_decision
        if focus.get("fresh") and focus.get("accept"):
            from workflows.cognitive_control.schemas import FocusProposal
            advice = FocusProposal.model_validate(focus)
            from workflows.cognitive_control.attention import stage_focus
            accepted_focus = stage_focus(changes, getattr(self, "_memory_view", view),
                instruction=advice.instruction, focus=focus, companion_name=self.ghost.get_companion_name(),
                expires_at=(datetime.utcnow() + timedelta(seconds=advice.seconds)).isoformat(),
                lineage=lineage.model_copy(update={"operation": "focus_acceptance"}))
            created_order.append(accepted_focus.id)
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
            active_game = self._active_game_request(dispatch, correlation, batch)
            if active_game is not None:
                rejected.append(dispatch.model_copy(update={"status": "rejected", "rejection_reason":
                    f"Game run {active_game} already owns this request. Use its management controls; "
                    "task progress cannot start another game run for the same request."}))
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
                action_type=ACTION_BY_NAME[dispatch.action_type],
                attention_frame_id=frame.id,
                intention_ids=list(goal_ids),
                lineage=lineage.model_copy(update={"operation": "action_dispatch",
                                                  "parent_record_ids": [selection.id]}),
                metadata={
                    "action_id": dispatch.action_id,
                    "route": dispatch.workflow_name,
                    "workflow_name": dispatch.workflow_name,
                    "depends_on": list(dispatch.depends_on),
                    "requires_completed": list(dispatch.requires_completed),
                    "arguments": dict(dispatch.arguments),
                    "affect_summary": parsed.affect_summary,
                    "appraisal_id": appraisal_feature.id,
                    "evidence_ids": list(self._stimulus_ids),
                    "self_origin": not any(item.source == "user_input" for item in batch),
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
                        "affect": parsed.affect_summary, "regulation": parsed.regulation,
                        "attention": attention_focus},
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
                internal_data={"control_generation": getattr(self, "_control_generation", 0)},
            ),
        )
        control = getattr(self, "cognitive_control", None)
        if control is None:
            receipt = self.ghost.commit(changes)
        else:
            from workflows.cognitive_control.schemas import WaitRequest
            receipt = control.commit(changes,
                WaitRequest.model_validate(parsed.wait_request) if parsed.wait_request is not None else None,
                getattr(self, "_control_generation", control.generation),
                envelope_ids={item.id for item in batch},
                wake_ids=[int(item.payload["knoxel_id"]) for item in batch if item.topic == "cognition.wake"],
                private_generation=getattr(self, "_control_generation", control.generation)
                    if getattr(self, "_private_only", False) else None,
                source_ids=list(self._stimulus_ids))
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
                    "attention": attention_focus,
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
        acknowledge = getattr(self, "private_acknowledge", None)
        if acknowledge is not None:
            acknowledge(list(self._stimulus_ids))

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

    def _active_game_request(self, dispatch, correlation: str, batch: list[Envelope]) -> str | None:
        """Keep progress for an active request from starting another game run.

        Read the current durable run table before committing an action. A new
        user input permits another request. Updates and cancellation still use
        the existing management controls. Completed runs release this boundary.
        """
        route = dispatch.workflow_name
        if route in {"minecraft", "minecraft_project"} and (
                dispatch.arguments.get("operation", "update" if dispatch.arguments.get("run_id") else "start") != "start"):
            return None
        if route == "self_management" and dispatch.arguments.get("operation") == "start":
            route = dispatch.arguments.get("workflow_name")
        if route not in {"minecraft", "minecraft_project"} or not correlation or any(
                item.source == "user_input" for item in batch):
            return None
        with self.ghost.lock:
            return next((run["run_id"] for run in self.ghost.managed_runs.values()
                         if run["workflow_name"] in {"minecraft", "minecraft_project"}
                         and run["parent_correlation"] == correlation
                         and run["state"] in {"accepted", "running", "waiting_for_input"}), None)

    def _existing_action_identity(self, dispatch) -> int | None:
        """Return the existing non-reply action with this identity, if any.

        Replies may repeat their content. A tool or start action keeps its
        durable identity, so a repeated request does not start another task.
        """
        if ACTION_BY_NAME[dispatch.action_type] == ActionType.Reply:
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
