"""Bus module and tool factory of the main cognition cycle.

``MainCogLoop`` turns one bus batch into one run of
``main_cognition.workflow.md``. It builds the typed
:class:`~workflows.main_cognition.cycle_work.CycleRequest` from committed
state, lets the phases run, and then commits the final
:class:`~workflows.main_cognition.cycle_work.CycleWork` in one write batch.
After the commit it publishes the bus events.

The application supplies one lazily activated completion callable.
:func:`build_main_cognition_tools` binds it as the ``llm_complete`` tool of
each run. The phases call the model only through that tool.
"""
from __future__ import annotations

import asyncio
import json
import threading
import traceback
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

from pydantic import BaseModel

from pm.bus.envelopes import Envelope, new_envelope, new_id
from pm.constants import CCQ_LENGTH, CYCLE_RETRY_DELAY_SECONDS, CYCLE_RETRY_MAX_DELAY_SECONDS
from pm.llm.completion import TextCompletion
from pm.llm.prompt_encoding import TurnInput, normalize_turns
from pm.memory.snapshot import GhostCancelled
from pm.records.cognitive_records import AttentionFrameRecord, AttentionRecord
from pm.records.cycle_records import ActionDispatchRef, BroadcastRef
from pm.records.cycle_state import CycleState
from pm.records.mental_state import PersonalityProfile, features_to_history
from pm.records.workspace import SELF_SHARD
from pm.records.enums import ActionType, FeatureType, SourceKind
from pm.records.global_workspace import GwtTurnRecord
from pm.records.memory_records import ActionSchemaRecord, new_feature
from pm.runs.workflow_module import CognitiveWorkflow
from workflows.cognitive_control.schemas import FocusState, MentalPlan
from workflows.inference_settings import REASONING
from workflows.main_cognition.codelet_schedule import LoadedCatalogue, load_catalogue
from workflows.main_cognition.cycle_work import CycleRequest, CycleWork, primary_group
from workflows.main_cognition.experience import consolidate_cycle, update_beliefs
from workflows.main_cognition.model_json import ModelFormatError
from workflows.main_cognition.phases.goals import proposal_consumers
from workflows.main_cognition.schemas import (
    CodeletHistory,
    CodeletRun,
    FocusProposalRef,
    GoalChange,
    GoalRecordView,
    ManagedRunView,
    MemoryCandidate,
    MentalProposal,
    PathwayPrecursor,
    ReflectionGuidance,
    Stimulus,
    WorkflowOption,
)
from workflows.main_cognition.stage_commits import (
    active_goal_texts, active_intentions, apply_appraisal, apply_belief_proposals, apply_goal_changes,
)
from workflows.public_speech.public_context import build_public_context


def build_main_cognition_tools(
    ghost,
    runtime_root: Path,
    llm_complete: TextCompletion,
    *,
    cancel_provider: Callable[[], threading.Event] | None = None,
) -> Callable[[object, object], dict]:
    """Return the ``tools_factory(environment, batch)`` of the main module.

    The factory grants one ``llm_complete`` tool to each run. The tool turns
    hidden reasoning off, because a reasoning budget exhausted real
    completions before. It passes the run's cancellation signal, so a failed
    parallel call or a cancelled run also stops its sibling calls.
    ``ghost`` and ``runtime_root`` stay in the signature for the application
    wiring. Phases never read the store through the tool.
    """

    def factory(environment, batch) -> dict:
        stop = cancel_provider() if cancel_provider is not None else threading.Event()

        def complete(messages: Sequence[TurnInput], **generation: object) -> str:
            generation["reasoning_budget"] = REASONING["off"]
            return llm_complete(normalize_turns(messages), cancel=stop, **generation)

        return {"llm_complete": complete}

    return factory


ACTION_BY_NAME = {member.value: member for member in ActionType}


class QueryRewrite(BaseModel):
    """The retrieval query that the rewrite provider derived from the stimuli."""

    query: str = ""
    evidence_use: str = "general"
    status: str = "literal"


class WorkflowQuestionPayload(BaseModel):
    """The payload of a ``workflow.question`` telemetry message."""

    question: str = ""


class WorkflowEventPayload(BaseModel):
    """The payload fields of a ``workflow.event`` message that cognition reads."""

    run_id: str = ""
    event: str = ""
    detail: str = ""
    message: str = ""
    artifact_path: str = ""
    executed_at: str | None = None


class KnoxelPayload(BaseModel):
    """The record reference of a data message. Some topics carry none."""

    knoxel_id: int | None = None


def _capabilities(ghost) -> list[str]:
    return [str(item) for item in ghost.ghost_config.supported_capabilities]


def _state_summary(ghost) -> str:
    """Return the summary of the newest broadcast, or an empty text before the first one."""
    turns = ghost.gwt_turns
    if not turns:
        return ""
    return turns[max(turns)].summary


def codelet_history(ghost) -> dict[str, CodeletHistory]:
    """Return per-codelet run history from committed attention frames.

    Each committed main ``AttentionFrameRecord`` stores the codelets that ran
    in its cycle under ``metadata["codelet_runs"]``. Frames of the private
    loop have no such entry. Runs that stayed silent count too, so a quiet LLM
    codelet still waits out its cooldown and does not spend a model call every
    cycle. The history is derived, never stored separately, so it survives
    restart and has one owner: the MemoryStore.
    """
    frames = sorted(ghost.all_attention_frames, key=lambda item: (item.tick_id, item.id))
    if not frames:
        return {}
    latest_tick = max(int(item.tick_id) for item in frames)
    now = datetime.utcnow()
    history: dict[str, CodeletHistory] = {}
    for frame in frames:
        if "codelet_runs" not in frame.metadata:
            continue
        for entry in frame.metadata["codelet_runs"]:
            run = CodeletRun.model_validate(entry)
            if not run.codelet or run.status not in {"ran", "silent", "failed"}:
                continue
            record = history.setdefault(run.codelet, CodeletHistory())
            record.runs += 1
            record.last_tick = int(frame.tick_id)
            record.cycles_since = latest_tick - int(frame.tick_id)
            record.seconds_since = max(0.0, (now - frame.timestamp_creation).total_seconds())
    return history


def pathway_precursors(ghost, loaded: LoadedCatalogue) -> tuple[list[PathwayPrecursor], int]:
    """Derive fresh pathway precursor evidence from committed attention.

    A precursor is a committed ``AttentionRecord`` that attention accepted
    (``selected``) from a codelet that belongs to a pathway. The records are
    re-read from the MemoryStore on every cycle, so a restart keeps the
    pathway state and no cycle can reuse stale pathway state. Each precursor
    carries the record's revision at read time; the trace keeps it visible.

    Returns ``(precursors, latest_tick)`` where ``latest_tick`` is the newest
    committed attention tick. The codelets phase measures tick age from it,
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
            # The main commit writes reality and family on every codelet record.
            precursors.append(PathwayPrecursor(
                codelet=codelet, record_id=int(record.id), tick_id=tick,
                attention_kind=str(record.attention_kind),
                reality=str(record.metadata["reality"]), family=str(record.metadata["family"]),
                pathways=list(membership[codelet]), selected=True,
                revision=int(ghost.revisions[int(record.id)]),
            ))
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
        for key, seen in nomination.source_revisions.items():
            if int(key) not in revisions or revisions[int(key)] != int(seen):
                stale.add(nomination.nomination_id)
                break
    return stale


class MainCogLoop(CognitiveWorkflow):
    """Real cognitive parent: one typed phase sequence, committed GWT, typed dispatches.

    The workflow owns the phase order and the dispatch ids. This module owns
    the bus and the store: it builds one typed request from the batch,
    commits one attention frame, GWT turn, cycle state, and action schema per
    released dispatch in one write batch, then publishes ``action.request``
    targeted at each dispatch's workflow name. Rejected dispatches publish an
    explicit receipt and never an action.

    The application sets ``cognitive_control``, ``release_private_results``,
    ``capability_provider``, and ``retrieval_query_provider`` after
    construction. Each stays None when the application does not use it.
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
        # Configuration supplies the action space. The decide phase uses its
        # default action space when this list is empty.
        self.workflow_catalogue = [WorkflowOption.model_validate(item) for item in workflow_catalogue or []]
        self.embedding_config_path = embedding_config_path
        self._embeddings = None
        self._cancelled_invocation_id: str | None = None
        self.invocation_id: str | None = None
        self.cognitive_control = None
        self.release_private_results: Callable | None = None
        self.capability_provider: Callable | None = None
        self.retrieval_query_provider: Callable | None = None
        # Per-batch state. ``prepare_input`` sets it and ``publish_output``
        # reads it for the same invocation.
        self._event_stimuli: dict[str, tuple[int | None, str, str]] = {}
        self._stimulus_ids: list[int] = []
        self._stimulus_revisions: dict[int, int] = {}
        self._focus_family_revisions: dict[int, int] = {}
        self._focus_source_revisions: dict[int, int] = {}
        self._batch_correlation = ""
        self._memory_view = None
        self._control_generation = 0
        self._private_only = False

    def cancel_run(self) -> None:
        """Cancel this invocation and prevent a format retry for the same request."""
        self._cancelled_invocation_id = self.invocation_id
        super().cancel_run()

    def accepts(self, envelope: Envelope) -> bool:
        if envelope.topic == "workflow.question" and envelope.kind == "telemetry":
            return envelope.target in {None, self.module_id}
        return super().accepts(envelope)

    def deliver(self, envelope: Envelope) -> None:
        """Record external wake causes before queueing the next cognition cycle."""
        if self.cognitive_control is not None:
            self.cognitive_control.observe(envelope)
        super().deliver(envelope)

    def handle_batch(self, batch: list[Envelope]) -> None:
        """Honor a wait and reject obsolete private stimuli before inference."""
        control = self.cognitive_control
        original = batch
        if control is not None:
            batch, self._control_generation = control.admit(batch)
            if not batch:
                control.release_wakes(original)
                if self.release_private_results is not None:
                    self.release_private_results(original)
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
            if self.release_private_results is not None:
                self.release_private_results(original)

    def _handle_awake_batch(self, batch: list[Envelope]) -> None:
        """Retry rejected model formats and revision conflicts without speech.

        A concurrent commit can invalidate the cycle's proposal. Each retry
        opens a new view and keeps the original causes. Three failed attempts
        raise the conflict, so contention cannot silently discard user input.
        Exhausted format repairs retain a diagnostic and retry the same batch
        after a delay that doubles from CYCLE_RETRY_DELAY_SECONDS up to
        CYCLE_RETRY_MAX_DELAY_SECONDS.
        Cancellation stops retries. Ordinary Python and provider errors propagate.
        """
        from pm.memory.snapshot import GhostConflict
        self._event_stimuli = {}
        attempt = 0
        delay = CYCLE_RETRY_DELAY_SECONDS
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
                # A failed parallel call sets run_stop to join its siblings.
                # Only an explicit run cancellation or module stop ends retry.
                # The same model often repeats the same mistake. A growing
                # delay stops a fast loop of model calls and run directories.
                if (self.stop_event.wait(delay)
                        or self._cancelled_invocation_id == self.invocation_id):
                    return
                delay = min(delay * 2, CYCLE_RETRY_MAX_DELAY_SECONDS)
            except GhostConflict as error:
                attempt += 1
                self.bus.put(new_envelope(self.module_id, "telemetry", "cycle.conflict",
                    batch[0].correlation_id if batch else "",
                    {"attempt": attempt, "reason": str(error)}))
                if self.stop_event.is_set() or self.run_stop.is_set():
                    return
                if attempt == 3:
                    raise

    def _retrieve(self, stimuli: list[Stimulus], memory_view, sources) -> tuple[list[MemoryCandidate], dict]:
        """Retrieve memory candidates for the stimuli by embedding.

        Returns the candidates and the retrieval trace of ``pm.memory``. The
        trace stays an opaque mapping. The cycle stores it in record metadata.
        A user message first goes through the query rewrite provider, when the
        application supplies one.
        """
        frontier = max(((item.tick_id, item.sub_tick_id) for item in sources), default=None)
        scopes = {(item.scope.reality_id, item.scope.frame.value) for item in sources} or None
        if self.embedding_config_path and self._embeddings is None:
            from pm.memory.embeddings import EmbeddingService
            from pm.llm.config import load_config
            self._embeddings = EmbeddingService(load_config(self.embedding_config_path).embedding)
        original_query = "\n".join(item.text for item in stimuli)
        if self.retrieval_query_provider is not None and any(item.source == "user_input" for item in stimuli):
            rewrite = QueryRewrite.model_validate(self.retrieval_query_provider(original_query))
        else:
            rewrite = QueryRewrite(query=original_query)
        retrieval = asyncio.run(self.ghost.retrieve_experiences(
            rewrite.query, self._embeddings, view=memory_view,
            frontier=frontier, scopes=scopes, cancel=self.run_stop,
        ))
        trace = retrieval["trace"]
        trace["original_query"] = original_query
        trace["query_rewrite"] = rewrite.model_dump(mode="json")
        matches = [MemoryCandidate.model_validate(item) for item in retrieval["matches"]]
        if rewrite.evidence_use == "event":
            from pm.memory.retrieval import evidence_role
            records = memory_view.serialize_knoxels()
            selected_sources = {key for item in matches for key in item.source_ids if key in records}
            original_sources = {key for key in selected_sources
                                if evidence_role(memory_view, records[key]) in {"user_report", "runtime_event", "world_observation"}}
            # An answer to a recall question cannot become event evidence for
            # its repetition. Keep speech available for explicit speech recall.
            rejected_ids = {item.knoxel_id for item in matches
                            if item.evidence_role in {"companion_utterance", "intent", "imagined", "interpretation"}
                            or item.evidence_role == "derived" and not original_sources.intersection(item.source_ids)}
            matches = [item for item in matches if item.knoxel_id not in rejected_ids]
            trace["speech_only_excluded_ids"] = sorted(rejected_ids)
        return matches, trace

    def prepare_input(self, batch: list[Envelope]) -> dict:
        """Build the typed cycle request from the batch and committed state.

        Side effects: event stimuli become new store records (see
        ``_stimulus_from``). The method records the revisions that the cycle
        read, so the commit can reject the cycle when one of them changed.
        """
        # The working target owns the frozen permitted view. Materialized host
        # events remain stimuli, but cannot expose concurrent future memory.
        memory_view = self.last_environment.ghost_view
        stimuli: list[Stimulus] = []
        self._stimulus_revisions = {}
        correlation = ""
        for envelope in batch:
            knoxel_id, text, kind = self._stimulus_from(envelope)
            if knoxel_id is None:
                continue
            stimuli.append(Stimulus(
                knoxel_id=knoxel_id, source=envelope.source, text=text,
                correlation_id=envelope.correlation_id, kind=kind,
            ))
            # Event records that the host created after the view froze are
            # not in the view. Their revision comes from the store.
            if knoxel_id in memory_view.revisions:
                self._stimulus_revisions[knoxel_id] = memory_view.revisions[knoxel_id]
            else:
                with self.ghost.lock:
                    self._stimulus_revisions[knoxel_id] = self.ghost.revisions[knoxel_id]
            if not correlation:
                correlation = envelope.correlation_id
        self._stimulus_ids = [item.knoxel_id for item in stimuli]
        # A queued task event cannot take the cause of a newer user request.
        # Individual stimuli retain their own correlation for attribution.
        user_correlations = [item.correlation_id for item in stimuli if item.source == "user_input"]
        if user_correlations:
            correlation = user_correlations[-1]
        self._batch_correlation = correlation
        sources = [memory_view.get_knoxel_by_id(item.knoxel_id) or self.ghost.get_knoxel_by_id(item.knoxel_id)
                   for item in stimuli]
        matches, trace = self._retrieve(stimuli, memory_view, sources)
        self._memory_view = memory_view

        from workflows.cognitive_control.attention import active_focus, focus_versions
        from workflows.reflection.experiments import effective_inputs
        control = self.cognitive_control
        maximum_focus_bias = control.config.maximum_focus_bias if control is not None else None
        # Reflection reads the stored advice mapping. The cycle reads it typed.
        stored_focus = active_focus(self.ghost, maximum_strength=maximum_focus_bias)
        focus = FocusState.model_validate(stored_focus) if stored_focus else None
        self._focus_family_revisions = focus_versions(memory_view)
        reflection = effective_inputs(memory_view, stored_focus)
        guidance = [ReflectionGuidance.model_validate(item) for item in reflection["guidance"]]
        mental_proposals: list[MentalProposal] = []
        focus_proposals: list[FocusProposalRef] = []
        for source in sources:
            if "mental_plan" not in source.metadata:
                continue
            plan = MentalPlan.model_validate(source.metadata["mental_plan"])
            if plan.disposition != "propose":
                continue
            mental_proposals.append(MentalProposal(
                record_id=source.id, instruction=plan.instruction, summary=plan.summary,
                disposition=plan.disposition, evidence_ids=list(plan.evidence_ids),
                goal_changes=[GoalChange.model_validate(item) for item in plan.goal_changes]))
            if plan.focus is not None:
                advice = plan.focus
                if maximum_focus_bias is not None:
                    advice = advice.model_copy(update={"strength": min(advice.strength, maximum_focus_bias)})
                focus_proposals.append(FocusProposalRef(record_id=source.id, focus=advice))
        # Deliberate recall can introduce sources that a literal stimulus
        # query missed. They remain original, revisioned memory candidates.
        focus_ids = set(focus.promote_ids + focus.reduce_ids) if focus is not None else set()
        focus_ids.update(reflection["source_ids"])
        for item in mental_proposals:
            focus_ids.update(item.evidence_ids)
        for item in focus_proposals:
            focus_ids.update(item.focus.promote_ids + item.focus.reduce_ids)
        self._focus_source_revisions = {}
        present = {item.knoxel_id for item in matches}
        from pm.memory.retrieval import evidence_role, memory_kind
        records = memory_view.serialize_knoxels() if focus_ids else {}
        for key in sorted(focus_ids):
            if key not in memory_view.revisions:
                continue
            record = memory_view.get(key)
            self._focus_source_revisions[key] = memory_view.revisions[key]
            if key not in present:
                raw = records[key]
                matches.append(MemoryCandidate(knoxel_id=key, text=record.content,
                    revision=memory_view.revisions[key], source_ids=[key],
                    memory_kind=memory_kind(raw), evidence_role=evidence_role(memory_view, raw),
                    reasons=["deliberate mental recall"]))
        previous = self.ghost.states[-1] if self.ghost.states else None
        # The appraise phase computes the EMA mental state from this history.
        # It reads the live store, like ``previous``: the cycle state and the
        # features that produced it come from the same committed ticks.
        with self.ghost.lock:
            history = features_to_history(self.ghost.all_features)
        capabilities = self.capability_provider() if self.capability_provider is not None else None
        public = build_public_context(self.ghost, {"stimuli": [item.model_dump(mode="json") for item in stimuli]},
                                      capabilities=capabilities)
        user_entity_id = self.ghost.get_entity_id(self.ghost.get_user_name())
        request = CycleRequest(
            run_id=self.invocation_id or "",
            correlation_id=correlation,
            companion_name=self.ghost.get_companion_name(),
            user_name=self.ghost.get_user_name(),
            user_entity_id=user_entity_id,
            state_summary=_state_summary(self.ghost),
            previous_state=previous,
            cycle_time=datetime.utcnow(),
            state_history=history,
            personality=PersonalityProfile.init_from_vector(self.ghost.ghost_config.personality_profile),
            active_goals=active_goal_texts(self.ghost)[:8],
            stimuli=stimuli,
            memory_candidates=matches,
            memory_trace=trace,
            active_focus=focus,
            focus_proposals=focus_proposals,
            mental_proposals=mental_proposals,
            reflection_guidance=guidance,
            wait_default_seconds=control.config.wait_seconds if control is not None else 1800,
            capabilities=_capabilities(self.ghost),
            available_workflows=self.workflow_catalogue,
            public_context=public,
            managed_runs=[ManagedRunView.model_validate(run)
                          for run in list(self.ghost.managed_runs.values())[-16:]],
            stimulus_revisions=dict(self._stimulus_revisions),
            active_goal_records=[
                GoalRecordView(knoxel_id=int(item.id), text=str(item.content),
                               revision=int(memory_view.revisions[int(item.id)]))
                for item in active_intentions(memory_view)[:8]
            ],
            codelet_history=codelet_history(self.ghost),
            seconds_since_user_input=seconds_since_user_input(self.ghost, self._stimulus_ids),
            codelet_catalogue=self.codelet_catalogue,
        )
        # The pathway catalogue is the same file the codelets phase reads. A
        # broken catalogue stops the cycle here, exactly as it would in the
        # phase, rather than silently running without pathways.
        loaded = load_catalogue(self.codelet_catalogue or None)
        request.pathway_precursors, request.pathway_latest_tick = pathway_precursors(self.ghost, loaded)
        return request.model_dump(mode="json")

    def _stimulus_from(self, envelope: Envelope) -> tuple[int | None, str, str]:
        """Resolve one envelope to ``(record id, text, kind)``.

        The record id is None when the envelope names no record. Workflow
        questions and workflow events become new ``WorldEvent`` records, with
        the store write as a side effect. Retrying a rejected cycle must reuse
        those records, so the result is cached for the handled batch. The cache
        does not survive restart.
        """
        cached = self._event_stimuli
        if envelope.id in cached:
            return cached[envelope.id]
        if envelope.topic == "workflow.question" and envelope.kind == "telemetry":
            question = WorkflowQuestionPayload.model_validate(envelope.payload).question.strip()
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
            event = WorkflowEventPayload.model_validate(envelope.payload)
            run_id = event.run_id.strip()
            state = event.event.strip()
            if not run_id or not state:
                return None, "", ""
            # Keep progress detail and evidence so the cycle can judge the task.
            detail = (event.detail or event.message).strip()
            artifact = event.artifact_path.strip()
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
                    "evidence": dict(envelope.payload),
                    "executed_at": event.executed_at or datetime.utcnow().isoformat(),
                },
            )
            result = feature.id, text, "workflow.event"
            self._event_stimuli = {**cached, envelope.id: result}
            return result
        knoxel_id = KnoxelPayload.model_validate(envelope.payload).knoxel_id
        if knoxel_id is None:
            return None, "", ""
        knoxel = self._input_knoxel(knoxel_id)
        if envelope.topic in {"cognition.wake", "system.lifecycle"}:
            return knoxel_id, str(knoxel.content), envelope.topic
        kind = "user.input"
        if envelope.topic == "minecraft.world":
            kind = "minecraft.world"
        elif knoxel.feature_type in {FeatureType.Thought, FeatureType.InnerMonologue}:
            kind = "thought.created"
        elif knoxel.feature_type == FeatureType.SystemMessage:
            kind = "system.lifecycle"
        elif knoxel.feature_type == FeatureType.WorldEvent:
            kind = "workflow.event"
        return knoxel_id, str(knoxel.content), kind

    def _input_knoxel(self, knoxel_id: int):
        """Resolve a supplied ID through the frozen invocation grant."""
        view = self.last_environment.ghost_view
        if knoxel_id not in view.revisions:
            raise KeyError(f"Stimulus is outside the invocation view: {knoxel_id}")
        return view.get(knoxel_id)

    def publish_output(self, report: dict, batch: list[Envelope]) -> None:
        """Commit one finished cycle in one write batch, then publish its events.

        The batch holds the appraisal, goal changes, beliefs, the episodic
        summary, every attention record, the attention frame, the action
        selection record, one action schema per released dispatch, the GWT
        turn, and the cycle state. Bus events go out only after the commit. A
        stale source or a concurrent focus change aborts the whole batch.
        """
        if self.stop_event.is_set() or self.run_stop.is_set():
            return
        work = CycleWork.model_validate(report)
        request = work.request
        stimulus_records = [item.model_dump(mode="json") for item in request.stimuli]
        correlation = self._batch_correlation or (batch[0].correlation_id if batch else "")
        cause = batch[0].id if batch else None
        changes = self.ghost.begin_batch(
            self.invocation_id or new_id(), new_tick=True, cancel=self.run_stop,
        )
        tick = changes.tick_id
        view = changes.view()
        changes.expect_metadata_version("control_kind", "attention_focus", self._focus_family_revisions)
        for source_id, revision in self._stimulus_revisions.items():
            changes.expect_revision(source_id, revision)
        for source_id, revision in self._focus_source_revisions.items():
            changes.expect_revision(source_id, revision)
        for memory in work.selected_memories:
            changes.expect_revision(memory.knoxel_id, memory.revision)
            for source_id in memory.source_ids:
                if source_id in self._memory_view.revisions:
                    changes.expect_revision(source_id, self._memory_view.revisions[source_id])

        # Apply the appraisal and goal proposals into this same batch before the
        # frame. A failure here aborts the whole batch, so stale proposals never
        # partly update accepted state. The appraise phase already computed the
        # appraisal affect and the EMA mental state; the commit stores them.
        memory_trace = {**request.memory_trace,
                        "context_selected_ids": [item.knoxel_id for item in work.selected_memories]}
        appraisal_feature = apply_appraisal(
            changes, work.appraisal, work.affect,
            evidence_ids=list(self._stimulus_ids),
            companion_name=self.ghost.get_companion_name(),
            memory_trace=memory_trace,
        )
        goal_records = apply_goal_changes(
            changes, view, work.goals.changes,
            evidence_ids=list(self._stimulus_ids),
            appraisal_id=appraisal_feature.id,
        )
        created_goal_ids = [
            int(record["intention_id"]) for record in goal_records
            if record["created"] and record["intention_id"] is not None
        ]
        goal_ids = [int(item.id) for item in active_intentions(changes.view())]
        belief_receipt = update_beliefs(changes, work.context.belief_updates, user_name=self.ghost.get_user_name(),
                                        permitted_source_ids=set(self._stimulus_ids))
        experience_ids = consolidate_cycle(changes, self._stimulus_ids, appraisal_feature, work.affect.felt)

        # Commit every nomination, winner or not, as an AttentionRecord. A
        # nomination whose read sources changed since the cycle began is
        # marked stale and cannot win. The cycle itself continues: one stale
        # proposal must not cost the user a reply.
        nominations = work.attention.nominations
        stale = stale_nominations(nominations, view.revisions)
        codelets = work.codelets
        # The belief owner consumes accepted belief proposals through the same
        # batch. A repeat of the same interpretation retains the prior record;
        # a changed one supersedes it and keeps its history. Nothing here
        # rewrites the observed records that the interpretation cites.
        codelet_beliefs = apply_belief_proposals(
            changes, nominations,
            companion_name=self.ghost.get_companion_name(),
            stale_ids=stale,
            catalogue_version=codelets.catalogue_version,
        )
        # The winning nominations are the workspace focus that the public
        # workflows plan their memory context from. Stale ones never win.
        attention_focus = [{"text": item.text, "target_ids": list(item.target_ids), "reality": item.reality}
                           for item in nominations
                           if item.status == "selected" and item.nomination_id not in stale and item.text]
        attention_ids: list[int] = []
        winning_ids: list[int] = []
        for nomination in nominations:
            status, reason = nomination.status, nomination.reason
            if nomination.nomination_id in stale:
                status, reason = "stale", "a source changed before commit; was " + (nomination.status or "unscored")
            record = changes.create(AttentionRecord(
                content=nomination.text or nomination.reasoning or nomination.attention_kind,
                tick_id=tick,
                target_knoxel_ids=list(nomination.target_ids),
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
                    "consumers": proposal_consumers(nomination.attention_kind) if status == "selected" else [],
                    "loop": codelets.loop,
                    "catalogue_version": codelets.catalogue_version,
                    "catalogue_digest": codelets.catalogue_digest,
                },
            ))
            attention_ids.append(record.id)
            if status == "selected":
                winning_ids.append(record.id)
        workspace = work.workspace
        frame = changes.create(AttentionFrameRecord(
            content=workspace.frame_reason or work.summary,
            tick_id=tick,
            focus_knoxel_ids=list(workspace.focus_ids),
            # A cycle without nominations keeps the ids that attention selected.
            winning_attention_ids=winning_ids if nominations else list(work.attention.selected_ids),
            active_goal_ids=goal_ids,
            control_mode=workspace.control_mode,
            frame_reason=workspace.frame_reason or work.summary,
            metadata={
                "codelet_runs": [run.model_dump(mode="json", include={"codelet", "kind", "status", "reason",
                                                                      "activation", "calls"})
                                 for run in codelets.runs],
                "codelet_cost": codelets.cost.model_dump(mode="json"),
                "catalogue_version": codelets.catalogue_version,
                "catalogue_digest": codelets.catalogue_digest,
                "stale_nominations": sorted(stale),
                "attention_record_ids": attention_ids,
                "belief_proposals": codelet_beliefs,
                "pathway_signals": dict(codelets.pathway_signals),
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
        focus = work.attention.competition.focus
        selected_ids = {memory.knoxel_id for memory in work.selected_memories}
        reflection_versions = {}
        for item in request.reflection_guidance:
            effective_id = item.effective_record_id
            if (item.owner == "attention" and focus is not None and focus.control_record_id == effective_id
                    or item.owner == "belief" and effective_id in selected_ids):
                reflection_versions[str(item.experiment_id)] = effective_id
        selection_dump = work.action_selection.model_dump(mode="json")
        selection = new_feature(self.ghost, work.action_selection.reasoning or work.summary,
            FeatureType.StateUpdate, source=self.ghost.get_companion_name(), source_kind=SourceKind.Self,
            changes=changes, metadata={"operation": "action_selection", "selection": selection_dump,
                                       "reflection_versions": reflection_versions},
            lineage=lineage.model_copy(update={"operation": "action_selection", "parent_record_ids": [frame.id]}))
        created_order.append(selection.id)
        if focus is not None and focus.fresh and focus.accept:
            from workflows.cognitive_control.attention import stage_focus
            accepted_focus = stage_focus(changes, self._memory_view,
                instruction=focus.instruction, focus=focus.model_dump(mode="json"),
                companion_name=self.ghost.get_companion_name(),
                expires_at=(datetime.utcnow() + timedelta(seconds=focus.seconds)).isoformat(),
                lineage=lineage.model_copy(update={"operation": "focus_acceptance"}))
            created_order.append(accepted_focus.id)
        memory_evidence = [item.model_dump(mode="json") for item in work.selected_memories]
        memory_texts = [item.text for item in work.selected_memories]
        experience_text = workspace.subjective_experience or workspace.frame_reason
        published: list[tuple] = []
        rejected = []
        queued = []
        duplicates = []
        for dispatch in work.dispatches:
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
            presentation = dispatch.presentation
            if presentation.mode == "deceive" and (
                not presentation.purpose or not presentation.known_fact_ids
                or any(key not in view.revisions for key in presentation.known_fact_ids)
            ):
                raise ValueError("The proposed deception lacks retained factual context.")
            arguments = dispatch.arguments.payload()
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
                    "arguments": arguments,
                    "affect_summary": work.affect.felt,
                    "appraisal_id": appraisal_feature.id,
                    "evidence_ids": list(self._stimulus_ids),
                    "self_origin": not any(item.source == "user_input" for item in batch),
                    "regulation": work.affect.regulation,
                    "memory_evidence": memory_evidence,
                    "memory_trace": memory_trace,
                    "belief_updates": belief_receipt,
                    "presentation": presentation.model_dump(mode="json"),
                    "dispatch_status": "pending",
                    "dispatch_target": dispatch.workflow_name,
                    "dispatch_correlation": correlation,
                    "dispatch_payload": {"action_id": dispatch.action_id, "workflow_name": dispatch.workflow_name,
                        "content": dispatch.content, "arguments": arguments,
                        "stimuli": stimulus_records, "memories": memory_texts,
                        "memory_evidence": memory_evidence,
                        "workspace": experience_text,
                        "affect": work.affect.felt, "regulation": work.affect.regulation,
                        "attention": attention_focus},
                },
            ))
            created_order.append(action.id)
            published.append((dispatch, action.id))
        summary = workspace.subjective_experience.strip() or work.affect.felt.strip() or work.summary
        state = self._accepted_state(work, tick, frame.id, summary, published, correlation)
        changes.set_gwt(
            GwtTurnRecord(
                tick_id=tick, attention_frame_id=frame.id,
                knoxel_ids=list(dict.fromkeys([*self._stimulus_ids, *created_order,
                                             *(item.knoxel_id for item in work.selected_memories)])),
                summary=summary,
            ),
            state,
        )
        control = self.cognitive_control
        if control is None:
            receipt = self.ghost.commit(changes)
        else:
            receipt = control.commit(changes, work.wait_request, self._control_generation,
                envelope_ids={item.id for item in batch},
                wake_ids=[int(item.payload["knoxel_id"]) for item in batch if item.topic == "cognition.wake"],
                private_generation=self._control_generation if self._private_only else None,
                source_ids=list(self._stimulus_ids))
        tick = receipt.tick_id

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
                    "arguments": dispatch.arguments.payload(),
                    "stimuli": stimulus_records,
                    "memories": memory_texts,
                    "memory_evidence": memory_evidence,
                    "workspace": experience_text,
                    "affect": work.affect.felt,
                    "regulation": work.affect.regulation,
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

    def _accepted_state(self, work: CycleWork, tick: int, frame_id: int,
                        summary: str, published: list[tuple], correlation: str) -> CycleState:
        """Finish the cycle state that the phases changed and return it.

        The phases changed the previous state in place. The appraise phase set
        the mental state. Here the state gets its tick, time, frame, dispatch
        references, and the new broadcast. The broadcast queue keeps the last
        ``CCQ_LENGTH`` turns.
        """
        state = work.state
        state.tick_id = tick
        state.timestamp = datetime.utcnow()
        state.attention_frame_id = frame_id
        state.published_slots = ["main"]
        state.stimuli_ids = list(self._stimulus_ids)
        state.control_generation = self._control_generation
        state.commit_hash = None
        # Every dispatch of the old single decision belongs to the primary
        # group, as in the decide phase. A cycle without stimuli acts for the self.
        primary = primary_group(work)
        key = primary.key if primary is not None else SELF_SHARD
        state.dispatches = [ActionDispatchRef(
            action_id=created_id, group=key, correlation_id=correlation,
            workflow=dispatch.workflow_name, action_type=dispatch.action_type,
        ) for dispatch, created_id in published]
        state.broadcasts = [*state.broadcasts, BroadcastRef(tick_id=tick, summary=summary)][-CCQ_LENGTH:]
        return state

    def _committed_action_id(self, dispatch, tick: int) -> int | None:
        """Return the accepted action id for this identity and tick, if any."""
        for knoxel in self.ghost.all_knoxels.values():
            if not isinstance(knoxel, ActionSchemaRecord):
                continue
            if int(knoxel.tick_id) != int(tick):
                continue
            if "action_id" in knoxel.metadata and knoxel.metadata["action_id"] == dispatch.action_id:
                return int(knoxel.id)
        return None

    def _active_game_request(self, dispatch, correlation: str, batch: list[Envelope]) -> str | None:
        """Keep progress for an active request from starting another game run.

        Read the current durable run table before committing an action. A new
        user input permits another request. Updates and cancellation still use
        the existing management controls. Completed runs release this boundary.
        An action without an operation is a start, or an update when it names
        a run id.
        """
        route = dispatch.workflow_name
        arguments = dispatch.arguments
        operation = arguments.operation or ("update" if arguments.run_id else "start")
        if route in {"minecraft", "minecraft_project"} and operation != "start":
            return None
        if route == "self_management" and arguments.operation == "start":
            route = arguments.workflow_name
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
            if "action_id" in knoxel.metadata and knoxel.metadata["action_id"] == dispatch.action_id:
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


__all__ = ["InnerThoughtLoop", "MainCogLoop", "build_main_cognition_tools", "codelet_history",
           "pathway_precursors", "seconds_since_user_input", "stale_nominations"]
