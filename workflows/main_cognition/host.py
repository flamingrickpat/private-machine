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

from pm import vclock

from pm.records.copying import pickle_copy

import asyncio
import json
import threading
import traceback
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, ValidationError

from pm.bus.envelopes import Envelope, new_envelope, new_id
from pm.constants import (
    CCQ_LENGTH, CSM_RECENT_DIALOGUE_TURNS, CYCLE_RETRY_DELAY_SECONDS, CYCLE_RETRY_MAX_DELAY_SECONDS,
    EXPECTATION_OPEN_LIMIT, WORKSPACE_TAIL_CANDIDATES,
)
from pm.llm.completion import TextCompletion
from pm.llm.http_client import ModelError
from pm.llm.structured import context_measure
from pm.llm.prompt_encoding import TurnInput, normalize_turns
from pm.memory.snapshot import GhostCancelled
from pm.records.base import Provenance
from pm.records.codelet_records import AffectMeta, CodeletPercept, Structure
from pm.records.cognitive_records import ExpectationRecord
from pm.records.cycle_records import ActionDispatchRef, BroadcastRef
from pm.records.cycle_state import CycleState
from pm.records.mental_state import PersonalityProfile, features_to_history
from pm.records.workspace import SELF_SHARD, ShardKey
from pm.records.enums import ActionType, FeatureType, SourceKind
from pm.records.global_workspace import CodeletRunRef, CycleTrace, GwtTurnRecord, PathwayBoostRef
from pm.records.memory_records import ActionSchemaRecord, new_feature
from pm.runs.workflow_module import CognitiveWorkflow
from workflows.cognitive_control.schemas import MentalPlan
from workflows.inference_settings import REASONING
from workflows.main_cognition.action_request import partner_key, request_shards
from workflows.main_cognition.codelet_agent import build_codelet_agent
from workflows.main_cognition.csm_view import record_view
from workflows.main_cognition.cycle_work import CycleRequest, CycleWork, provisional_tick
from workflows.main_cognition.experience import consolidate_cycle, update_beliefs
from workflows.main_cognition.expectations import SOURCE_REF as EXPECTATION_SOURCE
from workflows.main_cognition.model_json import ModelFormatError
from workflows.main_cognition.phases.codelets import RECOMMENDATION
from workflows.main_cognition.phases.intake import group_key
from workflows.main_cognition.schemas import (
    ActionDispatch,
    ActionMetadata,
    ActionRequest,
    FocusProposalRef,
    GoalChange,
    GoalRecordView,
    ManagedRunView,
    MemoryCandidate,
    MentalProposal,
    OpenExpectation,
    RecordView,
    ShardStream,
    Stimulus,
    WorkflowOption,
)
from workflows.main_cognition.story_prefix import shard_label
from workflows.main_cognition.cycle_commits import (
    active_goal_texts, active_intentions, apply_appraisal, apply_goal_changes,
)
from workflows.inner_thought.expectations import pending_expectations
from workflows.public_speech.public_context import build_public_context


def build_main_cognition_tools(
    ghost,
    runtime_root: Path,
    llm_complete: TextCompletion,
    *,
    cancel_provider: Callable[[], threading.Event] | None = None,
    embeddings=None,
) -> Callable[[object, object], dict]:
    """Return the ``tools_factory(environment, batch)`` of the main module.

    The factory grants these tools to each run:

    - ``llm_complete``: one model call. It turns hidden reasoning off,
      because a reasoning budget exhausted real completions before. It passes
      the run's cancellation signal, so a failed parallel call or a
      cancelled run also stops its sibling calls.
    - ``embed_texts``: the embeddings of a list of texts, through
      ``embeddings`` (a ``pm.memory.embeddings.EmbeddingService``). The
      service embeds temporary texts directly. Record views carry stored vectors.
    - ``codelet_agent``: one agent codelet session over the frozen grant of
      the run (:func:`~workflows.main_cognition.codelet_agent.build_codelet_agent`).

    Without ``embeddings`` the factory grants ``llm_complete`` only. The
    codelets phase then fails, so real cognition must pass the service.
    ``runtime_root`` stays in the signature for the application wiring.
    Phases never read the store through these tools; the agent session reads
    the frozen grant only.
    """

    def factory(environment, batch) -> dict:
        stop = cancel_provider() if cancel_provider is not None else threading.Event()

        def complete(messages: Sequence[TurnInput], **generation: object) -> str:
            generation["reasoning_budget"] = REASONING["off"]
            return llm_complete(normalize_turns(messages), cancel=stop, **generation)

        tools = {"llm_complete": complete}
        measure = context_measure(llm_complete)
        if measure is not None:
            complete.measure_context = measure
        if embeddings is not None:
            async def embed_all(texts: list[str]) -> list[list[float]]:
                return [await embeddings.embed(text, cancel=stop) for text in texts]

            def embed_texts(texts: list[str]) -> list[list[float]]:
                return asyncio.run(embed_all(list(texts)))

            tools["embed_texts"] = embed_texts
            tools["codelet_agent"] = build_codelet_agent(ghost, environment, embeddings, stop)
        return tools

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
    turns = ghost.read_workspace_turns()
    if not turns:
        return ""
    return turns[max(turns)].summary


def dialogue_partner(feature, entity_ids: dict[str, int], user_entity_id: int | None,
                     self_entity_id: int | None) -> int | None:
    """Return the entity that the companion talks with in one dialogue line.

    Only a companion reply names an addressee (``addressee_entity_id``), so
    a reply belongs to the entity that it answers. Another line belongs to
    its speaker: ``feature.entity_id``, or the entity of the source name in
    ``entity_ids`` (entity names and aliases). A line of the user, of the
    companion without an addressee, or of an unknown speaker belongs to the
    user, so dialogue without a partner stays in the user's shard.
    """
    if feature.addressee_entity_id:
        return int(feature.addressee_entity_id)
    speaker = feature.entity_id or (entity_ids[feature.source] if feature.source in entity_ids else 0)
    if speaker and speaker not in {self_entity_id, user_entity_id}:
        return int(speaker)
    return user_entity_id


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
    return max(0.0, (vclock.utcnow() - max(times)).total_seconds())


class MainCogLoop(CognitiveWorkflow):
    """Real cognitive parent: one typed phase sequence, committed GWT, typed dispatches.

    The workflow owns the phase order and the dispatch ids. This module owns
    the bus and the store: it builds one typed request from the batch,
    commits one GWT turn, cycle state, subjective
    experience, and action schema with its expectations per released
    dispatch in one write batch, then publishes ``action.request`` targeted
    at each dispatch's workflow name. Each action carries the correlation of
    its stimulus group, so the groups of one batch keep separate exchanges.
    Rejected dispatches publish an explicit receipt and never an action.

    The application sets ``cognitive_control``, ``release_private_results``,
    ``capability_provider``, and ``retrieval_query_provider`` after
    construction. Each stays None when the application does not use it.
    """

    WORKFLOW_NAME = "main_cognition"
    inference_admission = None
    DATA_TOPICS = frozenset({
        "user.input", "thought.created", "task.telemetry", "workflow.question", "workflow.event",
        "system.lifecycle",
        "minecraft.world", "environment.event",
        "cognition.wake",
    })

    def __init__(self, *args, workflow_catalogue=None, embedding_config_path=None,
                 codelet_definitions=None, **kwargs):
        super().__init__(*args, **kwargs)
        # The codelet definitions are a workflow resource. ``None`` selects
        # codelets.yaml; a gate or persona can name another file.
        self.codelet_definitions = str(codelet_definitions) if codelet_definitions else ""
        # Configuration supplies the action space. The decide phase uses its
        # default action space when this list is empty.
        self.workflow_catalogue = [WorkflowOption.model_validate(item) for item in workflow_catalogue or []]
        # An optional provider rebuilds the action space at each cycle, for
        # example with the MCP tools that are reachable now. Only the route
        # prompt renders the descriptions.
        self.workflow_catalogue_provider: Callable[[], list[WorkflowOption]] | None = None
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
        if self.inference_admission is not None and envelope.kind != "control":
            self.inference_admission.reserve(envelope.id)
        if self.cognitive_control is not None:
            self.cognitive_control.observe(envelope)
        super().deliver(envelope)

    def handle_batch(self, batch: list[Envelope]) -> None:
        """Hold foreground priority through the cycle and its reply dispatch."""
        try:
            self._handle_priority_batch(batch)
        finally:
            if self.inference_admission is not None:
                for envelope in batch:
                    self.inference_admission.release(envelope.id)

    def _handle_priority_batch(self, batch: list[Envelope]) -> None:
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
        CYCLE_RETRY_MAX_DELAY_SECONDS. A completion that the provider failed,
        filtered, cut off, or left empty (``ModelError``, for example
        ``ProviderCompletionError``) is outside our control and gets the same
        whole-cycle retry: the A19F step 11 gate lost two cycles to such
        completions. No partial text of it is used.
        Cancellation stops retries. Ordinary Python errors propagate.
        """
        from pm.memory.snapshot import GhostConflict
        self._event_stimuli = {}
        attempt = 0
        delay = CYCLE_RETRY_DELAY_SECONDS
        while not self.stop_event.is_set():
            try:
                super().handle_batch(batch)
                return
            except (ModelFormatError, ModelError) as error:
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
            selected_sources = {key for item in matches for key in item.source_ids if key in memory_view.revisions}
            original_sources = {key for key in selected_sources
                                if evidence_role(memory_view, memory_view.get(key)) in {"user_report", "runtime_event", "world_observation"}}
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
        self_entity_id = self.ghost.get_entity_id(self.ghost.get_companion_name())
        for stimulus, source in zip(stimuli, sources, strict=True):
            stimulus.entity_id = self._speaker_entity(stimulus, source, self_entity_id)
        matches, trace = self._retrieve(stimuli, memory_view, sources)
        self._memory_view = memory_view

        from workflows.cognitive_control.attention import active_focus, focus_versions
        from workflows.reflection.experiments import effective_inputs
        control = self.cognitive_control
        maximum_focus_bias = control.config.maximum_focus_bias if control is not None else None
        focus = active_focus(self.ghost, maximum_strength=maximum_focus_bias)
        self._focus_family_revisions = focus_versions(memory_view)
        reflection = effective_inputs(memory_view, focus)
        guidance = list(reflection.guidance)
        mental_proposals: list[MentalProposal] = []
        focus_proposals: list[FocusProposalRef] = []
        for source in sources:
            if "mental_plan" not in source.metadata:
                continue
            try:
                plan = MentalPlan.model_validate(source.metadata["mental_plan"])
                goal_changes = [GoalChange.model_validate(item) for item in plan.goal_changes]
            except ValidationError as error:
                # Old generated records can predate validation at the producer.
                # Preserve them as evidence and reject only their proposal.
                self.bus.put(new_envelope(self.module_id, "telemetry", "mental.rejected", correlation,
                    {"record_id": source.id, "reason": str(error)}))
                continue
            if plan.disposition != "propose":
                continue
            mental_proposals.append(MentalProposal(
                record_id=source.id, instruction=plan.instruction, summary=plan.summary,
                disposition=plan.disposition, evidence_ids=list(plan.evidence_ids),
                goal_changes=goal_changes))
            if plan.focus is not None:
                advice = plan.focus
                if maximum_focus_bias is not None:
                    advice = advice.model_copy(update={"strength": min(advice.strength, maximum_focus_bias)})
                focus_proposals.append(FocusProposalRef(record_id=source.id, focus=advice))
        # Deliberate recall can introduce sources that a literal stimulus
        # query missed. They remain original, revisioned memory candidates.
        focus_ids = set(focus.promote_ids + focus.reduce_ids) if focus is not None else set()
        focus_ids.update(reflection.source_ids)
        for item in mental_proposals:
            focus_ids.update(item.evidence_ids)
        for item in focus_proposals:
            focus_ids.update(item.focus.promote_ids + item.focus.reduce_ids)
        self._focus_source_revisions = {}
        present = {item.knoxel_id for item in matches}
        from pm.memory.retrieval import evidence_role, memory_kind
        for key in sorted(focus_ids):
            if key not in memory_view.revisions:
                continue
            record = memory_view.get(key)
            self._focus_source_revisions[key] = memory_view.revisions[key]
            if key not in present:
                raw = record.as_payload(include_embedding=False)
                matches.append(MemoryCandidate(knoxel_id=key, text=record.content,
                    revision=memory_view.revisions[key], source_ids=[key],
                    memory_kind=memory_kind(raw), evidence_role=evidence_role(memory_view, raw),
                    reasons=["deliberate mental recall"]))
        previous = self.ghost.read_cycles()[-1] if self.ghost.read_cycles() else None
        # The appraise phase computes the EMA mental state from this history.
        # It reads the live store, like ``previous``: the cycle state and the
        # features that produced it come from the same committed ticks.
        with self.ghost.lock:
            history = features_to_history(self.ghost.all_features, self.ghost.all_codelet_percepts)
        capabilities = self.capability_provider() if self.capability_provider is not None else None
        public = build_public_context(self.ghost, ActionRequest(stimuli=stimuli), capabilities=capabilities)
        user_entity_id = self.ghost.get_entity_id(self.ghost.get_user_name())
        keys = [group_key(item, CycleRequest(user_entity_id=user_entity_id)) for item in stimuli]
        group_entities = sorted({int(key.ref) for key in keys if key.kind == "entity"})
        entity_names = {entity_id: self.ghost.get_entity_name(entity_id) or f"entity {entity_id}"
                        for entity_id in group_entities}
        open_expectations = self._open_expectations(memory_view)
        narrative_ids = self._narrative_ids(memory_view, [*([self_entity_id] if self_entity_id else []),
                                                          *group_entities])
        known = [shard.key for shard in previous.workspace.all_shards()] if previous is not None else [SELF_SHARD]
        # The user's shard always gets a tail, so a reply to the user after a
        # thought or a workflow event still reads the latest dialogue.
        user_key = [ShardKey(kind="entity", ref=str(user_entity_id))] if user_entity_id is not None else []
        streams = self._streams(memory_view, list(dict.fromkeys([*known, *user_key, *keys])),
                                user_entity_id, self_entity_id)
        workspace_ids = ([*previous.workspace.item_ids(), *previous.workspace.tail_ids()]
                         if previous is not None else [])
        extra = [*narrative_ids, *workspace_ids, *(key for stream in streams for key in stream.record_ids),
                 *(int(item.id) for item in active_intentions(memory_view)[:8]),
                 *(item.record_id for item in open_expectations)]
        records, recent_dialogue = self._record_views(memory_view, sources, matches, previous, extra)
        request = CycleRequest(
            run_id=self.invocation_id or "",
            correlation_id=correlation,
            companion_name=self.ghost.get_companion_name(),
            user_name=self.ghost.get_user_name(),
            user_entity_id=user_entity_id,
            self_entity_id=self_entity_id,
            entity_names=entity_names,
            state_summary=_state_summary(self.ghost),
            previous_state=previous,
            cycle_time=vclock.utcnow(),
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
            post_reply_seconds=control.config.post_reply_seconds if control is not None else 0.0,
            capabilities=_capabilities(self.ghost),
            available_workflows=(self.workflow_catalogue_provider() if self.workflow_catalogue_provider is not None
                                 else self.workflow_catalogue),
            public_context=public,
            managed_runs=[ManagedRunView.model_validate(run)
                          for run in list(self.ghost.read_run_receipts().values())[-16:]],
            stimulus_revisions=dict(self._stimulus_revisions),
            active_goal_records=[
                GoalRecordView(knoxel_id=int(item.id), text=str(item.content),
                               revision=int(memory_view.revisions[int(item.id)]))
                for item in active_intentions(memory_view)[:8]
            ],
            seconds_since_user_input=seconds_since_user_input(self.ghost, self._stimulus_ids),
            character_card=self.ghost.ghost_config.character_card_story,
            codelet_definitions=self.codelet_definitions,
            records=records,
            recent_dialogue_ids=recent_dialogue,
            narrative_ids=narrative_ids,
            streams=streams,
            open_expectations=open_expectations,
        )
        return request.model_dump(mode="json")

    def _speaker_entity(self, stimulus: Stimulus, source, self_entity_id: int | None) -> int | None:
        """Return the entity that spoke a dialogue stimulus, or None.

        Only a dialogue stimulus (kind ``user.input``) has a speaker. The
        feature names its entity, or the entity follows from its source
        name. The companion itself and an unknown speaker give None, so the
        intake phase groups the stimulus by its source as before.
        """
        if stimulus.kind != "user.input":
            return None
        entity_id = source.entity_id or self.ghost.get_entity_id(source.source)
        return None if entity_id is None or entity_id == self_entity_id else int(entity_id)

    def _open_expectations(self, memory_view) -> list[OpenExpectation]:
        """Return the newest pending expectations that main cognition recorded, oldest first.

        The private loop keeps its own expectations. It resolves them itself
        until A19G moves it onto this cycle. At most
        ``EXPECTATION_OPEN_LIMIT`` are offered.
        """
        records = [record for record in pending_expectations(memory_view)
                   if record.provenance is not None and record.provenance.source_ref == EXPECTATION_SOURCE]
        return [OpenExpectation(record_id=record.id, content=record.content, expected_valence=record.expected_valence,
                                urgency=record.urgency, fulfillment=record.fulfillment, tick_id=record.tick_id,
                                revision=memory_view.revisions[record.id],
                                about_entity_ids=list(record.about_entity_ids))
                for record in records[-EXPECTATION_OPEN_LIMIT:]]

    def _narrative_ids(self, memory_view, entity_ids: list[int]) -> list[int]:
        """Return the newest narrative of each type for each entity in ``entity_ids``.

        The store keeps every refined version of a narrative. The newest one
        has the highest refinement tick, then the newest world time, then the
        highest id. Only records of the frozen view count.
        """
        newest: dict[tuple[str, int], tuple[tuple, int]] = {}
        with self.ghost.lock:
            for record in self.ghost.all_narratives:
                if record.entity_id not in entity_ids or record.id not in memory_view.revisions:
                    continue
                key = (record.narrative_type.value, record.entity_id)
                rank = (record.last_refined_with_tick or -1, record.timestamp_world_begin, record.id)
                if key not in newest or rank > newest[key][0]:
                    newest[key] = (rank, record.id)
        return [newest[key][1] for key in sorted(newest)]

    def _streams(self, memory_view, keys: list[ShardKey], user_entity_id: int | None,
                 self_entity_id: int | None) -> list[ShardStream]:
        """Return the newest stream records of each shard, oldest first.

        - Self shard: the companion's thoughts and experiences (``Thought``,
          ``ExternalThought``, ``InnerMonologue``, and ``SubjectiveExperience``
          features).
        - Entity shard: the dialogue with that entity
          (:meth:`_dialogue_partner`). A line of another speaker belongs to
          the speaker. A companion reply belongs to the entity that it
          addresses. Lines without a known partner belong to the user, so old
          dialogue stays in the user's shard.
        - Workflow shard: the ``WorldEvent`` features that the workflow
          reported.

        Each stream holds at most ``WORKSPACE_TAIL_CANDIDATES`` records. The
        intake phase cuts it to the tail budget.
        """
        inner = {FeatureType.Thought, FeatureType.ExternalThought, FeatureType.InnerMonologue,
                 FeatureType.SubjectiveExperience}
        with self.ghost.lock:
            features = sorted((item for item in self.ghost.all_features if item.id in memory_view.revisions),
                              key=lambda item: (item.timestamp_world_begin, item.id))
            names = {name: entity.id for entity in self.ghost.all_entities
                     for name in [*entity.aliases, entity.content]}
            partners = {item.id: dialogue_partner(item, names, user_entity_id, self_entity_id)
                        for item in features if item.feature_type == FeatureType.Dialogue}
        streams = []
        for key in keys:
            if key.kind == "self":
                chosen = [item for item in features if item.feature_type in inner]
            elif key.kind == "entity":
                chosen = [item for item in features if item.id in partners and partners[item.id] == int(key.ref)]
            else:
                chosen = [item for item in features
                          if item.feature_type == FeatureType.WorldEvent and item.source == key.ref]
            streams.append(ShardStream(key=key, record_ids=[item.id for item in chosen[-WORKSPACE_TAIL_CANDIDATES:]]))
        return streams

    def _record_views(self, memory_view, sources, matches: list[MemoryCandidate],
                      previous: CycleState | None, extra_ids: list[int]) -> tuple[dict[int, RecordView], list[int]]:
        """Return the record views that the CSM can hold, and the recent dialogue ids.

        The views cover the records of the previous CSM, the stimuli, the
        last ``CSM_RECENT_DIALOGUE_TURNS`` dialogue turns before the stimuli,
        the memory candidates, and ``extra_ids`` (narratives, goals, the
        previous workspace items and tails, and the stream records). The
        previous CSM and workspace hold committed records only, so each of
        them is in the frozen view. A missing record is a bug and raises
        KeyError.
        """
        views = {source.id: record_view(source) for source in sources}
        stimulus_ids = set(views)
        with self.ghost.lock:
            dialogue = sorted((item for item in self.ghost.all_features
                               if item.feature_type == FeatureType.Dialogue and item.id in memory_view.revisions
                               and item.id not in stimulus_ids),
                              key=lambda item: (item.timestamp_world_begin, item.id))[-CSM_RECENT_DIALOGUE_TURNS:]
        recent = [item.id for item in dialogue]
        held = list(previous.csm.items) if previous is not None else []
        for key in [*recent, *held, *(item.knoxel_id for item in matches), *extra_ids]:
            if key not in views:
                views[key] = record_view(memory_view.get(key))
        return views, recent

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
                    "executed_at": event.executed_at or vclock.utcnow().isoformat(),
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
        summary, the codelet percepts, the structures, the subjective
        experience, the decision record, one action
        schema per released dispatch with its expectations, the expectation
        check results, the GWT turn with the shard changes, and the cycle
        state. The backup follows the commit and precedes all bus events.
        A backup error propagates after memory acceptance. A stale source, a
        concurrently changed expectation, or a concurrent focus change aborts
        the whole batch.
        """
        if self.stop_event.is_set() or self.run_stop.is_set():
            return
        work = CycleWork.model_validate(report)
        request = work.request
        user_stimuli = {item.knoxel_id for item in request.stimuli if item.source == "user_input"}
        cause = batch[0].id if batch else None
        changes = self.ghost.begin_batch(
            self.invocation_id or new_id(), new_tick=True, cancel=self.run_stop,
        )
        changes.depend_on(self.last_environment.ghost_view.revisions,
                          manifest={"kind": "main_cognition", "invocation_id": self.invocation_id,
                                    "request": work.request.model_dump(mode="json")})
        changes.depend_on_commits(self.last_environment.ghost_view.commit_ids)
        tick = changes.tick_id
        view = changes.view()
        changes.expect_metadata_version("control_kind", "attention_focus", self._focus_family_revisions)
        for source_id, revision in self._stimulus_revisions.items():
            changes.expect_revision(source_id, revision)
        for source_id, revision in self._focus_source_revisions.items():
            changes.expect_revision(source_id, revision)
        for outcome in work.expectations:
            changes.expect_revision(outcome.record_id, outcome.revision)
        for memory in work.selected_memories:
            changes.expect_revision(memory.knoxel_id, memory.revision)
            for source_id in memory.source_ids:
                if source_id in self._memory_view.revisions:
                    changes.expect_revision(source_id, self._memory_view.revisions[source_id])

        # Apply the appraisal and goal proposals into this same batch before the
        # turn record. A failure here aborts the whole batch, so stale proposals never
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
        changes.depend_on(goal_ids)
        belief_receipt = update_beliefs(changes, work.context.belief_updates, user_name=self.ghost.get_user_name(),
                                        permitted_source_ids=set(self._stimulus_ids))
        experience_ids = consolidate_cycle(changes, self._stimulus_ids, appraisal_feature, work.affect.felt)
        # Codelet percepts, recommendations, and structures become records
        # now. The CSM, the codelet state, the workspace, and the attention
        # trace held pending ids until here. A percept that entered a shard
        # is staged as causal (see stage_percepts).
        pending = stage_percepts(changes, work)
        structure_ids = stage_structures(changes, work, pending)
        experience_feature = new_feature(self.ghost, work.experience.text, FeatureType.SubjectiveExperience,
            source=self.ghost.get_companion_name(), source_kind=SourceKind.Self, changes=changes,
            metadata={"memory_kind": "interpretation", "operation": "subjective_experience",
                      "seed": work.experience.seed})
        # A cycle with several significant groups also has one experience for each group.
        group_experiences = {item.record_id: new_feature(
            self.ghost, item.text, FeatureType.SubjectiveExperience, source=self.ghost.get_companion_name(),
            source_kind=SourceKind.Self, changes=changes,
            metadata={"memory_kind": "interpretation", "operation": "subjective_experience", "seed": item.seed,
                      "group": item.key.model_dump(mode="json")}).id
            for item in work.experience.groups}
        mapping = {**pending, **structure_ids, work.experience.record_id: experience_feature.id, **group_experiences}
        remap_pending(work, mapping)
        caused = mark_causal(changes, view, work)
        experience = work.experience
        created_order: list[int] = [appraisal_feature.id, *created_goal_ids, *experience_ids,
                                    *pending.values(), *structure_ids.values(), experience_feature.id,
                                    *group_experiences.values(),
                                    *belief_receipt["accepted_ids"]]
        from pm.records.lineage import RecordLineage
        parent = next((view.get(key) for key in self._stimulus_ids if view.get(key).lineage is not None), None)
        lineage = RecordLineage(
            episode_id=parent.lineage.episode_id if parent is not None else changes.commit_id,
            sequence=parent.lineage.sequence + 1 if parent is not None else 0,
            operation="main_appraisal", parent_record_ids=list(self._stimulus_ids),
            source_event_ids=list(self._stimulus_ids), intention_ids=goal_ids,
            topic_hints=list(parent.lineage.topic_hints) if parent is not None else [])
        for key in set(created_order) & changes.created_ids:
            changes.annotate_lineage(key, lineage)
        focus = work.attention.focus
        selected_ids = {memory.knoxel_id for memory in work.selected_memories}
        reflection_versions = {}
        for item in request.reflection_guidance:
            effective_id = item.effective_record_id
            if (item.owner == "attention" and focus is not None and focus.control_record_id == effective_id
                    or item.owner == "belief" and effective_id in selected_ids):
                reflection_versions[str(item.experiment_id)] = effective_id
        # The decision record keeps the operation name "action_selection",
        # because reflection finds its experiments' effects by it. The turn
        # record names it as ``decision_id``.
        decision_text = "\n".join(item.text for item in work.state.decisions)
        selection = new_feature(self.ghost, decision_text or work.summary,
            FeatureType.StateUpdate, source=self.ghost.get_companion_name(), source_kind=SourceKind.Self,
            changes=changes, metadata={"operation": "action_selection",
                                       "decisions": [item.model_dump(mode="json") for item in work.state.decisions],
                                       "routes": [item.model_dump(mode="json") for item in work.routes],
                                       "reflection_versions": reflection_versions},
            lineage=lineage.model_copy(update={"operation": "action_selection",
                                              "parent_record_ids": [experience_feature.id]}))
        created_order.append(selection.id)
        if focus is not None and focus.fresh and focus.accept:
            from workflows.cognitive_control.attention import stage_focus
            accepted_focus = stage_focus(changes, self._memory_view,
                instruction=focus.instruction, focus=focus.model_dump(mode="json"),
                companion_name=self.ghost.get_companion_name(),
                expires_at=(vclock.utcnow() + timedelta(seconds=focus.seconds)).isoformat(),
                lineage=lineage.model_copy(update={"operation": "focus_acceptance"}))
            created_order.append(accepted_focus.id)
        memory_evidence = [item.model_dump(mode="json") for item in work.selected_memories]
        # The check results come first, so a superseded expectation keeps
        # its checked fulfillment.
        apply_expectation_outcomes(changes, view, work)
        stimuli = {item.knoxel_id: item for item in request.stimuli}
        published: list[tuple[ActionDispatch, int, ActionRequest]] = []
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
            active_game = self._active_game_request(dispatch, dispatch.correlation_id, batch)
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
            partner = partner_key(dispatch.group, request.user_entity_id)
            action_request = ActionRequest(
                action_id=dispatch.action_id, workflow_name=dispatch.workflow_name,
                content=dispatch.content, arguments=dispatch.arguments, group=dispatch.group,
                partner_entity_id=int(partner.ref) if partner is not None else None,
                partner_name=shard_label(partner, request) if partner is not None else request.user_name,
                stimuli=[stimuli[key] for key in dispatch.stimulus_ids],
                shards=request_shards(work.state.workspace, dispatch.group, partner,
                                      lambda key: shard_label(key, request),
                                      hidden=experience.hidden_for(dispatch.group)),
                presentation=presentation, workspace=experience.text_for(dispatch.group),
                affect=work.affect.felt, regulation=work.affect.regulation)
            action = changes.create(ActionSchemaRecord(
                content=dispatch.content,
                action_type=ACTION_BY_NAME[dispatch.action_type],
                intention_ids=list(goal_ids),
                lineage=lineage.model_copy(update={"operation": "action_dispatch",
                                                  "parent_record_ids": [selection.id]}),
                metadata={
                    "action_id": dispatch.action_id,
                    "route": dispatch.workflow_name,
                    "workflow_name": dispatch.workflow_name,
                    "depends_on": list(dispatch.depends_on),
                    "requires_completed": list(dispatch.requires_completed),
                    "arguments": dispatch.arguments.payload(),
                    "affect_summary": work.affect.felt,
                    "appraisal_id": appraisal_feature.id,
                    "evidence_ids": list(dispatch.stimulus_ids),
                    "group": dispatch.group.model_dump(mode="json"),
                    "decision": dispatch.content,
                    "self_origin": not user_stimuli.intersection(dispatch.stimulus_ids),
                    "regulation": work.affect.regulation,
                    "memory_evidence": memory_evidence,
                    "memory_trace": memory_trace,
                    "belief_updates": belief_receipt,
                    "presentation": presentation.model_dump(mode="json"),
                    "dispatch_status": "pending",
                    "dispatch_target": dispatch.workflow_name,
                    "dispatch_correlation": dispatch.correlation_id,
                    "dispatch_payload": action_request.payload(),
                },
            ))
            created_order.append(action.id)
            created_order.extend(stage_expectations(changes, view, work, dispatch, action.id))
            published.append((dispatch, action.id, action_request))
        summary = experience.text.strip() or work.affect.felt.strip() or work.summary
        state = self._accepted_state(work, tick, summary, published, experience_feature.id)
        codelets = work.codelets
        trace = CycleTrace(
            codelet_budget=codelets.budget,
            codelet_runs=[CodeletRunRef.model_validate(run.model_dump(include=set(CodeletRunRef.model_fields)))
                          for run in codelets.runs],
            codelet_ratings=dict(codelets.ratings),
            selector_patterns=codelets.patterns, echo_ids=list(codelets.echo_ids),
            pathway_boosts=[PathwayBoostRef.model_validate(boost.model_dump(include=set(PathwayBoostRef.model_fields)))
                            for boost in codelets.pathway_boosts],
            definitions_version=codelets.definitions_version, definitions_digest=codelets.definitions_digest,
            percept_ids=sorted(pending.values()), structure_ids=sorted(structure_ids.values()),
            causal_percept_ids=caused, llm_calls=codelets.cost.llm_calls,
            input_tokens_estimate=codelets.cost.input_tokens_estimate,
            output_tokens_estimate=codelets.cost.output_tokens_estimate)
        # ``knoxel_ids`` keeps its meaning for its readers: the stimuli, the
        # created records, the selected memories, and every workspace item
        # and tail record. ``shard_changes`` is the v2 change record.
        changes.set_gwt(
            GwtTurnRecord(
                tick_id=tick, stimulus_ids=list(self._stimulus_ids),
                knoxel_ids=list(dict.fromkeys([*self._stimulus_ids, *created_order,
                                             *(item.knoxel_id for item in work.selected_memories),
                                             *state.workspace.item_ids(), *state.workspace.tail_ids()])),
                summary=summary,
                shard_changes=[pickle_copy(change) for change in work.attention.changes],
                experience_id=experience_feature.id,
                decision_id=selection.id,
                trace=trace,
            ),
            state,
        )
        control = self.cognitive_control
        if control is None:
            receipt = self.ghost.commit(changes)
        else:
            receipt = control.commit(changes, work.wait_request, self._control_generation,
                envelope_ids={item.id for item in batch},
                wake_ids=[KnoxelPayload.model_validate(item.payload).knoxel_id
                          for item in batch if item.topic == "cognition.wake"],
                private_generation=self._control_generation if self._private_only else None,
                source_ids=list(self._stimulus_ids))
        # Failed attempts must leave the last completed cycle's backup intact.
        self.ghost.backup()
        tick = receipt.tick_id

        self.bus.put(new_envelope(self.module_id, "data", "gwt.updated", "", {"gwt_tick": tick}))
        for dispatch, created_id, action_request in published:
            # A repeated commit returns the original receipt and does not apply
            # the new batch. Resolve the accepted action by durable identity and
            # tick so the published id is always committed.
            committed_id = self._committed_action_id(dispatch, tick) or created_id
            self.bus.put(new_envelope(
                self.module_id, "data", "action.request", dispatch.correlation_id,
                action_request.model_copy(update={"knoxel_id": committed_id, "gwt_tick": tick}).payload(),
                target=dispatch.workflow_name,
                causation_id=cause,
            ))
        for dispatch in queued:
            self._publish_receipt(dispatch.correlation_id, dispatch, "queued", cause)
        for dispatch in rejected:
            self._publish_receipt(dispatch.correlation_id, dispatch, "rejected", cause)
        for dispatch, existing_id in duplicates:
            self._publish_receipt(
                dispatch.correlation_id, dispatch, "duplicate", cause,
                detail=f"active action {existing_id} has identity {dispatch.action_id}",
            )

    def _accepted_state(self, work: CycleWork, tick: int, summary: str, published: list[tuple],
                        experience_id: int) -> CycleState:
        """Finish the cycle state that the phases changed and return it.

        The phases changed the previous state in place. The appraise phase set
        the mental state, and the decide phase set the decisions. Here the
        state gets its tick, time, dispatch references, and the new broadcast
        with the subjective experience. The broadcast queue keeps the last
        ``CCQ_LENGTH`` turns. A dispatch reference exists only for a
        published action, with its action record id.
        """
        state = work.state
        # The phases used the provisional tick. Items and codelets that this
        # cycle touched get the real tick.
        expected = provisional_tick(work)
        for item in state.csm.items.values():
            item.entered_tick = tick if item.entered_tick == expected else item.entered_tick
            item.last_boost_tick = tick if item.last_boost_tick == expected else item.last_boost_tick
        for runtime in state.codelets.codelets.values():
            runtime.last_fired_tick = tick if runtime.last_fired_tick == expected else runtime.last_fired_tick
        entered = [item for shard in state.workspace.all_shards() for item in shard.items]
        entered += [item for change in work.attention.changes for item in change.entered]
        for item in entered:
            item.entered_tick = tick if item.entered_tick == expected else item.entered_tick
        state.tick_id = tick
        state.timestamp = vclock.utcnow()
        state.control_generation = self._control_generation
        state.dispatches = [ActionDispatchRef(
            action_id=created_id, group=dispatch.group, correlation_id=dispatch.correlation_id,
            workflow=dispatch.workflow_name, action_type=dispatch.action_type,
        ) for dispatch, created_id, _ in published]
        state.broadcasts = [*state.broadcasts, BroadcastRef(tick_id=tick, summary=summary,
                                                            experience_id=experience_id)][-CCQ_LENGTH:]
        return state

    def _committed_action_id(self, dispatch, tick: int) -> int | None:
        """Return the accepted action id for this identity and tick, if any."""
        for knoxel in self.ghost.all_knoxels.values():
            if not isinstance(knoxel, ActionSchemaRecord):
                continue
            if int(knoxel.tick_id) != int(tick):
                continue
            if ActionMetadata.model_validate(knoxel.metadata).action_id == dispatch.action_id:
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
            if ActionMetadata.model_validate(knoxel.metadata).action_id == dispatch.action_id:
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


def stage_percepts(changes, work: CycleWork) -> dict[int, int]:
    """Stage one ``CodeletPercept`` record for each percept and recommendation.

    Returns ``{pending id: record id}``. Each percept record stores its
    appraisal state and its delta (the mini-appraisal). A recommendation
    record stores the mean appraisal state and the summed delta of its
    percepts.

    A record that is a workspace item after attention is staged with
    ``causal=True``: it entered the story. ``features_to_history`` then reads
    its delta, so it changes the EMA mental state of the next cycle. A
    percept that never entered a shard stays non-causal and changes nothing.
    """
    mapping: dict[int, int] = {}
    companion = work.request.companion_name
    conscious = set(work.state.workspace.item_ids())
    # Outputs are in run order. A variant can be about a percept of an
    # earlier run, and that percept already has its record id here.
    for output in work.codelets.outputs:
        for index, variant in enumerate(output.variants):
            shared = {"codelet": output.codelet, "mode": output.mode, "wave": output.wave, "variant": index,
                      "target_shard": variant.target_shard.model_dump(mode="json"), "source": companion}
            recommendation = changes.create(CodeletPercept(
                content=variant.recommendation, codelet_name=output.codelet,
                explanation=variant.recommendation, reasoning_trace=variant.recommendation,
                affect=AffectMeta(affective_valence=variant.valence, incentive_salience=variant.salience),
                mental_state_appraisal_vec=list(variant.recommendation_vec),
                mental_state_delta_vec=list(variant.recommendation_delta_vec),
                causal=variant.recommendation_id in conscious,
                provenance=Provenance(source_kind=SourceKind.Self, source_ref=f"codelet:{output.codelet}"),
                metadata={**shared, "percept_type": RECOMMENDATION,
                          "about_ids": [mapping.get(key, key) for key in variant.about_ids]}))
            mapping[variant.recommendation_id] = recommendation.id
            for percept in variant.percepts:
                record = changes.create(CodeletPercept(
                    content=percept.text, codelet_name=output.codelet,
                    explanation=variant.recommendation, reasoning_trace=variant.recommendation,
                    affect=AffectMeta(affective_valence=percept.valence, incentive_salience=percept.salience),
                    mental_state_appraisal_vec=list(percept.appraisal_vec),
                    mental_state_delta_vec=list(percept.delta_vec),
                    causal=percept.record_id in conscious,
                    provenance=Provenance(source_kind=SourceKind.Self, source_ref=f"codelet:{output.codelet}"),
                    metadata={**shared, "percept_type": percept.percept_type, "details": dict(percept.details),
                              "recommendation_id": recommendation.id,
                              "appraisal_general": percept.appraisal_general.model_dump(mode="json"),
                              "appraisal_social": percept.appraisal_social.model_dump(mode="json"),
                              "csm_activation": percept.activation}))
                mapping[percept.record_id] = record.id
    return mapping


def stage_structures(changes, work: CycleWork, pending: dict[int, int]) -> dict[int, int]:
    """Stage one ``Structure`` record for each structure of the cycle.

    ``pending`` maps the pending percept ids to record ids, so a structure
    names its percepts by record id. Returns ``{pending structure id: record
    id}``. The structure content is its theme and its reason. The metadata
    names the builder, the kind, and the attention scores of each shard.
    """
    scores: dict[int, list[dict]] = {}
    for shard in work.attention.ratings:
        for rated in shard.ratings:
            scores.setdefault(rated.structure_id, []).append(
                {"shard": shard.key.model_dump(mode="json"), "rating": rated.rating, "aux": rated.aux,
                 "score": rated.score, "won": rated.won, "reasoning": rated.reasoning})
    mapping: dict[int, int] = {}
    for draft in work.structures.structures:
        record = changes.create(Structure(
            content=f"{draft.theme}: {draft.reason}",
            internal_knoxel_ids=[pending.get(key, key) for key in draft.record_ids],
            relations=[relation.model_copy(update={"source_id": pending.get(relation.source_id, relation.source_id),
                                                   "target_id": pending.get(relation.target_id, relation.target_id)})
                       for relation in draft.relations],
            affect=AffectMeta(affective_valence=max(-1.0, min(1.0, draft.valence)),
                              incentive_salience=max(0.0, min(1.0, draft.salience))),
            target_shard=draft.target_shard,
            provenance=Provenance(source_kind=SourceKind.Self, source_ref=f"structures:{draft.builder}"),
            primitive_metadata={"builder": draft.builder, "kind": draft.kind, "theme": draft.theme,
                                "reason": draft.reason},
            metadata={"attention": scores[draft.record_id] if draft.record_id in scores else []}))
        mapping[draft.record_id] = record.id
    return mapping


def mark_causal(changes, view, work: CycleWork) -> list[int]:
    """Return percept IDs that entered a workspace shard in this cycle.

    The GWT trace records later participation without changing old percepts.
    Call after remap_pending, when entered items have durable record IDs.
    """
    created = changes.created_ids
    entered = [item.record_id for change in work.attention.changes for item in change.entered]
    caused = []
    for record_id in entered:
        if record_id in created:
            caused.append(record_id)
            continue
        if work.records[record_id].record_class != "CodeletPercept":
            continue
        # The GWT entry records this later participation. The percept stays immutable.
        caused.append(record_id)
    return [record_id for record_id in caused if work.records[record_id].record_class == "CodeletPercept"]


def remap_pending(work: CycleWork, mapping: dict[int, int]) -> None:
    """Replace pending percept and structure ids by record ids in the state and the traces.

    ``work.records`` keeps its pending keys and gets a second entry under
    each record id, so later readers can look up both.
    """
    work.state.csm.rekey(mapping)
    work.state.codelets.rekey(mapping)
    work.state.workspace.rekey(mapping)
    work.state.structure_ids = [mapping.get(key, key) for key in work.state.structure_ids]
    for boost in work.codelets.pathway_boosts:
        boost.precursor_ids = [mapping.get(key, key) for key in boost.precursor_ids]
    for change in work.attention.changes:
        change.rekey(mapping)
    for shard in work.attention.ratings:
        for rated in shard.ratings:
            rated.structure_id = mapping.get(rated.structure_id, rated.structure_id)
    for pending_id, record_id in mapping.items():
        if pending_id in work.records:
            work.records[record_id] = work.records[pending_id].model_copy(update={"record_id": record_id})
    work.experience.rekey(mapping)


def stage_expectations(changes, view, work: CycleWork, dispatch: ActionDispatch, action_id: int) -> list[int]:
    """Stage the expected reactions of one published action. Return the new record ids.

    Port of the record part of ``_generate_expectations_for_action``. Each
    record links the action and the group's stimuli. An open expectation
    that a new one supersedes gets the status ``superseded``, with the new
    record as its observation, unless the check of this cycle already
    resolved it. The cycle read its revision; the commit rejects a
    concurrent change.
    """
    created = []
    by_id = {item.record_id: item for item in work.request.open_expectations}
    checked = {item.record_id: item for item in work.expectations}
    for draft in dispatch.expectations:
        record = changes.create(ExpectationRecord(
            content=draft.content, anticipation=draft.content, expected_outcome=draft.content,
            status="pending", expected_valence=draft.expected_valence, urgency=draft.urgency,
            about_entity_ids=list(draft.about_entity_ids), created_from_action_id=action_id,
            evidence_ids=list(dispatch.stimulus_ids),
            provenance=Provenance(source_kind=SourceKind.Self, source_ref=EXPECTATION_SOURCE),
            metadata={"decision": dispatch.content, "action_id": dispatch.action_id}))
        created.append(record.id)
        for old_id in draft.supersedes_ids:
            if old_id in checked and checked[old_id].status != "pending":
                continue
            old = view.get(old_id).model_copy(deep=False)
            if old_id in checked:
                old.fulfillment = checked[old_id].fulfillment
            old.status = "superseded"
            old.observation_ids = [record.id]
            old.resolved_tick_id = changes.tick_id
            old.metadata = {**old.metadata, "resolution_reason": "a new expectation replaced it",
                            "resolved_by": EXPECTATION_SOURCE}
            changes.expect_revision(old_id, by_id[old_id].revision)
            changes.replace(old)
    return created


def apply_expectation_outcomes(changes, view, work: CycleWork) -> None:
    """Stage the check results of the appraise phase on the checked expectations.

    Each checked record gets its new fulfillment. A supported or
    contradicted record also gets its status, its observations (the stimuli
    of the check), and the resolution tick. The metadata keeps the impacts of
    the check. An outcome with no change and no new status changes nothing.
    """
    for outcome in work.expectations:
        if outcome.status == "pending" and outcome.change == 0.0:
            continue
        record = view.get(outcome.record_id).model_copy(deep=False)
        record.fulfillment = outcome.fulfillment
        if outcome.status != "pending":
            record.status = outcome.status
            record.observation_ids = list(outcome.observation_ids)
            record.resolved_tick_id = changes.tick_id
        record.metadata = {**record.metadata, "last_check": outcome.model_dump(mode="json"),
                           "last_check_tick": changes.tick_id, "resolved_by": EXPECTATION_SOURCE}
        changes.replace(record)


__all__ = ["InnerThoughtLoop", "MainCogLoop", "apply_expectation_outcomes", "build_main_cognition_tools",
           "dialogue_partner",
           "mark_causal", "remap_pending", "seconds_since_user_input", "stage_expectations", "stage_percepts",
           "stage_structures"]
