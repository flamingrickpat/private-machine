"""The private thought loop host binding.

The host freezes a permitted view and derives cheap private opportunities on
each timer tick. The parent runs nomination, separate shared attention, and
one connected inquiry operation. Empty nomination results and attention
decisions persist without a GWT turn. An executed operation first commits its
graph and episode state as append-only recovery roots. A second, repairable
commit projects that accepted operation into the inner GWT. Only an outcome
finding publishes ``thought.created`` for main appraisal.

Both loops share one MemoryStore and one companion identity. Every cross-loop
read carries a revision, and the commit rejects a proposal whose source
changed. The inner loop never writes the main loop's workspace and never changes
mental state directly: only the main appraisal path does that.

Reading map: schemas and request_views define named input and persisted
contracts; nominations retains generated batches; inquiry owns graph staging;
graph_host binds durable acceptance and derived GWT repair; graph_execution
coordinates the existing child kernels; graph_outcomes retains integration.
Existing linear episodes finish through the explicit legacy pipeline without
invented graph judgments or active computation time.

Ordinary input does not supersede private work. Accepted Critical
reconsideration does: the next opportunity closes every inquiry from the older
epoch as ``displaced`` and frees the slot. The closed question is offered to
shared attention and can start a new episode under the current grant; inquiry
owns that staging and graph_execution the offer. When several closed questions
wait, offer_salience chooses the one offered from what the private mind holds
now: shared original sources, embedding similarity, and seeded chance. That
choice is retained on the attention outcome. Deliberate mental work still
awaits its wiring to this slot.
"""
from __future__ import annotations

from pm import vclock

from pm.records.copying import pickle_copy
from pm.records.base import RecordBase

from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
import json
from pathlib import Path
from threading import Event
import traceback
from typing import TYPE_CHECKING, Any, NamedTuple
from pydantic import JsonValue

from workflows.cognitive_control.schemas import ControlConfig

from pm.bus.envelopes import Envelope, new_envelope, new_id
from pm.memory.bash_view import MemoryBashEnvironment
from pm.memory.snapshot import GhostCancelled, GhostConflict, Snapshot
from pm.records.enums import EntityClass, FeatureType, InterlocusType, SourceKind
from pm.records.cognitive_records import ExpectationRecord, Intention
from pm.records.cycle_state import CycleState
from pm.records.global_workspace import GwtTurnRecord
from pm.records.lineage import RecordLineage
from pm.records.mental_state import FullMentalState
from pm.records.memory_records import Entity, Feature, new_feature
from pm.records.qualia_records import QualeRecord
from pm.runs.workflow_module import CognitiveWorkflow

from workflows.main_cognition.host import seconds_since_user_input
from workflows.main_cognition.model_json import ModelFormatError
from pm.llm.concurrency import InferenceAdmission, background_inference
from pm.llm.structured import context_measure
from workflows.main_cognition.cycle_commits import active_intentions
from workflows.inner_thought.expectations import open_expectation, pending_expectations, resolve_expectation
from workflows.inner_thought.felt_state import (
    is_reportable, notable_channels, previous_cycle_sample, project_channels,
)
from workflows.inner_thought.operations import operation_lines
from workflows.inner_thought.pipeline import ModelProposalError
from workflows.inner_thought.schemas import (
    InnerReport, InnerRequest, LegacyEpisodeContext, LegacyEpisodeState, LegacyThoughtMetadata, ThoughtCandidate,
)
from workflows.inner_thought.request_views import (
    CapabilityInspection, ExpectationView, GoalView, MainFrameView, OperationSuggestion,
    PrivateInspection, RecentThoughtView,
)
from workflows.inner_thought.stage_runner import build_inner_stage_runner
from workflows.inner_thought.triggers import evaluate_triggers
from workflows.inner_thought.graph_execution import PrivateGraphResult, PrivateRecall, StageRunner, run_graph_opportunity
from pm.memory.dependencies import capture_source
from workflows.public_speech.public_context import EvidenceItem
from workflows.inner_thought.graph_host import PrivateInquiryOwner, pending_findings
from workflows.inner_thought.inquiry import INQUIRY_EPISODE_KEY, inquiry_history
from workflows.inner_thought.nominations import capture_inquiry_grant
from workflows.inner_thought.pipeline import citable_ids
from pm.memory.read_scope import ReadScope
from pm.memory.write_batch import WriteBatch
from workflows.reflection.schemas import ReflectionContext
from workflows.cognitive_control.attention import captured_private_focus

if TYPE_CHECKING:
    from pm.llm.prompt_encoding import TurnInput
    from pm.memory.embeddings import EmbeddingService
    from pm.memory.store import MemoryStore
    from pm.introspection.capabilities import CapabilityOverlay
    from workflows.cognitive_control.host import CognitiveControl

#: How many recent private thoughts feed the next cycle's context.
RECENT_THOUGHT_LIMIT = 8


class OpenLegacyEpisode(NamedTuple):
    """The last retained linear state and its actual carrier record."""

    state: LegacyEpisodeState | None
    thought: Feature | None


class ResolvedLegacyEpisode(NamedTuple):
    """A typed legacy context and whether acknowledgment blocks continuation."""

    context: LegacyEpisodeContext
    blocked: bool


def build_inner_thought_tools(
    ghost: MemoryStore,
    runtime_root: str | Path,
    llm_complete: Callable[..., str],
    *,
    cancel_provider: Callable[[], Event | None] | None = None,
    embeddings: EmbeddingService | None = None,
    yield_provider: Callable[[], bool] | None = None,
    inquiry_provider: Callable[[InnerRequest, StageRunner], PrivateGraphResult],
) -> Callable[[MemoryBashEnvironment, object], dict[str, Callable[..., object]]]:
    """Bind private stages and bounded recall to the invocation's frozen read grant.

    ``llm_complete`` is the stage model's synchronous completion. Every stage
    call checks ``yield_provider`` before and after inference, so a cooperative
    yield stops the cycle instead of keeping the model slot. ``cancel_provider``
    supplies the run's cancellation event; it is None until a run starts.

    ``inquiry_provider`` is required because the ``inner_thought`` manifest
    grants ``run_private_inquiry`` to its ``work`` role unconditionally. The
    kernel refuses a role whose declared tool is absent from the run grant, so
    an unwired loop must fail here at assembly rather than at the first private
    cycle. A private loop without inquiry authority is not a supported shape.

    The returned tools read only records inside the environment's frozen view.
    ``retrieve_memory`` raises ValueError for an empty query; a failed
    retrieval is an error, not an empty result.
    """
    from pathlib import Path

    def factory(environment: MemoryBashEnvironment, batch: object) -> dict[str, Callable[..., object]]:
        """Return the tools of one private invocation over its frozen view."""
        import asyncio

        view = environment.ghost_view
        assert view is not None
        scopes = {(raw["data"]["scope"]["reality_id"], raw["data"]["scope"]["frame"])
                  for raw in view.serialize_knoxels().values()}
        cancel = cancel_provider() if cancel_provider is not None else None

        def complete(messages: Sequence[TurnInput], **generation: object) -> str:
            """Complete one private stage unless the cycle has been yielded."""

            def check_yield() -> None:
                """Raise cancellation when a yield was requested mid-stage."""
                if yield_provider is not None and yield_provider():
                    if cancel is not None:
                        cancel.set()
                    raise asyncio.CancelledError

            check_yield()
            result = llm_complete(messages, **generation)
            check_yield()
            return result

        complete.measure_context = context_measure(llm_complete)

        def retrieve_memory(query: str) -> dict[str, object]:
            """Search the frozen view for experiences, with their source evidence."""
            if not query.strip():
                raise ValueError("Private recall requires a nonempty query.")
            result = asyncio.run(ghost.retrieve_experiences(
                query, embeddings, view=view, scopes=scopes, limit=8, cancel=cancel))
            from workflows.reflection.experiments import source_evidence
            ids = {key for item in result["matches"] for key in [item["knoxel_id"], *item.get("source_ids", [])]}
            result["reflection_sources"] = [source_evidence(view, view.get(key)).model_dump(mode="json") for key in sorted(ids)
                                            if key in view.revisions]
            return result

        runner = build_inner_stage_runner(ghost, complete, Path(runtime_root), read_environment=environment, cancel=cancel)
        def run_private_inquiry(payload: dict[str, object]) -> dict[str, object]:
            """Run the parent's typed graph path under its existing private owner."""
            return inquiry_provider(InnerRequest.model_validate(payload), runner).model_dump(mode="json")

        return {"run_stage": runner, "retrieve_memory": retrieve_memory,
                "run_private_inquiry": run_private_inquiry}

    return factory


class PrivateThoughtLoop(CognitiveWorkflow):
    """The second conscious loop: scheduled private thought with its own shard."""

    WORKFLOW_NAME = "inner_thought"
    ghost: MemoryStore
    inference_admission: InferenceAdmission | None = None
    cognitive_control: CognitiveControl | None = None
    inspector_provider: Callable[[], CapabilityOverlay] | None = None
    _overlay: CapabilityOverlay | None = None
    _control_generation: int
    # The loop is timer-driven. It never reacts to bus traffic, so a main
    # cycle cannot cascade into an inner cycle into another main cycle.
    DATA_TOPICS = frozenset()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Forward the module configuration and add the loop's own state."""
        super().__init__(*args, **kwargs)
        # Thoughts already put on the bus, so a resume never publishes twice.
        self._published_thoughts: set[int] = set()
        self.yield_requested = Event()
        self._episode_state: LegacyEpisodeState | None = None
        self._episode_thought_id: int | None = None
        self._read_revisions: dict[int, int] = {}
        self._active_goal_ids: list[int] = []

    def run_inquiry(self, request: InnerRequest, run_stage: StageRunner) -> PrivateGraphResult:
        """Bind the actual private invocation to nomination, attention, and graph staging."""
        control = self.cognitive_control
        with self.ghost.lock:
            if control is not None and (control.waiting() or control.reconsideration_required()):
                return PrivateGraphResult(summary="Private work waits for accepted input classification or reconsideration.")
            environment = self.last_environment
            if not isinstance(environment, MemoryBashEnvironment):
                raise TypeError("A private inquiry requires its actual frozen memory environment.")
            view = environment.ghost_view
            assert view is not None
            identities = [record for record in view.all_knoxels.values()
                          if isinstance(record, Entity) and record.entity_class == EntityClass.Self
                          and record.content == request.companion_name]
            if len(identities) != 1:
                raise ValueError("A private inquiry requires one exact companion identity anchor.")
            observer = identities[0]
            permitted = {record_id for record_id in citable_ids(request) if view.get(record_id).scope == observer.scope}
            # The request and its grant describe the same observer reality.
            # Foreign frame contents cannot remain in the model's evidence list.
            request = request.model_copy(update={
                "candidates": [item for item in request.candidates if item.knoxel_id in permitted],
                "active_goal_records": [item for item in request.active_goal_records if item.knoxel_id in permitted],
                "pending_expectations": [item for item in request.pending_expectations if item.knoxel_id in permitted],
                "recent_thoughts": [item for item in request.recent_thoughts if item.knoxel_id in permitted],
                "main_frame": request.main_frame if request.main_frame.frame_id in permitted else MainFrameView(),
                "reflection": ReflectionContext(),
                "inspection": None,
                "context_limits": (*request.context_limits,
                    "Legacy experiment and inspection summaries are excluded from this graph grant; their source scope is not captured here."),
            })
            if request.trigger_candidates:
                request.trigger_candidates = evaluate_triggers(request)
            focus = captured_private_focus(view, now_utc=vclock.utcnow(), maximum_strength=self._episode_limits().maximum_focus_bias)
            governing_ids = [item.knoxel_id for item in [*request.active_goal_records, *request.pending_expectations]]
            if focus.source_id is not None:
                governing_ids.append(focus.source_id)
            grant = capture_inquiry_grant(view, observer_entity_id=observer.id, scope=observer.scope,
                source_ids=sorted(permitted), governing_ids=governing_ids,
                critical_epoch=control.publication_epoch() if control is not None else 0)
            generation = control.generation if control is not None else 0

        def commit_private(changes: WriteBatch) -> None:
            """Retain explicit graph dependencies instead of ambient invocation over-collection."""
            with ReadScope() as reads:
                reads.collecting = False
                if control is None:
                    self.ghost.commit(changes)
                else:
                    control.commit_private(changes, generation)

        owner = PrivateInquiryOwner(self.ghost, commit=commit_private, cancel=self.run_stop, limits=self._episode_limits())
        def recall_memory(query: str) -> PrivateRecall:
            """Search the actual invocation view and retain its original source meanings."""
            import asyncio
            result = asyncio.run(self.ghost.retrieve_experiences(query, self.ghost.embeddings,
                view=view, scopes={(observer.scope.reality_id, observer.scope.frame.value)},
                limit=8, cancel=self.run_stop))
            matches = tuple(EvidenceItem.model_validate(item) for item in result["matches"])
            source_ids = {key for item in matches for key in [item.knoxel_id, *item.source_ids]}
            # The retrieval trace is a receipt, so it crosses this boundary as
            # JSON. Retrieval reports scope pairs as tuples, which the typed
            # contract does not accept and the prompt never needs.
            trace = json.loads(json.dumps(result["trace"]))
            return PrivateRecall(matches=matches, trace=trace, sources=tuple(
                capture_source(view, key, current_head=False) for key in sorted(source_ids)))
        embedding_service = self.ghost.embeddings

        def embed_texts(texts: Sequence[str]) -> list[list[float]]:
            """Embed question and cue texts through the run's one embedding endpoint."""
            import asyncio
            assert embedding_service is not None  # Passed on only when the endpoint exists.
            return asyncio.run(embedding_service.embed_many(list(texts), cancel=self.run_stop))

        try:
            return run_graph_opportunity(request, owner=owner, grant=grant, run_stage=run_stage,
                mental_state=self._mental_state, control_generation=generation,
                pending_input_ids=tuple(sorted(record.id for record in view.all_knoxels.values()
                    if isinstance(record, Feature) and record.metadata.get("exchange_status") == "pending")), focus=focus,
                recall_memory=recall_memory,
                embed_texts=embed_texts if embedding_service is not None else None)
        except GhostCancelled:
            return PrivateGraphResult(summary="Private graph work was held or cancelled before acceptance.")
        except GhostConflict as error:
            return PrivateGraphResult(summary=f"Private graph direction requires fresh admission: {error}")

    def yield_run(self) -> None:
        """Finish active inference before yielding. Cancel queued work now.

        The generation fence still rejects stale private proposals. Shutdown
        continues to use cancel_run and interrupts in-flight I/O immediately.
        """
        self.yield_requested.set()
        if self.inference_admission is None or not self.inference_admission.background_active:
            self.cancel_run()

    def resume(self) -> None:
        """Replay unappraised accepted thoughts after interrupted delivery.

        Main cycle source ids acknowledge thoughts. A wait delays replay, and
        a newer control generation discards superseded work without inference.
        """
        control = self.cognitive_control
        with self.ghost.lock:
            if control is not None and (control.waiting() or control.reconsideration_required()):
                return
            generation = control.generation if control is not None else 0
            view = self.ghost.snapshot(capture=False)
            episodes = {str(record.metadata[INQUIRY_EPISODE_KEY]) for record in view.all_knoxels.values()
                        if INQUIRY_EPISODE_KEY in record.metadata}

            def commit_projection(changes: WriteBatch) -> None:
                """Rebuild accepted inner views under the current scheduling hold."""
                with ReadScope() as reads:
                    reads.collecting = False
                    if control is None:
                        self.ghost.commit(changes)
                    else:
                        control.commit_private(changes, generation)

            owner = PrivateInquiryOwner(self.ghost, commit=commit_projection, cancel=self.run_stop, limits=self._episode_limits())
            for episode_id in sorted(episodes):
                for accepted in inquiry_history(view, episode_id):
                    if accepted.receipt.operation_record_id is not None:
                        owner.ensure_inner_turn(accepted)
            appraised = {key for cycle in self.ghost.read_cycles() for key in cycle.claimed_ids()}
            pending = [item for item in self._inner_thoughts()
                       if item.id not in appraised and item.id not in self._published_thoughts
                       and (control is None or item.metadata.get("control_generation", control.generation)
                            == control.generation)]
            graph_findings = pending_findings(self.ghost)
        for thought in pending:
            # Main can release an event during delivery. Mark it first so
            # that the release permits another attempt after a discarded cycle.
            self._published_thoughts.add(thought.id)
            self.bus.put(new_envelope(self.module_id, "data", "thought.created", "",
                                      {"knoxel_id": thought.id, "gwt_tick": thought.tick_id}))
        for finding in graph_findings:
            self._publish_graph_finding(finding.id)

    def release_results(self, envelopes: list[Envelope]) -> None:
        """Release queued thoughts after main acceptance or a discarded cycle."""
        for envelope in envelopes:
            if envelope.topic == "thought.created":
                self._published_thoughts.discard(envelope.payload.get("knoxel_id"))

    def handle_batch(self, batch: list[Envelope]) -> None:
        """Run one cycle and discard an explicitly rejected model proposal.

        The rejected proposal changes no committed private state. A rejection
        retains its diagnostic beside the parent kernel artifacts and emits
        telemetry only. It is not an idle cycle. Other errors propagate.
        """
        self.yield_requested.clear()
        control = self.cognitive_control
        if control is not None:
            with self.ghost.lock:
                if control.waiting():
                    return
                self._control_generation = control.generation
        self._end_expired_episode()
        self.resume()
        try:
            with background_inference():
                super().handle_batch(batch)
        except (ModelProposalError, ModelFormatError) as error:
            diagnostic = {
                "invocation_id": self.invocation_id,
                "error_type": type(error).__name__,
                "reason": str(error),
            }
            invocation_id = self.invocation_id
            assert invocation_id is not None
            path = self.runtime_root / "kernel" / invocation_id / "model-proposal-rejection.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({**diagnostic, "traceback": traceback.format_exc()}, indent=2),
                            encoding="utf-8")
            self.bus.put(new_envelope(self.module_id, "telemetry", "thought.rejected", "",
                                      {**diagnostic, "diagnostic_path": str(path)}))

    def prepare_input(self, batch: list[Envelope]) -> dict[str, Any]:
        """Freeze one private request from committed state.

        The returned mapping is the serialized prompt payload; the typed
        ``InnerRequest`` exists in this method. Every citable record carries the
        revision the loop read, so the commit can reject a proposal built on
        state the main loop already replaced.
        """
        view = getattr(self.last_environment, "ghost_view", None) or self.ghost.snapshot()
        with self.ghost.lock:
            tick = self.ghost.current_tick_id
            states = list(self.ghost.read_cycles())
            main_turns = self.ghost.gwt_turns_of("main", limit=3)
            thoughts = self._recent_private_thoughts()
            goals = active_intentions(view)
            expectations = pending_expectations(view)
            latest_appraisal = self._latest_appraisal()
            startup_gap = self._startup_gap_seconds()
            latest_user = self._latest_user_dialogue()
            private_thought_ids = self._private_thought_ids()
            open_episode, latest_thought = self._open_episode()
            now = vclock.utcnow()

        previous_cycle = states[-1] if states else None
        # The private cycle keeps the accepted EMA state of the main cycle
        # unchanged. A typed copy keeps the personality and the expression
        # readouts, which a vector round trip would lose. Before the first
        # cycle the state is the EMA readout of an empty history: the defaults.
        mental_state = (pickle_copy(previous_cycle.mental_state)
                        if previous_cycle is not None else FullMentalState())
        age_seconds = max(0.0, (vclock.utcnow() - previous_cycle.timestamp).total_seconds()) \
            if previous_cycle is not None else 0.0

        main_frame = MainFrameView()
        frame_age = 0
        if main_turns:
            newest = main_turns[0]
            surprise = 0.0
            if latest_appraisal is not None:
                surprise = float((latest_appraisal.metadata or {}).get("emotion_delta", {}).get(
                    "surprise", 0.0) or 0.0)
            # A frame whose only stimuli were the loop's own thoughts is
            # self-feedback. The trigger policy must not treat it as fresh
            # outside material and start another cycle from it.
            frame_cycle = next((item for item in reversed(states)
                                if item.tick_id == newest.tick_id), None)
            claimed = frame_cycle.claimed_ids() if frame_cycle is not None else []
            self_feedback = bool(claimed and all(int(key) in private_thought_ids for key in claimed))
            # The subjective experience of the newest main turn stands for
            # the main workspace. Before A19F the attention frame did.
            main_frame = MainFrameView(tick=newest.tick_id, frame_id=newest.experience_id,
                summary=newest.summary, surprise=surprise,
                gap_seconds=startup_gap, self_feedback=self_feedback)
            frame_age = max(0, tick - newest.tick_id)

        goal_records = [
            GoalView(knoxel_id=goal.id, text=goal.content,
                tick_id=goal.tick_id, fulfilment=goal.fulfilment,
                affective_valence=goal.affective_valence, revision=view.revisions[goal.id])
            for goal in goals
        ]
        expectation_records = [
            ExpectationView(knoxel_id=item.id, text=item.content,
                anticipation=item.anticipation, expected_outcome=item.expected_outcome,
                tick_id=item.tick_id, revision=view.revisions[item.id])
            for item in expectations
        ]
        request = InnerRequest(
            run_id=self.invocation_id or "",
            companion_name=self.ghost.get_companion_name(),
            character=self.ghost.get_simple_character_story_block(),
            user_name=self.ghost.get_user_name(),
            state_summary=self._state_summary(previous_cycle),
            mental_state=previous_cycle.mental_state if previous_cycle is not None else None,
            mental_state_age_seconds=age_seconds,
            seconds_since_user_input=seconds_since_user_input(self.ghost, []),
            tick_id=tick,
            main_frame=main_frame,
            main_frame_age_ticks=frame_age,
            active_goal_records=goal_records,
            pending_expectations=expectation_records,
            recent_thoughts=thoughts,
            candidates=self._candidates(view, goals, expectations, latest_appraisal,
                                        main_frame, thoughts, latest_user),
            operations=[OperationSuggestion(name=line) for line in operation_lines()],
        )
        from workflows.reflection.experiments import context as reflection_context, source_evidence
        request.reflection = reflection_context(view)
        reflection_sources = {item.knoxel_id for item in request.reflection.records}
        for candidate in request.candidates:
            if candidate.knoxel_id not in reflection_sources:
                request.reflection.records.append(source_evidence(view, view.get(candidate.knoxel_id)))
        self._reflection_context = request.reflection
        # A bounded episode carries its own identity and history. A continuing
        # step waits for main appraisal of the previous thought before it reads
        # new accepted state. Step, time, and stall limits close it explicitly.
        episode_context, blocked = self._resolve_episode(open_episode, latest_thought, latest_user, now)
        # Closed episodes still own their parking interval. The next fresh
        # request must retain that interval even without an open episode.
        suppressed = sorted(self._parked_sources(now, latest_user))
        request.episode = episode_context.model_copy(update={"suppressed_source_ids": tuple(suppressed)})
        self._episode_state = episode_context.state
        # Only a continuing step links to a parent thought. A fresh episode has
        # no parent, even when an older closed thought is the newest record.
        self._episode_thought_id = (int(latest_thought.id)
                                    if episode_context.state is not None and latest_thought is not None else None)
        if blocked:
            request.trigger_candidates = []
        else:
            request.trigger_candidates = evaluate_triggers(request)
            request.inspection = self._build_inspection()
            # Re-evaluate so a fresh capability doubt can fire with the evidence.
            if request.inspection:
                request.trigger_candidates = evaluate_triggers(request)
        # The felt channels of the last accepted cycle give the model context.
        # The new sample is computed at commit time from this frozen view.
        channels, _ = previous_cycle_sample(states)
        request.channels = dict(channels) if channels is not None else {}
        self._read_revisions = {
            item.knoxel_id: item.revision
            for item in [*goal_records, *expectation_records, *thoughts]
        }
        for candidate in request.candidates:
            self._read_revisions[int(candidate.knoxel_id)] = int(candidate.revision)
        self._mental_state = mental_state
        self._active_goal_ids = [item.knoxel_id for item in goal_records]
        return request.model_dump(mode="json")

    def publish_output(self, report: dict[str, JsonValue], batch: list[Envelope]) -> None:
        """Commit accepted private work, or record an idle or rejected cycle.

        An idle cycle (``operation == "none"``) commits nothing: the loop does
        not generate thoughts to satisfy a quota. A stale proposal is rejected
        with a diagnostic envelope, never committed on top of newer state. A
        cancellation discards the cycle silently; the store guarantees no
        late commit after ``stop_event`` is set.
        """
        if self.stop_event.is_set() or self.run_stop.is_set():
            return
        if report.get("stage_kind") == "private_inquiry_result":
            result = PrivateGraphResult.model_validate(report)
            if result.finding_id is not None:
                self._publish_graph_finding(result.finding_id)
            self.bus.put(new_envelope(self.module_id, "telemetry", "thought.graph", "",
                                      result.model_dump(mode="json")))
            return
        parsed = InnerReport.model_validate(report)
        if parsed.operation == "none":
            self.bus.put(new_envelope(self.module_id, "telemetry", "thought.idle", "",
                                      {"summary": parsed.summary}))
            return

        try:
            self._commit_cycle(parsed)
        except GhostCancelled:
            return
        except GhostConflict as error:
            self.bus.put(new_envelope(self.module_id, "telemetry", "thought.rejected", "",
                                      {"reason": f"{type(error).__name__}: {error}",
                                       "operation": parsed.operation}))
            return

    def _publish_graph_finding(self, record_id: int) -> None:
        """Deliver one accepted outcome identity; accepted internal branches stay private."""
        if record_id in self._published_thoughts:
            return
        self._published_thoughts.add(record_id)
        turns = [turn for turn in self.ghost.gwt_turns.values()
                 if turn.slot_id == "inner" and record_id in turn.knoxel_ids]
        if len(turns) != 1:
            raise ValueError("An accepted private finding requires one actual inner GWT projection.")
        self.bus.put(new_envelope(self.module_id, "data", "thought.created", "",
                                  {"knoxel_id": record_id, "gwt_tick": turns[0].tick_id}))

    def _commit_cycle(self, parsed: InnerReport) -> None:
        """Commit one accepted private cycle into the inner shard."""
        ghost = self.ghost
        view = getattr(self.last_environment, "ghost_view", None) or ghost.snapshot()
        changes = ghost.begin_batch(self.invocation_id or new_id(), new_tick=True, cancel=self.run_stop)
        tick = changes.tick_id
        for record_id, revision in self._read_revisions.items():
            if record_id in view.all_knoxels:
                changes.expect_head(record_id, revision)
            else:
                changes.expect_revision(record_id, revision)
        for memory in parsed.retrieved_memories:
            key = memory.knoxel_id
            if key not in view.revisions or memory.revision != view.revisions[key]:
                raise GhostConflict("Private recall is outside the frozen source revisions.")
            for source_id in {key, *memory.source_ids}:
                if source_id not in view.revisions:
                    raise GhostConflict("Private recall includes a source outside the invocation grant.")
                changes.expect_revision(source_id, view.revisions[source_id])

        experiment_record = None
        if parsed.reflection.action != "none":
            from workflows.reflection.experiments import source_evidence, stage_decision
            payload = self._reflection_context.model_copy(deep=True)
            known = {item.knoxel_id for item in payload.records}
            for item in parsed.retrieved_memories:
                for key in {item.knoxel_id, *item.source_ids} - known:
                    payload.records.append(source_evidence(view, view.get(key)))
                    known.add(key)
            try:
                experiment_record = stage_decision(changes, view, parsed.reflection, payload,
                    companion_name=ghost.get_companion_name(),
                    maximum_focus_bias=self._episode_limits().maximum_focus_bias)
            except ValueError as error:
                raise ModelProposalError(str(error)) from error

        episode_metadata = self._next_episode(parsed)
        thought = new_feature(
            ghost, parsed.thought, FeatureType.InnerMonologue,
            source=ghost.get_companion_name(), interlocus=InterlocusType.PrivateInternal,
            changes=changes, source_kind=SourceKind.Self,
            lineage=RecordLineage(
                episode_id=episode_metadata.episode_id, operation=parsed.operation,
                sequence=episode_metadata.sequence,
                parent_record_ids=[self._episode_thought_id] if self._episode_thought_id is not None else [],
                source_event_ids=list(parsed.evidence_ids), intention_ids=self._active_goal_ids,
                topic_hints=list(parsed.topic_hints), reality=parsed.reality),
            metadata={
                "loop": "inner", "operation": parsed.operation, "trigger": parsed.trigger,
                "reflection": parsed.reflection.model_dump(mode="json"),
                "reflection_experiment_id": experiment_record.id if experiment_record is not None else None,
                "episode": episode_metadata.model_dump(mode="json"),
                **({"control_generation": self._control_generation}
                   if self.cognitive_control is not None else {}),
                "trigger_reason": parsed.trigger_reason, "temporal_focus": parsed.temporal_focus,
                "uncertainty": parsed.uncertainty, "reality": parsed.reality,
                "evidence_ids": list(parsed.evidence_ids),
                "evidence_summary": parsed.evidence_summary,
                "interpretation": parsed.interpretation,
                "alternatives": [item.model_dump(mode="json") for item in parsed.alternatives],
                "continuation": parsed.continuation, "continuation_reason": parsed.continuation_reason,
                "focus_intention": parsed.noticing,
                "disclosure": parsed.disclosure.model_dump(mode="json"),
                "summary": parsed.summary,
                "mental_plan": {
                    "instruction": parsed.noticing or parsed.trigger_reason,
                    "summary": parsed.summary or parsed.thought,
                    "disposition": "propose",
                    "evidence_ids": list(parsed.evidence_ids),
                    "goal_changes": [item.model_dump(mode="json") for item in parsed.goal_changes],
                    "focus": None,
                },
                "capability_states": self._capability_states(),
            },
        )

        # The thought retains proposed goals. Main appraisal and its goals
        # stage own acceptance, so private work cannot change active goals.

        # A capability inspection commits a revisable capability belief in the
        # same batch. The belief cites the inspected evidence revision, so a
        # later source change can mark it stale.
        if parsed.operation == "capability_inspection" and self._overlay is not None:
            from pm.introspection.self_model import record_capability_beliefs
            record_capability_beliefs(
                changes, self._overlay, companion_name=ghost.get_companion_name(),
                source_revision=self._overlay.built_at.isoformat(),
                stale_names=self._overlay.stale_names(),
            )

        if parsed.expectation.action == "open":
            open_expectation(
                changes,
                anticipation=parsed.expectation.anticipation,
                expected_outcome=parsed.expectation.expected_outcome,
                expected_valence=parsed.expectation.expected_valence,
                uncertainty=parsed.expectation.uncertainty,
                goal_ids=parsed.expectation.goal_ids,
                evidence_ids=[int(item) for item in parsed.evidence_ids] or [thought.id],
            )
        elif parsed.expectation.action == "resolve":
            if parsed.expectation.expectation_id is None:
                raise ValueError("A resolving expectation requires its accepted identity.")
            record = view.get_knoxel_by_id(parsed.expectation.expectation_id)
            if isinstance(record, ExpectationRecord):
                resolve_expectation(
                    changes, record,
                    status=parsed.expectation.status,
                    observation_ids=[int(item) for item in parsed.expectation.observation_ids],
                    reason=parsed.expectation.reason,
                    revision=int(view.revisions.get(record.id, 1)),
                )

        # The felt-state sample is deterministic math over the frozen accepted
        # state. Its drivers name the records that moved the state, and
        # reportability applies the causal gate: a driver that never won
        # attention and was never causal leaves the snapshot dormant.
        channels, previous_time = previous_cycle_sample(ghost.read_cycles())
        state = self._mental_state
        drivers = [int(item) for item in parsed.evidence_ids]
        sample = project_channels(state, active_goals=active_intentions(view),
                                  previous=channels, previous_time=previous_time,
                                  now=vclock.utcnow(), driver_ids=drivers)
        gate_ids = {int(item) for item in parsed.evidence_ids} | {thought.id}
        quale_ids: list[int] = []
        for channel_id in notable_channels(sample):
            channel = sample[channel_id]
            quale = changes.create(QualeRecord(
                content=f"{channel_id} moved to {channel.position:.3f} "
                        f"(velocity {channel.velocity:.4f} per second).",
                channel_id=channel_id,
                position=channel.position, velocity=channel.velocity,
                acceleration=channel.acceleration,
                driver_ids=list(channel.last_driver_ids),
                reportable=is_reportable(channel.last_driver_ids,
                                         attention_ids=gate_ids, causal_ids=gate_ids),
            ))
            quale_ids.append(quale.id)

        # A private cycle claims no stimuli and writes no attention frame.
        # A19G moves the inner loop onto the shared cycle and its workspace.
        cycle = CycleState(tick_id=tick, mental_state=state, qualia_channels=sample)
        changes.set_gwt(GwtTurnRecord(
            tick_id=tick, slot_id="inner",
            knoxel_ids=[thought.id, *quale_ids],
            summary=parsed.summary or parsed.thought[:120],
        ), cycle)
        control = self.cognitive_control
        if control is None:
            ghost.commit(changes)
        else:
            control.commit_private(changes, self._control_generation)

        # The self-origin event. The main loop appraises it like any internal
        # stimulus; nothing here publishes speech or commands the body.
        self._published_thoughts.add(thought.id)
        self.bus.put(new_envelope(self.module_id, "data", "thought.created", "",
                                  {"knoxel_id": thought.id, "gwt_tick": tick}))
        self.bus.put(new_envelope(self.module_id, "data", "gwt.updated", "",
                                  {"gwt_tick": tick, "slot": "inner"}))

    # ------------------------------------------------------------------
    # Deterministic gathering helpers. All read committed state under the
    # caller's lock.

    def _build_inspection(self) -> PrivateInspection | None:
        """Freeze selected implementation evidence or the provider's explicit limit.

        The provider owns the real source grant. Missing configuration supplies
        no inspection. Filesystem failures remain visible evidence limits;
        provider logic errors propagate.
        """
        self._overlay = None
        provider = self.inspector_provider
        if provider is None:
            return None
        try:
            overlay = provider()
        except OSError as error:
            return PrivateInspection(status="unavailable", reason=f"{type(error).__name__}: {error}")
        self._overlay = overlay
        from workflows.introspection.snapshot import overlay_public
        from pm.introspection.episodes import EpisodeIndex, answer_questions
        index = EpisodeIndex(self.ghost)
        recent = index.recent(limit=5)
        answers = answer_questions(self.ghost, recent[0], overlay=overlay) if recent else {}
        return PrivateInspection(status="ok",
            capability_overlay=CapabilityInspection.model_validate(overlay_public(overlay)),
            episodes=[item.to_dict() for item in recent], newest_episode_answers=answers,
            observed_at=overlay.built_at)

    def _capability_states(self) -> dict[str, str]:
        """Return the strongest inspected state per capability, if available."""
        overlay = self._overlay
        if overlay is None:
            return {}
        return {name: finding.strongest.value for name, finding in overlay.findings.items()}

    def _episode_limits(self) -> ControlConfig:
        """Return the operator episode limits, or the shipped defaults."""
        control = self.cognitive_control
        return control.config if control is not None else ControlConfig()

    def _inner_thoughts(self) -> list[Feature]:
        """Return the accepted private thoughts in the order the store holds them."""
        return [knoxel for knoxel in self.ghost.all_features
                if knoxel.feature_type == FeatureType.InnerMonologue
                and knoxel.provenance and knoxel.provenance.source_kind == SourceKind.Self
                and (knoxel.metadata or {}).get("loop") == "inner"]

    def _open_episode(self) -> OpenLegacyEpisode:
        """Read the latest retained linear episode without inventing graph state."""
        thoughts = self._inner_thoughts()
        if not thoughts:
            return OpenLegacyEpisode(None, None)
        latest = max(thoughts, key=lambda item: item.tick_id)
        metadata = LegacyThoughtMetadata.model_validate(latest.metadata)
        if metadata.episode is None or metadata.episode.closed:
            return OpenLegacyEpisode(None, latest)
        return OpenLegacyEpisode(metadata.episode, latest)

    def _thought_acknowledged(self, thought_id: int) -> bool:
        """Return whether an accepted main cycle appraised this thought."""
        return any(thought_id in cycle.claimed_ids() for cycle in self.ghost.read_cycles())

    def _parked_sources(self, now: datetime, latest_user: Feature | None) -> set[int]:
        """Read actual legacy parking intervals without inferring offline work.

        Existing linear behavior releases parking on newer outside input.
        The graph migration replaces that policy together with control guards.
        """
        parked: set[int] = set()
        for thought in self._inner_thoughts():
            metadata = LegacyThoughtMetadata.model_validate(thought.metadata)
            episode = metadata.episode
            if episode is None or episode.parked_until is None or now >= episode.parked_until:
                continue
            if latest_user is not None and latest_user.tick_id > thought.tick_id:
                continue
            control = self.cognitive_control
            if (control is not None and metadata.control_generation is not None
                    and metadata.control_generation != control.generation):
                continue
            parked.update(episode.parked_sources)
        return parked

    def _end_expired_episode(self) -> None:
        """Close expired linear work while preserving its original history."""
        with self.ghost.lock:
            episode, thought = self._open_episode()
            if episode is None:
                return
            if thought is None:
                raise RuntimeError("A retained linear episode requires its carrier thought.")
            control = self.cognitive_control
            metadata = LegacyThoughtMetadata.model_validate(thought.metadata)
            user = self._latest_user_dialogue()
            elapsed = max(0.0, (vclock.utcnow() - episode.started_at).total_seconds())
            reason: str | None = None
            if (control is not None and metadata.control_generation is not None
                    and metadata.control_generation != control.generation):
                reason = "superseded generation"
            elif user is not None and user.tick_id > thought.tick_id:
                reason = "new outside input"
            elif elapsed >= self._episode_limits().episode_seconds:
                reason = "time limit"
            if reason is None:
                return
            changes = self.ghost.begin_batch(f"episode-end:{thought.id}:{reason}", cancel=self.stop_event)
            closed = episode.model_copy(update={"closed": True, "closed_reason": reason, "elapsed_seconds": elapsed})
            changes.set_lifecycle(thought.id, episode=closed.model_dump(mode="json"))
            self.ghost.commit(changes)

    def _resolve_episode(self, open_episode: LegacyEpisodeState | None,
                         latest_thought: Feature | None, latest_user: Feature | None,
                         now: datetime) -> ResolvedLegacyEpisode:
        """Resolve legacy admission without reclassifying duration as graph work."""
        empty = ResolvedLegacyEpisode(LegacyEpisodeContext(), False)
        if open_episode is None or latest_thought is None:
            return empty
        elapsed = max(0.0, (now - open_episode.started_at).total_seconds())
        limits = self._episode_limits()
        control = self.cognitive_control
        generation = LegacyThoughtMetadata.model_validate(latest_thought.metadata).control_generation
        if control is not None and generation is not None and generation != control.generation:
            return empty
        if latest_user is not None and latest_user.tick_id > latest_thought.tick_id:
            return empty
        state = open_episode.model_copy(update={"elapsed_seconds": elapsed})
        if not self._thought_acknowledged(latest_thought.id):
            if elapsed < max(limits.episode_seconds, 120.0):
                return ResolvedLegacyEpisode(LegacyEpisodeContext(provenance="legacy_linear",
                    state=state, awaiting_acknowledgment=True), True)
            return empty
        return ResolvedLegacyEpisode(LegacyEpisodeContext(provenance="legacy_linear", state=state,
            steps_remaining=max(0, limits.episode_steps - open_episode.sequence - 1)), False)

    def _next_episode(self, parsed: InnerReport) -> LegacyEpisodeState:
        """Build the next retained linear state from named original fields.

        This behavior-preserving migration retains world-clock elapsed time,
        legacy acknowledgment, and parking. It supplies no graph judgments.
        """
        prior = self._episode_state
        now = vclock.utcnow()
        limits = self._episode_limits()
        step = prior.sequence + 1 if prior is not None else 0
        started_at = prior.started_at if prior is not None else now
        elapsed = max(0.0, (now - started_at).total_seconds())
        operations = (*(prior.operations if prior is not None else ()), parsed.operation)
        evidence = tuple(dict.fromkeys([*(prior.evidence_ids if prior is not None else ()), *parsed.evidence_ids]))
        alternatives = tuple(dict.fromkeys([*(prior.alternatives if prior is not None else ()),
            *(item.text for item in parsed.alternatives if item.text)]))
        thoughts = tuple(dict.fromkeys([*(prior.thoughts if prior is not None else ()), parsed.thought]))[-8:]
        progress = (*(prior.progress if prior is not None else ()), parsed.progress)
        stalled = (prior.stalled if prior is not None else 0) + (0 if parsed.progress != "none" else 1)
        closed_reason: str | None = None
        if parsed.continuation != "continue":
            closed_reason = parsed.continuation
        elif step + 1 >= limits.episode_steps:
            closed_reason = "step limit"
        elif elapsed >= limits.episode_seconds:
            closed_reason = "time limit"
        elif stalled >= limits.stalled_steps:
            closed_reason = "stalled progress"
        parked_until = now + timedelta(seconds=limits.parked_seconds) if closed_reason in {"suspend", "stalled progress"} else None
        return LegacyEpisodeState(episode_id=prior.episode_id if prior is not None else new_id(),
            sequence=step, started_at=started_at, elapsed_seconds=elapsed,
            closed=closed_reason is not None, closed_reason=closed_reason,
            operations=operations, evidence_ids=evidence, alternatives=alternatives,
            thoughts=thoughts, progress=progress, stalled=stalled, continuation=parsed.continuation,
            next_operation=parsed.next_operation, parked_until=parked_until,
            parked_sources=tuple(sorted(evidence)) if parked_until is not None else (),
            parent_thought_id=self._episode_thought_id)

    def _private_thought_ids(self) -> set[int]:
        """Return every committed private thought id, for self-feedback detection."""
        return {int(knoxel.id) for knoxel in self.ghost.all_features
                if knoxel.feature_type == FeatureType.InnerMonologue
                and (knoxel.metadata or {}).get("loop") == "inner"}

    def _recent_private_thoughts(self) -> list[RecentThoughtView]:
        """Return named retained thought views with their actual evidence IDs."""
        thoughts: list[RecentThoughtView] = []
        for knoxel in self.ghost.all_features:
            if knoxel.feature_type != FeatureType.InnerMonologue:
                continue
            if knoxel.provenance is None or knoxel.provenance.source_kind != SourceKind.Self:
                continue
            metadata = LegacyThoughtMetadata.model_validate(knoxel.metadata)
            thoughts.append(RecentThoughtView(knoxel_id=knoxel.id, text=knoxel.content,
                tick_id=knoxel.tick_id, operation=metadata.operation, trigger=metadata.trigger,
                continuation=metadata.continuation, source_ids=list(metadata.evidence_ids),
                revision=self.ghost.revisions[knoxel.id]))
        thoughts.sort(key=lambda item: item.tick_id, reverse=True)
        return thoughts[:RECENT_THOUGHT_LIMIT]

    def _latest_appraisal(self) -> Feature | None:
        """Return the newest committed appraisal record, or None before one exists."""
        latest: Feature | None = None
        for knoxel in self.ghost.all_features:
            if (knoxel.metadata or {}).get("origin") == "appraisal":
                if latest is None or knoxel.tick_id > latest.tick_id:
                    latest = knoxel
        return latest

    def _startup_gap_seconds(self) -> float:
        """Return the offline gap that the newest startup record measured, in seconds."""
        latest: Feature | None = None
        for knoxel in self.ghost.all_features:
            metadata = knoxel.metadata or {}
            if metadata.get("event") == "startup" and "gap_seconds" in metadata:
                if latest is None or knoxel.tick_id > latest.tick_id:
                    latest = knoxel
        return float((latest.metadata or {}).get("gap_seconds", 0.0) or 0.0) if latest else 0.0

    def _latest_user_dialogue(self) -> Feature | None:
        """Return the newest dialogue record the user spoke, or None."""
        # The user proxy names the user as the feature source; the provenance
        # of a user message carries no source kind. The source name is the
        # stable identity, exactly as the main loop's idle measurement uses.
        latest: Feature | None = None
        user = self.ghost.get_user_name()
        for knoxel in self.ghost.all_features:
            if knoxel.feature_type != FeatureType.Dialogue:
                continue
            if knoxel.source != user:
                continue
            if latest is None or knoxel.tick_id > latest.tick_id:
                latest = knoxel
        return latest

    def _state_summary(self, cycle: CycleState | None) -> str:
        """Return the one-line accepted-state summary the private prompt carries."""
        if cycle is None:
            return "(no accepted state yet)"
        turn = self.ghost.gwt_turns[cycle.tick_id] if cycle.tick_id in self.ghost.gwt_turns else None
        return f"tick {cycle.tick_id}, loop: {turn.slot_id if turn is not None else 'none'}"

    def _candidates(
        self,
        view: Snapshot,
        goals: Sequence[Intention],
        expectations: Sequence[ExpectationRecord],
        latest_appraisal: Feature | None,
        main_frame: MainFrameView,
        thoughts: Sequence[RecentThoughtView],
        latest_user: Feature | None,
    ) -> list[ThoughtCandidate]:
        """Derive the loop's private attention deterministically.

        The selection weights inner material above outside material: goals,
        expectations, and recent thoughts outrank the latest outside message.
        The model can narrow this list, never widen it.
        """
        candidates: list[ThoughtCandidate] = []
        revisions = view.revisions

        def add(record: RecordBase, kind: str, urgency: float, reason: str) -> None:
            """Offer one record as a private candidate when the frozen view holds it."""
            # A live record that is absent from the frozen view must not enter
            # the candidate list. A concurrent append cannot widen this grant.
            if int(record.id) not in revisions:
                return
            candidates.append(ThoughtCandidate(
                knoxel_id=int(record.id), kind=kind, text=str(record.content)[:400],
                revision=int(revisions.get(int(record.id), 1)),
                tick_id=int(record.tick_id), urgency=urgency, reason=reason,
            ))

        for goal in goals[:4]:
            add(goal, "goal", 0.6, "an active goal without recent progress")
        for item in expectations[:3]:
            add(item, "expectation", 0.55, "a recorded anticipation awaits its outcome")
        if latest_appraisal is not None:
            add(latest_appraisal, "appraisal", 0.4, "the newest accepted internal state change")
        frame_id = main_frame.frame_id
        if frame_id is not None:
            record = view.get_knoxel_by_id(int(frame_id))
            if record is not None:
                add(record, "frame", 0.35, "the main loop's latest accepted workspace")
        for thought in thoughts[:3]:
            record = view.get_knoxel_by_id(thought.knoxel_id)
            if record is not None:
                add(record, "thought", 0.3, "a recent private thought, for continuity")
        if latest_user is not None:
            add(latest_user, "dialogue", 0.25, "the latest outside message, as background")
        candidates.sort(key=lambda item: (-item.urgency, -item.tick_id))
        return candidates[:10]


__all__ = ["PrivateThoughtLoop", "build_inner_thought_tools"]
