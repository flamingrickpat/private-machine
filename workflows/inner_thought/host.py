"""The private thought loop host binding.

The host owns the loop's commit boundary. Each timer tick it freezes a
permitted view, derives the deterministic private candidates and triggers,
and runs the parent workflow. A cycle with no trigger performs nothing. An
accepted cycle commits its thought in an inner GWT turn and publishes one
``thought.created`` self-origin event, which the main loop appraises like any
other internal event.

Both loops share one MemoryStore and one companion identity. Every cross-loop
read carries a revision, and the commit rejects a proposal whose source
changed. The inner loop never writes the main loop's workspace and never changes
mental state directly: only the main appraisal path does that.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import json
from threading import Event
import traceback

from workflows.cognitive_control.schemas import ControlConfig

from pm.bus.envelopes import Envelope, new_envelope, new_id
from pm.memory.snapshot import GhostCancelled, GhostConflict
from pm.records.enums import FeatureType, InterlocusType, SourceKind
from pm.records.cognitive_records import ExpectationRecord
from pm.records.cycle_state import CycleState
from pm.records.global_workspace import GwtTurnRecord
from pm.records.lineage import RecordLineage
from pm.records.mental_state import FullMentalState
from pm.records.memory_records import new_feature
from pm.records.qualia_records import QualeRecord
from pm.runs.workflow_module import CognitiveWorkflow

from workflows.main_cognition.host import seconds_since_user_input
from workflows.main_cognition.model_json import ModelFormatError
from pm.llm.concurrency import background_inference
from pm.llm.structured import context_measure
from workflows.main_cognition.cycle_commits import active_intentions
from workflows.inner_thought.expectations import open_expectation, pending_expectations, resolve_expectation
from workflows.inner_thought.felt_state import (
    is_reportable, notable_channels, previous_cycle_sample, project_channels,
)
from workflows.inner_thought.operations import operation_lines
from workflows.inner_thought.pipeline import ModelProposalError
from workflows.inner_thought.schemas import InnerReport, InnerRequest, ThoughtCandidate
from workflows.inner_thought.stage_runner import build_inner_stage_runner
from workflows.inner_thought.triggers import evaluate_triggers

#: How many recent private thoughts feed the next cycle's context.
RECENT_THOUGHT_LIMIT = 8


def build_inner_thought_tools(ghost, runtime_root, llm_complete, *, cancel_provider=None, embeddings=None,
                              yield_provider=None):
    """Bind stages and bounded recall to the invocation's frozen read grant."""
    from pathlib import Path

    def factory(environment, batch):
        import asyncio

        view = environment.ghost_view
        scopes = {(raw["data"]["scope"]["reality_id"], raw["data"]["scope"]["frame"])
                  for raw in view.serialize_knoxels().values()}
        cancel = cancel_provider() if cancel_provider is not None else None

        def complete(messages, **generation):
            def check_yield():
                if yield_provider is not None and yield_provider():
                    if cancel is not None:
                        cancel.set()
                    raise asyncio.CancelledError

            check_yield()
            result = llm_complete(messages, **generation)
            check_yield()
            return result

        complete.measure_context = context_measure(llm_complete)

        def retrieve_memory(query: str) -> dict:
            if not query.strip():
                raise ValueError("Private recall requires a nonempty query.")
            result = asyncio.run(ghost.retrieve_experiences(
                query, embeddings, view=view, scopes=scopes, limit=8, cancel=cancel))
            from workflows.reflection.experiments import source_payload
            ids = {key for item in result["matches"] for key in [item["knoxel_id"], *item.get("source_ids", [])]}
            result["reflection_sources"] = [source_payload(view, view.get(key)) for key in sorted(ids)
                                            if key in view.revisions]
            return result

        return {"run_stage": build_inner_stage_runner(
            ghost, complete, Path(runtime_root), read_environment=environment, cancel=cancel),
            "retrieve_memory": retrieve_memory}

    return factory


class PrivateThoughtLoop(CognitiveWorkflow):
    """The second conscious loop: scheduled private thought with its own shard."""

    WORKFLOW_NAME = "inner_thought"
    inference_admission = None
    # The loop is timer-driven. It never reacts to bus traffic, so a main
    # cycle cannot cascade into an inner cycle into another main cycle.
    DATA_TOPICS = frozenset()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._published_thoughts: set[int] = set()
        self.yield_requested = Event()

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
        control = getattr(self, "cognitive_control", None)
        with self.ghost.lock:
            if control is not None and control.waiting():
                return
            appraised = {key for cycle in self.ghost.states for key in cycle.claimed_ids()}
            pending = [item for item in self._inner_thoughts()
                       if item.id not in appraised and item.id not in self._published_thoughts
                       and (control is None or item.metadata.get("control_generation", control.generation)
                            == control.generation)]
        for thought in pending:
            # Main can release an event during delivery. Mark it first so
            # that the release permits another attempt after a discarded cycle.
            self._published_thoughts.add(thought.id)
            self.bus.put(new_envelope(self.module_id, "data", "thought.created", "",
                                      {"knoxel_id": thought.id, "gwt_tick": thought.tick_id}))

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
        control = getattr(self, "cognitive_control", None)
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
            path = self.runtime_root / "kernel" / self.invocation_id / "model-proposal-rejection.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({**diagnostic, "traceback": traceback.format_exc()}, indent=2),
                            encoding="utf-8")
            self.bus.put(new_envelope(self.module_id, "telemetry", "thought.rejected", "",
                                      {**diagnostic, "diagnostic_path": str(path)}))

    def prepare_input(self, batch: list[Envelope]) -> dict:
        """Freeze one private request from committed state.

        Every citable record carries the revision the loop read, so the commit
        can reject a proposal built on state the main loop already replaced.
        """
        view = getattr(self.last_environment, "ghost_view", None) or self.ghost.snapshot()
        with self.ghost.lock:
            tick = self.ghost.current_tick_id
            states = list(self.ghost.states)
            main_turns = self.ghost.gwt_turns_of("main", limit=3)
            thoughts = self._recent_private_thoughts()
            goals = active_intentions(view)
            expectations = pending_expectations(view)
            latest_appraisal = self._latest_appraisal()
            startup_gap = self._startup_gap_seconds()
            latest_user = self._latest_user_dialogue()
            private_thought_ids = self._private_thought_ids()
            open_episode, latest_thought = self._open_episode()
            now = datetime.utcnow()

        previous_cycle = states[-1] if states else None
        # The private cycle keeps the accepted EMA state of the main cycle
        # unchanged. A typed copy keeps the personality and the expression
        # readouts, which a vector round trip would lose. Before the first
        # cycle the state is the EMA readout of an empty history: the defaults.
        mental_state = (previous_cycle.mental_state.model_copy(deep=True)
                        if previous_cycle is not None else FullMentalState())
        age_seconds = max(0.0, (datetime.utcnow() - previous_cycle.timestamp).total_seconds()) \
            if previous_cycle is not None else 0.0

        main_frame: dict = {}
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
            main_frame = {
                "tick": newest.tick_id, "frame_id": newest.experience_id,
                "summary": newest.summary, "surprise": surprise,
                "gap_seconds": startup_gap, "self_feedback": self_feedback,
            }
            frame_age = max(0, tick - newest.tick_id)

        goal_records = [
            {"knoxel_id": int(goal.id), "text": str(goal.content),
             "tick_id": int(goal.tick_id), "fulfilment": float(goal.fulfilment),
             "affective_valence": float(goal.affective_valence),
             "revision": int(view.revisions.get(int(goal.id), 1))}
            for goal in goals
        ]
        expectation_records = [
            {"knoxel_id": int(item.id), "content": str(item.content),
             "anticipation": item.anticipation, "expected_outcome": item.expected_outcome,
             "tick_id": int(item.tick_id),
             "revision": int(view.revisions.get(int(item.id), 1))}
            for item in expectations
        ]
        request = InnerRequest(
            run_id=self.invocation_id or "",
            companion_name=self.ghost.get_companion_name(),
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
            operations=[{"name": line} for line in operation_lines()],
        )
        from workflows.reflection.experiments import context as reflection_context, source_payload
        request.reflection = reflection_context(view)
        reflection_sources = {item["knoxel_id"] for item in request.reflection["records"]}
        for candidate in request.candidates:
            if candidate.knoxel_id not in reflection_sources:
                request.reflection["records"].append(source_payload(view, view.get(candidate.knoxel_id)))
        self._reflection_context = request.reflection
        # A bounded episode carries its own identity and history. A continuing
        # step waits for main appraisal of the previous thought before it reads
        # new accepted state. Step, time, and stall limits close it explicitly.
        episode_state, blocked = self._resolve_episode(open_episode, latest_thought, latest_user, now)
        # Closed episodes still own their parking interval. The next fresh
        # request must retain that interval even without an open episode.
        suppressed = sorted(self._parked_sources(now, latest_user))
        request.episode = {**(episode_state or {}), "suppressed_source_ids": suppressed}
        self._episode_state = episode_state
        # Only a continuing step links to a parent thought. A fresh episode has
        # no parent, even when an older closed thought is the newest record.
        self._episode_thought_id = (int(latest_thought.id)
                                    if episode_state is not None and latest_thought is not None else None)
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
        request.channels = {key: value.model_dump(mode="json") for key, value in (channels or {}).items()}
        self._read_revisions = {
            int(item["knoxel_id"]): int(item["revision"])
            for item in [*goal_records, *expectation_records, *thoughts]
        }
        for candidate in request.candidates:
            self._read_revisions[int(candidate.knoxel_id)] = int(candidate.revision)
        self._mental_state = mental_state
        self._active_goal_ids = [int(item["knoxel_id"]) for item in goal_records]
        return request.model_dump(mode="json")

    def publish_output(self, report: dict, batch: list[Envelope]) -> None:
        """Commit accepted private work, or record an idle or rejected cycle.

        An idle cycle (``operation == "none"``) commits nothing: the loop does
        not generate thoughts to satisfy a quota. A stale proposal is rejected
        with a diagnostic envelope, never committed on top of newer state. A
        cancellation discards the cycle silently; the store guarantees no
        late commit after ``stop_event`` is set.
        """
        if self.stop_event.is_set() or self.run_stop.is_set():
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

    def _commit_cycle(self, parsed: InnerReport) -> None:
        """Commit one accepted private cycle into the inner shard."""
        ghost = self.ghost
        view = getattr(self.last_environment, "ghost_view", None) or ghost.snapshot()
        changes = ghost.begin_batch(self.invocation_id or new_id(), new_tick=True, cancel=self.run_stop)
        tick = changes.tick_id
        for record_id, revision in (getattr(self, "_read_revisions", {}) or {}).items():
            changes.expect_revision(record_id, revision)
        for memory in parsed.retrieved_memories:
            key = int(memory["knoxel_id"])
            if key not in view.revisions or memory["revision"] != view.revisions[key]:
                raise GhostConflict("Private recall is outside the frozen source revisions.")
            for source_id in {key, *memory.get("source_ids", [])}:
                if source_id not in view.revisions:
                    raise GhostConflict("Private recall includes a source outside the invocation grant.")
                changes.expect_revision(source_id, view.revisions[source_id])

        experiment_record = None
        if parsed.reflection.action != "none":
            from workflows.reflection.experiments import source_payload, stage_decision
            payload = dict(self._reflection_context)
            payload["records"] = list(payload["records"])
            known = {item["knoxel_id"] for item in payload["records"]}
            for item in parsed.retrieved_memories:
                for key in {item["knoxel_id"], *item.get("source_ids", [])} - known:
                    payload["records"].append(source_payload(view, view.get(key)))
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
                episode_id=episode_metadata["episode_id"], operation=parsed.operation,
                sequence=episode_metadata["sequence"],
                parent_record_ids=([int(self._episode_thought_id)] if getattr(self, "_episode_thought_id", None) else []),
                source_event_ids=list(parsed.evidence_ids), intention_ids=list(getattr(self, "_active_goal_ids", []) or []),
                topic_hints=list(parsed.topic_hints), reality=parsed.reality),
            metadata={
                "loop": "inner", "operation": parsed.operation, "trigger": parsed.trigger,
                "reflection": parsed.reflection.model_dump(mode="json"),
                "reflection_experiment_id": experiment_record.id if experiment_record is not None else None,
                "episode": episode_metadata,
                **({"control_generation": getattr(self, "_control_generation", self.cognitive_control.generation)}
                   if getattr(self, "cognitive_control", None) is not None else {}),
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
        if parsed.operation == "capability_inspection" and getattr(self, "_overlay", None) is not None:
            from pm.introspection.self_model import record_capability_beliefs
            record_capability_beliefs(
                changes, self._overlay, companion_name=ghost.get_companion_name(),
                source_revision=str(getattr(self, "_overlay", None).built_at.isoformat()),
                stale_names=getattr(self, "_overlay", None).stale_names(),
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
            record = view.get_knoxel_by_id(int(parsed.expectation.expectation_id))
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
        channels, previous_time = previous_cycle_sample(ghost.states)
        state = self._mental_state
        drivers = [int(item) for item in parsed.evidence_ids]
        sample = project_channels(state, active_goals=active_intentions(view),
                                  previous=channels, previous_time=previous_time,
                                  now=datetime.utcnow(), driver_ids=drivers)
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
        control = getattr(self, "cognitive_control", None)
        if control is None:
            ghost.commit(changes)
        else:
            control.commit_private(changes, getattr(self, "_control_generation", control.generation))

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

    def _build_inspection(self) -> dict:
        """Freeze the selected implementation evidence for this cycle.

        The provider owns the source grant and the real records. When no
        provider is configured, the cycle carries no inspection evidence and
        the inspector questions report an explicit limit. Filesystem access
        failures become evidence limits. Provider logic errors stop the runtime.
        """
        self._overlay = None
        provider = getattr(self, "inspector_provider", None)
        if provider is None:
            return {}
        try:
            overlay = provider()
        except OSError as error:
            return {"status": "unavailable", "reason": f"{type(error).__name__}: {error}"}
        self._overlay = overlay
        from workflows.introspection.snapshot import overlay_public
        from pm.introspection.episodes import EpisodeIndex, answer_questions
        index = EpisodeIndex(self.ghost)
        recent = index.recent(limit=5)
        answers = answer_questions(self.ghost, recent[0], overlay=overlay) if recent else {}
        return {
            "status": "ok",
            "capability_overlay": overlay_public(overlay),
            "episodes": [item.to_dict() for item in recent],
            "newest_episode_answers": answers,
            "observed_at": overlay.built_at.isoformat(),
        }

    def _capability_states(self) -> dict:
        """Return the strongest inspected state per capability, if available."""
        overlay = getattr(self, "_overlay", None)
        if overlay is None:
            return {}
        return {name: finding.strongest.value for name, finding in overlay.findings.items()}

    def _episode_limits(self) -> ControlConfig:
        """Return the operator episode limits, or the shipped defaults."""
        control = getattr(self, "cognitive_control", None)
        return control.config if control is not None else ControlConfig()

    def _inner_thoughts(self) -> list:
        """Return the accepted private thoughts in the order the store holds them."""
        return [knoxel for knoxel in self.ghost.all_features
                if knoxel.feature_type == FeatureType.InnerMonologue
                and knoxel.provenance and knoxel.provenance.source_kind == SourceKind.Self
                and (knoxel.metadata or {}).get("loop") == "inner"]

    def _open_episode(self) -> tuple[dict | None, object | None]:
        """Return the newest open private episode and the thought that carries it.

        The state is durable in the thought metadata, so a restart reconstructs
        the same episode without inventing offline activity.
        """
        thoughts = self._inner_thoughts()
        if not thoughts:
            return None, None
        latest = max(thoughts, key=lambda item: item.tick_id)
        episode = (latest.metadata or {}).get("episode")
        if not episode or episode.get("closed"):
            return None, latest
        return dict(episode), latest

    def _thought_acknowledged(self, thought_id: int) -> bool:
        """Return whether an accepted main cycle appraised this thought."""
        return any(thought_id in cycle.claimed_ids() for cycle in self.ghost.states)

    def _parked_sources(self, now: datetime, latest_user) -> set[int]:
        """Return parked source ids whose review time has not arrived.

        A newer outside message is new evidence and releases the parking early.
        """
        parked: set[int] = set()
        for thought in self._inner_thoughts():
            episode = (thought.metadata or {}).get("episode") or {}
            until = episode.get("parked_until")
            if not until or now >= datetime.fromisoformat(until):
                continue
            if latest_user is not None and latest_user.tick_id > thought.tick_id:
                continue
            control = getattr(self, "cognitive_control", None)
            if (control is not None and thought.metadata.get("control_generation") is not None
                    and thought.metadata["control_generation"] != control.generation):
                continue
            parked.update(int(item) for item in episode.get("parked_sources", []))
        return parked

    def _end_expired_episode(self) -> None:
        """Close expired or superseded work before the next snapshot opens.

        This controller update performs no inference. The thought content and
        lineage stay unchanged. Its episode metadata owns the terminal state.
        """
        with self.ghost.lock:
            episode, thought = self._open_episode()
            if episode is None:
                return
            control = getattr(self, "cognitive_control", None)
            user = self._latest_user_dialogue()
            elapsed = max(0.0, (datetime.utcnow() - datetime.fromisoformat(episode["started_at"])).total_seconds())
            reason = None
            if (control is not None and thought.metadata.get("control_generation") is not None
                    and thought.metadata["control_generation"] != control.generation):
                reason = "superseded generation"
            elif user is not None and user.tick_id > thought.tick_id:
                reason = "new outside input"
            elif elapsed >= self._episode_limits().episode_seconds:
                reason = "time limit"
            if reason is None:
                return
            changes = self.ghost.begin_batch(f"episode-end:{thought.id}:{reason}", cancel=self.stop_event)
            current = changes.view().get(thought.id)
            current.metadata["episode"] = {**episode, "closed": True, "closed_reason": reason,
                                           "elapsed_seconds": elapsed}
            changes.replace(current)
            self.ghost.commit(changes)

    def _resolve_episode(self, open_episode: dict | None, latest_thought, latest_user,
                         now: datetime) -> tuple[dict | None, bool]:
        """Return the episode context for this cycle and whether it must wait.

        An open episode continues only after main appraisal. Superseded work, a
        newer outside message, or an expired acknowledgment wait ends it. The
        caller must hold the store lock.
        """
        if open_episode is None or latest_thought is None:
            return None, False
        started_at = datetime.fromisoformat(open_episode["started_at"])
        elapsed = max(0.0, (now - started_at).total_seconds())
        limits = self._episode_limits()
        control = getattr(self, "cognitive_control", None)
        generation = latest_thought.metadata.get("control_generation")
        if control is not None and generation is not None and generation != control.generation:
            return None, False
        if latest_user is not None and latest_user.tick_id > latest_thought.tick_id:
            return None, False
        if not self._thought_acknowledged(int(latest_thought.id)):
            # Main must accept the previous step before the next one reads it.
            if elapsed < max(limits.episode_seconds, 120.0):
                return ({**open_episode, "awaiting_acknowledgment": True,
                         "elapsed_seconds": elapsed}, True)
            return None, False
        return ({**open_episode, "awaiting_acknowledgment": False,
                 "elapsed_seconds": elapsed,
                 "steps_remaining": max(0, limits.episode_steps - int(open_episode.get("sequence", 0)) - 1)}, False)

    def _next_episode(self, parsed: InnerReport) -> dict:
        """Compute the durable episode state after this accepted step.

        A non-continuing step, a step or time limit, or repeated stalled
        progress closes the episode. A suspension or a stall parks its sources
        for a bounded review time.
        """
        prior = getattr(self, "_episode_state", None) or {}
        now = datetime.utcnow()
        limits = self._episode_limits()
        step = int(prior.get("sequence", -1)) + 1
        started_at = prior.get("started_at") or now.isoformat()
        elapsed = max(0.0, (now - datetime.fromisoformat(started_at)).total_seconds())
        operations = list(prior.get("operations", [])) + [parsed.operation]
        evidence = list(dict.fromkeys([*prior.get("evidence_ids", []), *parsed.evidence_ids]))
        alternatives = list(dict.fromkeys([*prior.get("alternatives", []),
                                           *(item.text for item in parsed.alternatives if item.text)]))
        thoughts = list(dict.fromkeys([*prior.get("thoughts", []), parsed.thought]))[-8:]
        progress = list(prior.get("progress", [])) + [parsed.progress]
        stalled = int(prior.get("stalled", 0)) + (0 if parsed.progress != "none" else 1)
        closed_reason = None
        if parsed.continuation != "continue":
            closed_reason = parsed.continuation
        elif step + 1 >= limits.episode_steps:
            closed_reason = "step limit"
        elif elapsed >= limits.episode_seconds:
            closed_reason = "time limit"
        elif stalled >= limits.stalled_steps:
            closed_reason = "stalled progress"
        parked_until = None
        if closed_reason in {"suspend", "stalled progress"}:
            parked_until = (now + timedelta(seconds=limits.parked_seconds)).isoformat()
        return {
            "episode_id": prior.get("episode_id") or new_id(),
            "sequence": step, "started_at": started_at, "elapsed_seconds": elapsed,
            "closed": closed_reason is not None, "closed_reason": closed_reason,
            "operations": operations, "evidence_ids": evidence,
            "alternatives": alternatives, "thoughts": thoughts, "progress": progress,
            "stalled": stalled, "continuation": parsed.continuation,
            "next_operation": parsed.next_operation,
            "parked_until": parked_until,
            "parked_sources": sorted(evidence) if parked_until else [],
            "parent_thought_id": getattr(self, "_episode_thought_id", None),
        }

    def _private_thought_ids(self) -> set[int]:
        """Return every committed private thought id, for self-feedback detection."""
        return {int(knoxel.id) for knoxel in self.ghost.all_features
                if knoxel.feature_type == FeatureType.InnerMonologue
                and (knoxel.metadata or {}).get("loop") == "inner"}

    def _recent_private_thoughts(self) -> list[dict]:
        thoughts = []
        for knoxel in self.ghost.all_features:
            if knoxel.feature_type != FeatureType.InnerMonologue:
                continue
            if not knoxel.provenance or knoxel.provenance.source_kind != SourceKind.Self:
                continue
            metadata = knoxel.metadata or {}
            thoughts.append({
                "knoxel_id": int(knoxel.id), "text": str(knoxel.content),
                "tick_id": int(knoxel.tick_id), "operation": metadata.get("operation", ""),
                "trigger": metadata.get("trigger", ""),
                "continuation": metadata.get("continuation", ""),
                "source_ids": [int(item) for item in metadata.get("evidence_ids", [])],
                "revision": int(self.ghost.revisions.get(knoxel.id, 1)),
            })
        thoughts.sort(key=lambda item: item["tick_id"], reverse=True)
        return thoughts[:RECENT_THOUGHT_LIMIT]

    def _latest_appraisal(self):
        latest = None
        for knoxel in self.ghost.all_features:
            if (knoxel.metadata or {}).get("origin") == "appraisal":
                if latest is None or knoxel.tick_id > latest.tick_id:
                    latest = knoxel
        return latest

    def _startup_gap_seconds(self) -> float:
        latest = None
        for knoxel in self.ghost.all_features:
            metadata = knoxel.metadata or {}
            if metadata.get("event") == "startup" and "gap_seconds" in metadata:
                if latest is None or knoxel.tick_id > latest.tick_id:
                    latest = knoxel
        return float((latest.metadata or {}).get("gap_seconds", 0.0) or 0.0) if latest else 0.0

    def _latest_user_dialogue(self):
        # The user proxy names the user as the feature source; the provenance
        # of a user message carries no source kind. The source name is the
        # stable identity, exactly as the main loop's idle measurement uses.
        latest = None
        user = self.ghost.get_user_name()
        for knoxel in self.ghost.all_features:
            if knoxel.feature_type != FeatureType.Dialogue:
                continue
            if knoxel.source != user:
                continue
            if latest is None or knoxel.tick_id > latest.tick_id:
                latest = knoxel
        return latest

    def _state_summary(self, cycle) -> str:
        if cycle is None:
            return "(no accepted state yet)"
        turn = self.ghost.gwt_turns[cycle.tick_id] if cycle.tick_id in self.ghost.gwt_turns else None
        return f"tick {cycle.tick_id}, loop: {turn.slot_id if turn is not None else 'none'}"

    def _candidates(self, view, goals, expectations, latest_appraisal, main_frame,
                    thoughts, latest_user) -> list[ThoughtCandidate]:
        """Derive the loop's private attention deterministically.

        The selection weights inner material above outside material: goals,
        expectations, and recent thoughts outrank the latest outside message.
        The model can narrow this list, never widen it.
        """
        candidates: list[ThoughtCandidate] = []
        revisions = view.revisions

        def add(record, kind, urgency, reason):
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
        frame_id = main_frame.get("frame_id")
        if frame_id is not None:
            record = view.get_knoxel_by_id(int(frame_id))
            if record is not None:
                add(record, "frame", 0.35, "the main loop's latest accepted workspace")
        for thought in thoughts[:3]:
            record = view.get_knoxel_by_id(int(thought["knoxel_id"]))
            if record is not None:
                add(record, "thought", 0.3, "a recent private thought, for continuity")
        if latest_user is not None:
            add(latest_user, "dialogue", 0.25, "the latest outside message, as background")
        candidates.sort(key=lambda item: (-item.urgency, -item.tick_id))
        return candidates[:10]


__all__ = ["PrivateThoughtLoop", "build_inner_thought_tools"]
