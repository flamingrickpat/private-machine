"""The private thought loop host binding.

The host owns the loop's commit boundary. Each timer tick it freezes a
permitted view, derives the deterministic private candidates and triggers,
and runs the parent workflow. A cycle with no trigger performs nothing. An
accepted cycle commits its thought into the inner GWT shard and publishes one
``thought.created`` self-origin event, which the main loop appraises like any
other internal event.

Both loops share one MemoryStore and one companion identity. Every cross-loop
read carries a revision, and the commit rejects a proposal whose source
changed. The inner loop never writes the main loop's frame and never changes
mental state directly: only the main appraisal path does that.
"""
from __future__ import annotations

from datetime import datetime
import json
import traceback

from pm.bus.envelopes import Envelope, new_envelope, new_id
from pm.memory.snapshot import GhostCancelled, GhostConflict
from pm.records.enums import FeatureType, InterlocusType, SourceKind
from pm.records.cognitive_records import AttentionFrameRecord, ExpectationRecord
from pm.records.cycle_state import CycleState
from pm.records.global_workspace import GwtTurnRecord
from pm.records.mental_state import FullMentalState, create_empty_state
from pm.records.memory_records import new_feature
from pm.records.qualia_records import QualeRecord
from pm.runs.workflow_module import CognitiveWorkflow

from workflows.main_cognition.host import seconds_since_user_input
from workflows.main_cognition.stage_commits import active_intentions, apply_goal_changes
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


def build_inner_thought_tools(ghost, runtime_root, llm_complete, *, cancel_provider=None):
    """Bind the parent manifest's ``run_stage`` tool to the shared store."""
    from pathlib import Path

    def factory(environment, batch):
        return {"run_stage": build_inner_stage_runner(
            ghost, llm_complete, Path(runtime_root), read_environment=environment,
            cancel=cancel_provider() if cancel_provider is not None else None)}

    return factory


class PrivateThoughtLoop(CognitiveWorkflow):
    """The second conscious loop: scheduled private thought with its own shard."""

    WORKFLOW_NAME = "inner_thought"
    # The loop is timer-driven. It never reacts to bus traffic, so a main
    # cycle cannot cascade into an inner cycle into another main cycle.
    DATA_TOPICS = frozenset()

    def handle_batch(self, batch: list[Envelope]) -> None:
        """Run one cycle and discard an explicitly rejected model proposal.

        The rejected proposal changes no committed private state. A rejection
        retains its diagnostic beside the parent kernel artifacts and emits
        telemetry only. It is not an idle cycle. Other errors propagate.
        """
        try:
            super().handle_batch(batch)
        except ModelProposalError as error:
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
            main_shard = self.ghost.gwt_shard("main", limit=3)
            thoughts = self._recent_private_thoughts()
            goals = active_intentions(view)
            expectations = pending_expectations(view)
            latest_appraisal = self._latest_appraisal()
            startup_gap = self._startup_gap_seconds()
            latest_user = self._latest_user_dialogue()

        previous_cycle = states[-1] if states else None
        mental_vec = list(previous_cycle.mental_state_latent_vec) if previous_cycle is not None else []
        age_seconds = max(0.0, (datetime.utcnow() - previous_cycle.timestamp).total_seconds()) \
            if previous_cycle is not None else 0.0

        main_frame: dict = {}
        frame_age = 0
        if main_shard:
            newest = main_shard[0]
            surprise = 0.0
            if latest_appraisal is not None:
                surprise = float((latest_appraisal.metadata or {}).get("emotion_delta", {}).get(
                    "surprise", 0.0) or 0.0)
            main_frame = {
                "tick": newest.tick_id, "frame_id": newest.attention_frame_id,
                "summary": newest.summary, "surprise": surprise,
                "gap_seconds": startup_gap,
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
            mental_state_latent_vec=mental_vec,
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
        request.trigger_candidates = evaluate_triggers(request)
        request.inspection = self._build_inspection()
        # Re-evaluate so a fresh capability doubt can fire with the evidence.
        if request.inspection:
            request.trigger_candidates = evaluate_triggers(request)
        # The felt channels of the last accepted cycle give the model context.
        # The new sample is computed at commit time from this frozen view.
        channels, _ = previous_cycle_sample(states)
        request.channels = dict(channels or {})
        self._read_revisions = {
            int(item["knoxel_id"]): int(item["revision"])
            for item in [*goal_records, *expectation_records, *thoughts]
        }
        for candidate in request.candidates:
            self._read_revisions[int(candidate.knoxel_id)] = int(candidate.revision)
        self._mental_vec = mental_vec
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
        changes = ghost.begin_batch(self.invocation_id or new_id(), new_tick=True, cancel=self.stop_event)
        tick = changes.tick_id
        for record_id, revision in (getattr(self, "_read_revisions", {}) or {}).items():
            changes.expect_revision(record_id, revision)

        thought = new_feature(
            ghost, parsed.thought, FeatureType.InnerMonologue,
            source=ghost.get_companion_name(), interlocus=InterlocusType.PrivateInternal,
            changes=changes, source_kind=SourceKind.Self,
            metadata={
                "loop": "inner", "operation": parsed.operation, "trigger": parsed.trigger,
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
                "capability_states": self._capability_states(),
            },
        )

        # Goal changes go through the shared owner, so lifecycle evidence,
        # keys, and history stay identical across both loops.
        apply_goal_changes(changes, view, parsed.goal_changes,
                           evidence_ids=[thought.id], appraisal_id=None, origin="private_loop")

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

        frame = changes.create(AttentionFrameRecord(
            content=parsed.summary or parsed.thought[:120],
            tick_id=tick,
            focus_knoxel_ids=[int(item) for item in parsed.candidate_ids][:5],
            active_goal_ids=list(getattr(self, "_active_goal_ids", []) or []),
            control_mode="reflective",
            frame_reason=parsed.trigger_reason or f"private operation {parsed.operation}",
        ))

        # The felt-state sample is deterministic math over the frozen accepted
        # state. Its drivers name the records that moved the state, and
        # reportability applies the causal gate: a driver that never won
        # attention and was never causal leaves the snapshot dormant.
        channels, previous_time = previous_cycle_sample(ghost.states)
        state = (FullMentalState.init_from_list(self._mental_vec)
                 if self._mental_vec else create_empty_state())
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

        cycle = CycleState(
            tick_id=tick, attention_frame_id=frame.id, published_slots=["inner"],
            stimuli_ids=[], mental_state_latent_vec=list(self._mental_vec or []),
            internal_data={"qualia_channels": {key: value.model_dump(mode="json")
                                               for key, value in sample.items()},
                           "operation": parsed.operation},
        )
        changes.set_gwt(GwtTurnRecord(
            tick_id=tick, slot_id="inner", attention_frame_id=frame.id,
            knoxel_ids=[thought.id, frame.id, *quale_ids],
            summary=parsed.summary or parsed.thought[:120],
        ), cycle)
        ghost.commit(changes)

        # The self-origin event. The main loop appraises it like any internal
        # stimulus; nothing here publishes speech or commands the body.
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
        return f"tick {cycle.tick_id}, slots published: {', '.join(cycle.published_slots) or 'none'}"

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
