"""Commit a private agent proposal before routing it through main appraisal.

The module runs one scoped pm-coder mental session, commits its proposal as a
self-origin feature, and publishes it for ordinary appraisal. It owns no
emotion, goal, or workspace change: main cognition still accepts or rejects
the advice, and a source conflict discards it silently by design.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from pm.bus.envelopes import Envelope, new_envelope
from pm.memory.agent_tools import MemoryReadSession, MemoryToolConfig, MemoryView
from pm.memory.snapshot import GhostConflict
from pm.records.enums import FeatureType, InterlocusType, SourceKind
from pm.records.lineage import RecordLineage
from pm.records.memory_records import new_feature
from pm.runs.workflow_module import CognitiveWorkflow
from workflows.cognitive_control.schemas import MentalPlan
from workflows.focus_management.schemas import MentalWorkRequest
from workflows.main_cognition.schemas import ActionRequest
from workflows.public_speech.shard_context import shard_story

if TYPE_CHECKING:
    from pm.memory.bash_view import MemoryBashEnvironment
    from pm.memory.embeddings import EmbeddingService


class FocusManagement(CognitiveWorkflow):
    """Offer a deliberate mental action through a real pm-coder workflow.

    The agent reads a frozen grant and proposes changes. It never owns the
    main workspace, goals, or affect. Source conflicts discard the proposal.
    """

    DATA_TOPICS = frozenset({"action.request"})
    WORKFLOW_NAME = "focus_management"

    def __init__(self, *args: Any, embeddings: EmbeddingService | None = None,
                 **kwargs: Any) -> None:
        """Forward the module configuration and bind the tool factory."""
        super().__init__(*args, tools_factory=self._tools, **kwargs)
        self.embeddings = embeddings
        self._published_results = set()

    def handle_batch(self, batch: list[Envelope]) -> None:
        """Run new actions once. Timer ticks replay unappraised durable results."""
        if not batch:
            self.resume()
            return
        for envelope in batch:
            action = self.ghost.get_knoxel_by_id(ActionRequest.model_validate(envelope.payload).knoxel_id)
            if action.metadata.get("mental_result_id") is not None:
                self.resume()
                continue
            super().handle_batch([envelope])

    def resume(self) -> None:
        """Replay results after interruption without rerunning an accepted agent.

        Accepted main cycles acknowledge result ids. A wait delays publication.
        The process remembers queued ids. Restart rebuilds this set from state.
        """
        control = getattr(self, "cognitive_control", None)
        with self.ghost.lock:
            if control is not None and control.waiting():
                return
            appraised = {key for cycle in self.ghost.states for key in cycle.claimed_ids()}
            pending = [item for item in self.ghost.all_features
                       if item.metadata.get("mental_action_id") is not None
                       and item.id not in appraised and item.id not in self._published_results]
        for result in pending:
            # Main can discard an event during delivery. Mark it first so
            # that immediate release cannot leave a stale publication claim.
            self._published_results.add(result.id)
            self.bus.put(new_envelope(self.module_id, "data", "thought.created",
                result.metadata.get("mental_correlation", ""), {"knoxel_id": result.id}))

    def release_results(self, envelopes: list[Envelope]) -> None:
        """Permit replay after main cognition discards or cancels queued work."""
        for envelope in envelopes:
            if envelope.topic == "thought.created":
                self._published_results.discard(envelope.payload.get("knoxel_id"))

    def prepare_input(self, batch: list[Envelope]) -> dict:
        """Read the selected mental instruction and the shards of the cycle that chose it.

        The instruction is ``arguments.instruction``, else
        ``arguments.instructions``, else the decision text. The workspace is
        the shard story of the choosing cycle, so the agent knows what was on
        the companion's mind.
        """
        request = ActionRequest.model_validate(batch[0].payload)
        self._action_id = request.knoxel_id
        self._correlation = batch[0].correlation_id
        control = getattr(self, "cognitive_control", None)
        instruction = request.arguments.instruction or request.arguments.instructions or request.content
        if not instruction.strip():
            raise ValueError("Mental work requires a nonempty instruction.")
        companion = self.ghost.get_companion_name()
        return MentalWorkRequest(
            instruction=instruction, companion=companion,
            timeout_seconds=control.config.mental_timeout_seconds if control is not None else 600,
            character=self.ghost.ghost_config.character_card_story,
            workspace=shard_story(self.ghost.snapshot(), request, companion),
            experience=request.workspace, affect=request.affect,
        ).model_dump(mode="json")

    def _tools(self, environment: MemoryBashEnvironment,
               batch: object) -> dict[str, Callable[..., object]]:
        """Bind scoped memory, goal, and session tools to one frozen view."""
        ids = frozenset(environment.visible_knoxels)
        scopes = frozenset((raw["data"]["scope"]["reality_id"], raw["data"]["scope"]["frame"])
                           for raw in environment.visible_knoxels.values())
        memory = MemoryReadSession(MemoryToolConfig(self.ghost, MemoryView(ids, scopes),
            semantic=self.embeddings is not None, lexical=True, graph=True,
            embeddings=self.embeddings, cancel=self.run_stop))
        self._goal_revisions = {}

        def mental_goals(offset: int = 0, limit: int = 16) -> dict:
            """Read a page of goals, including deferred concerns. No goal changes occur."""
            if offset < 0 or not 1 <= limit <= 32:
                raise ValueError("Use a nonnegative offset and a limit from 1 to 32.")
            memory.check_cancelled()
            goals = [memory.snapshot.get(key) for key, raw in memory.records.items() if raw["__type__"] == "Intention"]
            goals.sort(key=lambda item: (-float(item.incentive_salience), item.id))
            selected = goals[offset:offset + limit]
            for goal in selected:
                self._goal_revisions[goal.id] = memory.snapshot.revisions[goal.id]
            return {"goals": [{"id": item.id, "text": item.content, "status": item.status,
                               "priority": item.incentive_salience} for item in selected],
                    "next_offset": offset + limit if offset + limit < len(goals) else None,
                    "inspected_ids": list(self._goal_revisions)}

        def inspected_goal_ids() -> list[int]:
            """Return goal ids already read, without granting a new inspection."""
            return list(self._goal_revisions)

        return {"mental_memory": lambda: memory, "mental_goals": mental_goals,
                "mental_inspected": inspected_goal_ids}

    def publish_output(self, report: dict, batch: list[Envelope]) -> None:
        """Commit the action receipt and private result in one durable batch.

        Source conflicts retain a failed outcome. No goal or affect changes
        occur here. Restart replays the result without another agent session.
        """
        if report.get("status") == "failed":
            self._commit_failure(report)
            return
        plan = MentalPlan.model_validate(report["plan"])
        changes = self.ghost.begin_batch(self.invocation_id, new_tick=True, cancel=self.run_stop)
        for key, revision in {**report["source_revisions"], **self._goal_revisions}.items():
            changes.expect_revision(int(key), int(revision))
        action = changes.view().get(self._action_id).copy_for_edit()
        changes.expect_revision(action.id, self.last_environment.ghost_view.revisions[action.id])
        proposal = new_feature(self.ghost, plan.summary, FeatureType.Thought,
            source=self.ghost.get_companion_name(), source_kind=SourceKind.Self,
            interlocus=InterlocusType.PrivateInternal, changes=changes,
            metadata={"mental_plan": plan.model_dump(mode="json"), "agent_session": report["agent_session"],
                      "mental_action_id": action.id, "mental_correlation": self._correlation,
                      "memory_kind": "interpretation", "source_revisions": report["source_revisions"]},
            lineage=RecordLineage(episode_id=action.lineage.episode_id if action.lineage else self.invocation_id,
                operation="direct_attention", parent_record_ids=[action.id], source_event_ids=plan.evidence_ids,
                topic_hints=plan.focus.topic_hints if plan.focus else []))
        changes.depend_on(report["source_revisions"], authority=action.id)
        changes.set_lifecycle(action.id, dispatch_status="admitted", mental_result_id=proposal.id,
                              mental_status="proposed", execution_status="completed")
        try:
            self.ghost.commit(changes)
        except GhostConflict as error:
            self.bus.put(new_envelope(self.module_id, "telemetry", "mental.rejected", self._correlation,
                                      {"reason": str(error)}))
            self._commit_failure({"error_type": type(error).__name__, "reason": str(error)})
            return
        self._receipt(proposal.id, "proposed")
        self.resume()

    def _commit_failure(self, report: dict) -> None:
        """Retain an expected failure without committing the rejected proposal."""
        with self.ghost.lock:
            changes = self.ghost.begin_batch(self.invocation_id + ":failure", new_tick=True, cancel=self.run_stop)
            action = changes.view().get(self._action_id).copy_for_edit()
            failure = new_feature(self.ghost, "Mental work failed: " + report["reason"], FeatureType.Thought,
                source=self.ghost.get_companion_name(), source_kind=SourceKind.Self,
                interlocus=InterlocusType.PrivateInternal, changes=changes,
                metadata={"mental_action_id": action.id, "mental_correlation": self._correlation,
                          "mental_status": "failed", "error_type": report["error_type"],
                          "memory_kind": "interpretation"},
                lineage=RecordLineage(episode_id=self.invocation_id, operation="mental_failure",
                                     parent_record_ids=[action.id]))
            changes.depend_on([action.id], authority=action.id)
            changes.set_lifecycle(action.id, dispatch_status="admitted", mental_status="failed",
                                  mental_result_id=failure.id, execution_status="failed")
            self.ghost.commit(changes)
        self._receipt(failure.id, "failed")
        self.resume()

    def _receipt(self, result_id: int, status: str) -> None:
        """Publish diagnostic admission evidence. This receipt requests no speech."""
        self.bus.put(new_envelope(self.module_id, "telemetry", "action.receipt", self._correlation,
            {"workflow_name": self.WORKFLOW_NAME, "status": status,
             "action_schema_id": self._action_id, "result_id": result_id,
             "public_reply_requested": False}))
