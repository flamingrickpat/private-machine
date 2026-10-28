"""``PublicDialogue``: the shared bus module of Story and Assistant replies.

It validates evidence and commits each utterance before publication. Story
and Assistant subclass it. Each draft receives one review. Story revises rejected
drafts. Declared model-output exhaustion leaves the exchange pending. Other errors reach the
application. Host failure strings never become companion speech.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from pm.host.durable_files import atomic_json
from pm.llm.http_client import EmptyCompletionError, ProviderCompletionError
from pm.llm.structured import ModelFormatError
from pm.utils import yaml_util

from pm.bus.envelopes import Envelope, new_id
from workflows.main_cognition.schemas import ActionMetadata, ActionRequest, Stimulus
from workflows.public_speech.public_context import build_public_context
from workflows.public_speech.shard_context import shard_context
from workflows.public_speech.speech_text import SpeechOutputError
from workflows.public_speech.publication import commit_public_reply, publish_reply
from pm.bus.envelopes import new_envelope
from pm.runs.workflow_module import CognitiveWorkflow

if TYPE_CHECKING:
    from pydantic import JsonValue

    from workflows.public_speech.context_planning import PlannedContext
    from workflows.public_speech.public_context import PublicContext


def _clip(text: str, limit: int = 500) -> str:
    text = str(text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "\u2026"


class PublicDialogue(CognitiveWorkflow):
    """Shared evidence guards and visible failures for public workflows."""

    capability_provider = None
    inference_admission = None

    # Shared embedding service for context planning. The assembly sets it;
    # without it, planning scores lexically.
    context_embeddings = None

    # Input token budget from the model's serving context * context_input_fraction.
    # The assembly sets it once at startup. Without it, the plan YAML value is used.
    context_budget: int | None = None

    def partner_stimuli(self, request: ActionRequest) -> list[Stimulus]:
        """Return the spoken lines of the request's stimuli, with their accepted text.

        A spoken line is a ``Dialogue`` feature from the user or another
        agent, such as a second companion. Attribution comes from the
        accepted record, not from the request. Private thoughts and system
        events never become speech, even when their text resembles dialogue.
        A stimulus that does not exist raises KeyError.
        """
        from pm.records.enums import FeatureType, SourceKind
        result = []
        with self.ghost.lock:
            for item in request.stimuli:
                record = self.ghost.get_knoxel_by_id(item.knoxel_id)
                if record is None:
                    raise KeyError(f"Public stimulus does not exist: {item.knoxel_id}")
                if (record.feature_type == FeatureType.Dialogue
                        and record.provenance is not None
                        and record.provenance.source_kind in {SourceKind.User, SourceKind.Agent}):
                    result.append(item.model_copy(update={"text": str(record.content)}))
        return result

    def public_context(self, request: ActionRequest, planned: PlannedContext | None = None,
                       late_ids: list[int] = ()) -> PublicContext:
        """Freeze the public context of one reply and keep it for the publication guard.

        ``late_ids`` are partner lines that arrived after the deciding cycle.
        They join the stimulus set of the context, so the commit resolves
        them and the revision guard covers them.
        """
        # Refresh the durable private ontology before the snapshot. The review
        # reads the terms and bindings from the frozen public context.
        from workflows.public_speech.ontology import observe_ontology
        observe_ontology(self.ghost, authority_id=request.knoxel_id)
        capabilities = self.capability_provider() if self.capability_provider else None
        with self.ghost.lock:
            self._public_context = build_public_context(self.ghost, request, capabilities=capabilities,
                                                        planned=planned, late_ids=list(late_ids))
        return self._public_context

    def reply_context(self, request: ActionRequest, late_ids: list[int] = ()) -> PlannedContext:
        """Return the reply context: shard dialogue plus planned retrieval from memory.

        The shard tail provides the verbatim transcript with the partner.
        The lane planner fills the remaining context with embedding-scored
        memories, facts, narratives, and hierarchical summaries from the
        broader store. The focus query comes from the workspace text, the
        current stimuli, and the active goals.

        When no embedding service is available, the planner scores lexically.
        The shard tail stops at the cycle that chose the action. Partner lines
        that arrived after that cycle are read from the live store and added.
        """
        import asyncio
        with self.ghost.lock:
            view = self.ghost.snapshot()
        shard = shard_context(view, request, self.ghost.get_companion_name(), late_ids=list(late_ids))
        if self.context_embeddings is None:
            return shard
        # Build the focus from the workspace, stimuli, and goals.
        from workflows.public_speech.context_planning import FocusInput, plan_reply_context
        stimulus_texts = [s.text for s in request.stimuli if s.text]
        focus = FocusInput(
            workspace=request.workspace or "",
            current_message=" ".join(stimulus_texts),
            attention_texts=[request.content] if request.content else [],
            exclude_ids=[s.knoxel_id for s in request.stimuli if s.knoxel_id],
        )
        # Exclude records already shown in the shard transcript.
        shard_ids = {t.knoxel_id for t in shard.dialogue if t.knoxel_id}
        focus.exclude_ids = list(set(focus.exclude_ids) | shard_ids)
        try:
            planned = asyncio.run(plan_reply_context(
                view, "story", focus, self.context_embeddings,
                total_tokens=self.context_budget,
            ))
        except Exception:
            return shard
        # Merge: shard dialogue stays, planned evidence fills the rest.
        existing_ids = {item.knoxel_id for item in shard.evidence if item.knoxel_id}
        merged_evidence = list(shard.evidence)
        for item in planned.evidence:
            if item.knoxel_id not in existing_ids:
                merged_evidence.append(item)
                existing_ids.add(item.knoxel_id)
        shard.evidence = merged_evidence
        # Merge planned utterances into the spoken transcript. The shard tail
        # holds the recent window; the planner adds older real utterances. Keep
        # one copy per record and order by world time so the assistant turn
        # reads chronologically. Thoughts never reach here (they stay evidence).
        seen = {t.knoxel_id for t in shard.dialogue if t.knoxel_id}
        merged_dialogue = list(shard.dialogue)
        for turn in planned.dialogue:
            if turn.knoxel_id and turn.knoxel_id in seen:
                continue
            merged_dialogue.append(turn)
            if turn.knoxel_id:
                seen.add(turn.knoxel_id)
        merged_dialogue.sort(key=lambda t: (t.timestamp or "", t.knoxel_id or 0))
        shard.dialogue = merged_dialogue
        shard.trace["planned_lanes"] = planned.trace.get("lanes", [])
        return shard

    def prepare_reply(self, request: ActionRequest) -> tuple[list[Stimulus], PlannedContext, PublicContext]:
        """Return the partner lines, the shard context, and the public context of one action.

        The reply answers the stimuli of the action. Partner lines that
        arrived after the deciding cycle join them, so a slow cycle does not
        answer an old line. Both public workflows use this one path, so they
        cannot drift apart.
        """
        late = self.late_partner_lines(request)
        spoken = self.partner_stimuli(request) + self.late_stimuli(request, late)
        planned = self.reply_context(request, late)
        public = self.public_context(request, planned, late_ids=late)
        return spoken, planned, public

    def late_stimuli(self, request: ActionRequest, late_ids: list[int]) -> list[Stimulus]:
        """Return a stimulus for each late partner line, oldest first.

        The text comes from the accepted record, as in :meth:`partner_stimuli`.
        """
        result = []
        with self.ghost.lock:
            for record_id in late_ids:
                record = self.ghost.get_knoxel_by_id(record_id)
                if record is None:
                    raise KeyError(f"Late partner line does not exist: {record_id}")
                result.append(Stimulus(knoxel_id=record.id, source=str(record.source or ""),
                                       text=" ".join(str(record.content).split()),
                                       entity_id=request.partner_entity_id))
        return result

    def cycle_frontier(self, request: ActionRequest) -> int:
        """Return the newest record id that the deciding cycle saw.

        The stimuli and the shard contents of the request are all committed at
        or before that cycle. Record ids increase with time, so the maximum is
        the boundary between what the cycle knew and what arrived later.
        """
        ids = [item.knoxel_id for item in request.stimuli]
        for item in request.shards:
            ids += [*item.shard.item_ids(), *item.shard.tail_ids]
        return max(ids, default=0)

    def late_partner_lines(self, request: ActionRequest) -> list[int]:
        """Return the partner Dialogue ids that arrived after the deciding cycle, oldest first.

        The partner is the entity the reply addresses. Only spoken lines from
        the user or another agent count, as in :meth:`partner_stimuli`. A
        recovery or task reply has no cycle, so it has no late lines.
        """
        if request.recovery_outcome or not request.shards:
            return []
        frontier = self.cycle_frontier(request)
        partner = request.partner_name or self.ghost.get_user_name()
        from pm.records.enums import FeatureType, SourceKind
        with self.ghost.lock:
            return sorted(record.id for record in self.ghost.all_features
                          if record.id > frontier and record.feature_type == FeatureType.Dialogue
                          and (record.entity_id == request.partner_entity_id if request.partner_entity_id is not None
                               else record.override_source == partner) and record.content.strip()
                          and record.provenance is not None
                          and record.provenance.source_kind in {SourceKind.User, SourceKind.Agent})

    def stale_reply_reason(self, envelope: Envelope, request: ActionRequest) -> str:
        """Return why this action must not produce a reply now, or an empty string.

        Two rules protect the conversation:

        - One reply per exchange per cycle. A cycle can release two actions
          for the same partner line. The second one would answer a line the
          first one already answered.
        - A reply that answers nothing current. When every stimulus of the
          action is already resolved and no newer partner line arrived, the
          action has no open question left.

        A newer partner line does not make the reply stale. The reply
        re-anchors to that line instead (``late_partner_lines``).
        """
        from workflows.public_speech.publication import exchange_record
        if request.recovery_outcome:
            return ""
        exchange = exchange_record(self.ghost, envelope.correlation_id) if envelope.correlation_id else None
        if exchange is None:
            return ""
        resolver = exchange.metadata.get("public_reply_id")
        if resolver and exchange.metadata.get("exchange_status") == "resolved":
            reply = self.ghost.get_knoxel_by_id(int(resolver))
            answered_tick = (reply.metadata.get("generation_state_tick") if reply is not None else None)
            if answered_tick is not None and answered_tick == request.gwt_tick:
                return (f"Exchange {envelope.correlation_id} already got its reply in cycle "
                        f"{request.gwt_tick} (record {resolver}).")
            answered = {int(key) for key in (reply.metadata.get("responding_to_ids") or [])} if reply else set()
            open_ids = [item.knoxel_id for item in request.stimuli if item.knoxel_id not in answered]
            if not open_ids and not self.late_partner_lines(request):
                return (f"Every line of exchange {envelope.correlation_id} is already answered "
                        f"(record {resolver}).")
        return ""

    def deliver(self, envelope: Envelope) -> None:
        """Reserve reply priority before queueing the action."""
        if self.inference_admission is not None and envelope.kind != "control":
            self.inference_admission.reserve(envelope.id)
        super().deliver(envelope)

    def handle_batch(self, batch: list[Envelope]) -> None:
        """Release reply reservations even after rejection or cancellation."""
        try:
            self._handle_priority_batch(batch)
        finally:
            if self.inference_admission is not None:
                for envelope in batch:
                    self.inference_admission.release(envelope.id)

    def _handle_priority_batch(self, batch: list[Envelope]) -> None:
        # Each public action owns one report and one exchange resolution.
        # A second envelope must not disappear inside a merged mailbox batch.
        for envelope in batch:
            request = ActionRequest.model_validate(envelope.payload)
            action = self.ghost.get_knoxel_by_id(request.knoxel_id) if request.knoxel_id is not None else None
            if action is not None and ActionMetadata.model_validate(action.metadata).public_reply_id is not None:
                continue
            stale = self.stale_reply_reason(envelope, request)
            if stale:
                # Do not spend a model call on a line that is already answered.
                self.bus.put(new_envelope(self.module_id, "telemetry", "action.receipt", envelope.correlation_id,
                    {"status": "superseded", "workflow_name": self.WORKFLOW_NAME,
                     "rejection_reason": stale, "recovery_outcome": "not_published"}, causation_id=envelope.id))
                continue
            # Preparation can fail before a new snapshot exists. A prior task
            # receipt must never become the recovery evidence for this action.
            self._public_context = None
            try:
                super().handle_batch([envelope])
            except (EmptyCompletionError, ProviderCompletionError, SpeechOutputError, ModelFormatError) as error:
                self.defer_reply(envelope, error)

    def defer_reply(self, envelope: Envelope,
                    error: EmptyCompletionError | ProviderCompletionError | SpeechOutputError | ModelFormatError) -> None:
        """Retain exhausted model-output evidence and keep the exchange pending.

        Only declared model-output errors reach this method. Snapshot, tool,
        filesystem, cancellation, and programming errors keep their boundaries.
        No reply commits, and no tool action repeats. A later action or restart
        can retry the pending exchange. Saving evidence must succeed.
        """
        environment = self.last_environment
        files = {"inbox/request.yaml": environment.read_text("inbox/request.yaml")}
        files.update({path: environment.read_text(path) for path in environment.list_files("scratch")})
        path = self.runtime_root / self.module_id / self.invocation_id / "speech-failure.json"
        atomic_json(path, {"error_type": type(error).__name__, "reason": str(error),
                           "correlation_id": envelope.correlation_id, "files": files})
        self.bus.put(new_envelope(self.module_id, "telemetry", "action.receipt", envelope.correlation_id,
            {"status": "generation_failed", "workflow_name": self.WORKFLOW_NAME,
             "rejection_reason": str(error), "recovery_outcome": "not_published", "artifact_path": str(path)},
            causation_id=envelope.id))

    def reject_reply(self, batch: list[Envelope], reason: str, *, status: str = "rejected") -> None:
        """Retain a rejection and stop execution without invented speech.

        The workflow artifacts retain the rejected draft. The application owns
        the error and shutdown. The user request remains pending for recovery.
        """
        envelope = batch[0]
        self.bus.put(new_envelope(self.module_id, "telemetry", "action.receipt", envelope.correlation_id,
                               {"status": status, "workflow_name": self.WORKFLOW_NAME,
                                "rejection_reason": reason, "recovery_outcome": "not_published"},
                               causation_id=envelope.id))
        raise RuntimeError(f"Public reply {status} in {self.module_id}: {reason}")

    def accept_reply(self, text: str, batch: list[Envelope], *,
                     tool_receipts: list[dict[str, JsonValue]] | None = None, recovery_outcome: str = "",
                     review_reason: str = "", review_judgment: str = "") -> None:
        """Commit an evidence-guarded reply, or raise with its rejection evidence.

        ``tool_receipts`` are already JSON-shaped receipt mappings; they cross
        the bus with the published reply. ``review_reason`` records why the
        agentic review accepted this reply.
        """
        from pm.memory.snapshot import GhostConflict
        envelope = batch[0]
        request = ActionRequest.model_validate(envelope.payload)
        try:
            if self.capability_provider is not None:
                current = self.capability_provider().get("minecraft", {})
                prior = self._public_context.capabilities.get("minecraft", {})
                if (current.get("status"), current.get("tools")) != (prior.get("status"), prior.get("tools")):
                    raise GhostConflict("Minecraft capability changed during public generation.")
            if self._public_context.task_receipt.get("task_verified"):
                from workflows.public_speech.reply_recovery import task_evidence
                try:
                    current_task = task_evidence(self._public_context.task_receipt)
                except (OSError, ValueError, KeyError, TypeError, yaml_util.YAMLError) as error:
                    self._public_context.task_receipt = {**self._public_context.task_receipt,
                        "task_verified": False, "verification_error": type(error).__name__}
                    raise GhostConflict("Task evidence became unreadable during public generation.") from error
                if not current_task.get("task_verified"):
                    self._public_context.task_receipt = current_task
                    raise GhostConflict("Verified task bytes changed during public generation.")
            reply_id = commit_public_reply(
                self.ghost, text, envelope.correlation_id, commit_id=self.invocation_id or new_id(),
                action_id=None if recovery_outcome else request.knoxel_id, context=self._public_context,
                addressee_entity_id=request.partner_entity_id,
                recovery=recovery_outcome, review_reason=review_reason, review_judgment=review_judgment,
                cancel=self.stop_event,
            )
        except GhostConflict as error:
            self.reject_reply(batch, str(error), status="stale")
            return
        if reply_id is not None:
            publish_reply(self.bus, self.module_id, envelope.correlation_id, reply_id, cause=envelope.id,
                          action_id=request.action_id, tool_receipts=tool_receipts or [],
                          recovery_outcome=recovery_outcome, review_reason=review_reason,
                          review_judgment=review_judgment)

