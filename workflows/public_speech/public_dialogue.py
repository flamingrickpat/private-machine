"""``PublicDialogue``: the shared bus module of Story and Assistant replies.

It validates evidence and commits each utterance before publication. Story
and Assistant subclass it. Each workflow reviews its draft once. Errors reach
the application; host failure strings never become companion speech.
"""
from __future__ import annotations

from pm.utils import yaml_util

from pm.bus.envelopes import Envelope, new_id
from workflows.main_cognition.schemas import ActionMetadata, ActionRequest, Stimulus
from workflows.public_speech.public_context import build_public_context
from workflows.public_speech.shard_context import shard_context
from workflows.public_speech.publication import commit_public_reply, publish_reply
from pm.bus.envelopes import new_envelope
from pm.runs.workflow_module import CognitiveWorkflow


def _clip(text: str, limit: int = 500) -> str:
    text = str(text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "\u2026"


class PublicDialogue(CognitiveWorkflow):
    """Shared evidence guards and visible failures for public workflows."""

    capability_provider = None

    # Shared embedding service for context planning. The assembly sets it;
    # without it, planning scores lexically.
    context_embeddings = None

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

    def public_context(self, request: ActionRequest, planned=None):
        """Freeze the public context of one reply and keep it for the publication guard."""
        # Refresh the durable private ontology before the snapshot. The review
        # reads the terms and bindings from the frozen public context.
        from workflows.public_speech.ontology import observe_ontology
        observe_ontology(self.ghost)
        capabilities = self.capability_provider() if self.capability_provider else None
        with self.ghost.lock:
            self._public_context = build_public_context(self.ghost, request, capabilities=capabilities,
                                                        planned=planned)
        return self._public_context

    def reply_context(self, request: ActionRequest):
        """Return the reply context from the workspace shards of the request.

        The context holds the verbatim transcript with the partner, the shard
        story, and the evidence (:func:`~workflows.public_speech.shard_context.shard_context`).
        No memory is retrieved again.
        """
        with self.ghost.lock:
            view = self.ghost.snapshot()
        return shard_context(view, request, self.ghost.get_companion_name())

    def handle_batch(self, batch: list[Envelope]) -> None:
        # Each public action owns one report and one exchange resolution.
        # A second envelope must not disappear inside a merged mailbox batch.
        for envelope in batch:
            request = ActionRequest.model_validate(envelope.payload)
            action = self.ghost.get_knoxel_by_id(request.knoxel_id) if request.knoxel_id is not None else None
            if action is not None and ActionMetadata.model_validate(action.metadata).public_reply_id is not None:
                continue
            # Preparation can fail before a new snapshot exists. A prior task
            # receipt must never become the recovery evidence for this action.
            self._public_context = None
            super().handle_batch([envelope])

    def reject_reply(self, batch, reason: str, *, status: str = "rejected") -> None:
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

    def accept_reply(self, text: str, batch, *, tool_receipts=None, recovery_outcome="",
                     review_reason="", review_judgment="") -> None:
        """Commit an evidence-guarded reply, or raise with its rejection evidence.

        ``review_reason`` records why the agentic review accepted this reply.
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

