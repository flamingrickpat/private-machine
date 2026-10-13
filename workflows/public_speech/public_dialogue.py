"""``PublicDialogue``: the shared bus module of Story and Assistant replies.

It guards stale replies and commits each reply before publication. Story
and Assistant subclass it. The one agentic review happens inside each
workflow role (``speech_review``), so a rejected reply here is rare: it means
a stale snapshot, a failed run, or a false task result.
"""
from __future__ import annotations

import yaml

from pm.bus.envelopes import Envelope, new_id
from workflows.public_speech.public_context import build_public_context
from workflows.public_speech.publication import commit_public_reply, exchange_record, publish_reply
from pm.bus.envelopes import new_envelope
from pm.runs.workflow_module import CognitiveWorkflow


def _recency(knoxel) -> tuple:
    return (getattr(knoxel, "tick_id", 0), getattr(knoxel, "id", 0))


def _clip(text: str, limit: int = 500) -> str:
    text = str(text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "\u2026"


class PublicDialogue(CognitiveWorkflow):
    """Shared stale guards and observable recovery for public workflows."""

    capability_provider = None

    # Shared embedding service for context planning. The assembly sets it;
    # without it, planning scores lexically.
    context_embeddings = None

    def public_context(self, payload, planned=None):
        # Refresh the durable private ontology before the snapshot. The review
        # reads the terms and bindings from the frozen public context.
        from workflows.public_speech.ontology import observe_ontology
        observe_ontology(self.ghost)
        capabilities = self.capability_provider() if self.capability_provider else None
        with self.ghost.lock:
            self._public_context = build_public_context(self.ghost, payload, capabilities=capabilities,
                                                        planned=planned)
        return self._public_context

    def plan_context(self, payload: dict, plan_name: str):
        """Plan the reply's memory context from the cycle's workspace.

        The workspace experience, the current message, the winning attention
        nominations, and the active goals form the focus. Records that the
        nominations target, and records that cognition selected, are required.
        The current stimuli are excluded; the prompt shows them as the current line.
        Runs on the module thread; embedding calls happen outside the lock.
        """
        import asyncio
        from workflows.main_cognition.stage_commits import active_goal_texts
        from workflows.public_speech.context_planning import FocusInput, plan_reply_context
        stimuli = [item for item in payload.get("stimuli", []) if "knoxel_id" in item]
        attention = list(payload.get("attention", []))
        with self.ghost.lock:
            view = self.ghost.snapshot()
            goals = active_goal_texts(self.ghost)[:6]
        focus = FocusInput(
            workspace=str(payload.get("workspace", "")),
            current_message=str(stimuli[-1].get("text", "")) if stimuli else str(payload.get("content", "")),
            attention_texts=[str(item.get("text", "")) for item in attention],
            goals=goals,
            required_ids=[int(key) for item in attention for key in item.get("target_ids", [])]
                         + [int(item["knoxel_id"]) for item in payload.get("memory_evidence", [])],
            exclude_ids=[int(item["knoxel_id"]) for item in stimuli],
        )
        return asyncio.run(plan_reply_context(view, plan_name, focus, self.context_embeddings,
                                              cancel=getattr(self, "stop_event", None)))

    def handle_batch(self, batch: list[Envelope]) -> None:
        # Each public action owns one report and one exchange resolution.
        # A second envelope must not disappear inside a merged mailbox batch.
        for envelope in batch:
            exchange = exchange_record(self.ghost, envelope.correlation_id)
            if exchange is not None and exchange.metadata.get("exchange_status") == "resolved":
                continue
            # Preparation can fail before a new snapshot exists. A prior task
            # receipt must never become the recovery evidence for this action.
            self._public_context = None
            try:
                super().handle_batch([envelope])
            except Exception as error:
                if self.stop_event.is_set() or self.run_stop.is_set():
                    return
                self.reject_reply([envelope], f"{type(error).__name__}: {error}", status="failed")

    def reject_reply(self, batch, reason: str, *, status: str = "rejected") -> None:
        """Retain a failed receipt and close the exchange with explicit recovery."""
        envelope = batch[0]
        stale = status == "stale"
        text = ("That reply became stale before I could send it. Please ask again."
                if stale else "That reply failed before I could send it. Please ask again."
                if status == "failed" else "I could not ground that reply in the available evidence. Please ask again.")
        context = getattr(self, "_public_context", None)
        if status in {"rejected", "failed"} and context is not None:
            from pm.memory.snapshot import GhostConflict
            from workflows.public_speech.publication import validate_public_context
            try:
                with self.ghost.lock:
                    validate_public_context(self.ghost, context)
            except GhostConflict as error:
                status, stale, reason = "stale", True, str(error)
                text = "That reply became stale before I could send it. Please ask again."
        if context is not None and context.task_receipt:
            task = context.task_receipt
            if task.get("task_verified"):
                spec = task["report"]["note_spec"]
                text = f"The note passed exact byte verification at {spec['path']}: {spec['text']!r}. My conversational reply failed validation."
            else:
                text = (f"The {task['workflow_name']} worker is {task['state']}. "
                        "Its receipt does not verify the requested note. My conversational reply failed validation.")
        reply_id = commit_public_reply(self.ghost, text, envelope.correlation_id,
                                      commit_id=f"public-recovery:{envelope.id}", recovery=status,
                                      cancel=self.stop_event)
        self.bus.put(new_envelope(self.module_id, "telemetry", "action.receipt", envelope.correlation_id,
                               {"status": status, "workflow_name": self.WORKFLOW_NAME,
                                "rejection_reason": reason, "recovery_outcome": "published" if reply_id else "already_resolved",
                                "public_reply_id": reply_id}, causation_id=envelope.id))
        if reply_id is not None:
            publish_reply(self.bus, self.module_id, envelope.correlation_id, reply_id,
                          cause=envelope.id, recovery_outcome=status)

    def accept_reply(self, text: str, batch, *, tool_receipts=None, recovery_outcome="",
                     review_reason="", review_judgment="") -> None:
        """Commit an evidence-guarded reply, or expose its stale recovery result.

        ``review_reason`` records why the agentic review accepted this reply.
        """
        from pm.memory.snapshot import GhostConflict
        envelope = batch[0]
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
                except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
                    self._public_context.task_receipt = {**self._public_context.task_receipt,
                        "task_verified": False, "verification_error": type(error).__name__}
                    raise GhostConflict("Task evidence became unreadable during public generation.") from error
                if not current_task.get("task_verified"):
                    self._public_context.task_receipt = current_task
                    raise GhostConflict("Verified task bytes changed during public generation.")
            reply_id = commit_public_reply(
                self.ghost, text, envelope.correlation_id, commit_id=self.invocation_id or new_id(),
                action_id=None if recovery_outcome else envelope.payload.get("knoxel_id"), context=self._public_context,
                recovery=recovery_outcome, review_reason=review_reason, review_judgment=review_judgment,
                cancel=self.stop_event,
            )
        except GhostConflict as error:
            self.reject_reply(batch, str(error), status="stale")
            return
        if reply_id is not None:
            publish_reply(self.bus, self.module_id, envelope.correlation_id, reply_id, cause=envelope.id,
                          action_id=envelope.payload.get("action_id", ""), tool_receipts=tool_receipts or [],
                          recovery_outcome=recovery_outcome, review_reason=review_reason,
                          review_judgment=review_judgment)

