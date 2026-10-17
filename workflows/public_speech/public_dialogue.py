"""``PublicDialogue``: the shared bus module of Story and Assistant replies.

It validates evidence and commits each utterance before publication. Story
and Assistant subclass it. Each workflow reviews its draft once. Errors reach
the application; host failure strings never become companion speech.
"""
from __future__ import annotations

from pm.utils import yaml_util

from pm.bus.envelopes import Envelope, new_id
from workflows.public_speech.public_context import build_public_context
from workflows.public_speech.publication import commit_public_reply, publish_reply
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
    """Shared evidence guards and visible failures for public workflows."""

    capability_provider = None

    # Shared embedding service for context planning. The assembly sets it;
    # without it, planning scores lexically.
    context_embeddings = None

    def user_stimuli(self, payload: dict) -> list[dict]:
        """Return actual user dialogue from the action's stimuli.

        Resolve attribution from accepted records. Old pending actions can lack
        the source fields added to newer dispatches. Private and system events
        never become user speech, even when their text resembles dialogue.
        """
        from pm.records.enums import FeatureType, SourceKind
        result = []
        with self.ghost.lock:
            for item in payload.get("stimuli", []):
                record = self.ghost.get_knoxel_by_id(int(item["knoxel_id"]))
                if record is None:
                    raise KeyError(f"Public stimulus does not exist: {item['knoxel_id']}")
                if (record.feature_type == FeatureType.Dialogue
                        and record.provenance is not None
                        and record.provenance.source_kind == SourceKind.User):
                    result.append({**item, "text": str(record.content)})
        return result

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
        users = self.user_stimuli(payload)
        attention = list(payload.get("attention", []))
        with self.ghost.lock:
            view = self.ghost.snapshot()
            goals = active_goal_texts(self.ghost)[:6]
        focus = FocusInput(
            workspace=str(payload.get("workspace", "")),
            current_message="\n".join(item["text"] for item in users),
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
            action_id = envelope.payload.get("knoxel_id")
            action = self.ghost.get_knoxel_by_id(action_id) if action_id is not None else None
            if action is not None and action.metadata.get("public_reply_id") is not None:
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

