"""Ordinary conversational selection while main cognition calculates asynchronously.

The host reads recovery-safe accepted guidance and independently classified
partner input. Its real kernel reuses character decision and routing. Local
speech and quiet choices become append-only recovery roots. Consequential
choices remain proposals for main cognition. This host changes no intention,
appraisal, wait, or main workspace.

The critical publication epoch and source meaning fence selection under the
store lock. Speech still belongs to PublicDialogue. Its accepted action request
survives restart without another choice. Quiet receipts suppress reply recovery.
Consequential receipts return as distinct main stimuli, so original dialogue
does not receive a second appraisal merely to request deliberation.

Read schemas.py for frozen contracts, steps.py for choice, and records.py for
the shared durable receipt projection. Provider failures retry visibly. Source
conflicts requeue original causes with newer input. Internal errors propagate.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import JsonValue

from pm.bus.envelopes import Envelope, new_envelope
from pm.constants import CYCLE_RETRY_DELAY_SECONDS, CYCLE_RETRY_MAX_DELAY_SECONDS
from pm.llm.concurrency import InferenceRole, inference_role
from pm.llm.http_client import ModelError
from pm.llm.structured import ModelFormatError
from pm.memory.bash_view import MemoryBashEnvironment
from pm.memory.checkpoint import CommitReceipt
from pm.memory.read_scope import ReadScope
from pm.memory.snapshot import GhostConflict
from pm.memory.store import MemoryStore
from pm.records.base import Provenance, RecordScope
from pm.records.enums import ActionType, FeatureType, InterlocusType, SourceKind
from pm.records.lineage import RecordLineage
from pm.records.memory_records import ActionSchemaRecord, Feature
from pm.runs.workflow_module import CognitiveWorkflow
from workflows.input_triage.host import build_input_request
from workflows.input_triage.schemas import InputSourcePayload
from workflows.interaction.records import interaction_owned_sources, interaction_receipts
from workflows.interaction.schemas import GuidanceDisposition, InteractionMetadata, InteractionOutcome, InteractionReceipt, InteractionRequest
from workflows.interaction.steps import prepare_work
from workflows.main_cognition.action_request import partner_key, request_shards
from workflows.main_cognition.cycle_commits import active_intentions
from workflows.main_cognition.schemas import ActionRequest, WorkflowOption
from workflows.main_cognition.story_prefix import shard_label
from workflows.public_speech.dependencies import capture_dependencies, validate_dependencies
from workflows.public_speech.publication_control import ReplySuperseded

if TYPE_CHECKING:
    from workflows.cognitive_control.host import CognitiveControl


@dataclass(frozen=True, slots=True)
class PartnerScope:
    """One ordinary exchange's identity and reality boundary."""

    entity_id: int | None
    scope: RecordScope


class Interaction(CognitiveWorkflow):
    """Own ordinary local choices, leaving main state and public effects to their owners."""

    WORKFLOW_NAME = "interaction"
    DATA_TOPICS = frozenset({"user.input"})
    ghost: MemoryStore
    cognitive_control: CognitiveControl
    workflow_catalogue_provider: Callable[[], list[WorkflowOption]]

    def validate_selection(self, selection_id: str) -> None:
        """Reject a selected action after a newer local choice replaces its sources."""
        if not selection_id:
            return
        with self.ghost.lock:
            view = self.ghost.recovery_snapshot(capture=False)
            owners = interaction_receipts(view)
            matching = [record for record in view.all_knoxels.values()
                        if isinstance(record, Feature) and "interaction_receipt" in record.metadata
                        and InteractionMetadata.model_validate(record.metadata).interaction_receipt.selection_id == selection_id]
            if len(matching) != 1:
                raise ValueError("A selected ordinary action requires exactly one durable selection receipt.")
            receipt = InteractionMetadata.model_validate(matching[0].metadata).interaction_receipt
            if any(owners[key].selection_id != selection_id for key in receipt.source_ids):
                raise ReplySuperseded("A newer ordinary choice superseded this selected reply.")

    def remaining_inputs(self, batch: list[Envelope]) -> list[Envelope]:
        """Exclude classified Critical inputs and sources already selected by an owner."""
        with self.ghost.lock:
            view = self.ghost.recovery_snapshot(capture=False)
            receipts = interaction_receipts(view)
            owned = interaction_owned_sources(view)
            claimed = {source_id for state in self.ghost.read_cycles()
                       if state.tick_id in view.gwt_turns and view.gwt_turns[state.tick_id].slot_id == "main"
                       for source_id in state.claimed_ids()}
            result: list[Envelope] = []
            for envelope in batch:
                source_id = InputSourcePayload.model_validate(envelope.payload).knoxel_id
                if view.get(source_id).metadata.get("exchange_status") == "resolved":
                    continue
                if source_id in receipts:
                    continue
                assessment = self.cognitive_control.triage_result([source_id])
                if assessment is None or assessment.items[0].category == "Critical":
                    continue
                if source_id in claimed and source_id not in owned:
                    # Cold-start and pre-B03 input already passed main selection.
                    continue
                result.append(envelope)
            return result

    def handle_batch(self, batch: list[Envelope]) -> None:
        """Wait for real guidance, then select each partner's coalesced exchange."""
        source_ids = [InputSourcePayload.model_validate(item.payload).knoxel_id for item in batch]
        if not self.cognitive_control.wait_for_triage(source_ids, self.stop_event):
            return
        while not self.stop_event.is_set():
            with self.ghost.lock:
                view = self.ghost.recovery_snapshot(capture=False)
                has_guidance = any(turn.slot_id == "main" for turn in view.gwt_turns.values())
            if has_guidance:
                break
            self.stop_event.wait(0.05)
        remaining = self.remaining_inputs(batch)
        if not remaining:
            return
        # The shared builder derives attribution and scope from actual records.
        with self.ghost.lock:
            view = self.ghost.recovery_snapshot(capture=False)
            request = build_input_request(self.ghost, view, remaining, run_id="")
        groups: dict[PartnerScope, list[Envelope]] = {}
        for envelope, stimulus in zip(remaining, request.cycle.stimuli, strict=True):
            key = PartnerScope(stimulus.entity_id, view.get(stimulus.knoxel_id).scope)
            groups.setdefault(key, []).append(envelope)
        for partner_batch in groups.values():
            self._select_batch(self.coalesce_pending(partner_batch))

    def coalesce_pending(self, batch: list[Envelope]) -> list[Envelope]:
        """Carry an older unsent local reply into the newest exchange's selection."""
        with self.ghost.lock:
            view = self.ghost.recovery_snapshot(capture=False)
            base = build_input_request(self.ghost, view, batch, run_id="")
            partner_id = base.cycle.stimuli[0].entity_id
            scopes = [view.get(item.knoxel_id).scope for item in base.cycle.stimuli]
            sources = {InputSourcePayload.model_validate(item.payload).knoxel_id: item for item in batch}
            for source_id, receipt in interaction_receipts(view).items():
                if receipt.outcome != InteractionOutcome.Reply or source_id in sources:
                    continue
                source = view.get(source_id)
                if (receipt.dependencies.partner_entity_id != partner_id or source.scope not in scopes
                        or source.metadata.get("exchange_status") != "pending"):
                    continue
                sources[source_id] = new_envelope(self.module_id, "data", "user.input",
                    str(source.metadata["exchange_correlation"]), {"knoxel_id": source_id})
            return [sources[source_id] for source_id in sorted(sources)]

    def _select_batch(self, batch: list[Envelope]) -> None:
        """Retry provider output, or coalesce a source conflict with queued newer input."""
        delay = CYCLE_RETRY_DELAY_SECONDS
        while not self.stop_event.is_set():
            if not self.cognitive_control.wait_for_reconsideration(self.stop_event):
                return
            remaining = self.remaining_inputs(batch)
            if not remaining:
                return
            batch = self.coalesce_pending(remaining)
            try:
                with inference_role(InferenceRole.Interaction):
                    super().handle_batch(batch)
                return
            except (GhostConflict, ReplySuperseded) as error:
                self.bus.put(new_envelope(self.module_id, "telemetry", "interaction.superseded", "",
                    {"reason": str(error), "source_ids": [InputSourcePayload.model_validate(item.payload).knoxel_id
                                                         for item in batch]}))
                for envelope in batch:
                    self.deliver(envelope)
                return
            except (ModelError, ModelFormatError) as error:
                self.bus.put(new_envelope(self.module_id, "telemetry", "interaction.rejected", "",
                    {"reason": str(error), "retry_delay_seconds": delay}))
                if self.stop_event.wait(delay):
                    return
                delay = min(delay * 2, CYCLE_RETRY_MAX_DELAY_SECONDS)

    def prepare_input(self, batch: list[Envelope]) -> dict[str, JsonValue]:
        """Freeze recovery-safe evidence and accepted guidance for the real kernel."""
        assert isinstance(self.last_environment, MemoryBashEnvironment)
        view = self.last_environment.ghost_view
        assert view is not None, "Ordinary interaction requires its frozen memory view."
        with self.ghost.lock:
            base = build_input_request(self.ghost, view, batch, run_id=self.invocation_id or "")
            previous = base.cycle.previous_state
            if previous is None:
                raise ValueError("Ordinary interaction requires an accepted main cycle.")
            source_ids = [item.knoxel_id for item in base.cycle.stimuli]
            base.cycle.early_triage = self.cognitive_control.triage_result(source_ids)
            base.cycle.available_workflows = self.workflow_catalogue_provider()
            base.cycle.capabilities = list(self.ghost.ghost_config.supported_capabilities)
            partner = base.cycle.stimuli[0].entity_id
            dependencies = capture_dependencies(view, sorted(base.evidence.record_fingerprints),
                guidance_tick=previous.tick_id, critical_epoch=self.cognitive_control.publication_epoch(),
                partner_entity_id=partner, scopes=[view.get(key).scope for key in source_ids])
            identity = json.dumps(base.source_fingerprints, sort_keys=True).encode("utf-8")
            selection_id = "interaction-" + hashlib.sha256(identity).hexdigest()[:24]
            intention_ids = [goal.id for goal in active_intentions(view) if goal.scope in dependencies.scopes][:8]
            request = InteractionRequest(selection_id=selection_id, cycle=base.cycle,
                                         dependencies=dependencies, intention_ids=intention_ids)
            self._request: InteractionRequest = request
        return request.model_dump(mode="json")

    def publish_output(self, report: dict[str, JsonValue], batch: list[Envelope]) -> None:
        """Accept selection before any speech dispatch or reconsideration event."""
        proposed = InteractionReceipt.model_validate(report)
        accepted = self.accept_selection(proposed)
        record = self.ghost.snapshot({accepted.added_ids[-1]}).get(accepted.added_ids[-1])
        receipt = InteractionMetadata.model_validate(record.metadata).interaction_receipt
        if receipt.action_request is not None:
            request = receipt.action_request
            self.bus.put(new_envelope(self.module_id, "data", "action.request",
                self._request.cycle.correlation_id, request.payload(), target=request.workflow_name,
                causation_id=batch[0].id))
        if receipt.outcome == InteractionOutcome.Reconsider or receipt.decision.guidance_disposition != GuidanceDisposition.Follow:
            self.bus.put(new_envelope(self.module_id, "data", "interaction.reconsider",
                self._request.cycle.correlation_id, {"knoxel_id": record.id}, causation_id=batch[0].id))
        self.bus.put(new_envelope(self.module_id, "telemetry", "interaction.accepted",
            self._request.cycle.correlation_id, {"knoxel_id": record.id, "selection_id": receipt.selection_id,
                                               "outcome": receipt.outcome.value}))

    def accept_selection(self, receipt: InteractionReceipt) -> CommitReceipt:
        """Atomically fence the epoch and source meaning, then retain an append-only root."""
        with self.ghost.lock, ReadScope() as acceptance_reads:
            acceptance_reads.collecting = False
            if receipt.selection_id in self.ghost.applied_commits:
                return self.ghost.applied_commits[receipt.selection_id]
            control = self.cognitive_control
            control.validate_publication(receipt.dependencies.critical_epoch, receipt.dependencies.guidance_tick)
            validate_dependencies(self.ghost, receipt.dependencies, current_partner_input=True)
            current_intentions = [goal.id for goal in active_intentions(self.ghost.recovery_snapshot(capture=False))
                                  if goal.scope in receipt.dependencies.scopes][:8]
            if current_intentions != self._request.intention_ids:
                raise GhostConflict("Accepted governing intentions changed during ordinary selection.")
            if all(self.ghost.snapshot({key}).get(key).metadata.get("exchange_status") == "resolved"
                   for key in receipt.source_ids):
                raise ReplySuperseded("Every source of this ordinary choice already has a public reply.")
            batch = self.ghost.begin_batch(receipt.selection_id, embed=False)
            batch.preserve_on_recovery()
            batch.depend_on([source.record_id for source in receipt.dependencies.sources],
                            manifest={"kind": "interaction", "dependencies": receipt.dependencies.model_dump(mode="json")})
            batch.depend_on_commits(receipt.dependencies.input_commit_ids)
            for source in receipt.dependencies.sources:
                batch.expect_source(source)
            request = self._request.cycle
            if request.previous_state is None:
                raise ValueError("Ordinary selection lost its accepted main state.")
            lineage = RecordLineage(episode_id=receipt.selection_id, operation="ordinary_interaction",
                                    source_event_ids=receipt.source_ids, intention_ids=list(self._request.intention_ids))
            scope = self.ghost.snapshot({receipt.source_ids[0]}).get(receipt.source_ids[0]).scope
            if receipt.dispatch is not None:
                dispatch = receipt.dispatch
                partner = partner_key(dispatch.group, request.user_entity_id)
                action_request = ActionRequest(action_id=dispatch.action_id, workflow_name=dispatch.workflow_name,
                    content=dispatch.content, arguments=dispatch.arguments, group=dispatch.group,
                    partner_entity_id=int(partner.ref) if partner is not None else None,
                    partner_name=shard_label(partner, request) if partner is not None else request.user_name,
                    stimuli=list(request.stimuli),
                    shards=request_shards(request.previous_state.workspace, dispatch.group, partner,
                                          lambda key: shard_label(key, request)),
                    gwt_tick=receipt.dependencies.guidance_tick,
                    presentation=dispatch.presentation, workspace=request.state_summary,
                    action_origin="interaction", interaction_selection_id=receipt.selection_id,
                    interaction_dependencies=receipt.dependencies)
                action = batch.create(ActionSchemaRecord(content=dispatch.content, action_type=ActionType.Reply,
                    scope=scope, intention_ids=list(self._request.intention_ids),
                    lineage=lineage, metadata={"action_id": dispatch.action_id, "workflow_name": dispatch.workflow_name,
                        "dispatch_status": "pending", "dispatch_target": dispatch.workflow_name,
                        "dispatch_correlation": request.correlation_id,
                        "presentation": dispatch.presentation.model_dump(mode="json"),
                        "evidence_ids": receipt.source_ids}))
                action_request.knoxel_id = action.id
                action.metadata["dispatch_payload"] = action_request.payload()
                from workflows.main_cognition.host import stage_expectations
                stage_expectations(batch, batch.view(), prepare_work(self._request), dispatch, action.id)
                receipt = receipt.model_copy(update={"action_request": action_request})
            original = "\n".join(f"[{item.knoxel_id}] {item.source}: {item.text}" for item in request.stimuli)
            decision = receipt.decision.variants[0] if receipt.decision.variants else "No local move."
            route_summary = ""
            if receipt.route is not None:
                route_summary = (f"\nProposed action: {receipt.route.action_type}, {receipt.route.workflow_name}. "
                                 f"{receipt.route.reasoning}")
            batch.create(Feature(content=f"Ordinary character choice: {decision}\nEvidence:\n{original}\n"
                                 f"Guidance use: {receipt.decision.guidance_disposition.value}. "
                                 f"{receipt.decision.guidance_reason}{route_summary}",
                feature_type=FeatureType.StateUpdate, interlocus=InterlocusType.PrivateInternal,
                provenance=Provenance(source_kind=SourceKind.Self, source_ref="interaction"),
                scope=scope,
                lineage=lineage, metadata={"interaction_receipt": receipt.model_dump(mode="json"),
                                           "exchange_correlation": request.correlation_id}))
            return self.ghost.commit(batch)

    def resume(self) -> None:
        """Republish unapplied consequential nominations without another local choice."""
        with self.ghost.lock:
            view = self.ghost.recovery_snapshot(capture=False)
            claimed = {key for state in self.ghost.read_cycles() for key in state.claimed_ids()}
            pending = [record for record in view.all_knoxels.values()
                       if isinstance(record, Feature) and "interaction_receipt" in record.metadata
                       and record.id not in claimed]
        for record in pending:
            receipt = InteractionMetadata.model_validate(record.metadata).interaction_receipt
            if receipt.outcome == InteractionOutcome.Reconsider or receipt.decision.guidance_disposition != GuidanceDisposition.Follow:
                self.bus.put(new_envelope(self.module_id, "data", "interaction.reconsider",
                    str(record.metadata["exchange_correlation"]), {"knoxel_id": record.id}))
        # A local choice does not claim main appraisal. Restore that independent
        # cause even when quiet selection or pending speech suppresses reply recovery.
        for source_id in interaction_receipts(view):
            if source_id not in claimed:
                source = view.get(source_id)
                self.bus.put(new_envelope("user_input", "data", "user.input",
                    str(source.metadata["exchange_correlation"]), {"knoxel_id": source_id}))
