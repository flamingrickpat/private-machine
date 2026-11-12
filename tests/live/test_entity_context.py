"""Four real OpenRouter inputs through immediate entity maintenance, before main.

The explicit synthetic fixture contains Spamuel's card, two identity anchors,
and a labelled legacy Berlin prior. This gate samples evidence maintenance;
it does not establish character quality or full-cycle action selection.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback
import unittest
from urllib.parse import urlsplit
from uuid import uuid4

from pydantic import BaseModel

from pm import vclock
from pm.bus.bus import BusSystem
from pm.llm.activation import ModelActivation
from pm.llm.config import load_config, resolve_connection
from pm.llm.stage_model import LazyStageModel
from pm.memory.bash_view import MemoryBashTarget
from pm.memory.store import MemoryStore
from pm.records.base import RecordBase, RecordScope
from pm.records.copying import pickle_copy
from pm.records.cycle_state import CycleStateModel
from pm.utils.operating_system import OperatingSystem
from workflows.cognitive_control.host import CognitiveControl
from workflows.entity_context.owner import EntityContextOwner, active_claims, receipt_chain
from workflows.entity_context.projection import project_entity_context, render_entity_context
from workflows.entity_context.schemas import EntityContextKey, EntityContextView, EntityProjectionQuery
from workflows.input_triage.host import InputTriage
from workflows.input_triage.schemas import EarlyTriageAssessment
from workflows.main_cognition.host import build_main_cognition_tools
from workflows.public_speech.user_input import UserInputProxy


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class MessageScenario:
    """One genuine incoming message and its supported speaker path."""

    label: str
    speaker: str | None
    text: str


class MessageReceipt(BaseModel):
    """Actual accepted sources and compact interpretation before any main cycle."""

    label: str
    source_id: int
    speaker_entity_id: int
    correlation_id: str
    assessment: EarlyTriageAssessment
    entity_view: EntityContextView
    rendered_view: str


class LiveEntityContextTests(unittest.TestCase):
    """Exercise real generated reports, corrections, freshness, peers, and reload."""

    def test_four_immediate_entity_updates_and_clean_restart(self) -> None:
        config = load_config(ROOT / "config.yaml")
        connection = resolve_connection(config)
        if (urlsplit(connection.base_url).hostname != "openrouter.ai"
                or connection.startup_script is not None
                or urlsplit(config.embedding.base_url).hostname != "openrouter.ai"):
            raise ValueError("NOT RUN — this gate requires OpenRouter chat/embeddings without startup scripts.")
        out = ROOT / ".runtime/beta-b04/live" / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8])
        out.mkdir(parents=True)
        print(f"B04 four-message live evidence: {out}", flush=True)
        OperatingSystem.environment["PM_COMPLETION_CACHE"] = "0"
        OperatingSystem.environment["PM_PROMPT_LOG_DIR"] = str(out / "prompts")
        canonical = MemoryStore.from_yaml(ROOT / "test_data/personas/spamuel/canon.yaml")
        identities: list[RecordBase] = [pickle_copy(record) for record in canonical.all_entities
                      if record.content in {canonical.get_user_name(), canonical.get_companion_name()}]
        for identity in identities:
            identity.embedding = []
        persona = canonical.ghost_config.model_copy(deep=True)
        persona.character_card_story += "\n\nExplicit synthetic legacy background prior for this gate: Jesse lives in Berlin."
        seed = CycleStateModel(knoxels=identities, cycles=[], **persona.model_dump())
        store = MemoryStore.from_state(seed, checkpoint=out / "ghost.pickle")
        (out / "fixture.json").write_text(json.dumps({
            "kind": "Explicit synthetic two-identity fixture; no historical dialogue",
            "character_card": persona.character_card_story,
            "identities": [item.model_dump(mode="json") for item in identities],
            "original_checkpoint_used": False,
        }, indent=2), encoding="utf-8")
        revision = OperatingSystem.output_text(["git", "rev-parse", "HEAD"], cwd=ROOT).strip()
        (out / "run.json").write_text(json.dumps({"started_at_utc": datetime.now(timezone.utc).isoformat(),
            "revision": revision, "model": connection.model, "endpoint": connection.base_url,
            "embedding_model": config.embedding.model, "embedding_endpoint": config.embedding.base_url,
            "scope": __doc__, "prior_known_b03_cost_usd": 0.21543439976,
            "cancelled_prior_usage": "unknown", "main_cycles_enabled": False}, indent=2), encoding="utf-8")
        model = LazyStageModel(connection, ModelActivation(), live_test=True)
        clock = vclock.VirtualClock(datetime(2026, 11, 12, 12, 0, tzinfo=timezone.utc))
        previous_clock = vclock.active()
        vclock.install(clock)
        bus = BusSystem()
        control = CognitiveControl("control", bus, self.fail, time.monotonic, None, ghost=store)
        control.enable_input_triage()
        owner = EntityContextOwner(store)
        control.entity_context_owner = owner
        module = InputTriage("input_triage", bus, self.fail, time.monotonic, None, ghost=store,
            working_target=MemoryBashTarget(store, "read", recovery_only=True),
            workflow=ROOT / "workflows/input_triage/input_triage.workflow.md", runtime_root=out,
            tools_factory=build_main_cognition_tools(store, out, model.complete,
                                                   cancel_provider=lambda: module.run_stop))
        module.cognitive_control = control
        module.entity_context_owner = owner
        bus.subscribe(module.module_id, module.accepts, module.deliver)
        intake = UserInputProxy("user_input", bus, self.fail, time.monotonic, store, lambda text: None)
        intake.register_triage_input = control.register_input
        observer_id = store.get_entity_id(store.get_companion_name())
        user_id = store.get_entity_id(store.get_user_name())
        assert observer_id is not None and user_id is not None
        receipts: list[MessageReceipt] = []

        def capture(subject_id: int, *, selected_store: MemoryStore = store) -> EntityContextView:
            """Project all accepted claims from this exact subject family."""
            view = selected_store.recovery_snapshot(capture=False)
            key = EntityContextKey(subject_entity_id=subject_id, observer_entity_id=observer_id, scope=RecordScope())
            claims = active_claims(view, key=key)
            return project_entity_context(view, query=EntityProjectionQuery(key=key,
                claim_keys=list(dict.fromkeys(item.claim_key for item in claims)), token_budget=2400),
                as_of_utc=vclock.utcnow())

        messages = [
            MessageScenario("away", None, "I am stepping away and cannot answer for the next ten minutes, until 12:10 UTC today. "
             "I can still receive your messages, but I will not be reading or replying during that interval."),
            MessageScenario("brief_return", None, "Correction: I can answer this now, briefly, but I am still occupied. "
             "That corrects my earlier claim that I cannot respond. This brief availability ends at 12:10 UTC today."),
            MessageScenario("city_correction", None, "I am back and can respond now. Also, the old background saying I live in Berlin "
             "is outdated: I now live in Hamburg. Berlin was my previous city."),
            MessageScenario("sparse_peer", "Mara", "Hi Spamuel, Mara here. I am only dropping off this note; I cannot respond until 14:00 UTC today."),
        ]
        try:
            for scenario in messages:
                label = scenario.label
                if label == "city_correction":
                    clock.set(datetime(2026, 11, 12, 12, 11, tzinfo=timezone.utc))
                    stale = capture(user_id)
                    rendered = render_entity_context(stale)
                    (out / "time_advance.json").write_text(stale.model_dump_json(indent=2), encoding="utf-8")
                    (out / "time_advance.txt").write_text(rendered, encoding="utf-8")
                    self.assertIn("Current ability to respond: unknown", rendered)
                    self.assertTrue(any(item.applicability in {"expired", "due_for_review"} for item in stale.claims))
                before_user = capture(user_id)
                print(f"B04 genuine input: {label}", flush=True)
                correlation = intake.submit(scenario.text, speaker=scenario.speaker)
                source = next(item for item in store.all_features
                              if item.metadata.get("exchange_correlation") == correlation)
                self.assertTrue(module.step())
                assessment = next(EarlyTriageAssessment.model_validate(item.metadata["assessment"])
                    for item in store.all_features
                    if item.metadata.get("control_kind") == "input_triage"
                    and source.id in EarlyTriageAssessment.model_validate(item.metadata["assessment"]).source_ids)
                entity_view = capture(source.entity_id)
                receipt = MessageReceipt(label=label, source_id=source.id, speaker_entity_id=source.entity_id,
                    correlation_id=correlation, assessment=assessment, entity_view=entity_view,
                    rendered_view=render_entity_context(entity_view))
                receipts.append(receipt)
                (out / f"{label}.json").write_text(receipt.model_dump_json(indent=2), encoding="utf-8")
                print(receipt.rendered_view, flush=True)
                outcomes = assessment.entity_outcomes
                assert outcomes is not None
                self.assertTrue(outcomes)
                self.assertTrue(all(item.disposition == "staged" for item in outcomes))
                new_claims = [item.claim for item in entity_view.claims
                              if any(evidence.source.record_id == source.id for evidence in item.claim.evidence)]
                self.assertTrue(new_claims, "Real extraction produced no accepted claim for the new message.")
                for claim in new_claims:
                    self.assertNotEqual(claim.epistemic_kind, "observation")
                    if claim.epistemic_kind == "report":
                        self.assertTrue(any(item.exact_quote is not None and item.exact_quote in source.content
                                            for item in claim.evidence))
                if label == "away":
                    self.assertIn("Current ability to respond: cannot", receipt.rendered_view)
                elif label == "brief_return":
                    chain = receipt_chain(store.recovery_snapshot(capture=False), entity_view.key)
                    old_ids = {item.claim.claim_id for item in before_user.claims
                               if item.claim.interaction is not None and item.claim.interaction.dimension == "respond"}
                    self.assertTrue(old_ids.intersection(chain[-1].receipt.proposal.retract_claim_ids),
                                    "The explicit correction must name an existing responding claim.")
                    self.assertIn("Current ability to respond: can", receipt.rendered_view)
                elif label == "city_correction":
                    self.assertTrue(any("Hamburg" in item.statement for item in new_claims))
                    self.assertIn("Berlin", store.ghost_config.character_card_story)
                elif label == "sparse_peer":
                    self.assertNotEqual(source.entity_id, user_id)
                    self.assertEqual(before_user, capture(user_id), "Peer maintenance changed Jesse's family.")
                reopened = MemoryStore.from_checkpoint(out / "ghost.pickle")
                self.assertEqual(entity_view, capture(source.entity_id, selected_store=reopened))
                rebuilt = CognitiveControl("rebuilt", BusSystem(), self.fail, time.monotonic, None, ghost=reopened)
                rebuilt.enable_input_triage()
                self.assertTrue(rebuilt.has_assessment(source.id))
                self.assertFalse(store.states, "No main cycle may run in this narrow gate.")
            (out / "result.json").write_text(json.dumps({"status": "passed", "model": connection.model,
                "endpoint": connection.base_url, "revision": revision, "requests": model.request_count,
                "source_ids": [item.source_id for item in receipts], "clean_reload_after_each_input": True,
                "main_cycles": 0, "original_checkpoint_used": False,
                "scope": __doc__}, indent=2), encoding="utf-8")
            print(f"B04 live gate passed; requests={model.request_count}; evidence={out}", flush=True)
        except BaseException as error:
            (out / "failure.json").write_text(json.dumps({"error_type": type(error).__name__,
                "reason": str(error), "traceback": traceback.format_exc(), "requests": model.request_count}, indent=2),
                encoding="utf-8")
            raise
        finally:
            vclock.install(previous_clock)
