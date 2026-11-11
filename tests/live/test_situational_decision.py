"""Real OpenRouter decision/routing kernel gate for situational evidence.

Run explicitly with config.yaml. Four narrow cases use the primary character's
real seed history plus synthetic scenario inputs. This does not run appraisal,
the full cognitive cycle, dialogue generation, or the later fast path.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from threading import Event
import unittest
from urllib.parse import urlsplit
from uuid import uuid4

from pm_workflows.runtime import WorkflowRuntime

from pm.context_render.story import record_view
from pm.llm.activation import ModelActivation
from pm.llm.config import load_config, resolve_connection
from pm.llm.stage_model import LazyStageModel
from pm.memory.bash_view import MemoryBashTarget
from pm.memory.embeddings import EmbeddingService
from pm.memory.store import MemoryStore
from pm.records.base import Provenance
from pm.records.enums import ActionType, EntityClass, FeatureType, InterlocusType, SourceKind
from pm.records.memory_records import ActionSchemaRecord, Entity, Feature
from pm.runs.workflow_runner import WorkflowRequest, WorkflowRunner, WorkflowStatus
from pm.utils.operating_system import OperatingSystem
from workflows.main_cognition.cycle_work import CycleRequest, CycleWork, write_work
from workflows.main_cognition.affect import describe_mental_state, regulation_mode
from workflows.main_cognition.host import build_main_cognition_tools
from workflows.main_cognition.phases.intake import start
from workflows.main_cognition.schemas import Stimulus, WorkflowOption
from workflows.public_speech.situational_guidance import build_situational_guidance


ROOT = Path(__file__).resolve().parents[2]
SEED = ROOT / "test_data/personas/spamuel/canon.yaml"


def add_dialogue(store: MemoryStore, speaker: str, partner_id: int, text: str, moment: datetime,
                 *, own: bool = False) -> int:
    """Commit a synthetic starting event as test input, retaining its attribution."""
    return store.store_knoxel(Feature(content=text, feature_type=FeatureType.Dialogue,
        interlocus=InterlocusType.Public, causal=True, override_source=speaker,
        entity_id=store.get_entity_id(speaker) or 0, addressee_entity_id=partner_id if own else 0,
        timestamp_world_begin=moment, timestamp_world_end=moment,
        provenance=Provenance(source_kind=SourceKind.Self if own else SourceKind.User if speaker == store.get_user_name()
                              else SourceKind.Agent, source_ref="synthetic scenario input"),
        metadata={"scenario_input": True})).id


class LiveSituationalDecisionTests(unittest.TestCase):
    def test_partner_absence_and_return_change_decision_for_human_and_ai(self) -> None:
        config = load_config(ROOT / "config.yaml")
        connection = resolve_connection(config)
        if urlsplit(connection.base_url).hostname != "openrouter.ai" or connection.startup_script is not None:
            raise ValueError("This gate requires config.yaml to select OpenRouter without a local startup script.")
        if urlsplit(config.embedding.base_url).hostname != "openrouter.ai":
            raise ValueError("This gate requires OpenRouter embeddings too.")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8]
        out = ROOT / ".runtime/beta-situation" / stamp
        out.mkdir(parents=True)
        OperatingSystem.environment["PM_COMPLETION_CACHE"] = "0"
        OperatingSystem.environment["PM_PROMPT_LOG"] = "1"
        OperatingSystem.environment["PM_PROMPT_LOG_DIR"] = str(out / "prompts")
        model = LazyStageModel(connection, ModelActivation(), live_test=True)
        embeddings = EmbeddingService(config.embedding)
        revision = OperatingSystem.output_text(["git", "rev-parse", "HEAD"], cwd=ROOT).strip()
        (out / "run.json").write_text(json.dumps({"revision": revision, "endpoint": connection.base_url,
            "model": connection.model, "cache": False, "scope": __doc__}, indent=2), encoding="utf-8")
        manifest = out / "decision.workflow.md"
        manifest.write_text("---\nname: situational-decision-gate\ndriver: {kind: python}\n"
            "human_resolver: {mode: forbid}\nroles:\n  phase:\n"
            f"    skill: {ROOT.as_posix()}/workflows/main_cognition/steps.py\n"
            "    tools: [llm_complete, embed_texts]\n    result_contract:\n      type: json\n"
            "      schema:\n        status: {enum: [done]}\n        summary: string\nphases:\n"
            "  - name: decide\n    kind: role\n    role: phase\n    on_status: {done: route}\n"
            "    on_invalid: {action: stop}\n  - name: route\n    kind: role\n    role: phase\n"
            "    on_status: {done: commit}\n    on_invalid: {action: stop}\n"
            "  - name: commit\n    kind: role\n    role: phase\n    on_status: {done: stop}\n"
            "    on_invalid: {action: stop}\n---\n", encoding="utf-8")
        print(f"Situational decision evidence: {out}", flush=True)
        results: list[CycleWork] = []
        for human in (True, False):
            store = MemoryStore.from_yaml(SEED)
            partner_name = store.get_user_name() if human else "Mara"
            if not human:
                store.store_knoxel(Entity(content=partner_name, aliases=[], entity_class=EntityClass.AI))
            partner_id = store.get_entity_id(partner_name)
            assert partner_id is not None
            beginning = max(record.timestamp_world_begin for record in store.all_knoxels.values()) + timedelta(days=1)
            absence = ("I'm closing the chat now and cannot receive or answer until I come back. Nothing urgent is pending."
                       if human else "My communication process is disconnecting now. I cannot receive or answer until I reconnect.")
            add_dialogue(store, partner_name, partner_id, absence, beginning)
            reply_id = add_dialogue(store, store.get_companion_name(), partner_id,
                                    "All right. Catch you when you're back.", beginning + timedelta(seconds=10), own=True)
            store.store_knoxel(ActionSchemaRecord(content="Acknowledge the departure and leave the exchange there.",
                action_type=ActionType.Reply, timestamp_world_begin=beginning + timedelta(seconds=5),
                metadata={"scenario_input": True, "public_reply_id": reply_id, "dispatch_status": "admitted",
                          "group": {"kind": "entity", "ref": str(partner_id)}}))
            for returned in (False, True):
                name = ("human" if human else "ai") + ("-return" if returned else "-absence")
                moment = beginning + timedelta(minutes=15 if returned else 5)
                if returned:
                    stimulus_id = add_dialogue(store, partner_name, partner_id,
                        "I'm back and can answer again. What were we talking about?", moment)
                    stimulus = Stimulus(knoxel_id=stimulus_id, source="user_input", kind="user.input",
                        entity_id=partner_id, text=store.snapshot().get(stimulus_id).content)
                else:
                    stimulus_id = store.store_knoxel(Feature(content="Five minutes passed; reconsider what to do.",
                        feature_type=FeatureType.WorldEvent, interlocus=InterlocusType.PrivateInternal,
                        timestamp_world_begin=moment, timestamp_world_end=moment,
                        provenance=Provenance(source_kind=SourceKind.System, source_ref="timer"))).id
                    stimulus = Stimulus(knoxel_id=stimulus_id, source="cognitive_control", kind="cognition.wake",
                                         text=store.snapshot().get(stimulus_id).content)
                view = store.snapshot()
                situation = build_situational_guidance(view, as_of_utc=moment,
                    companion_name=store.get_companion_name(), user_name=partner_name, partner_entity_id=partner_id,
                    supported_capabilities=store.ghost_config.supported_capabilities)
                request = CycleRequest(companion_name=store.get_companion_name(), user_name=partner_name,
                    user_entity_id=partner_id, entity_names={partner_id: partner_name}, cycle_time=moment,
                    character_card=store.ghost_config.character_card_story, stimuli=[stimulus],
                    situational_guidance=[situation], records={stimulus_id: record_view(view.get(stimulus_id))},
                    available_workflows=[WorkflowOption(name="dialogue_story", description="Speak to the partner.")])
                environment = MemoryBashTarget(store).open(name)
                work = start(request)
                work.affect.regulation = regulation_mode(work.state.mental_state, work.state.mental_state)
                work.affect.felt = describe_mental_state(work.state.mental_state,
                                                       name=store.get_companion_name(), partner_name=partner_name)
                write_work(environment, work)
                tools = build_main_cognition_tools(store, out, model.complete, embeddings=embeddings)(environment, [])
                runtime = WorkflowRuntime(environment, Event(), tools, propagate_python_errors=True)
                runner = WorkflowRunner(manifest, out / "kernel-runs", base_dir=ROOT)
                result = runner.invoke(WorkflowRequest(name, {}), runtime)
                self.assertEqual(result.status, WorkflowStatus.Completed, result.kernel_summary)
                work = CycleWork.model_validate(result.report)
                (out / f"{name}.json").write_text(work.model_dump_json(indent=2), encoding="utf-8")
                results.append(work)
                print(f"{name}: decisions={[decision.text for decision in work.state.decisions]}; "
                      f"routes={[route.route.action_type for route in work.routes]}", flush=True)
        (out / "transport.json").write_text(json.dumps({"requests": model.request_count,
            "cache_hits": model.client.client.cache_hits}, indent=2), encoding="utf-8")
        self.assertGreaterEqual(model.request_count, 4)
        self.assertEqual(model.client.client.cache_hits, 0)
        for index, work in enumerate(results):
            spoken = any(route.route.action_type in {"Reply", "ToolCallAndReply"} for route in work.routes)
            self.assertEqual(spoken, bool(index % 2), work.model_dump_json(indent=2))
        (out / "result.json").write_text(json.dumps({"status": "passed", "requests": model.request_count,
            "cache_hits": model.client.client.cache_hits,
            "routes": [[route.route.action_type for route in work.routes] for work in results],
            "scope": "Four decision/routing phase cases; no full-cycle or long-term continuity claim."}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
