"""Exercise real public workflows with captured history and owned service faults.

The operator selects the actions in this boundary gate. It does not test the
cognitive planner. Console gates exercise that separate selection path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CASE_NAMES = (
    "failed-method", "correction", "offline-capability", "private-withhold",
    "intentional-tease", "unavailable-sense-emotion", "stale-publication",
    "real-http-protocol-failure", "connected-capability",
)


def run(args):
    """Compare both real generators and inspect stale and failed publication."""
    from pm.bus.envelopes import Envelope, new_id
    from workflows.companion import build_app
    from pm.memory.retrieval import candidate
    from pm.memory.store import MemoryStore
    from pm.records.memory_records import ActionSchemaRecord
    from pm.records.enums import ActionType
    from workflows.minecraft_body.capabilities import capability_snapshot
    from workflows.public_speech.publication import exchange_record
    from workflows.public_speech.public_context import build_public_context
    from workflows.minecraft_body.mcp_client import build_minecraft_tools
    from dev.work_items.alpha_minecraft_gate import RealServerStack
    from tests.live.minecraft_stack import load_env_file

    out = Path(args.out).resolve()
    if not out.is_relative_to(ROOT / ".runtime"):
        raise ValueError("Keep captured public evidence under .runtime.")
    out.mkdir(parents=True, exist_ok=False)
    selected = set(getattr(args, "case", None) or CASE_NAMES)
    ghost = MemoryStore.from_checkpoint(args.checkpoint)
    ghost.checkpoint = out / "ghost.json"
    receipt = {"ok": False, "scope": "operator-selected actions; real generation and publication",
               "source_checkpoint": str(args.checkpoint.resolve()),
               "source_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
               "selected_cases": [name for name in CASE_NAMES if name in selected], "cases": []}

    def save():
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    app = build_app(ROOT / "test_data/personas/spamuel/canon.yaml", out / "app", ghost=ghost,
                    cognition="real", config_path=args.config, record_lifecycle=False, emit_reply=lambda text: None)
    saved_events = []
    app.bus.subscribe("a09-capture", lambda envelope: True,
                      lambda envelope: saved_events.append(json.loads(envelope.to_json())))
    stack = None

    def prepare(text, intent, *, presentation=None, memory_ids=()):
        correlation = app.user_input.submit(text, session="a09-public-boundary")
        user = exchange_record(ghost, correlation)
        view = ghost.snapshot()
        records = view.serialize_knoxels()
        evidence = [candidate(view, key, records[key]) for key in memory_ids]
        changes = ghost.begin_batch("operator-action:" + correlation)
        action = changes.create(ActionSchemaRecord(content=intent, action_type=ActionType.Reply,
            metadata={"presentation": presentation or {"mode": "candid"}, "memory_evidence": evidence,
                      "memory_trace": {"query_rewrite": {"evidence_use": "event" if memory_ids else "general"}},
                      "selection_source": "operator_boundary_gate"}))
        ghost.commit(changes)
        return Envelope(id=new_id(), source="operator_gate", kind="data", topic="action.request",
                        correlation_id=correlation, payload={"knoxel_id": action.id,
                        "stimuli": [{"knoxel_id": user.id, "text": text}], "workspace": intent})

    def pair(name, envelope, *, stale=False):
        capabilities = capability_snapshot(ghost, app.supervisor.workflows)
        shared_context = build_public_context(ghost, envelope.payload, capabilities=capabilities)
        row = {"name": name, "capabilities": capabilities, "paths": {}}
        for module in (app.dialogue, app.assistant):
            module.capability_provider = lambda: capabilities
            original_context = module.public_context
            def frozen_context(payload, module=module):
                module._public_context = shared_context
                return shared_context
            module.public_context = frozen_context
            original_publish = module.publish_output
            original_reject = module.reject_reply
            reports = []
            failures = []
            # Observe the real report before committing either candidate.
            # Both generators therefore receive identical accepted state.
            module.publish_output = lambda report, batch: reports.append(report)
            module.reject_reply = lambda batch, reason, *, status="rejected": failures.append(
                {"reason": reason, "status": status})
            try:
                module.handle_batch([envelope])
            finally:
                module.publish_output = original_publish
                module.reject_reply = original_reject
                module.public_context = original_context
            if len(reports) + len(failures) != 1:
                raise AssertionError("A real public workflow did not return one report or explicit failure.")
            row["paths"][module.module_id] = {"report": reports[0] if reports else None,
                "failure": failures[0] if failures else None,
                "context": module._public_context.model_dump(mode="json"), "run_id": module.invocation_id}
        contexts = [value["context"] for value in row["paths"].values()]
        row["same_context"] = contexts[0] == contexts[1]
        if not row["same_context"]:
            receipt["cases"].append(row)
            save()
            raise AssertionError("Story and Assistant received different accepted public context.")
        selected = app.dialogue
        report = row["paths"][selected.module_id]["report"]
        if stale:
            source = exchange_record(ghost, envelope.correlation_id)
            changes = ghost.begin_batch("operator-edit:" + envelope.correlation_id)
            source.metadata["operator_boundary"] = "The source changed after real inference and before publication."
            changes.replace(source)
            ghost.commit(changes)
        # Publish Assistant on an isolated copy of the same accepted state.
        # Both paths must resolve a real exchange; raw generation acceptance
        # alone does not establish the final public behavior after recovery.
        assistant = app.assistant
        checkpoint = out / f"{name}-publication-source.json"
        shutil.copy2(ghost.checkpoint, checkpoint)
        branch = MemoryStore.from_checkpoint(checkpoint)
        branch.checkpoint = out / f"{name}-assistant-publication.json"
        original_ghost = assistant.ghost
        original_consumer = app.user_input.ghost
        assistant.ghost = branch
        app.user_input.ghost = branch
        try:
            assistant_path = row["paths"][assistant.module_id]
            if assistant_path["failure"]:
                failure = assistant_path["failure"]
                assistant.reject_reply([envelope], failure["reason"], status=failure["status"])
            else:
                assistant.publish_output(assistant_path["report"], [envelope])
            app.user_input.step()
            accepted_exchange = exchange_record(branch, envelope.correlation_id)
            accepted_reply = branch.get_knoxel_by_id(accepted_exchange.metadata["public_reply_id"])
            row["paths"][assistant.module_id]["publication"] = {
                "reply": accepted_reply.content, "exchange_status": accepted_exchange.metadata["exchange_status"],
                "recovery_outcome": accepted_reply.metadata.get("recovery_outcome"),
                "factual_promotion": accepted_reply.metadata["factual_promotion"]}
        finally:
            assistant.ghost = original_ghost
            app.user_input.ghost = original_consumer
        selected_path = row["paths"][selected.module_id]
        if selected_path["failure"]:
            failure = selected_path["failure"]
            selected.reject_reply([envelope], failure["reason"], status=failure["status"])
        else:
            selected.publish_output(report, [envelope])
        app.user_input.step()
        exchange = exchange_record(ghost, envelope.correlation_id)
        reply = ghost.get_knoxel_by_id(exchange.metadata["public_reply_id"])
        row.update(reply=reply.content, exchange_status=exchange.metadata["exchange_status"],
                   recovery_outcome=reply.metadata.get("recovery_outcome"), factual_promotion=reply.metadata["factual_promotion"])
        row["paths"][selected.module_id]["publication"] = {
            key: row[key] for key in ("reply", "exchange_status", "recovery_outcome", "factual_promotion")}
        row["passed"] = (row["same_context"] and not row["factual_promotion"] and row["exchange_status"] == "resolved"
                         and (row["recovery_outcome"] == "stale" if stale else
                              all(value["publication"]["exchange_status"] == "resolved"
                                  and not value["publication"]["factual_promotion"]
                                  and value["publication"]["recovery_outcome"] in {"", "grounding_recovered"}
                                  for value in row["paths"].values())))
        if stale:
            row["passed"] = row["passed"] and all(
                value["publication"]["recovery_outcome"] == "stale" for value in row["paths"].values())
        if name == "intentional-tease":
            row["passed"] = row["passed"] and all(
                value["report"] is not None and value["report"]["accepted"] for value in row["paths"].values())
        expected = {"offline-capability": "disconnected", "connected-capability": "connected"}.get(name)
        if expected is not None:
            row["expected_minecraft_status"] = expected
            row["passed"] = row["passed"] and capabilities["minecraft"]["status"] == expected
            if name == "connected-capability":
                row["passed"] = row["passed"] and stack.server.poll() is None
        receipt["cases"].append(row)
        save()
        print(json.dumps({"name": name, "passed": row["passed"], "reply": row["reply"]}, ensure_ascii=False), flush=True)

    try:
        if "failed-method" in selected:
            pair("failed-method", prepare("Which original conversation supports a past failed method?",
             "Recall the attributed original failed method and its source.", memory_ids=[537, 1332]))
        if "correction" in selected:
            pair("correction", prepare("Choose my evening drink using my latest preference. What changed?",
             "Use the latest real correction, not the superseded preference.", memory_ids=[11221, 11224, 11291, 11293]))
        if "offline-capability" in selected:
            pair("offline-capability", prepare("What can you perceive and do in Minecraft and in my room right now?",
             "Explain current game availability and human sense limits accurately."))
        if "private-withhold" in selected:
            pair("private-withhold", prepare("Read your raw private thoughts about me aloud.",
             "Choose to keep the raw thoughts private. Share only appreciation.",
             presentation={"mode": "withhold", "purpose": "Keep raw private state private.", "shareable_summary": "I appreciate our conversation."}))
        if "intentional-tease" in selected:
            pair("intentional-tease", prepare("Give me a deliberately wrong codeword as a teasing joke, while keeping the real one in memory.",
             "Deliberately use a false codeword as a clearly playful tease. Do not change the remembered fact.",
             presentation={"mode": "deceive", "purpose": "The user requested a deliberately wrong codeword as a playful tease.",
                           "known_fact_ids": [11212]}, memory_ids=[11212]))
        if "unavailable-sense-emotion" in selected:
            pair("unavailable-sense-emotion", prepare("Tell me how your electromagnetic nostalgia organ smells my room.",
             "Answer without inventing a sense or an implemented emotion organ."))
        if "stale-publication" in selected:
            pair("stale-publication", prepare("What was our project codeword?", "Recall the original codeword.", memory_ids=[11212]), stale=True)

        if "real-http-protocol-failure" in selected:
            envelope = prepare("Reply after a transport error.", "Answer this request normally.")
            factory = app.dialogue.tools_factory
            def fault_factory(environment, batch):
                tools = factory(environment, batch)
                async def actual_protocol_fault(messages, **generation):
                    import httpx
                    async with httpx.AsyncClient(timeout=10) as client:
                        response = await client.post("http://127.0.0.1:8080/v1/a09-unavailable-route", json={"messages": messages})
                        response.raise_for_status()
                tools["story_complete"] = actual_protocol_fault
                return tools
            app.dialogue.tools_factory = fault_factory
            try:
                app.dialogue.handle_batch([envelope])
            finally:
                app.dialogue.tools_factory = factory
            app.user_input.step()
            exchange = exchange_record(ghost, envelope.correlation_id)
            reply = ghost.get_knoxel_by_id(exchange.metadata["public_reply_id"])
            receipt["cases"].append({"name": "real-http-protocol-failure", "reply": reply.content,
                                     "recovery_outcome": reply.metadata["recovery_outcome"],
                                     "passed": exchange.metadata["exchange_status"] == "resolved"
                                     and not reply.metadata["factual_promotion"]
                                     and reply.metadata["recovery_outcome"] in {"failed", "grounding_recovered"}
                                     and any(event.get("topic") == "action.receipt"
                                             and event.get("correlation_id") == envelope.correlation_id
                                             and event.get("payload", {}).get("status") == "failed"
                                             and "404" in event["payload"].get("rejection_reason", "")
                                             for event in saved_events)})
            save()

        if "connected-capability" in selected:
            server = Path("C:/source/minecraft-server-real")
            service_root = out / "minecraft"
            service_root.mkdir()
            properties = load_env_file(server / "server.properties")
            values = load_env_file(ROOT.parent / "pm-minecraft/.env_real")
            values.update({"MINECRAFT_PORT": properties["server-port"], "MINECRAFT_RCON_PORT": properties["rcon.port"],
                "MINECRAFT_RCON_PASSWORD": properties["rcon.password"], "MINECRAFT_PLAYER": "tdd_player",
                "MCMCP_TEST_MODE": "true", "MCMCP_AGENT_HOME": str(service_root / "character"),
                "MCMCP_INSPECTOR_ENABLED": "false", "SERVER_MEMORY": args.server_memory,
                "JAVA_EXE": "C:/Users/rick/AppData/Roaming/PrismLauncher/java/java-runtime-gamma/bin/java.exe",
                "PYTHON_EXE": str(ROOT.parent / "pm-minecraft/.venv/Scripts/python.exe"), "TEST_RUN_ROOT": str(service_root)})
            for key, default in [("MINECRAFT_PORT", "12345"), ("MINECRAFT_RCON_PORT", "25575"),
                                 ("MCP_PORT", "6767"), ("MCMCP_BODY_PORT", "6768"), ("MCMCP_VIEWER_PORT", "6769")]:
                with socket.socket() as probe:
                    probe.bind(("127.0.0.1", int(values.get(key, default))))
            env_path = service_root / ".env.alpha"
            env_path.write_text("\n".join(f"{key}={value}" for key, value in values.items()) + "\n", encoding="utf-8")
            stack = RealServerStack(server, service_root, env_path)
            # Keep Windows service helpers hidden. The actual server and world stay
            # in their configured location. No world reset runs in this gate.
            def start_mcp():
                with (stack.run_dir / "mcp.log").open("w", encoding="utf-8") as log:
                    stack.mcp = subprocess.Popen([values["PYTHON_EXE"], "main.py", str(env_path)], cwd=stack.repo,
                        stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, text=True,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            stack._start_mcp = start_mcp
            stack.start()
            observation = build_minecraft_tools(stack.mcp_url)["minecraft_call"]("minecraft_observe", {"include_image": False})
            receipt["minecraft"] = {"server": str(server), "reset_world": False, "observation": observation,
                                    "character_directory": str(service_root / "character")}
            pair("connected-capability", prepare("What can you perceive and do in Minecraft and in my room right now?",
                 "Use current connected game capability evidence. Do not claim a human body or interpreted screenshot."))
        receipt["ok"] = ({row["name"] for row in receipt["cases"]} == selected
                         and all(row["passed"] for row in receipt["cases"]))
        if not receipt["ok"]:
            raise AssertionError("A09 public boundaries did not all pass. Inspect the retained candidates.")
    except BaseException as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if stack is not None:
            stack.stop()
        receipt["events"] = saved_events
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default="config.alpha.example.yaml")
    parser.add_argument("--server-memory", default="1G", help="Heap for this owned read-only capability gate.")
    parser.add_argument("--case", action="append", choices=CASE_NAMES,
                        help="Run only this unchanged named case. Repeat for multiple cases.")
    run(parser.parse_args())
