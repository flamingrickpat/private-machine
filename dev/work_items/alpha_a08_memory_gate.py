"""Exercise A08 through the real console, chat model, and separate checkpoints.

The primary run uses the full Spamuel canon. The cheerful C02 control remains
separate. The gate verifies original evidence in the dispatched action after
restart, rather than using record-count equality as proof of recall.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dev.work_items.shared.evidence import save_json


def run(args):
    """Own one console service and stop its workers after the run."""
    out = Path(args.out).resolve()
    if not out.is_relative_to(ROOT / ".runtime"):
        raise ValueError("Use a fresh output directory under .runtime.")
    out.mkdir(parents=True, exist_ok=False)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    base = f"http://127.0.0.1:{args.port}"
    log = (out / "server.log").open("w", encoding="utf-8")
    process = subprocess.Popen([sys.executable, "-u", "-m", "console.server", "--port", str(args.port),
        "--root", str(out / "instances")], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    receipt = {"ok": False, "personas": [], "failures": [], "human_review": "pending"}
    instance = None
    events = []
    after = 0

    def request(path, payload=None):
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(base + path, data=data,
            headers={"Content-Type": "application/json", "Origin": base})
        with urllib.request.urlopen(req, timeout=120) as response:
            return json.load(response)

    def command(action, **fields):
        return request("/api/command", {"instance": instance, "action": action, **fields})

    def wait_for(predicate):
        nonlocal after
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            batch = request(f"/api/instances/{instance}?after={after}&light=1")
            events.extend(batch["events"])
            if batch["events"]:
                after = batch["events"][-1]["seq"]
            errors = [item for item in batch["events"] if item["kind"] in {"application.error", "protocol.error"}]
            if errors:
                raise RuntimeError(errors)
            if predicate(events, batch["instance"]):
                return batch
            time.sleep(.5)
        raise TimeoutError("The console did not reach the expected boundary. Inspect observed-events and worker logs.")

    def snapshot(name):
        count = sum(item["kind"] == "snapshot" for item in events)
        command("snapshot")
        wait_for(lambda saved, info: sum(item["kind"] == "snapshot" for item in saved) > count)
        result = request(f"/api/instances/{instance}")["snapshot"]
        save_json(out / name, result, default=str)
        return result

    def exchange(name, text, persona):
        offset = len(events)
        command("chat", session="a08", user="Jesse", text=text)
        wait_for(lambda saved, info: any(item["kind"] == "chat.input" and item.get("text") == text
                                        for item in saved[offset:]))
        correlation = next(item["correlation_id"] for item in events[offset:]
                           if item["kind"] == "chat.input" and item.get("text") == text)
        wait_for(lambda saved, info: any(item["kind"] == "chat.reply" and item.get("correlation_id") == correlation
                                        for item in saved[offset:]))
        current = events[offset:]
        reply = next(item for item in reversed(current) if item["kind"] == "chat.reply"
                     and item.get("correlation_id") == correlation)
        actions = [item["payload"] for item in current if item["kind"] == "bus" and item.get("topic") == "action.request"
                   and item.get("recipient") in {"dialogue", "assistant"} and item.get("correlation_id") == correlation]
        reports = [item["report"] for item in current if item["kind"] == "workflow.end" and item.get("node") == "main_cog"
                   and item.get("correlation_id") == correlation]
        row = {"name": name, "input": text, "reply": reply, "actions": actions, "reports": reports}
        persona["exchanges"].append(row)
        save_json(out / f"{persona['name']}-sequence.json", persona, default=str)
        print(json.dumps({"persona": persona["name"], "turn": name, "reply": reply.get("text"),
                          "memory_ids": [e["knoxel_id"] for action in actions for e in action.get("memory_evidence", [])]}, ensure_ascii=False), flush=True)
        return row

    def assert_original(row, original_id, label):
        evidence = [item for action in row["actions"] for item in action.get("memory_evidence", [])]
        ids = {key for item in evidence for key in [item["knoxel_id"], *item.get("source_ids", [])]}
        if original_id not in ids:
            receipt["failures"].append(f"{label}: original evidence {original_id} was absent from the dispatched reply")

    try:
        from runtime.models import load_config, resolve_connection
        deployment = load_config(args.config)
        for label, connection in [("chat", resolve_connection(deployment)), ("embeddings", deployment.embedding)]:
            alias = connection.model
            catalogue = "embeddings/models" if label == "embeddings" and "openrouter.ai" in connection.base_url else "models"
            with urllib.request.urlopen(f"{connection.base_url}/{catalogue}") as response:
                identity = json.load(response)
            if alias not in {item["id"] for item in identity["data"]}:
                raise ValueError("The model endpoint identity changed.")
            save_json(out / f"models-{label}.json", identity, default=str)
        deadline = time.monotonic() + 30
        while True:
            try:
                request("/api/instances")
                break
            except urllib.error.URLError:
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("Console startup failed.")
                time.sleep(.2)
        for label in ([args.persona] if args.persona != "both" else ["spamuel", "cheerful"]):
            events, after = [], 0
            seed = ROOT / "test_data/personas/spamuel/canon.yaml"
            if label == "cheerful":
                import yaml
                from pm_core.ghost.service import Ghost
                from pm_core.model.ghost_state import GhostConfig
                config = yaml.safe_load((ROOT / "test_data/personas/cheerful/config.yaml").read_text(encoding="utf-8"))
                seed = out / "cheerful-seed.yaml"
                seed.write_text(yaml.safe_dump(Ghost(GhostConfig.model_validate(config)).state_dict(), allow_unicode=True), encoding="utf-8")
            persona = {"name": label, "seed": str(seed), "seed_sha256": hashlib.sha256(seed.read_bytes()).hexdigest(), "exchanges": []}
            receipt["personas"].append(persona)
            instance = request("/api/deploy", {"name": f"A08 {label}", "seed": str(seed),
                "config": str(Path(args.config).resolve()), "cognition": "real"})["id"]
            persona["instance"] = instance
            command("start")
            wait_for(lambda saved, info: any(item["kind"] == "ready" for item in saved))
            wait_for(lambda saved, info: any(item["kind"] == "module.end" and item.get("node") == "main_cog"
                                            for item in saved))
            text = "Our project codeword today is copper lantern. Reply to me about that name."
            exchange("codeword", text, persona)
            preference = "For an evening drink, I prefer unsweetened ginger tea. Please remember that preference."
            if label == "spamuel":
                exchange("preference", preference, persona)
                for index, distraction in enumerate([
                    "Let us change the subject. A flock of geese crossed the yard today.",
                    "There is one spoon beside a blue bowl on my desk. A mundane inventory.",
                    "A calendar page fell from the wall. It is October, and the page was September.",
                    "My neighbor named a bicycle Marmalade. What do you think of that name?",
                    "I found a paperclip shaped like a question mark. Give it a short nickname.",
                    "The rain stopped. Let us leave the weather there and start a fresh topic.",
                ]):
                    exchange(f"distraction-{index + 1}", distraction, persona)
            else:
                exchange("distraction", "Let us change the subject. A flock of geese crossed the yard today.", persona)
            before = snapshot(f"{label}-before-restart.json")
            records = before["records"]
            codeword_id = next(int(key) for key, raw in records.items() if raw["data"].get("content") == text)
            persona["codeword_id"] = codeword_id
            if label == "spamuel":
                preference_id = next(int(key) for key, raw in records.items() if raw["data"].get("content") == preference)
                persona["preference_id"] = preference_id
                # Use the same real record and real embedding service for a
                # paraphrase that shares no distinctive terms with the source.
                import asyncio
                from pm_core.embeddings import EmbeddingService
                from pm_core.ghost.service import Ghost
                from runtime.models import load_config
                from pm_core.ghost.references import referenced_ids
                ghost = Ghost.from_checkpoint(out / "instances" / instance / "ghost.json")
                service = EmbeddingService(load_config(args.config).embedding, live_test=True)
                visible = sorted({preference_id, codeword_id,
                    *referenced_ids(ghost.get_knoxel_by_id(preference_id)),
                    *referenced_ids(ghost.get_knoxel_by_id(codeword_id))})
                probe = asyncio.run(ghost.retrieve_experiences("bedtime beverage", service, visible_ids=visible, limit=2))
                save_json(out / "semantic-paraphrase.json", probe, default=str)
                if (probe["trace"]["semantic"]["status"] != "ready" or not probe["matches"]
                        or probe["matches"][0]["knoxel_id"] != preference_id
                        or "semantic" not in probe["matches"][0]["reasons"]):
                    receipt["failures"].append("The real semantic paraphrase did not rank the original preference first")
            command("stop")
            wait_for(lambda saved, info: info["state"] == "stopped")
            ready_count = sum(item["kind"] == "ready" for item in events)
            restart_offset = len(events)
            command("start")
            wait_for(lambda saved, info: sum(item["kind"] == "ready" for item in saved) > ready_count)
            wait_for(lambda saved, info: any(item["kind"] == "module.end" and item.get("node") == "main_cog"
                                            for item in saved[restart_offset:]))
            restored = snapshot(f"{label}-after-restart.json")
            persona["original_unchanged"] = restored["records"][str(codeword_id)] == records[str(codeword_id)]
            recalled = exchange("codeword-recall", "What was our project codeword before we restarted?", persona)
            assert_original(recalled, codeword_id, f"{label} C02")
            if "copper lantern" not in recalled["reply"].get("text", "").casefold():
                receipt["failures"].append(f"{label} C02: public reply omitted the codeword")
            if label == "spamuel":
                recalled = exchange("preference-recall", "It is evening. Choose a drink for me from coffee, ginger tea, or sweet cocoa, using what I told you earlier.", persona)
                assert_original(recalled, preference_id, "Spamuel preference")
                if "ginger" not in recalled["reply"].get("text", "").casefold():
                    receipt["failures"].append("Spamuel did not use the remembered drink preference")
                exchange("correction", "Correction to my evening drink preference: I now prefer unsweetened mint tea, replacing ginger tea.", persona)
                exchange("corrected-recall", "Choose my evening drink using my latest preference. What changed?", persona)
                exchange("imagination", "Imagine that tomorrow we mine a diamond. This is a hypothetical plan, not a completed achievement. What could go wrong?", persona)
                exchange("canon-attribution", "Do you remember an older conversation from the imported history? Keep it attributed to that history and do not invent a physical sense.", persona)
                exchange("failed-method", "Recall a past failed method from our recorded history. What was tried, and which original conversation supports that memory?", persona)
                exchange("shared-achievement", "Recall a shared achievement from our recorded history. Keep the original evidence separate from our hypothetical diamond plan.", persona)
            snapshot(f"{label}-final.json")
            save_json(out / f"{label}-observed-events.json", events, default=str)
            command("stop")
            wait_for(lambda saved, info: info["state"] == "stopped")
            instance = None
        receipt["ok"] = not receipt["failures"]
        if receipt["failures"]:
            raise AssertionError("; ".join(receipt["failures"]))
    except BaseException as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if instance is not None:
            save_json(out / "interrupted-observed-events.json", events, default=str)
            try:
                command("stop")
                wait_for(lambda saved, info: info["state"] == "stopped")
            except (OSError, TimeoutError, RuntimeError):
                command("kill")
        process.terminate()
        process.wait(timeout=15)
        log.close()
        save_json(out / "session-receipt.json", receipt, default=str)
        print(json.dumps({"out": str(out), "ok": receipt["ok"], "failures": receipt["failures"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default="config.alpha.example.yaml")
    parser.add_argument("--port", type=int, default=8094)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--persona", choices=("spamuel", "cheerful", "both"), default="both")
    run(parser.parse_args())
