"""Run real console exchanges, inspect accepted evidence, and restart the worker."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--persona", choices=("spamuel", "cheerful"), default="spamuel")
    parser.add_argument("--port", type=int, default=8093)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--request-task", action="store_true", help="Include the task-request scenario, which can expose a missing public reply.")
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT / ".runtime"):
        raise ValueError("Use a fresh console directory under .runtime.")
    out.mkdir(parents=True, exist_ok=False)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    seed = ROOT / "test_data/personas/spamuel/canon.yaml"
    if args.persona == "cheerful":
        import yaml
        from pm.memory.store import MemoryStore
        from pm.records.cycle_state import PersonaConfig
        seed = out / "cheerful-seed.yaml"
        ghost = MemoryStore(PersonaConfig.model_validate(yaml.safe_load(
            (ROOT / "test_data/personas/cheerful/config.yaml").read_text(encoding="utf-8"))))
        seed.write_text(yaml.safe_dump(ghost.state_dict()), encoding="utf-8")
    base = f"http://127.0.0.1:{args.port}"
    log = (out / "server.log").open("w", encoding="utf-8")
    process = subprocess.Popen([sys.executable, "-u", "-m", "pm.console.server", "--port", str(args.port),
        "--root", str(out / "instances")], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    instance = None
    events = []
    after = 0
    receipt = {"persona": args.persona, "seed": str(seed), "seed_sha256": hashlib.sha256(seed.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(), "exchanges": [],
        "scope": "console HTTP and worker smoke; human immersion and full W7 acceptance are separate"}

    def request(path: str, payload=None):
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(base + path, data=data,
            headers={"Content-Type": "application/json", "Origin": base})
        with urllib.request.urlopen(req, timeout=120) as response:
            return json.load(response)

    def wait_for(predicate):
        nonlocal after
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            batch = request(f"/api/instances/{instance}?after={after}&light=1")
            events.extend(batch["events"])
            if batch["events"]:
                after = batch["events"][-1]["seq"]
            if predicate(events, batch["instance"]):
                return batch["instance"]
            errors = [event for event in events if event["kind"] in {"application.error", "protocol.error"}]
            if errors or batch["instance"]["exit_code"] not in (None, 0):
                raise RuntimeError(errors or batch["instance"])
            time.sleep(.5)
        raise TimeoutError("The real console did not reach the expected boundary. Inspect events and worker.log.")

    def command(action: str, **fields):
        return request("/api/command", {"instance": instance, "action": action, **fields})

    def exchange(text: str):
        count = sum(event["kind"] == "chat.reply" for event in events)
        start = time.monotonic()
        command("chat", session="a01", user="Jesse", text=text)
        wait_for(lambda saved, info: sum(event["kind"] == "chat.reply" for event in saved) > count)
        reply = [event for event in events if event["kind"] == "chat.reply"][-1]
        receipt["exchanges"].append({"input": text, "reply": reply, "seconds": time.monotonic() - start})
        print(json.dumps(receipt["exchanges"][-1], ensure_ascii=False), flush=True)

    try:
        deadline = time.monotonic() + 30
        while True:
            try:
                request("/api/instances")
                break
            except urllib.error.URLError:
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("Console HTTP startup failed. Inspect server.log.")
                time.sleep(.2)
        instance = request("/api/deploy", {"name": f"A01 {args.persona}", "seed": str(seed),
            "config": str(args.config.resolve()), "cognition": "real"})["id"]
        receipt["instance"] = instance
        command("start")
        wait_for(lambda saved, info: any(event["kind"] == "ready" for event in saved))
        if args.request_task:
            exchange("Hey. Our project codeword is copper lantern. Start the local_task workflow to write a note that says copper lantern.")
            exchange("How did the copper lantern task go? Tell me its real status.")
        else:
            exchange("Hey. Our project codeword today is copper lantern. Reply to me about that name.")
            exchange("What is our codeword? Do you like the name?")
        command("snapshot")
        before = request(f"/api/instances/{instance}")["snapshot"]
        (out / "before-restart.json").write_text(json.dumps(before), encoding="utf-8")
        command("stop")
        wait_for(lambda saved, info: info["state"] == "stopped")
        ready_count = sum(event["kind"] == "ready" for event in events)
        command("start")
        wait_for(lambda saved, info: sum(event["kind"] == "ready" for event in saved) > ready_count)
        command("snapshot")
        restored = request(f"/api/instances/{instance}")["snapshot"]
        (out / "after-restart.json").write_text(json.dumps(restored), encoding="utf-8")
        receipt["restart"] = {"records_equal": before["records"] == restored["records"],
            "gwt_equal": before["gwt"] == restored["gwt"], "runs_equal": before["runs"] == restored["runs"],
            "records": len(restored["records"]), "runs": restored["runs"]}
        if not all(receipt["restart"][key] for key in ("records_equal", "gwt_equal", "runs_equal")):
            raise AssertionError("Accepted memory or task evidence changed across restart.")
        exchange("What was our project codeword before we restarted?")
        receipt["status"] = "passed"
    except BaseException as error:
        receipt["status"] = "failed"
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if instance is not None:
            try:
                command("stop")
                wait_for(lambda saved, info: info["state"] == "stopped")
            except (OSError, TimeoutError, RuntimeError):
                command("kill")
        process.terminate()
        process.wait(timeout=15)
        log.close()
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
        (out / "observed-events.json").write_text(json.dumps(events, ensure_ascii=False), encoding="utf-8")
        print(out / "receipt.json", flush=True)


if __name__ == "__main__":
    main()
