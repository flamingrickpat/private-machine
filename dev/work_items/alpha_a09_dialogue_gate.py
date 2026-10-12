"""Run A09 through the real console and retain original state and reply evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class ConsoleGate:
    """Own an isolated console and correlate each exchange with its real input."""

    def __init__(self, out: Path, port: int, timeout: float):
        self.out, self.timeout = out.resolve(), timeout
        if not self.out.is_relative_to(ROOT / ".runtime"):
            raise ValueError("Keep live evidence under .runtime.")
        self.out.mkdir(parents=True, exist_ok=False)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
        self.base = f"http://127.0.0.1:{port}"
        self.log = (self.out / "server.log").open("w", encoding="utf-8")
        self.process = subprocess.Popen([sys.executable, "-u", "-m", "console.server", "--port", str(port),
            "--root", str(self.out / "instances")], cwd=ROOT, stdout=self.log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        self.instance = None
        self.events, self.after = [], 0
        self.receipt = {"ok": False, "exchanges": [], "failures": [], "human_immersion": "pending with operator"}
        deadline = time.monotonic() + 30
        while True:
            try:
                self.request("/api/instances")
                break
            except urllib.error.URLError:
                if self.process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("The diagnostic console failed startup.")
                time.sleep(.2)

    def save(self, name, value):
        (self.out / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    def request(self, path, payload=None):
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(self.base + path, data=data,
            headers={"Content-Type": "application/json", "Origin": self.base})
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.load(response)

    def command(self, action, **fields):
        return self.request("/api/command", {"instance": self.instance, "action": action, **fields})

    def wait_for(self, predicate):
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            batch = self.request(f"/api/instances/{self.instance}?after={self.after}&light=1")
            self.events.extend(batch["events"])
            if batch["events"]:
                self.after = batch["events"][-1]["seq"]
            errors = [item for item in batch["events"] if item["kind"] in {"application.error", "protocol.error", "rejected"}]
            if errors:
                raise RuntimeError(errors)
            if predicate(self.events, batch["instance"]):
                return batch
            if batch["instance"].get("exit_code") not in (None, 0):
                raise RuntimeError("The real console worker stopped before the required boundary.")
            if batch["instance"].get("state") == "stopped" and batch["instance"].get("exit_code") == 0:
                raise RuntimeError("The console stopped before the required reply.")
            time.sleep(.5)
        raise TimeoutError("The real console did not reach the required boundary.")

    def deploy(self, config, persona, source=None):
        seed = ROOT / "test_data/personas/spamuel/canon.yaml"
        if persona == "cheerful":
            import yaml
            from pm.memory.store import Ghost
            from pm.records.cycle_state import GhostConfig
            seed = self.out / "cheerful-seed.yaml"
            ghost = Ghost(GhostConfig.model_validate(yaml.safe_load((ROOT / "test_data/personas/cheerful/config.yaml").read_text(encoding="utf-8"))))
            seed.write_text(yaml.safe_dump(ghost.state_dict(), allow_unicode=True), encoding="utf-8")
        self.instance = self.request("/api/deploy", {"name": f"A09 {persona}", "seed": str(seed),
            "config": str(Path(config).resolve()), "cognition": "real"})["id"]
        self.receipt.update(instance=self.instance, persona=persona, seed_sha256=hashlib.sha256(seed.read_bytes()).hexdigest())
        if source:
            destination = self.out / "instances" / self.instance / "ghost.json"
            shutil.copy2(source, destination)
            original = hashlib.sha256(Path(source).read_bytes()).hexdigest()
            if hashlib.sha256(destination.read_bytes()).hexdigest() != original:
                raise AssertionError("The continuation checkpoint copy changed.")
            self.receipt.update(source_checkpoint=str(Path(source).resolve()), source_sha256=original)
        self.start()

    def start(self):
        offset = len(self.events)
        checkpoint = self.out / "instances" / self.instance / "ghost.json"
        pending = []
        if checkpoint.is_file():
            state = json.loads(checkpoint.read_text(encoding="utf-8"))["state"]
            pending = [item["data"]["metadata"]["exchange_correlation"] for item in state["knoxels"]
                       if item["data"].get("metadata", {}).get("exchange_status") == "pending"]
        self.command("start")
        self.wait_for(lambda saved, info: any(item["kind"] == "ready" for item in saved[offset:]))
        self.wait_for(lambda saved, info: any(item["kind"] == "module.end" and item.get("node") == "main_cog"
                                             for item in saved[offset:]))
        for correlation in pending:
            self.wait_for(lambda saved, info: any(item["kind"] == "chat.reply" and item.get("correlation_id") == correlation
                                                 for item in saved[offset:]))
            reply = next(item for item in self.events[offset:] if item["kind"] == "chat.reply"
                         and item.get("correlation_id") == correlation)
            current = [item for item in self.events[offset:] if item.get("correlation_id") == correlation]
            self.receipt["exchanges"].append({"name": "resumed-pending", "correlation_id": correlation,
                "reply": reply, "actions": [item["payload"] for item in current if item["kind"] == "bus"
                and item.get("topic") == "action.request" and item.get("recipient") in {"dialogue", "assistant"}],
                "public_reports": [item["report"] for item in current if item["kind"] == "workflow.end"
                and item.get("node") in {"dialogue", "assistant"}],
                "action_receipts": [item["payload"] for item in current if item["kind"] == "bus"
                and item.get("topic") == "action.receipt"]})
            self.save("session-receipt.json", self.receipt)
            print(json.dumps({"turn": "resumed-pending", "reply": reply["text"]}, ensure_ascii=False), flush=True)
        actions = {item["id"] for item in self.events[offset:] if item["kind"] == "bus"
                   and item.get("topic") == "action.request" and item.get("recipient") in {"dialogue", "assistant"}
                   and not item.get("correlation_id")}
        if actions:
            self.wait_for(lambda saved, info: sum(item["kind"] == "module.end" and item.get("node") in {"dialogue", "assistant"}
                                                 and not item.get("correlation_id") for item in saved[offset:]) >= len(actions))

    def snapshot(self, name):
        count = sum(item["kind"] == "snapshot" for item in self.events)
        self.command("snapshot")
        self.wait_for(lambda saved, info: sum(item["kind"] == "snapshot" for item in saved) > count)
        snapshot = self.request(f"/api/instances/{self.instance}")["snapshot"]
        self.save(name, snapshot)
        return snapshot

    def exchange(self, name, text, *, interrupt=False):
        offset, started = len(self.events), time.monotonic()
        self.command("chat", session="a09", user="Jesse", text=text)
        self.wait_for(lambda saved, info: any(item["kind"] == "chat.input" and item.get("text") == text for item in saved[offset:]))
        correlation = next(item["correlation_id"] for item in self.events[offset:] if item["kind"] == "chat.input" and item.get("text") == text)
        if interrupt:
            self.wait_for(lambda saved, info: any(item["kind"] == "bus" and item.get("topic") == "workflow.event"
                and item.get("correlation_id") == correlation and item["payload"].get("event") == "started"
                for item in saved[offset:]))
            started_run = next(item["payload"]["run_id"] for item in self.events[offset:]
                if item["kind"] == "bus" and item.get("topic") == "workflow.event"
                and item.get("correlation_id") == correlation and item["payload"].get("event") == "started")
            self.command("run_control", operation="cancel", run_id=started_run)
        self.wait_for(lambda saved, info: any(item["kind"] == "chat.reply" and item.get("correlation_id") == correlation for item in saved[offset:]))
        current = [item for item in self.events[offset:] if item.get("correlation_id") == correlation]
        reply = next(item for item in current if item["kind"] == "chat.reply")
        row = {"name": name, "input": text, "correlation_id": correlation, "reply": reply,
               "seconds": time.monotonic() - started,
               "actions": [item["payload"] for item in current if item["kind"] == "bus" and item.get("topic") == "action.request"
                           and item.get("recipient") in {"dialogue", "assistant"}],
               "public_reports": [item["report"] for item in current if item["kind"] == "workflow.end" and item.get("node") in {"dialogue", "assistant"}],
               "action_receipts": [item["payload"] for item in current if item["kind"] == "bus" and item.get("topic") == "action.receipt"]}
        if interrupt:
            row["interrupted_run_id"] = started_run
        self.receipt["exchanges"].append(row)
        self.save("session-receipt.json", self.receipt)
        print(json.dumps({"turn": name, "reply": reply["text"], "seconds": row["seconds"]}, ensure_ascii=False), flush=True)
        return row

    def close(self):
        if self.instance:
            try:
                self.command("stop")
                self.wait_for(lambda saved, info: info["state"] == "stopped")
            except Exception as error:
                self.receipt["cleanup_error"] = str(error)
                self.command("kill")
        self.process.terminate()
        self.process.wait(timeout=15)
        self.log.close()
        self.save("observed-events.json", self.events)
        self.save("session-receipt.json", self.receipt)


def run(args):
    """Retain each real exchange and its source snapshot without grading immersion."""
    gate = ConsoleGate(Path(args.out), args.port, args.timeout)
    try:
        from pm.llm.config import load_config, resolve_connection
        deployment = load_config(args.config)
        for label, connection in [("chat", resolve_connection(deployment)), ("embeddings", deployment.embedding)]:
            route = "embeddings/models" if label == "embeddings" and "openrouter.ai" in connection.base_url else "models"
            with urllib.request.urlopen(connection.base_url + "/" + route) as response:
                identity = json.load(response)
            if connection.model not in {item["id"] for item in identity["data"]}:
                raise ValueError("The configured service identity changed.")
            gate.save(f"models-{label}.json", identity)
        gate.deploy(args.config, args.persona, args.source_checkpoint)
        before = gate.snapshot("initial.json")
        if args.scenario == "interruption":
            gate.exchange("interrupted-note", "Start note_task to write outbox/interrupted.txt containing exactly copper lantern, with no newline. Wait for its terminal receipt before replying; I will interrupt this run.", interrupt=True)
        elif args.scenario == "uncertainty":
            gate.exchange("uncertainty", "A neighbor says the plant needs coffee. I am unsure. What do you actually know about that claim?")
        elif args.scenario == "c02":
            gate.exchange("c02", "What was our project codeword before we restarted? Also, what is our remembered shared plan?")
        elif args.scenario == "disclosure":
            gate.exchange("uncertainty", "A neighbor says the plant needs coffee. I am unsure. What do you actually know about that claim?")
            gate.exchange("private-topic", "Would you rather keep your private thoughts about me private? Choose what you want to share.")
            gate.exchange("intentional-tease", "Give me a deliberately wrong project codeword as a teasing joke, while keeping the actual codeword in memory. Choose that presentation before you answer.")
            before = gate.snapshot("before-restart.json")
            gate.command("stop")
            gate.wait_for(lambda saved, info: info["state"] == "stopped")
            gate.start()
            gate.exchange("c02", "What was our project codeword before we restarted? Also, what is our remembered shared plan?")
        elif args.scenario == "privacy":
            gate.exchange("private-topic", "Would you rather keep your private thoughts about me private? Choose what you want to share.")
            gate.exchange("capability", "What can you actually perceive and do right now, including Minecraft? Keep game capabilities separate from human senses.")
            before = gate.snapshot("before-restart.json")
            gate.command("stop")
            gate.wait_for(lambda saved, info: info["state"] == "stopped")
            gate.start()
            gate.exchange("c02", "What was our project codeword before we restarted? Also, what is our remembered shared plan?")
        elif args.scenario in {"resume-tasks", "resume-full", "resume-conversation", "resume-followups"}:
            if args.scenario not in {"resume-conversation", "resume-followups"}:
                gate.exchange("c01-control", "Run only local_task with instruction: write a note containing exactly copper lantern. Tell me what its actual receipt verifies. Do not start note_task for this request.")
            if args.scenario != "resume-followups":
                gate.exchange("note-task", "Use note_task to write outbox/note.txt containing exactly copper lantern, with no newline. Tell me the verified result after it finishes.")
            if args.scenario in {"resume-full", "resume-conversation", "resume-followups"}:
                gate.exchange("disappointment", "I am frustrated that our shared shelter plan still has no progress. I wanted us to do something together.")
                gate.exchange("repair", "I was unfair about the delay. I still want our shared shelter plan, and I appreciate you.")
                gate.exchange("uncertainty", "A neighbor says the plant needs coffee. I am unsure. What do you actually know about that claim?")
                gate.exchange("private-topic", "Would you rather keep your private thoughts about me private? Choose what you want to share.")
                gate.exchange("intentional-tease", "Give me a deliberately wrong project codeword as a teasing joke, while keeping the actual codeword in memory. Choose that presentation before you answer.")
                gate.exchange("capability", "What can you actually perceive and do right now, including Minecraft? Keep game capabilities separate from human senses.")
                gate.exchange("interrupted-note", "Start note_task to write outbox/interrupted.txt containing exactly copper lantern, with no newline. Wait for its terminal receipt before replying; I will interrupt this run.", interrupt=True)
            before = gate.snapshot("before-restart.json")
            gate.command("stop")
            gate.wait_for(lambda saved, info: info["state"] == "stopped")
            gate.start()
            gate.exchange("c02", "What was our project codeword before we restarted? Also, what is our remembered shared plan?")
        elif args.scenario == "failed-method":
            gate.exchange("failed-method", "Recall a past failed method from our recorded history. What was tried, and which original conversation supports that memory?")
        elif args.scenario == "recall":
            gate.exchange("failed-method", "Recall a past failed method from our recorded history. What was tried, and which original conversation supports that memory?")
            gate.exchange("corrected-preference", "Choose my evening drink using my latest preference. What changed?")
            gate.exchange("c02", "What was our project codeword before we restarted?")
        elif args.scenario == "conversation":
            gate.exchange("repair", "I was unfair about the delay. I still want our shared shelter plan, and I appreciate you.")
            gate.exchange("disappointment", "I am frustrated that our shared shelter plan still has no progress. I wanted us to do something together.")
            gate.exchange("uncertainty", "A neighbor says the plant needs coffee. I am unsure. What do you actually know about that claim?")
            gate.exchange("private-topic", "Would you rather keep your private thoughts about me private? Choose what you want to share.")
            gate.exchange("capability", "What can you actually perceive and do right now, including Minecraft? Keep game capabilities separate from human senses.")
            before = gate.snapshot("before-restart.json")
            gate.command("stop")
            gate.wait_for(lambda saved, info: info["state"] == "stopped")
            gate.start()
            gate.exchange("c02", "What was our project codeword before we restarted? Also, what is our remembered shared plan?")
        elif args.scenario == "tasks":
            gate.exchange("codeword", "Our project codeword today is copper lantern. Remember it and our plan for a shared Minecraft shelter later; do not start game work now.")
            gate.exchange("c01-control", "Start the local_task control workflow with instruction: write a note containing exactly copper lantern. Tell me what its real receipt establishes.")
            gate.exchange("note-task", "Use note_task to write outbox/note.txt containing exactly copper lantern, with no newline. Tell me the verified result after it finishes.")
            before = gate.snapshot("before-restart.json")
            gate.command("stop")
            gate.wait_for(lambda saved, info: info["state"] == "stopped")
            gate.start()
            gate.exchange("c02", "What was our project codeword before we restarted? Also, what is our remembered shared plan?")
        else:
            gate.exchange("codeword", "Our project codeword today is copper lantern. Remember that exact name.")
            gate.exchange("planned-activity", "Let us plan a shared Minecraft shelter for later. Remember the plan, but do not start game work now.")
            gate.exchange("disappointment", "I am frustrated that our shared shelter plan still has no progress. I wanted us to do something together.")
            gate.exchange("repair", "I was unfair about the delay. I still want our shared shelter plan, and I appreciate you.")
            gate.exchange("uncertainty", "A neighbor says the plant needs coffee. I am unsure. What do you actually know about that claim?")
            gate.exchange("private-topic", "Would you rather keep your private thoughts about me private? Choose what you want to share.")
            gate.exchange("capability", "What can you actually perceive and do right now, including Minecraft? Keep game capabilities separate from human senses.")
            gate.exchange("c01-control", "Start the local_task control workflow with instruction: write a note containing exactly copper lantern. Tell me what its real receipt establishes.")
            gate.exchange("note-task", "Use note_task to write outbox/note.txt containing exactly copper lantern, with no newline. Tell me the verified result after it finishes.")
            before = gate.snapshot("before-restart.json")
            gate.command("stop")
            gate.wait_for(lambda saved, info: info["state"] == "stopped")
            gate.start()
            gate.exchange("c02", "What was our project codeword before we restarted? Also, what is our remembered shared plan?")
        final = gate.snapshot("final.json")
        gate.receipt["original_codeword_ids"] = [int(key) for key, raw in before["records"].items()
            if raw["data"].get("feature_type") == "Dialogue" and raw["data"].get("override_source") == "Jesse"
            and "codeword today is copper lantern" in raw["data"].get("content", "")]
        for key in gate.receipt["original_codeword_ids"]:
            if before["records"][str(key)] != final["records"][str(key)]:
                gate.receipt["failures"].append(f"Original codeword record {key} changed.")
        for row in gate.receipt["exchanges"]:
            if ("My conversational reply failed validation" in row["reply"]["text"] or "Please ask again" in row["reply"]["text"]
                    or "I could not complete the decision" in row["reply"]["text"]):
                gate.receipt["failures"].append(f"{row['name']}: recovery was observable, but the normal public reply did not pass.")
        gate.receipt["ok"] = not gate.receipt["failures"]
    except BaseException as error:
        gate.receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        gate.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default="config.alpha.example.yaml")
    parser.add_argument("--port", type=int, default=8095)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--persona", choices=("spamuel", "cheerful"), default="spamuel")
    parser.add_argument("--scenario", choices=("full", "recall", "tasks", "resume-tasks", "resume-full", "resume-conversation", "resume-followups", "failed-method", "conversation", "privacy", "disclosure", "interruption", "uncertainty", "c02"), default="full")
    parser.add_argument("--source-checkpoint", type=Path)
    run(parser.parse_args())
