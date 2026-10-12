"""Record real A02 decisions, limits, contention, and owned-service failures."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from typing import Callable

import httpx
import yaml

from pm_decision import (
    AsyncDecisionClient, ChoiceQuestion, DecisionCancelled, DecisionClient,
    DecisionError, DecisionResult, DecisionTransportError,
    ProbabilityQuestion, ScoreQuestion,
)

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / ".runtime/alpha/a01/console-spamuel/instances/fd71e4655da44430b9779eb6739d45f5/managed_tasks/dd8a80caec16462b97e0e85cbdf87346/outbox/report.yaml"
CHECKPOINT = REPORT.parents[3] / "ghost.json"
CRITERIA = {"COMPLETE": "The task completed.", "FAILED": "The task failed.", "INVALID": "The report cannot establish completion or failure."}
QUESTION = ChoiceQuestion(instructions="What is the task status in the final task report?", criteria=CRITERIA)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save(path: Path, value: object) -> None:
    """Store evidence immediately so later failures do not erase earlier results."""
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def record(out: Path, name: str, call: Callable[[], DecisionResult]) -> dict:
    """Retain both success and failure receipts without treating labels as facts."""
    try:
        result = call()
        value = result.receipt.to_dict()
    except (DecisionError, DecisionCancelled) as error:
        value = error.receipt.to_dict() if error.receipt else {"outcome": type(error).__name__}
        value["exception"] = type(error).__name__
    save(out / f"{name}.json", value)
    print(json.dumps({"case": name, "outcome": value["outcome"], "seconds": value.get("seconds")}), flush=True)
    return value


async def contention(out: Path, state: str, identity: dict, report: str) -> dict:
    """Compare three warm chat calls with three calls during real Laya work."""
    decision = AsyncDecisionClient(provider_identity=identity)
    body = {"model": "qwen", "messages": [{"role": "user", "content": "Summarize this real task report in one sentence:\n" + report}],
            "max_tokens": 96, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    async with httpx.AsyncClient(timeout=120, trust_env=False) as connection:
        async def chat() -> dict:
            start = time.monotonic()
            response = await connection.post("http://127.0.0.1:8080/v1/chat/completions", json=body)
            response.raise_for_status()
            return {"seconds": time.monotonic() - start, "response": response.json()}

        await chat()  # Warm the unchanged prompt before both measurements.
        alone = [await chat() for _ in range(3)]
        concurrent = []
        for _ in range(3):
            async def load() -> list:
                return [(await decision.decide(state, {"status": QUESTION})).receipt.to_dict() for _ in range(8)]
            reply, decisions = await asyncio.gather(chat(), load())
            concurrent.append({"chat": reply, "decisions": decisions})
    value = {"chat_alone": alone, "chat_with_decisions": concurrent,
             "scope": "Three warm samples per condition. Shared workstation timing, not a latency guarantee."}
    save(out / "contention.json", value)
    return value


async def owned_failures(out: Path, binary: Path, model: Path, port: int, state: str) -> dict:
    """Cancel and disconnect real inference in an owned CPU service.

    The CPU process uses the same binary and model as the live GPU service.
    It avoids a second GPU allocation. Only this child process is terminated.
    Slot observations establish that each failure occurred after dispatch.
    """
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))
    command = [str(binary), "--model", str(model), "--alias", "laya", "--host", "127.0.0.1",
               "--port", str(port), "--device", "none", "--n-gpu-layers", "0", "--parallel", "1",
               "--ctx-size", "8192", "--batch-size", "8192", "--ubatch-size", "8192", "--threads", "4", "--jinja", "--slots"]
    value = {"command": command, "binary_sha256": sha(binary.read_bytes()), "model_sha256": sha(model.read_bytes())}
    log = (out / "owned-service.log").open("wb")
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    url = f"http://127.0.0.1:{port}"
    task: asyncio.Task | None = None
    try:
        async with httpx.AsyncClient(timeout=2, trust_env=False) as control:
            deadline = time.monotonic() + 60
            while True:
                if process.poll() is not None:
                    raise RuntimeError("Owned Laya process exited during startup. Inspect owned-service.log.")
                try:
                    response = await control.get(url + "/health")
                    if response.status_code == 200: break
                except httpx.RequestError: pass
                if time.monotonic() >= deadline: raise TimeoutError("Owned Laya startup exceeded 60 seconds.")
                await asyncio.sleep(0.1)
            value["props"] = (await control.get(url + "/props")).json()
            client = AsyncDecisionClient(url + "/v1/systemone", timeout=60, provider_identity={"build": value["props"]["build_info"]})
            many = {f"status_{i}": QUESTION for i in range(8)}

            async def active() -> list:
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    slots = (await control.get(url + "/slots")).json()
                    if any(slot.get("is_processing") for slot in slots): return slots
                    if task is not None and task.done(): raise RuntimeError("Request ended before active-slot observation.")
                    await asyncio.sleep(0.01)
                raise TimeoutError("No active decision slot appeared.")

            task = asyncio.create_task(client.decide(state, many))
            value["cancel_active_slots"] = await active()
            task.cancel()
            try: await task
            except DecisionCancelled as error: value["task_cancel"] = error.receipt.to_dict()
            else: raise AssertionError("Task cancellation did not reach the client.")
            # Wait for the server to accept a new request after its cancelled inference.
            recovery = await client.choice(state.split("Final task report:\n")[-1], QUESTION.instructions, CRITERIA)
            value["recovery"] = recovery.receipt.to_dict()

            cancel = threading.Event()
            task = asyncio.create_task(asyncio.to_thread(DecisionClient(url + "/v1/systemone", timeout=60).decide, state, many, cancel=cancel))
            value["sync_cancel_active_slots"] = await active()
            cancel.set()
            try: await task
            except DecisionCancelled as error: value["sync_cancel"] = error.receipt.to_dict()
            else: raise AssertionError("Sync event cancellation did not reach the client.")

            task = asyncio.create_task(client.decide(state, many))
            value["loss_active_slots"] = await active()
            process.terminate()
            await asyncio.to_thread(process.wait, 10)
            try: await task
            except DecisionTransportError as error: value["process_loss"] = error.receipt.to_dict()
            else: raise AssertionError("Process loss did not surface as a transport error.")
            value["exit_code"] = process.returncode
    finally:
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if process.poll() is None:
            process.terminate()
            await asyncio.to_thread(process.wait, 10)
        log.close()
        save(out / "owned-failures.json", value)
    return value


def main() -> None:
    """Run real gates and retain evidence under a new local output directory.

    Invalid paths raise ValueError before inference. Failed service startup or
    gate assertions leave earlier evidence intact and stop the command.
    The owned fault process stops in its own cleanup boundary.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=Path, default=ROOT / "test_data/personas/spamuel/canon.yaml")
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--binary", type=Path, default=Path("C:/workspace/llama-systemone-b11374/llama-server.exe"))
    parser.add_argument("--model-file", type=Path, default=Path("C:/source/ai/Laya-BF16.gguf"))
    parser.add_argument("--fault-port", type=int, default=8084)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT / ".runtime"):
        raise ValueError("Store inherited history and live receipts under .runtime.")
    out.mkdir(parents=True, exist_ok=False)
    props = httpx.get("http://127.0.0.1:8082/props").json()
    identity = {"build": props["build_info"], "model_path": props["model_path"],
                "binary_sha256": sha(args.binary.read_bytes()), "model_sha256": sha(args.model_file.read_bytes())}
    save(out / "identity.json", {"props": props, "identity": identity,
         "packages": {name: {"version": metadata.version(name), "origin": metadata.distribution(name).read_text("direct_url.json")} for name in ("pm-coder", "pm-workflows")}})
    seed = yaml.load(args.seed.read_text(encoding="utf-8"), Loader=yaml.CSafeLoader)
    report = args.report.read_text(encoding="utf-8")
    checkpoint = json.loads(args.checkpoint.read_text(encoding="utf-8"))
    # Bind the report to its actual producer instead of the first local task.
    run_id = args.report.parent.parent.name
    run = checkpoint["managed_runs"][run_id]
    if run["workflow_name"] != "local_task":
        raise ValueError("This gate expects a captured local_task report and its matching checkpoint.")
    task_report = json.dumps({"captured_run_state": run["state"], "report_yaml": report}, ensure_ascii=False)
    records = [{"id": k["data"]["id"], "content": k["data"].get("content", "")} for k in seed["knoxels"] if k["data"].get("content")]
    history = "\n".join(r["content"] for r in records)
    save(out / "provenance.json", {"seed": str(args.seed), "seed_sha256": sha(args.seed.read_bytes()), "records": len(seed["knoxels"]),
         "report": str(args.report), "report_sha256": sha(args.report.read_bytes()), "checkpoint": str(args.checkpoint),
         "checkpoint_sha256": sha(args.checkpoint.read_bytes()), "run_id": run["run_id"],
         "scope": "Captured application report plus real inherited history. Added layout labels are generated. No reviewed quality labels or holdout inputs."})
    client = DecisionClient(provider_identity=identity)
    results = {}
    results["choice"] = record(out, "choice", lambda: client.choice(task_report, QUESTION.instructions, CRITERIA))
    results["score"] = record(out, "score", lambda: client.score(report, "How complete is the answer field?", ["Empty answer", "Partial answer", "Complete answer"]))
    results["probability"] = record(out, "probability", lambda: client.probability(report, "Does the report contain a nonempty answer field?"))
    questions = {"status": QUESTION, "answer": ProbabilityQuestion(instructions="Does the report contain a nonempty answer field?"),
                 "completeness": ScoreQuestion(instructions="How complete is the answer field?", criteria=["Empty answer", "Partial answer", "Complete answer"])}
    results["multi"] = record(out, "multi", lambda: client.decide(task_report, questions))
    long_state = ""
    for budget in (256, 1024, 4096, 7800, 8300):
        # Binary search selects an exact prefix of real history by observed token count.
        lo, hi = 0, min(len(history), budget * 8)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            tokens = httpx.post("http://127.0.0.1:8082/tokenize", json={"content": history[:mid]}, timeout=30).json()["tokens"]
            if len(tokens) <= budget: lo = mid
            else: hi = mid - 1
        prefix = history[:lo]
        for placement in ("start", "end"):
            state = ("Task report:\n" + task_report + "\nHistorical context:\n" + prefix) if placement == "start" else ("Historical context:\n" + prefix + "\nFinal task report:\n" + task_report)
            name = f"history-{budget}-{placement}"
            save(out / f"{name}-input.json", {"state": state, "questions": {"status": QUESTION.model_dump()}})
            results[name] = record(out, name, lambda: client.decide(state, {"status": QUESTION}))
            if budget == 4096 and placement == "end": long_state = state
    results["wrong-endpoint"] = record(out, "wrong-endpoint", lambda: DecisionClient("http://127.0.0.1:8080/v1/systemone").decide(task_report, {"status": QUESTION}))
    results["wrong-model"] = record(out, "wrong-model", lambda: DecisionClient(model="missing-model").decide(task_report, {"status": QUESTION}))
    results["timeout"] = record(out, "timeout", lambda: DecisionClient(timeout=0.001).decide(long_state, {"status": QUESTION}))
    results["malformed"] = record(out, "malformed", lambda: client.decide({"bad": float("nan")}, {"status": QUESTION}))
    raw = httpx.post(client.endpoint, json={"state": task_report, "questions": {}}, timeout=30)
    save(out / "server-malformed.json", {"http_status": raw.status_code, "response": raw.json()})
    assert raw.status_code == 400
    assert all(results[k]["outcome"] == "success" for k in ("choice", "score", "probability", "multi"))
    assert results["wrong-model"]["exception"] == "DecisionSchemaError"
    assert results["wrong-endpoint"]["exception"] == "DecisionHTTPError"
    assert results["timeout"]["exception"] == "DecisionTransportError"
    assert results["malformed"]["exception"] == "DecisionInputError"
    asyncio.run(contention(out, long_state, identity, task_report))
    asyncio.run(owned_failures(out, args.binary, args.model_file, args.fault_port, long_state))
    save(out / "receipt.json", {"status": "passed", "results": results,
         "holdout_evaluated": False, "human_review": "pending", "scope": "API and real failure gates. Model quality remains a measured observation."})


if __name__ == "__main__":
    main()
