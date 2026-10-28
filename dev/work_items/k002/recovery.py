"""Exercise clean recovery with a real PM process and a deliberate process crash.

This gate forks the supplied checkpoint. It never terminates an existing PM
or a model service. Real inference must finish the checkpoint's approved reply.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import time

from pm.app.transport import event_cursor
from pm.host.durable_files import atomic_json
from pm.memory.store import MemoryStore
from simulation.client import PMProcess
from simulation.evidence import capture_participant
from simulation.personas import fork_home
from simulation.source import freeze_source


def run(args) -> dict:
    """Resume approved speech, crash during a newer cognition run, and reopen cleanly."""
    receipt_path = args.out / "result.json"
    def progress(stage, **fields):
        atomic_json(receipt_path, {"status": "running", "stage": stage, **fields})
    progress("fork")
    source = freeze_source(args.out / "source", args.config)
    home = fork_home(args.home, args.out / "participant", snapshot=args.snapshot)
    initial = MemoryStore.from_checkpoint(home.ghost_pickle)
    selected = [record.id for record in initial.all_knoxels.values()
                if record.metadata.get("dispatch_status") == "pending"
                and record.metadata.get("workflow_name") == "dialogue_story"]
    if not selected:
        raise ValueError("The checkpoint needs an unfinished approved dialogue action.")
    first_id = initial.current_knoxel_id + 1
    first_tick = initial.current_tick_id
    cursor = event_cursor(initial)
    process = PMProcess("recovery", home.root, source["config"], args.out / "endpoint.json", source_root=Path(source["source"]))
    process.launch(args.clock)
    process.listen(cursor)
    progress("waiting_for_approved_reply", action_ids=selected, original_tick=first_tick)
    while True:
        process.state()
        try:
            event = process.events.get(timeout=.1)
            break
        except queue.Empty:
            pass
    progress("delivering_provisional_input", reply=event)
    delivery = "k002-crash-probe"
    ack = process.post("/v1/chat/completions", {"delivery_id": delivery, "user": initial.get_user_name(),
        "messages": [{"role": "user", "content": "Recovery test: this latest message must disappear after the forced restart."}]})
    progress("waiting_for_real_cognition", reply=event, stimulus=ack)
    while True:
        process.state()
        pending = []
        for path in (home.root / "logs/prompts").glob("*.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                continue
            if record.get("module") == "main_cog" and record.get("route") == "chat/completions":
                pending.append(path.name)
        if pending:
            break
        time.sleep(.05)
    # The readiness receipt identifies only the child created by this gate.
    endpoint = json.loads(process.endpoint_file.read_text(encoding="utf-8"))
    process.reader_stop.set()
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x0001, False, endpoint["pid"])
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not kernel.TerminateProcess(handle, 137):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            kernel.CloseHandle(handle)
    else:
        import signal
        os.kill(endpoint["pid"], signal.SIGKILL)
    process.process.wait()
    process.reader.join()
    process.http.close()
    process._log.close()
    clean = MemoryStore.from_checkpoint(home.ghost_pickle)
    raw = MemoryStore.from_checkpoint(home.ghost_pickle, raw=True)
    latest = next(record for record in raw.all_features if record.metadata.get("delivery_id") == delivery)
    if clean.get_knoxel_by_id(latest.id) is not None:
        raise AssertionError("The new cognition run finished before the crash. The exclusion gate did not execute.")
    if clean.get_knoxel_by_id(event["output_id"]) is None:
        raise AssertionError("Clean recovery lost the approved completed reply.")
    if clean.current_tick_id != first_tick:
        raise AssertionError("The forced interruption unexpectedly admitted a new main cycle.")
    progress("restart", reply=event, excluded_input_id=latest.id)
    restarted = PMProcess("reopened", home.root, source["config"], args.out / "reopened-endpoint.json", source_root=Path(source["source"]))
    restarted.launch(args.clock)
    restarted.state()
    reopened = MemoryStore.from_checkpoint(home.ghost_pickle)
    if reopened.get_knoxel_by_id(latest.id) is not None:
        raise AssertionError("A normal process restart restored the provisional input.")
    restarted.close()
    progress("capture", reply_id=event["output_id"], excluded_input_id=latest.id)
    captured = capture_participant(home.root, args.out / "evidence", first_id)
    result = {"status": "passed", "original_tick": first_tick, "recovered_tick": clean.current_tick_id,
        "approved_action_ids": selected, "reply_id": event["output_id"], "excluded_input_id": latest.id,
        "raw_input_preserved": raw.get_knoxel_by_id(latest.id) is not None,
        "real_cognition_prompt_files": pending, "forced_process_id": endpoint["pid"], "evidence": captured}
    atomic_json(receipt_path, result)
    return result


def main() -> int:
    """Run the isolated real gate. Retain a failure receipt and propagate errors."""
    from pm.host.processes import ensure_process_cleanup
    ensure_process_cleanup()
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.alpha.example.yaml"))
    parser.add_argument("--clock", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        print(json.dumps(run(args), ensure_ascii=False), flush=True)
        return 0
    except BaseException as error:
        atomic_json(args.out / "failure.json", {"error": repr(error), "at": datetime.now(timezone.utc).isoformat()})
        receipt = args.out / "result.json"
        previous = json.loads(receipt.read_text(encoding="utf-8")) if receipt.exists() else {}
        atomic_json(receipt, {**previous, "status": "failed", "error": repr(error)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
