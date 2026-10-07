"""Expose live schemas through a leased, journaled Minecraft MCP gateway."""
from __future__ import annotations

import asyncio
import base64
import json
import os
import threading
import time
from typing import Any

from fastmcp import FastMCP
from fastmcp.tools import Tool, ToolResult
from pydantic import Field

from runtime.body_lease import BODY_LEASES, StaleEpoch

# These tools do not move or rotate the body. Horizon scans do rotate it.
READ_TOOLS = frozenset({
    "minecraft_observe", "minecraft_count_inventory", "minecraft_inspect_block",
    "minecraft_find_block", "minecraft_find_interactables", "minecraft_raytrace",
    "minecraft_recipe_search", "minecraft_info", "minecraft_list_capabilities",
    "minecraft_list_waypoints", "minecraft_relative", "minecraft_distance",
    "minecraft_screenshot", "minecraft_analyze_area", "minecraft_find_path",
    "minecraft_is_safe_to_dig", "minecraft_recall",
})


class ActorGateway:
    """Serialize actions, retain receipts, and apply instructions before dispatch.

    Cleanup waits for a dispatched synchronous call. Cancellation cannot undo
    its world effects. A changed instruction rejects the already planned call.
    The model must read that revision before choosing another action.
    """

    def __init__(self, endpoint: str, owner: str, environment, runtime, raw_tools: dict,
                 *, visual: bool = False) -> None:
        self.endpoint, self.owner = endpoint, owner
        self.environment, self.runtime = environment, runtime
        self.raw_tools = dict(raw_tools)
        transport = raw_tools["minecraft_call"]
        for name, attribute in (("minecraft_claim", "claim"), ("minecraft_release", "release"),
                                ("minecraft_fenced_call", "fenced_call")):
            capability = getattr(transport, attribute, None)
            if capability is not None:
                self.raw_tools.setdefault(name, capability)
        authorization = getattr(raw_tools.get("managed_instructions"), "authorize", None)
        if authorization is not None:
            self.raw_tools.setdefault("managed_authorize", authorization)
        cancellation = getattr(raw_tools.get("managed_instructions"), "cancelled", None)
        if cancellation is not None:
            self.raw_tools.setdefault("managed_cancelled", cancellation)
        self.visual = visual
        self._lock = threading.Lock()
        self._journal_lock = threading.RLock()
        self.receipts: list[dict] = []
        self.token: dict | None = None
        self.observed_epoch = False
        path = environment._path("scratch/actor-journal.jsonl")
        self.previous = read_actor_journal(path)
        self.receipts = [row for row in self.previous if row.get("event") == "result"]
        self.sequence = max((row.get("seq", 0) for row in self.previous), default=0)
        self.revision = runtime.tools["managed_instructions"]()["revision"]

    def boundary(self) -> dict:
        """Return the current instruction and revision without a world call."""
        self.runtime.check_cancelled()
        current = self.runtime.tools["managed_instructions"]()
        self.revision = current["revision"]
        return {**current, "body_owner": BODY_LEASES.owner(self.endpoint)}

    def call(self, name: str, arguments: dict, *, close: bool = False) -> dict:
        """Reject overlapping actor plans and wait for a direct read boundary."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Another actor call is still active.")
        try:
            with BODY_LEASES.command(self.endpoint, self.runtime.stop_event):
                return self._dispatch(name, arguments, close=close)
        finally:
            self._lock.release()

    def _dispatch(self, name: str, arguments: dict, *, close: bool = False) -> dict:
        """Dispatch one call and persist its original result before returning.

        Concurrent calls are rejected. A transport error preserves an uncertain
        outcome and cancels future work. No automatic action retry occurs.
        """
        self.runtime.check_cancelled()
        if self.token is not None:
            BODY_LEASES.validate(self.endpoint, self.token)
        elif BODY_LEASES.owner(self.endpoint) == self.owner:
            self.token = BODY_LEASES.token(self.endpoint, self.owner)
            if "minecraft_claim" in self.raw_tools:
                claim = self.raw_tools["minecraft_claim"](self.token)
                if not claim["ok"]:
                    raise StaleEpoch("The Minecraft server refused this execution epoch.")
            self._append({"event": "epoch_acquired", "epoch": self.token["epoch"],
                          "session": self.token["session"], "owner": self.owner, "time": time.time()})
        current = self.runtime.tools["managed_instructions"](close=close)
        if current["revision"] != self.revision:
            self.revision = current["revision"]
            self._append({"event": "instruction_changed", "tool": name, "arguments": arguments,
                          "revision": self.revision, "instruction": current["instruction"], "time": time.time()})
            return {"ok": False, "reason": "instruction_changed", **current,
                    "message": "No action was sent. Choose again from the new instruction."}
        if name not in READ_TOOLS:
            if self.token is None:
                raise StaleEpoch("This session does not own the Minecraft body.")
            if not self.observed_epoch:
                raise StaleEpoch("A fresh observation must precede the first mutation in this epoch.")
        self.runtime.check_cancelled()
        self.sequence += 1
        seq = self.sequence
        start = {"seq": seq, "event": "dispatch", "tool": name,
                 "arguments": arguments, "revision": self.revision, "time": time.time(),
                 "epoch": self.token["epoch"] if self.token else None,
                 "session": self.token["session"] if self.token else None}
        authorize = self.raw_tools.get("managed_authorize")
        if authorize is not None and not authorize(self.revision, lambda: self._append(start)):
            current = self.runtime.tools["managed_instructions"]()
            self.revision = current["revision"]
            return {"ok": False, "reason": "instruction_changed", **current,
                    "message": "No action was sent. Choose again from the new instruction."}
        if authorize is None:
            self._append(start)
        finished = threading.Event()
        watcher = None
        if name not in READ_TOOLS and self.token is not None and "minecraft_fenced_call" in self.raw_tools:
            def cancel_command():
                """Stop only this epoch's active command after accepted cancellation."""
                while not finished.wait(0.05):
                    poll = self.raw_tools.get("managed_cancelled")
                    if self.runtime.stop_event.is_set() or (poll is not None and poll()):
                        self._append({"event": "stop_dispatch", "epoch": self.token["epoch"], "time": time.time()})
                        try:
                            stopped = self.raw_tools["minecraft_fenced_call"]("minecraft_stop", {"scope": "both"}, self.token)
                            self._append({"event": "stop_result", "epoch": self.token["epoch"],
                                          "result": stopped, "time": time.time()})
                        except Exception as error:
                            self._append({"event": "stop_uncertain", "epoch": self.token["epoch"],
                                          "error": f"{type(error).__name__}: {error}", "time": time.time()})
                        return
            watcher = threading.Thread(target=cancel_command, name="body-cancel", daemon=True)
            watcher.start()
        try:
            if self.token is not None and "minecraft_fenced_call" in self.raw_tools:
                result = self.raw_tools["minecraft_fenced_call"](name, arguments, self.token)
            else:
                result = self.raw_tools["minecraft_call"](name, arguments)
        except Exception as error:
            self._append({**start, "event": "uncertain", "error": f"{type(error).__name__}: {error}"})
            self.runtime.stop_event.set()
            raise
        finally:
            finished.set()
            if watcher is not None:
                watcher.join()
        images = []
        for index, frame in enumerate(result.get("images") or []):
            suffix = "png" if "png" in frame["media_type"] else "jpg"
            path = f"screenshots/call-{seq:05d}-{index}.{suffix}"
            self.environment.write_binary(path, base64.b64decode(frame["base64"]))
            images.append({"path": path, "media_type": frame["media_type"]})
        receipt = {**start, "event": "result", "time": time.time(),
                   "ok": result["ok"], "result": result["result"], "images": images}
        self.receipts.append(receipt)
        self._append(receipt)
        if name == "minecraft_observe" and result["ok"]:
            self.observed_epoch = True
        return {**result, "images": images, "instruction_revision": self.revision}

    def _append(self, receipt: dict) -> None:
        """Serialize a complete journal append before exposing its result."""
        with self._journal_lock:
            self._write_receipt(receipt)

    def _write_receipt(self, receipt: dict) -> None:
        """Retain a torn append marker, then flush the next complete receipt."""
        path = self.environment._path("scratch/actor-journal.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size:
            with path.open("rb") as existing:
                existing.seek(-1, os.SEEK_END)
                torn = existing.read(1) != b"\n"
            if torn:
                tail = path.read_text(encoding="utf-8").splitlines()[-1]
                with path.open("a", encoding="utf-8") as stream:
                    stream.write("\n" + json.dumps({"event": "torn_tail", "text": tail}) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(receipt, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def reconcile(self, sources: list) -> dict:
        """Read uncertain target cells after the new epoch's body observation.

        Original dispatches and results remain separate evidence. A current
        cell can establish its state, but cannot establish who changed it.
        """
        from runtime.body_lease import atomic_json
        retained = [{"source": "scratch/actor-journal.jsonl", "calls": self.previous}]
        for path in sources:
            retained.append({"source": str(path), "calls": read_actor_journal(path)})
        pending, positions = [], {}
        for source in retained:
            rows = source["calls"]
            results = {row["seq"]: row for row in rows if row.get("event") == "result"}
            for row in rows:
                if row.get("event") != "dispatch" or row["tool"] in READ_TOOLS:
                    continue
                result = results.get(row["seq"])
                pending.append({"source": source["source"], "dispatch": row,
                                "result": result, "outcome": "recorded" if result else "uncertain"})
                for key in ("position", "target", "base"):
                    point = row.get("arguments", {}).get(key)
                    if isinstance(point, dict) and set(point) >= {"x", "y", "z"}:
                        positions[json.dumps(point, sort_keys=True)] = point
        reads = [self.call("minecraft_inspect_block", {"position": point}) for point in positions.values()]
        if any(not observed_target(row, point) for row, point in zip(reads, positions.values())):
            raise RuntimeError("An interrupted action target could not be observed. Do not resume mutation.")
        receipt = {"schema": "pm.body-reconciliation.v1", "epoch": self.token["epoch"] if self.token else None,
                   "instruction_revision": self.revision, "observation_seq": next(
                       row["seq"] for row in reversed(self.receipts) if row["tool"] == "minecraft_observe"),
                   "retained_actions": pending, "target_reads": reads, "automatic_mutation_replay": False,
                   "time": time.time()}
        atomic_json(self.environment._path(f"scratch/reconciliation-{receipt['epoch']}.json"), receipt)
        return receipt

    def wait_idle(self) -> None:
        """Wait for an external call before releasing its body lease."""
        with self._lock:
            if self.token is not None and "minecraft_release" in self.raw_tools:
                try:
                    self.raw_tools["minecraft_release"](self.token)
                except Exception as error:
                    self._append({"event": "release_uncertain", "epoch": self.token["epoch"],
                                  "error": f"{type(error).__name__}: {error}", "time": time.time()})

    def server(self, schemas: list[dict]) -> FastMCP:
        """Expose the upstream argument schemas and a private boundary tool."""
        server = FastMCP("minecraft_actor")
        for schema in schemas:
            server.add_tool(ActorTool(name=schema["name"], description=schema["description"],
                                      parameters=schema["parameters"], gateway=self))
        server.add_tool(Tool.from_function(self.boundary, name="actor_instructions"))
        return server


def observed_target(receipt: dict, point: dict) -> bool:
    """Accept a typed block observation, including explicitly observed air.

    The live tool reports air as ok=false, reason=not_found. That result still
    proves the cell is empty. Connection and unloaded-cell failures do not.
    """
    result = receipt.get("result", {})
    return bool(receipt.get("ok") or (
        result.get("reason") == "not_found" and result.get("block_name") == "air"
        and result.get("position") == point))


def read_actor_journal(path) -> list[dict]:
    """Read complete receipts. Retain a torn final append as an uncertain tail."""
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                marker = json.loads(lines[index + 1])
                if marker != {"event": "torn_tail", "text": line}:
                    raise
            rows.append({"event": "torn_tail", "text": line})
    return rows

class ActorTool(Tool):
    """Forward one upstream schema through its owning actor gateway."""

    gateway: Any = Field(exclude=True)

    async def run(self, arguments: dict) -> ToolResult:
        """Await external cleanup even if the MCP caller disconnects."""
        task = asyncio.create_task(asyncio.to_thread(self.gateway.call, self.name, arguments))
        try:
            result = await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise
        content = {key: value for key, value in result.items() if key != "images"}
        # A text actor gets metadata and paths. Pixels stay in separate files.
        blocks = [self.convert_result(content).content[0]]
        if self.gateway.visual:
            from mcp.types import ImageContent
            for frame in result.get("images", []):
                data = self.gateway.environment._path(frame["path"]).read_bytes()
                blocks.append(ImageContent(type="image", mimeType=frame["media_type"],
                                           data=base64.b64encode(data).decode("ascii")))
        return ToolResult(content=blocks)
