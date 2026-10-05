"""Expose live schemas through a leased, journaled Minecraft MCP gateway."""
from __future__ import annotations

import asyncio
import base64
import json
import threading
import time
from typing import Any

from fastmcp import FastMCP
from fastmcp.tools import Tool, ToolResult
from pydantic import Field

from runtime.body_lease import BODY_LEASES

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
        self.raw_tools = raw_tools
        self.visual = visual
        self._lock = threading.Lock()
        self.receipts: list[dict] = []
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
        current = self.runtime.tools["managed_instructions"](close=close)
        if current["revision"] != self.revision:
            self.revision = current["revision"]
            self._append({"event": "instruction_changed", "tool": name, "arguments": arguments,
                          "revision": self.revision, "instruction": current["instruction"], "time": time.time()})
            return {"ok": False, "reason": "instruction_changed", **current,
                    "message": "No action was sent. Choose again from the new instruction."}
        if name not in READ_TOOLS and BODY_LEASES.owner(self.endpoint) != self.owner:
            raise PermissionError("This session does not own the Minecraft body.")
        self.runtime.check_cancelled()
        seq = len(self.receipts) + 1
        start = {"seq": seq, "event": "dispatch", "tool": name,
                 "arguments": arguments, "revision": self.revision, "time": time.time()}
        self._append(start)
        try:
            result = self.raw_tools["minecraft_call"](name, arguments)
        except Exception as error:
            self._append({**start, "event": "uncertain", "error": f"{type(error).__name__}: {error}"})
            self.runtime.stop_event.set()
            raise
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
        return {**result, "images": images, "instruction_revision": self.revision}

    def _append(self, receipt: dict) -> None:
        path = self.environment._path("scratch/actor-journal.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(receipt, ensure_ascii=False) + "\n")
            stream.flush()

    def wait_idle(self) -> None:
        """Wait for an external call before releasing its body lease."""
        with self._lock:
            pass

    def server(self, schemas: list[dict]) -> FastMCP:
        """Expose the upstream argument schemas and a private boundary tool."""
        server = FastMCP("minecraft_actor")
        for schema in schemas:
            server.add_tool(ActorTool(name=schema["name"], description=schema["description"],
                                      parameters=schema["parameters"], gateway=self))
        server.add_tool(Tool.from_function(self.boundary, name="actor_instructions"))
        return server


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
