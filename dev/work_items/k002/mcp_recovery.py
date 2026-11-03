"""Exercise actual memory MCP transport loss and real OpenRouter recovery.

The TCP relay forwards real bytes and resets one connection after the actual
server starts its response. It supplies no fabricated tool or model response.
Only read-only memory calls run. The canonical source remains unchanged.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import struct
import sys
from urllib.parse import urlparse
import httpx

DEPENDENCY = Path(__file__).resolve().parents[4] / "pm-coder"
sys.path.insert(0, str(DEPENDENCY))

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset
from pm.host.durable_files import atomic_json
from pm.host.mcp_listeners import serve_services
from pm.llm.config import load_config, resolve_connection
from pm.memory.agent_tools import MemoryReadSession, MemoryToolConfig, MemoryView
from pm.memory.store import MemoryStore
from pm.records.enums import FeatureType, InterlocusType
from pm.utils.operating_system import OperatingSystem


@dataclass
class ResetRelay:
    """Forward actual MCP traffic, with one explicit TCP reset after a read."""

    target_port: int
    armed: bool = True
    resets: int = 0
    tasks: set[asyncio.Task[None]] = field(default_factory=set[asyncio.Task[None]])

    async def handle(self, incoming: asyncio.StreamReader, outgoing: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("Relay connection has no task.")
        self.tasks.add(task)
        upstream_reader, upstream_writer = await asyncio.open_connection("127.0.0.1", self.target_port)
        calling_tool = False

        async def upload() -> None:
            nonlocal calling_tool
            observed = b""
            while data := await incoming.read(65536):
                observed = (observed + data)[-65536:]
                if b'"tools/call"' in observed:
                    calling_tool = True
                upstream_writer.write(data)
                await upstream_writer.drain()

        async def download() -> None:
            while data := await upstream_reader.read(65536):
                if self.armed and calling_tool:
                    self.armed = False
                    self.resets += 1
                    wire_socket = outgoing.get_extra_info("socket")
                    wire_socket.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("HH", 1, 0))
                    outgoing.transport.abort()
                    print("Reset actual MCP response connection", flush=True)
                    return
                outgoing.write(data)
                await outgoing.drain()

        transfers = [asyncio.create_task(upload()), asyncio.create_task(download())]
        try:
            done, pending = await asyncio.wait(transfers, return_when=asyncio.FIRST_COMPLETED)
            for transfer in pending:
                transfer.cancel()
            for transfer in done:
                try:
                    transfer.result()
                except (ConnectionError, OSError):
                    pass  # Expected consequences of the requested TCP reset.
        finally:
            for transfer in transfers:
                transfer.cancel()
            await asyncio.gather(*transfers, return_exceptions=True)
            upstream_writer.close()
            outgoing.close()
            self.tasks.remove(task)


async def exercise(evidence: Path, dependency: Path) -> None:
    """Import the actual repaired dependency and drive its public session path."""
    sys.path.insert(0, str(dependency.resolve()))
    from pm_coder import SessionStore, async_run_auto_with_bash_machine, build_settings, connected_agent, make_model, open_session, run_turn
    from pm_bash_machine import BashMachine
    import pm_coder
    if Path(pm_coder.__file__).resolve().parent != dependency.resolve():
        raise RuntimeError("The live check imported a different dependency checkout.")

    config = load_config("config.yaml")
    connection = resolve_connection(config)
    if connection.base_url != "https://openrouter.ai/api/v1":
        raise ValueError("This check requires the owner's real OpenRouter endpoint.")
    OperatingSystem.environment["PM_COMPLETION_CACHE"] = "0"
    store = MemoryStore.from_checkpoint(Path.home() / ".pm/private-machine/spamuel-simulation-a/ghost.pickle")
    public = [row for row in store.all_features if row.feature_type == FeatureType.Dialogue
              and row.interlocus == InterlocusType.Public and row.source == "Jesse"]
    selected = max(public, key=lambda row: row.timestamp_world_begin)
    records = list(store.all_knoxels.values())
    view = MemoryView(frozenset(store.all_knoxels),
        frozenset((record.scope.reality_id, record.scope.frame) for record in records))
    with MemoryReadSession(MemoryToolConfig(store, view)) as memory:
        if selected.id not in memory.records:
            raise ValueError("The live test's selected record is outside its actual memory grant.")
        with serve_services({"memory": memory.server()}) as descriptors:
            target = urlparse(descriptors["memory"]["url"])
            if target.port is None:
                raise ValueError("The real MCP listener supplied no port.")
            relay = ResetRelay(target.port)
            server = await asyncio.start_server(relay.handle, "0.0.0.0", 0)
            async with server:
                port = server.sockets[0].getsockname()[1]
                mcp_config = evidence / "mcp.json"
                mcp_config.write_text(json.dumps({"mcpServers": {"memory": {
                    "url": f"http://127.0.0.1:{port}/mcp", "type": "http"}}}), encoding="utf-8")
                settings = build_settings(cwd=evidence, base_url=connection.base_url, model=connection.model,
                    api_key=connection.api_key.get_secret_value(), context_window=1_048_576,
                    enable_thinking=False, enable_write=False, workspace_discovery=False, mcp_config=mcp_config)
                async with open_session(settings, run_id="recovered", log_root=evidence / "agents") as (agent, _, session):
                    result = await run_turn(agent, settings, session,
                        f"Call memory_memory_get for record_id {selected.id}. Quote only the exact content field "
                        "and record ID in your final answer, without other metadata. If a read is interrupted, "
                        "retry the read. Do not use local files.")
                    if relay.resets != 1 or selected.id not in memory.accessed:
                        raise AssertionError("The actual memory read and connection reset did not both occur.")
                    if selected.content not in result.response:
                        raise AssertionError("The real model answer did not quote the actual retrieved content.")
                    print("Recovered real model answer:", result.response[:300], flush=True)
                    atomic_json(evidence / "recovered.json", {"run_id": result.run_id,
                        "response": result.response, "record_id": selected.id, "resets": relay.resets})
                    # Repeat a real model request after reconnect to detect a closed provider client.
                    followup = await run_turn(agent, settings, session,
                        "State the record ID you just read, using the existing transcript. Do not call more tools.")
                    print("Follow-up:", followup.response, flush=True)
                if session.agent_connection is not None:
                    raise AssertionError("The closed session retained its lifecycle owner.")
                transcript = session.messages_path.read_text(encoding="utf-8")
                if 'interrupted' not in transcript or 'outcome is unknown' not in transcript:
                    raise AssertionError("The uncertain read outcome was lost from the transcript.")

                # A completed model answer must survive an old MCP task's exit error.
                completed = SessionStore.open(evidence, "completed-before-cleanup", log_root=evidence / "agents")
                completed.context_window = 1_048_576
                mcp: MCPToolset[None] = MCPToolset(f"http://127.0.0.1:{port}/mcp")
                real_agent = Agent(make_model(settings, "agent", completed), toolsets=[mcp], deps_type=type(None))
                async with connected_agent(real_agent, completed):
                    saved = await run_turn(real_agent, settings, completed,
                        "Reply briefly that this turn is complete. Do not call any tools.")
                    relay.armed = True
                    try:
                        await mcp.direct_call_tool("memory_get", {"record_id": selected.id})
                    except httpx.TransportError:
                        pass  # The test deliberately poisons the actual tool connection after completion.
                    else:
                        raise AssertionError("The requested post-completion MCP reset did not occur.")
                if not saved.response or len(completed.runs_path.read_text(encoding="utf-8").splitlines()) != 1:
                    raise AssertionError("Cleanup lost or reran the completed model answer.")
                lifecycle = (completed.path / "agent-connection.jsonl").read_text(encoding="utf-8")
                if 'cleanup_transport_error' not in lifecycle:
                    raise AssertionError("No real post-completion cleanup error was observed.")
                print("Completed answer survived actual MCP cleanup failure without rerunning.", flush=True)

                relay.armed = True
                bash_result = await async_run_auto_with_bash_machine(
                    f"Use memory_memory_get to read record {selected.id}. If its read is interrupted, retry it. "
                    "Quote only its exact content field and record ID. Do not use file tools or the shell.",
                    BashMachine(), cwd=evidence, run_id="bash-recovered", log_root=evidence / "agents",
                    base_url=connection.base_url, model=connection.model,
                    api_key=connection.api_key.get_secret_value(), context_window=1_048_576,
                    enable_thinking=False, enable_write=True, workspace_discovery=False, mcp_config=mcp_config)
                if relay.resets != 3 or selected.content not in bash_result.response:
                    raise AssertionError("The real BashMachine agent entry did not recover the actual record.")
                print("Actual BashMachine auto-session recovered after the third real TCP reset.", flush=True)

                # Stop only the transparent relay to cause a real connect refusal.
                server.close()
                await server.wait_closed()
                cancelled = SessionStore.open(evidence, "cancel-reconnect", log_root=evidence / "agents")
                cancelled.context_window = 1_048_576
                unavailable: MCPToolset[None] = MCPToolset(f"http://127.0.0.1:{port}/mcp")
                waiting_agent = Agent(make_model(settings, "agent", cancelled), toolsets=[unavailable], deps_type=type(None))

                async def wait_for_connection() -> None:
                    async with connected_agent(waiting_agent, cancelled):
                        raise AssertionError("A stopped TCP relay unexpectedly accepted a connection.")

                wait_task = asyncio.create_task(wait_for_connection())
                retry_path = cancelled.path / "agent-connection.jsonl"
                deadline = asyncio.get_running_loop().time() + 20
                while not retry_path.exists():
                    if wait_task.done():
                        await wait_task
                    if asyncio.get_running_loop().time() > deadline:
                        wait_task.cancel()
                        await asyncio.gather(wait_task, return_exceptions=True)
                        raise AssertionError("The real refused connection produced no retry receipt within 20 seconds.")
                    await asyncio.sleep(.05)
                wait_task.cancel()
                try:
                    await wait_task
                except asyncio.CancelledError:
                    pass
                else:
                    raise AssertionError("Connection recovery swallowed cancellation.")
                if cancelled.agent_connection is not None:
                    raise AssertionError("Cancellation left its connection owner attached.")
                print("Real unavailable MCP connection retried and remained cancellable.", flush=True)
                for task in list(relay.tasks):
                    task.cancel()
                await asyncio.gather(*list(relay.tasks), return_exceptions=True)
    atomic_json(evidence / "result.json", {"status": "passed", "at": datetime.now(timezone.utc).isoformat(),
        "endpoint": connection.base_url, "model": connection.model, "scope": "real memory MCP and TCP reset; real model recovery"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    args.evidence.mkdir(parents=True, exist_ok=False)
    asyncio.run(exercise(args.evidence.resolve(), DEPENDENCY))


if __name__ == "__main__":
    main()
