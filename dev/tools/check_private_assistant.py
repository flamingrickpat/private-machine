"""Run the private assistant through the kernel against a live model and Minecraft body."""
from __future__ import annotations

import argparse
from pathlib import Path
from fastmcp import FastMCP

from pm.host.mcp_listeners import serve_services
from pm.llm.mcp_config import ToolCallConfig, McpServerConfig
from pm.memory.store import MemoryStore
from workflows.companion import build_app
from workflows.tool_call.report import load_report


def next_event() -> str:
    """Read the next appointment on the synthetic test calendar."""
    return "Copper Teapot at 18:45"


def check(config: Path, checkpoint: Path, minecraft: str, evidence: Path) -> None:
    evidence.mkdir(parents=True, exist_ok=False)
    store = MemoryStore.load(checkpoint, checkpoint=evidence / "private.pickle")
    calendar = FastMCP("calendar")
    calendar.tool(next_event)  # pm-lint: allow PM003 -- FastMCP registers a named Python tool
    suites_by_name = {"calendar": calendar}
    with serve_services(suites_by_name) as servers:  # pm-lint: allow PM005 -- MCP suites indexed by service name
        suites = ToolCallConfig(mcp_servers=[McpServerConfig(name="calendar", url=servers["calendar"]["url"]),
                                             McpServerConfig(name="minecraft", url=minecraft)])
        app = build_app(None, evidence / "runtime", ghost=store, cognition="real", config_path=str(config),
                        enable_minecraft=False, tool_call=suites)
        supervisor = app.supervisor
        run = supervisor.start(action_id="inspect-calendar-and-body", workflow_name="assistant",
            instructions="Privately read the next calendar event and observe the current Minecraft position. "
                         "Use both services, remain stationary, then finish with the time and coordinates.")
        supervisor.join(run["run_id"])
        receipt = supervisor.status(run["run_id"])
        report = load_report((Path(receipt["artifact_path"]) / "outbox/report.yaml").read_text(encoding="utf-8"))
        print(receipt["state"], report.summary, flush=True)
        assert report.accepted and {call.server for call in report.tool_calls} == {"calendar", "minecraft"}
        assert "18:45" in report.summary
        outcomes = [f for f in store.all_features if f.metadata.get("run_id") == run["run_id"]
                    and f.metadata.get("tool_call_report")]
        assert len(outcomes) == 1 and report.summary in outcomes[0].content
        print(f"Private outcome {outcomes[0].id}. Session receipts in {receipt['artifact_path']}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--minecraft", required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    check(args.config, args.checkpoint, args.minecraft, args.evidence.resolve())
