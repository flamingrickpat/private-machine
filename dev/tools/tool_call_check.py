"""Run the tool_call workflow once through the real managed-run path. Development tool, not product code.

The check uses a real model configuration, a real MCP server (for example
``dev/tools/demo_mcp.py``), a fresh memory store from a synthetic persona
seed, the real supervisor, and the real consolidator as its terminal hook. It
runs no cognition cycle, so the run has a ``launch`` context, not a pinned
source action. The model configuration file is not changed: the MCP server
comes from the command line.

    uv run --locked python dev/tools/tool_call_check.py --config config.alpha.openrouter.yaml \\
        --mcp demo=http://127.0.0.1:9881/mcp --instruction "roll a 20-sided die and make a red test image"

The output folder holds ``summary.json`` with the receipt, the report, and
the consolidated records (image bytes left out), plus the run artifacts.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from pm.llm.activation import ModelActivation
from pm.llm.config import load_config
from pm.llm.mcp_config import ToolCallConfig
from pm.llm.workflow_sessions import WorkflowModels
from pm.memory.store import MemoryStore
from pm.paths import REPO_ROOT
from pm.records.media import record_media
from pm.runs.supervisor import ManagedRunSupervisor
from workflows.memory_consolidation.artifact_consolidation import Consolidator
from workflows.tool_call.host import WORKFLOW_NAME, build_workflow_spec

SEED = REPO_ROOT / "test_data" / "personas" / "spamuel" / "canon.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--mcp", action="append", required=True, help="name=url of one MCP server")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / ".runtime" / "toolcall-test")
    args = parser.parse_args()
    out = (args.out / time.strftime("run-%Y%m%d-%H%M%S")).resolve()
    out.mkdir(parents=True)
    os.environ.setdefault("PM_PROMPT_LOG_DIR", str(out / "llm_logs"))
    servers = [dict(zip(("name", "url"), item.split("=", 1))) for item in args.mcp]
    tool_call = ToolCallConfig.model_validate({"mcp_servers": servers})
    config = load_config(args.config)
    store = MemoryStore.from_yaml(SEED, checkpoint=out / "ghost.json")
    consolidator = Consolidator(store, clock=time.time)
    consolidations = []
    supervisor = ManagedRunSupervisor(
        store, runtime_root=out / "runtime", workflows={WORKFLOW_NAME: build_workflow_spec(tool_call)},
        clock=time.time, task_root=out / "managed_tasks",
        workflow_models=WorkflowModels(config, ModelActivation()),
        on_terminal=lambda receipt: consolidations.append(
            consolidator.consolidate(receipt["run_id"], receipt["artifact_path"])))
    started = supervisor.start(action_id="toolcall-check", workflow_name=WORKFLOW_NAME,
                               instructions=args.instruction)
    print("started", started["run_id"], flush=True)
    supervisor.join(started["run_id"])
    receipt = supervisor.status(started["run_id"])
    # A repeated terminal hook must reuse the first consolidation.
    consolidations.append(consolidator.consolidate(receipt["run_id"], receipt["artifact_path"]))
    report_file = Path(receipt["artifact_path"]) / "outbox" / "report.yaml"
    records = []
    for receipt_row in consolidations[:1]:
        for record_id in receipt_row["memory_ids"]:
            record = store.get_knoxel_by_id(record_id)
            media = record_media(record)
            records.append({"id": record.id, "content": record.content,
                            "images": [image.model_dump(mode="json", exclude={"data_b64"}) for image in media.images],
                            "links": {key: record.metadata[key] for key in
                                      ("action_id", "source_action_schema_id", "correlation_id", "run_id")
                                      if key in record.metadata}})
    summary = {"receipt": {key: value for key, value in receipt.items() if key != "events"},
               "events": [row["message"] for row in receipt["events"]],
               "report": report_file.read_text(encoding="utf-8") if report_file.is_file() else None,
               "consolidations": consolidations, "records": records}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
