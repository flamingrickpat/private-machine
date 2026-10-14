"""Compare small real decisions with preselected context and direct memory tools.

The gate keeps the full Spamuel seed in isolated checkpoints. Added preferences
are synthetic user inputs through UserInputProxy. It performs no Minecraft
actions and assigns no human ratings. Failed trials remain in the receipt.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import statistics
import shutil
import sys
from threading import Event
import time
from typing import Literal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict
from pm.bus.bus import BusSystem
from pm.host.folder_target import FolderEnvironment
from pm.llm.activation import ModelActivation, ChatClient
from pm.llm.config import load_config, resolve_connection
from pm.llm.workflow_sessions import WorkflowModels
from pm.memory.agent_tools import MemoryReadSession
from pm.memory.embeddings import EmbeddingService
from pm.memory.retrieval import retrieve
from pm.memory.store import MemoryStore
from pm.runs.draft_session import DraftValidation, run_draft_session
from pm_workflows.runtime import WorkflowRuntime
from workflows.minecraft.host import project_memory_config
from workflows.public_speech.user_input import UserInputProxy


class NoteDecision(BaseModel):
    """A small editable decision. Static checks do not grade its explanation."""

    model_config = ConfigDict(extra="forbid")
    layout: Literal["compact", "verbatim", "ask"]
    effort: Literal["quick", "strict"]
    evidence_ids: list[int] = []
    explanation: str


CASES = [
    {"id": "paraphrase", "query": "How detailed does Jesse want the A14B pocket note?",
     "history": ["For the A14B pocket note, terse layout is fine. I value speed over presentation polish."],
     "request": "Prepare a small A14B pocket note. Choose its layout using my earlier preference.",
     "expected": [("compact", "quick")]},
    {"id": "new_constraint", "query": "A14B pocket note layout preference",
     "history": ["For the A14B pocket note, compact layout is fine.",
                 "Correction: the A14B pocket note now needs the source wording without any edits."],
     "request": "This A14B pocket note is a critical transcript. Preserve the source wording exactly and use strict review.",
     "expected": [("verbatim", "strict")]},
    {"id": "absent_preference", "query": "A14B zinnia note layout preference", "history": [],
     "request": "Make an A14B zinnia note. Either layout is acceptable. Prefer a quick practical choice without more questions.",
     "expected": [("compact", "quick"), ("verbatim", "quick")]},
]


def write(path: Path, value: object) -> None:
    """Retain a JSON receipt under the caller's ignored evidence directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def trial(case: dict, mode: str, number: int, out: Path, models, embeddings, *, repair=False) -> dict:
    """Run one real inference trial and inspect the accepted backend result."""
    target = out / f"{case['id']}-{mode}-{number:02d}"
    target.mkdir()
    cancel = Event()
    source_ids = json.loads((out / "fixtures" / f"{case['id']}-sources.json").read_text(encoding="utf-8"))
    shutil.copyfile(out / "fixtures" / f"{case['id']}.json", target / "checkpoint.json")
    # Each trial restores its own stopped owner. No live checkpoint is reopened.
    store = MemoryStore.from_checkpoint(target / "checkpoint.json")
    config = project_memory_config(store, cancel, embeddings)
    traces = []
    before = store.state_dict()
    started = time.monotonic()
    with MemoryReadSession(replace(config, trace=traces.append)) as memory:
        setup = time.monotonic() - started
        runtime = WorkflowRuntime(FolderEnvironment(target / "workspace", "bash"), cancel, {}, propagate_python_errors=True)
        base = ("Decide the note layout and effort for the current request. Memory is attributed evidence. "
                "Interpret original statements and newer corrections. A current explicit requirement takes precedence. "
                "Use practical choices when preference evidence is absent. Match effort to the request's consequences. "
                "Do not claim that you wrote a note. This decision has no file or world effects.\nRequest: " + case["request"])
        with models.session("Default", False, cancel, receipt_path=target / "model.jsonl") as model:
            if mode == "preselected":
                context = asyncio.run(retrieve(memory.snapshot, case["query"], embeddings, limit=8, cancel=cancel,
                                               prepare_limit=memory.grant.embedding_batch_limit))
                write(target / "retrieval.json", context)
                connection = models.connections["Default"]
                client = ChatClient(connection, models.owner, live_test=True)
                schema = NoteDecision.model_json_schema()
                messages = [{"role": "system", "content": base + "\nReturn one JSON decision."},
                            {"role": "user", "content": json.dumps({"memories": context["matches"]})}]
                response = asyncio.run(client.complete(messages, enable_thinking=False, parallel_tool_calls=False,
                    response_format={"type": "json_schema", "json_schema": {"name": "NoteDecision", "strict": True, "schema": schema}}))
                raw = response.choices[0].message.content
                decision = NoteDecision.model_validate_json(raw)
                write(target / "response.json", response.model_dump(mode="json"))
                stats = {"requests": client.client.request_count, "tokens": response.model_dump(mode="json").get("usage"),
                         "output_characters": len(raw), "repair_turns": 0}
            else:
                draft = {}
                server = FastMCP("note")
                @server.tool()
                def decision_set(layout: Literal["compact", "verbatim", "ask"], effort: Literal["quick", "strict"],
                                 explanation: str, evidence_ids: list[int] = []) -> dict:
                    """Set one typed note decision. It remains an uncommitted draft."""
                    candidate = NoteDecision(layout=layout, effort=effort, explanation=explanation, evidence_ids=evidence_ids)
                    if any(key not in memory.records for key in evidence_ids):
                        return {"ok": False, "error": "An evidence ID is outside the permitted memory view."}
                    draft["decision"] = candidate
                    write(target / "draft.json", candidate.model_dump(mode="json"))
                    return {"ok": True}
                attempts = 0
                def validate(text):
                    nonlocal attempts
                    attempts += 1
                    if repair and attempts == 1:
                        return DraftValidation(False, feedback="Retain the same draft. Add the suffix 'reviewed' to its explanation through decision_set.")
                    if "decision" not in draft:
                        return DraftValidation(False, feedback="Set the decision with decision_set.")
                    if any(key not in memory.accessed for key in draft["decision"].evidence_ids):
                        return DraftValidation(False, feedback="Inspect each cited evidence ID with memory_get before acceptance.")
                    return DraftValidation(True, draft["decision"])
                result = run_draft_session(replace(runtime, agent_model=replace(model, enable_thinking=False)), run_id=target.name,
                    prompt=base + "\nUse decision_set to set your draft. Optional memory tools can search original statements. "
                        "Inspect original records that support your choice. Search is unnecessary when the request is sufficient. "
                        "End the turn with brief prose. Do not use shell tools.",
                    servers={"note": server}, memory=memory, validate=validate, artifact_dir=target / "native", timeout_seconds=240)
                decision = result.value
                turns = [json.loads(line) for line in (target / "native/draft-turns.jsonl").read_text(encoding="utf-8").splitlines()]
                stats = {"requests": sum(row["tokens_used"].get("requests", 0) for row in turns),
                         "tool_calls": sum(row["tokens_used"].get("tool_calls", 0) for row in turns),
                         "tokens": [row["tokens_used"] for row in turns], "repair_turns": result.validation_attempts - 1,
                         "same_session": len({row["session_ref"] for row in turns}) == 1}
            checks = {"independent_expected_fields": (decision.layout, decision.effort) in case["expected"],
                      "accepted_memory_unchanged": store.state_dict() == before,
                      "cited_ids_visible": all(key in memory.records for key in decision.evidence_ids)}
            if case["id"] == "paraphrase" and mode == "tools":
                checks["original_preference_inspected"] = source_ids[0] in memory.accessed
                checks["original_preference_cited"] = source_ids[0] in decision.evidence_ids
            if repair:
                checks["retained_repair_session"] = stats["repair_turns"] >= 1 and stats["same_session"]
            result = {"case": case["id"], "mode": mode, "number": number,
                      "elapsed_seconds": time.monotonic() - started, "setup_seconds": setup,
                      "decision": decision.model_dump(mode="json"), "stats": stats, "checks": checks,
                      "passed": all(checks.values()), "synthetic_source_ids": source_ids,
                      "snapshot_id": memory.snapshot_id, "accessed_revisions": dict(memory.accessed)}
    write(target / "memory.json", traces)
    write(target / "receipt.json", result)
    print(json.dumps({key: result[key] for key in ("case", "mode", "elapsed_seconds", "passed")}), flush=True)
    return result


def run(config_path: Path, out: Path, trials: int, fixtures: Path | None = None) -> None:
    """Retain each trial, its failures, and medians without a gameplay claim."""
    out = out.resolve()
    if not out.is_relative_to(ROOT / ".runtime"):
        raise ValueError("Keep inference evidence under .runtime.")
    out.mkdir(parents=True, exist_ok=False)
    os.environ["PM_PROMPT_LOG_DIR"] = str(out / "prompts")
    config = load_config(config_path)
    models = WorkflowModels(config, ModelActivation())
    embeddings = EmbeddingService(config.embedding, live_test=True)
    receipt = {"seed_sha256": hashlib.sha256((ROOT / "test_data/personas/spamuel/canon.yaml").read_bytes()).hexdigest(),
               "model": config.chat[next(iter(config.chat))].model, "trials": [], "human_review": "pending",
               "generation": {"enable_thinking": False, "temperature": "provider_default", "parallel_tool_calls": False},
               "scope": "Small note decisions with full history. Synthetic preferences. No gameplay or public dialogue."}
    for case in CASES:
        fixture = out / "fixtures" / f"{case['id']}.json"
        fixture.parent.mkdir(exist_ok=True)
        if fixtures is not None:
            shutil.copyfile(fixtures / fixture.name, fixture)
            shutil.copyfile(fixtures / f"{case['id']}-sources.json", out / "fixtures" / f"{case['id']}-sources.json")
        else:
            store = MemoryStore.from_yaml(ROOT / "test_data/personas/spamuel/canon.yaml", checkpoint=fixture)
            user = UserInputProxy("user_input", BusSystem(), lambda error: None, time.time, store, lambda text: None)
            source_ids = []
            for text in case["history"]:
                user.submit(text, session="synthetic-a14b")
                source_ids.append(max(store.revisions))
            if source_ids:
                for index in range(3):
                    user.submit(f"The unrelated page counter is {index}.", session="synthetic-a14b")
            else:
                store.commit(store.begin_batch("fixture-save"))
            write(out / "fixtures" / f"{case['id']}-sources.json", source_ids)
            del store, user
        print(f"Prepared full-history fixture: {case['id']}", flush=True)
        for number in range(1, trials + 1):
            modes = ("preselected", "tools") if number % 2 else ("tools", "preselected")
            for mode in modes:
                try:
                    result = trial(case, mode, number, out, models, embeddings, repair=case["id"] == "paraphrase" and mode == "tools" and number == 1)
                except Exception as error:
                    result = {"case": case["id"], "mode": mode, "number": number,
                              "passed": False, "error": f"{type(error).__name__}: {error}"}
                    print(json.dumps(result), flush=True)
                receipt["trials"].append(result)
                write(out / "receipt.json", receipt)
    receipt["summary"] = {}
    for mode in ("preselected", "tools"):
        durations = [row["elapsed_seconds"] for row in receipt["trials"] if row["mode"] == mode and "elapsed_seconds" in row]
        receipt["summary"][mode] = {"completed": len(durations), "median_seconds": statistics.median(durations) if durations else None,
                                    "range_seconds": [min(durations), max(durations)] if durations else None}
    receipt["passed"] = all(row["passed"] for row in receipt["trials"])
    write(out / "receipt.json", receipt)
    if not receipt["passed"]:
        raise AssertionError("A small real decision comparison failed. Inspect the retained trials.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--fixtures", type=Path)
    args = parser.parse_args()
    run(args.config, args.out, args.trials, args.fixtures)
