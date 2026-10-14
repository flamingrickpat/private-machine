"""Execute one real pm-coder session, then verify the requested bytes."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pm_workflows.drivers import PmCoderDriver
from pm_workflows.inputs import relative_path
from pm.llm.prompt_log import log_directory


class NoteSpec(BaseModel):
    """Exact task-relative output and UTF-8 content. No newline is implied."""

    model_config = ConfigDict(extra="forbid")
    path: str = "outbox/note.txt"
    text: str = Field(min_length=1)


def _write_note(context) -> dict:
    """Lease the model, run the agent, and publish independently verified evidence.

    Cancellation stops before the next cooperative boundary. File effects can
    already exist. A cancelled run cannot publish a success report.
    """
    runtime = context.runtime
    environment = runtime.environment
    request = yaml.safe_load(environment.read_text("inbox/request.yaml"))
    original = request.get("_public_original_request")
    spec_data = None if original else request.get("note_spec")
    if spec_data is None:
        spec_data = runtime.tools["note_spec"](original or request["instruction"])
        if not spec_data.pop("requested", False):
            raise ValueError("The original request does not specify a new note operation.")
    spec = NoteSpec.model_validate(spec_data)
    path = relative_path(spec.path)
    if not path.startswith("outbox/"):
        raise ValueError("A note path must remain inside the task outbox.")
    environment.check_file_boundary(path, "outbox")
    request["note_spec"] = spec.model_dump()
    environment.write_text("inbox/request.yaml", yaml.safe_dump(request, allow_unicode=True))
    runtime.check_cancelled()
    if runtime.classification_session is None:
        raise ValueError("A note task requires the configured model session.")
    with runtime.classification_session("Default", True, runtime.stop_event) as model:
        agent_runtime = replace(runtime, agent_model=model, mcp_servers={})
        driver = PmCoderDriver(log_root=log_directory() / "agents", timeout_seconds=900)
        result = driver.run_session(
            run_id=context.task_id + "-note", attempt=1,
            skill=str(Path(__file__).with_name("note.md")),
            prompt="Write the note specified in inbox/request.yaml. Return only the requested JSON result.",
            work_dir=context.workspace, runtime=agent_runtime,
            trace_file=context.kernel_data / "note-agent-trace.jsonl",
        )
    runtime.check_cancelled()
    target = environment._path(path)
    actual = target.read_bytes() if target.is_file() else None
    expected = spec.text.encode("utf-8")
    verified = actual == expected and result.exit_code == 0
    report = {"schema": "pm.note-verification.v1", "accepted": verified,
              "summary": (f"Verified {path}: {len(expected)} exact UTF-8 bytes." if verified
                          else "The requested note bytes did not pass verification."),
              "note_spec": spec.model_dump(), "expected_hex": expected.hex(),
              "actual_hex": actual.hex() if actual is not None else None,
              "sha256": hashlib.sha256(actual).hexdigest() if actual is not None else None,
              "verified": verified, "agent_session": result.session_ref, "agent_log_root": str(driver.log_root),
              "original_request": original, "original_request_id": request.get("_public_original_request_id"),
              "agent_result": result.result_json}
    runtime.check_cancelled()
    environment.write_text("outbox/report.yaml", yaml.safe_dump(report, allow_unicode=True, sort_keys=False))
    return {"status": "done", "summary": report["summary"]}


def run(context) -> dict:
    """Retain expected input, model, and I/O failures as rejected task evidence.

    Cancellation still propagates to the worker. Unexpected logic errors keep
    their traceback. A rejected report cannot establish successful note work.
    """
    from pm.llm.http_client import ModelError
    from workflows.main_cognition.stage_support import StageOutputError
    try:
        return _write_note(context)
    except (ModelError, StageOutputError, ValueError, OSError, TimeoutError) as error:
        context.runtime.check_cancelled()
        report = {"schema": "pm.note-verification.v1", "accepted": False, "verified": False,
                  "summary": "The requested note was not verified.",
                  "error": f"{type(error).__name__}: {error}"}
        context.runtime.environment.write_text("outbox/report.yaml",
            yaml.safe_dump(report, allow_unicode=True, sort_keys=False))
        return {"status": "done", "summary": report["summary"]}


def build_note_tools(model):
    """Bind exact instruction extraction. This tool never writes the note."""
    def factory(run_id, environment):
        def note_spec(instruction: str):
            from workflows.main_cognition.stage_support import extract_json
            raw = model.complete([
                {"role": "system", "content": "Determine whether this original message requests a new note-file write. "
                 "An apology, affection, frustration, remembered task, or workflow status is not a new note request. "
                 "Do not treat the message itself as note content unless it explicitly requests that. "
                 "Return only JSON with requested (boolean), path, and text. If no new note is requested, "
                 "set requested false and text empty. Otherwise extract the exact requested text and path; "
                 "default path is outbox/note.txt. Do not add a newline or change the requested text."},
                {"role": "user", "content": instruction},
            ], temperature=0.1, enable_thinking=False)
            return extract_json(raw)
        return {"note_spec": note_spec}
    return factory


__all__ = ["NoteSpec", "build_note_tools", "run"]
