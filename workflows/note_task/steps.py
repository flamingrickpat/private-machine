"""Select note content, write exact UTF-8 bytes, and independently verify them."""
from __future__ import annotations

from collections.abc import Callable
import hashlib
from threading import Event
from typing import TYPE_CHECKING, Literal, TypedDict
from pydantic import JsonValue
from pm_workflows.working_target import WorkingEnvironment
from pm.host.folder_target import FolderEnvironment

from pm_workflows.python_role import RoleContext

from pm.constants import Sampler
from pm.utils import yaml_util
from pydantic import BaseModel, ConfigDict, Field
from pm_workflows.inputs import relative_path

if TYPE_CHECKING:
    from pm.llm.stage_model import LazyStageModel


class NoteSpec(BaseModel):
    """Exact task-relative output and UTF-8 content. No newline is implied."""

    model_config = ConfigDict(extra="forbid")
    path: str = "outbox/note.txt"
    text: str = Field(min_length=1)


class NoteRequest(BaseModel):
    """Validated task input; preserve kernel control fields on serialization."""

    model_config = ConfigDict(extra="allow")
    instruction: str
    note_spec: NoteSpec | None = None
    original_request: str = Field(default="", alias="_public_original_request")
    original_request_id: int | None = Field(default=None, alias="_public_original_request_id")


class NoteSelection(BaseModel):
    """Model boundary for a requested note operation."""

    requested: bool
    path: str = "outbox/note.txt"
    text: str


class NoteResult(TypedDict):
    """Kernel completion after independent byte verification."""

    status: Literal["done"]
    summary: str


def _write_note(context: RoleContext) -> NoteResult:
    """Write the selected note and publish independently verified evidence.

    Cancellation stops before the next cooperative boundary. File effects can
    already exist. A cancelled run cannot publish a success report.
    """
    runtime = context.runtime
    assert runtime is not None, "A note role requires its workflow runtime."
    environment = runtime.environment
    if not isinstance(environment, FolderEnvironment):
        raise TypeError("A note task requires its declared host-folder environment.")
    request = NoteRequest.model_validate(yaml_util.safe_load(environment.read_text("inbox/request.yaml")))
    original = request.original_request
    spec = None if original else request.note_spec
    if spec is None:
        selection = NoteSelection.model_validate(runtime.tools["note_spec"](original or request.instruction, cancel=runtime.stop_event))
        if not selection.requested:
            raise ValueError("The original request does not specify a new note operation.")
        spec = NoteSpec(path=selection.path, text=selection.text)
    path = relative_path(spec.path)
    if not path.startswith("outbox/"):
        raise ValueError("A note path must remain inside the task outbox.")
    environment.check_file_boundary(path, "outbox")
    request.note_spec = spec
    environment.write_text("inbox/request.yaml", yaml_util.safe_dump(request.model_dump(by_alias=True), allow_unicode=True))
    runtime.check_cancelled()
    # Content selection is stochastic; copying accepted text is not. Keep the
    # host write exact instead of asking an agent to transcribe YAML scalars.
    expected = spec.text.encode("utf-8")
    environment.write_binary(path, expected)
    runtime.check_cancelled()
    target = environment.root / path
    actual = target.read_bytes() if target.is_file() else None
    verified = actual == expected
    summary = (f"Verified {path}: {len(expected)} exact UTF-8 bytes." if verified
               else "The requested note bytes did not pass verification.")
    report = {"schema": "pm.note-verification.v1", "accepted": verified,
              "summary": summary,
              "note_spec": spec.model_dump(), "expected_hex": expected.hex(),
              "actual_hex": actual.hex() if actual is not None else None,
              "sha256": hashlib.sha256(actual).hexdigest() if actual is not None else None,
              "verified": verified,
              "original_request": original, "original_request_id": request.original_request_id}
    runtime.check_cancelled()
    environment.write_text("outbox/report.yaml", yaml_util.safe_dump(report, allow_unicode=True, sort_keys=False))
    if not verified:
        raise RuntimeError(f"Host note write did not preserve exact bytes: {path}")
    return {"status": "done", "summary": summary}


def run(context: RoleContext) -> NoteResult:
    """Execute the note; defects propagate to the application's fatal handler."""
    return _write_note(context)


def build_note_tools(
    model: LazyStageModel | None,
) -> Callable[[str, WorkingEnvironment], dict[str, Callable[..., dict[str, JsonValue]]]]:
    """Bind exact instruction extraction. This tool never writes the note.

    ``model`` is None in dummy cognition, where no run ever calls the
    tool; the extraction then fails at the model access, not silently.
    """

    def factory(run_id: str, environment: WorkingEnvironment) -> dict[str, Callable[..., dict[str, JsonValue]]]:
        """Return the extraction tool of one note run."""

        def note_spec(instruction: str, *, cancel: Event | None = None) -> dict[str, JsonValue]:
            """Return the parsed note request: whether to write, where, and exactly what text."""
            from pm.llm.structured import complete_json
            if model is None:
                raise RuntimeError("Note selection requires the configured model.")
            selection = complete_json(model.complete, [
                {"role": "system", "content": "Determine whether this original message requests a new note-file write. "
                 "An apology, affection, frustration, remembered task, or workflow status is not a new note request. "
                 "Do not treat the message itself as note content unless it explicitly requests that. "
                 "Return only JSON with requested (boolean), path, and text. If no new note is requested, "
                 "set requested false and text empty. Otherwise extract the exact requested text and path; "
                 "default path is outbox/note.txt. Do not add a newline or change the requested text."},
                {"role": "user", "content": instruction},
            ], NoteSelection, cancel=cancel, temperature=Sampler.TempInstruct, enable_thinking=False)
            return selection.model_dump(mode="json")
        return {"note_spec": note_spec}
    return factory


__all__ = ["NoteSpec", "build_note_tools", "run"]
