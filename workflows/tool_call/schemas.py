"""Inputs, agent claims, host evidence, and the terminal report of one tool goal.

The launch context (:class:`ToolCallContext`) is pinned to the cycle that
chose the action when the run has a source action. The report
(:class:`ToolCallReport`, ``pm.tool-call-report.v1``) keeps the agent's own
assessment beside the host evidence. The host never rates quality, and
``verified`` stays false: no generic verifier exists for a tool goal.
"""
from __future__ import annotations

from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

REPORT_SCHEMA = "pm.tool-call-report.v1"
CONTEXT_SCHEMA = "pm.tool-call-context.v1"
#: The run file of the host report. The agent cannot claim it as an artifact.
REPORT_PATH = "outbox/report.yaml"

Outcome = Literal["succeeded", "failed", "blocked", "uncertain", "cancelled", "interrupted"]


class PinnedDispatch(BaseModel):
    """The dispatch identity that self-management pinned into the run arguments.

    ``action_schema_id`` is the stored action record of the cycle that chose
    the action; it holds the frozen action request. It is None for a run that
    no cycle started, for example a direct operator start. ``tool_goal`` is
    the short optional goal that routing can add to the decision text.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    action_schema_id: int | None = Field(default=None, alias="_public_action_schema_id")
    public_reply_requested: bool = Field(default=False, alias="_public_reply_requested")
    original_request_id: int | None = Field(default=None, alias="_public_original_request_id")
    tool_goal: str = ""


class SourceStimulus(BaseModel):
    """One stimulus of the stimulus group that the source decision answered."""

    model_config = ConfigDict(extra="forbid")

    knoxel_id: int
    source: str = ""
    kind: str = ""
    text: str = ""


class SourceAction(BaseModel):
    """The action that started the run, as the source cycle stored it.

    ``decision`` is the companion's own decision text, unchanged.
    ``shard_story`` is the shard part of the source cycle's story prefix: the
    conscious content that the decision read. ``workspace`` is the cycle's
    subjective experience for the group, and ``affect`` its feeling in words.
    """

    model_config = ConfigDict(extra="forbid")

    action_schema_id: int
    action_id: str
    tick_id: int
    correlation_id: str = ""
    decision: str
    action_type: str
    group: str
    partner_name: str = ""
    stimuli: list[SourceStimulus] = Field(default_factory=list)
    shard_names: list[str] = Field(default_factory=list)
    shard_story: str = ""
    workspace: str = ""
    affect: str = ""


class ContextRecord(BaseModel):
    """One accepted record that the session context shows, with its revision at capture."""

    model_config = ConfigDict(extra="forbid")

    record_id: int
    revision: int
    record_class: str


class TurnSummary(BaseModel):
    """One complete GWT turn: its tick and its broadcast summary."""

    model_config = ConfigDict(extra="forbid")

    tick_id: int
    summary: str


class ToolCallContext(BaseModel):
    """The frozen context of one tool session.

    ``context_source`` is ``dispatch`` when the context comes from the source
    action's cycle, and ``launch`` when the run has no source action. A launch
    context is never labeled as the context that caused the action.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_id: Literal["pm.tool-call-context.v1"] = Field(default=CONTEXT_SCHEMA, alias="schema")
    context_source: Literal["dispatch", "launch"]
    captured_tick_id: int
    companion_name: str
    user_name: str
    character_card: str
    source_action: SourceAction | None = None
    gwt_turns: list[TurnSummary] = Field(default_factory=list)
    records: list[ContextRecord] = Field(default_factory=list)
    missing_record_ids: list[int] = Field(default_factory=list)


class ToolCallRequest(BaseModel):
    """The run request: the instruction, the pinned dispatch, and the context.

    ``instruction`` is the decision text, unchanged, or the operator's
    instruction for a direct start. Other managed-run fields stay as extras.
    """

    model_config = ConfigDict(extra="allow")

    run_id: str = Field(min_length=1)
    instruction: str = Field(min_length=1)
    tool_goal: str = ""
    tool_call_context: ToolCallContext

    @field_validator("run_id", "instruction")
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("The run identifier and instruction must contain text.")
        return value


class ArtifactClaim(BaseModel):
    """A run file or an external reference that the agent names. A claim does not prove existence.

    A local path names a file directly inside the run's outbox. A claim of
    the host report path is kept, but the host ignores it (see
    ``steps.inspect_artifacts``). A URI is an
    unverified external reference, for example ``mcp://blender/C:/tmp/cat.jpg``.
    ``server`` names the MCP server that made the artifact, when known.
    """

    model_config = ConfigDict(extra="forbid")

    path: str | None = None
    uri: str | None = None
    media_type: str = "application/octet-stream"
    description: str = ""
    server: str = ""

    @model_validator(mode="after")
    def _one_location(self) -> ArtifactClaim:
        if (self.path is None) == (self.uri is None):
            raise ValueError("An artifact requires exactly one path or URI.")
        if self.path is not None:
            path = PurePosixPath(self.path)
            if (
                "\\" in self.path
                or ":" in self.path
                or path.is_absolute()
                or ".." in path.parts
                or len(path.parts) != 2
                or path.parts[0] != "outbox"
                or self.path != path.as_posix()
            ):
                raise ValueError("An artifact path must name a file directly inside outbox.")
        if self.uri is not None and not self.uri.strip():
            raise ValueError("An artifact URI must contain text.")
        return self


class AgentOutcome(BaseModel):
    """The agent's terminal assessment. ``quality_assessment`` is its own opinion."""

    model_config = ConfigDict(extra="forbid")

    outcome: Literal["succeeded", "failed", "blocked", "uncertain"]
    summary: str = Field(min_length=1)
    artifacts: list[ArtifactClaim] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)
    quality_assessment: str = ""
    uncertainty: list[str] = Field(default_factory=list)


class ArtifactEvidence(BaseModel):
    """Host evidence for local bytes, or an unverified external reference.

    A digest proves byte identity. It does not prove visual quality or goal completion.
    """

    model_config = ConfigDict(extra="forbid")

    claim: ArtifactClaim
    file_verified: bool = False
    local_path: str | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    reason: str = ""


class ToolCallSeen(BaseModel):
    """One MCP tool call that the native session record shows. ``arguments`` is the raw JSON text."""

    model_config = ConfigDict(extra="forbid")

    server: str
    tool: str
    arguments: str = ""


class ToolImage(BaseModel):
    """One image that an MCP tool returned in the session, kept as a run file.

    The host wrote the bytes to ``path`` in the outbox. ``sha256`` and
    ``size_bytes`` describe those bytes.
    """

    model_config = ConfigDict(extra="forbid")

    path: str
    server: str
    tool: str
    media_type: str
    sha256: str
    size_bytes: int


class ToolCallReport(BaseModel):
    """The host-owned terminal report for consolidation and a later public reply.

    ``accepted`` means that the agent reported success without a known local
    artifact error. The supervisor maps ``accepted: false`` to a failed
    receipt; ``outcome`` keeps the more specific goal outcome. ``verified``
    is always false. The source fields link the report to the action that
    started the run.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_id: Literal["pm.tool-call-report.v1"] = Field(default=REPORT_SCHEMA, alias="schema")
    run_id: str
    instruction: str
    tool_goal: str = ""
    outcome: Outcome
    summary: str
    accepted: bool = False
    verified: Literal[False] = False
    rejection_reason: str = ""
    agent_outcome: AgentOutcome | None = None
    artifacts: list[ArtifactEvidence] = Field(default_factory=list)
    tool_images: list[ToolImage] = Field(default_factory=list)
    tool_calls: list[ToolCallSeen] = Field(default_factory=list)
    context_source: Literal["dispatch", "launch", ""] = ""
    action_id: str = ""
    action_schema_id: int | None = None
    source_tick_id: int | None = None
    correlation_id: str = ""
    stimulus_ids: list[int] = Field(default_factory=list)
    context_tick_id: int | None = None
    context_turn_ids: list[int] = Field(default_factory=list)
    context_record_revisions: dict[int, int] = Field(default_factory=dict)
    instruction_revision: int | None = None
    final_instruction_revision: int | None = None
    mcp_server_names: list[str] = Field(default_factory=list)
    agent_session: str = ""
    trace_path: str = ""
    error_type: str = ""


__all__ = [
    "AgentOutcome", "ArtifactClaim", "ArtifactEvidence", "CONTEXT_SCHEMA", "ContextRecord", "Outcome",
    "PinnedDispatch", "REPORT_PATH", "REPORT_SCHEMA", "SourceAction", "SourceStimulus", "ToolCallContext",
    "ToolCallReport", "ToolCallRequest", "ToolCallSeen", "ToolImage", "TurnSummary",
]
