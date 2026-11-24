from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pm.memory.temporal_context import TemporalCoverageSelection
import pickle
import traceback as traceback_module
from types import TracebackType

from pydantic import BaseModel, ConfigDict, Field

from pm.context_render.budget import RequiredContextTooLarge, TokenEstimator
from pm.context_render.turns import Media, Text, Turn


class RenderSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    context_window: int = Field(ge=32000)
    input_fraction: float = Field(gt=0, lt=1)
    reserved_output_tokens: int = Field(gt=0)
    protocol_tokens: int = Field(default=0, ge=0)
    correction_tokens: int = Field(default=1024, ge=0,
        description="Input tokens held back before evidence selection for bounded validation feedback.")
    estimate_factor: float = Field(default=1.5, ge=1)

    @property
    def input_ceiling(self) -> int:
        return self.transport_input_ceiling - self.correction_tokens

    @property
    def transport_input_ceiling(self) -> int:
        return min(int(self.context_window * self.input_fraction),
                   self.context_window - self.reserved_output_tokens) - self.protocol_tokens

    def without_correction_reserve(self) -> RenderSettings:
        return RenderSettings(context_window=self.context_window, input_fraction=self.input_fraction,
            reserved_output_tokens=self.reserved_output_tokens, protocol_tokens=self.protocol_tokens,
            correction_tokens=0, estimate_factor=self.estimate_factor)


@dataclass(frozen=True, slots=True)
class EvidenceDecision:
    record_id: int
    score: float
    included: bool
    reason: str


@dataclass(frozen=True, slots=True)
class TemporalHistoryMeasurement:
    mandatory_tokens: int
    minimum_cover_prompt_tokens: int
    selected_history_prompt_tokens: int
    available_history_tokens: int
    refinement_attempts: int


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    operation: str
    turns: tuple[Turn, ...]
    raw_text: str
    estimated_tokens: int
    input_ceiling: int
    evidence: tuple[EvidenceDecision, ...]
    selected_groups: tuple[str, ...] = ()
    temporal_history: TemporalCoverageSelection | None = None
    temporal_history_measurement: TemporalHistoryMeasurement | None = None
    visible_source_ids: tuple[int, ...] = ()
    required_source_ids: tuple[int, ...] = ()


def require_rendered_prompt(operation: str, turns: tuple[Turn, ...], settings: RenderSettings,
                            evidence: tuple[EvidenceDecision, ...] = (), *,
                            selected_groups: tuple[str, ...] = ()) -> RenderedPrompt:
    estimator = TokenEstimator(estimate_factor=settings.estimate_factor)
    estimated_tokens = estimator.estimate(turns)
    if estimated_tokens > settings.input_ceiling:
        raise RequiredContextTooLarge(
            f"{operation}: estimated input {estimated_tokens}, ceiling {settings.input_ceiling}, "
            f"context {settings.context_window}, reserved output {settings.reserved_output_tokens}, "
            f"protocol {settings.protocol_tokens}, correction reserve {settings.correction_tokens}.")
    rendered_turns: list[str] = []
    for turn in turns:
        parts: list[str] = list(turn.content.notes)
        for part in turn.content.parts:
            if isinstance(part, Text):
                parts.append(part.text)
            elif isinstance(part, Media):
                parts.append(f"[attachment {part.kind.value}: {part.name}; {part.mime_type}]")
            else:
                for frame in part.frames:
                    parts.append(f"[video frame at {frame.seconds:g}s: {frame.image.mime_type}]")
        state = turn.provider_state
        if state is not None:
            if state.reasoning is not None:
                parts.append("Provider reasoning:\n" + state.reasoning)
            if state.reasoning_content is not None:
                parts.append("Provider reasoning content:\n" + state.reasoning_content)
            if state.reasoning_details_json is not None:
                parts.append("Opaque provider reasoning details:\n" + state.reasoning_details_json)
        if turn.tool_calls:
            parts.append(str(turn.tool_calls))
        rendered_turns.append(f"[{turn.role.value}]\n" + "\n".join(parts))
    raw_text = "\n\n".join(rendered_turns)
    return RenderedPrompt(operation, turns, raw_text, estimated_tokens, settings.input_ceiling, evidence, selected_groups,
                          visible_source_ids=tuple(sorted({row.record_id for row in evidence if row.included})))


@dataclass(frozen=True, slots=True)
class RendererCaptureSettings:
    directory: Path
    capture_input: bool = False
    capture_output: bool = False


class RendererCaptureOptions(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    directory: Path = Path(".runtime/renderer-captures")
    capture_input: bool = False
    capture_output: bool = False


@dataclass(frozen=True, slots=True)
class RendererInput[T]:
    operation: str
    input: T
    settings: RenderSettings


@dataclass(frozen=True, slots=True)
class RendererFailure:
    exception_type: str
    message: str
    traceback_text: str


class RendererCapture[T]:
    def __init__(self, capture: RendererCaptureSettings, input: RendererInput[T]) -> None:
        self.capture = capture
        self.input = input

    def __enter__(self) -> RendererCapture[T]:
        if self.capture.capture_input:
            self.capture.directory.mkdir(parents=True, exist_ok=True)
            with (self.capture.directory / "input.pickle").open("xb") as stream:
                pickle.dump(self.input, stream, protocol=pickle.HIGHEST_PROTOCOL)
        return self

    def save_output(self, output: RenderedPrompt) -> None:
        if self.capture.capture_output:
            self.capture.directory.mkdir(parents=True, exist_ok=True)
            with (self.capture.directory / "output.pickle").open("xb") as stream:
                pickle.dump(output, stream, protocol=pickle.HIGHEST_PROTOCOL)

    def __exit__(self, exception_type: type[BaseException] | None, exception: BaseException | None,
                 traceback: TracebackType | None) -> None:
        if exception is not None and self.capture.capture_output:
            self.capture.directory.mkdir(parents=True, exist_ok=True)
            failure = RendererFailure(type(exception).__qualname__, str(exception),
                                      "".join(traceback_module.format_exception(exception_type, exception, traceback)))
            with (self.capture.directory / "failure.pickle").open("xb") as stream:
                pickle.dump(failure, stream, protocol=pickle.HIGHEST_PROTOCOL)
