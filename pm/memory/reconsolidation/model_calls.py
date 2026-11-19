"""Validated model calls of memory reconsolidation.

Every call returns one JSON object that a pydantic schema validates. An
invalid answer gets a repair turn with the validation error. A provider
failure (``ModelError``: a filtered, cut-off, or empty completion) repeats
the same call. Transport errors and HTTP 429/5xx already retry inside
``pm.llm.http_client.JsonClient``. Other errors and cancellation propagate.

Calls of one stage can run in parallel with :meth:`ModelCalls.map`. The
number of workers is bounded. A failed call stops the calls that did not
start yet, and the first error propagates after all workers end.
"""
from __future__ import annotations

from threading import Event, Lock
from dataclasses import dataclass
from pm.llm.generation import GenerationSettings
from typing import TypeVar
from pydantic import BaseModel, ValidationError
from pm.constants import RECONSOLIDATION_TEMPERATURE, RECONSOLIDATION_WORKERS
from pm.llm.stage_model import LazyStageModel
from pm.llm.session import ModelSession
from pm.context_render.prompt import RenderedPrompt, RendererCapture, RendererInput
from pm.context_render.correction import CorrectionHistory, JsonCorrectionInput, render_json_correction
from pm.llm.structured import parse_response, JsonOutputError, extract_json as extract_object, ModelFormatError as ReconsolidationOutputError
from pm.memory.reconsolidation.schemas import SummaryAnswer, MentalStateAnswer, FactsAnswer, FilingAnswer, NarrativeAnswer, FactExtractionValidation, FactClassificationValidation
from pm.memory.reconsolidation.rendering import (
    SceneSummaryInput, TemporalSummaryInput, RecordedStateInput, FactExtractionInput, FactClassificationInput, NarrativeRenderInput,
    render_scene_summary, render_temporal_summary, render_recorded_state, render_fact_extraction, render_fact_classification, render_narrative,
)
T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class NarrativeRefinement:
    answer: NarrativeAnswer
    selected_ids: tuple[int, ...]


class ModelCalls:
    """Make validated model calls and count them.

    ``complete`` is a ``TextCompletion``, for example
    ``LazyStageModel.complete``. ``cancel`` reaches every call, so a stopped
    run also stops the calls in flight. ``calls`` counts every request that
    this object sent, repairs and repeats included.
    """

    def __init__(self, complete: LazyStageModel, *, cancel: Event | None = None,
                 temperature: float = RECONSOLIDATION_TEMPERATURE, workers: int = RECONSOLIDATION_WORKERS) -> None:
        self.model = complete
        self.cancel = cancel if cancel is not None else Event()
        self.session = ModelSession(complete, self.cancel)
        self.temperature = temperature
        self.generation = GenerationSettings(temperature=temperature, enable_thinking=False)
        self.workers = workers
        self.calls = 0
        self._lock = Lock()

    def json(self, rendered: RenderedPrompt, schema: type[T], *, context: BaseModel | None = None) -> T:
        original = rendered.turns
        operation = rendered.operation
        corrections = CorrectionHistory()
        settings = self.session.renderer_settings_for(self.generation)
        while True:
            with self._lock:
                self.calls += 1
            raw = self.session.complete(rendered.turns, settings=self.generation)
            try:
                return parse_response(raw, schema, validation_context=context)
            except (JsonOutputError, ValidationError) as error:
                diagnostic = corrections.record_rejection(str(error))
                input = JsonCorrectionInput(operation, original, raw, diagnostic)
                with RendererCapture(self.session.capture_settings("memory.correction"),
                        RendererInput("memory.correction", input, settings)) as capture:
                    rendered = render_json_correction(input, settings)
                    capture.save_output(rendered)

    def summarize_scene(self, input: SceneSummaryInput) -> SummaryAnswer:
        settings = self.session.renderer_settings_for(self.generation)
        with RendererCapture(self.session.capture_settings("memory.scene_summary"), RendererInput("memory.scene_summary", input, settings)) as capture:
            rendered = render_scene_summary(input, settings)
            capture.save_output(rendered)
        return self.json(rendered, SummaryAnswer)

    def summarize_temporal(self, input: TemporalSummaryInput) -> SummaryAnswer:
        settings = self.session.renderer_settings_for(self.generation)
        with RendererCapture(self.session.capture_settings("memory.temporal_summary"), RendererInput("memory.temporal_summary", input, settings)) as capture:
            rendered = render_temporal_summary(input, settings)
            capture.save_output(rendered)
        return self.json(rendered, SummaryAnswer)

    def describe_recorded_state(self, input: RecordedStateInput) -> MentalStateAnswer:
        settings = self.session.renderer_settings_for(self.generation)
        with RendererCapture(self.session.capture_settings("memory.recorded_state"), RendererInput("memory.recorded_state", input, settings)) as capture:
            rendered = render_recorded_state(input, settings)
            capture.save_output(rendered)
        return self.json(rendered, MentalStateAnswer)

    def extract_facts(self, input: FactExtractionInput, line_count: int) -> FactsAnswer:
        settings = self.session.renderer_settings_for(self.generation)
        with RendererCapture(self.session.capture_settings("memory.fact_extraction"), RendererInput("memory.fact_extraction", input, settings)) as capture:
            rendered = render_fact_extraction(input, settings)
            capture.save_output(rendered)
        return self.json(rendered, FactsAnswer, context=FactExtractionValidation(line_count=line_count))

    def classify_facts(self, input: FactClassificationInput) -> FilingAnswer:
        settings = self.session.renderer_settings_for(self.generation)
        with RendererCapture(self.session.capture_settings("memory.fact_classification"), RendererInput("memory.fact_classification", input, settings)) as capture:
            rendered = render_fact_classification(input, settings)
            capture.save_output(rendered)
        return self.json(rendered, FilingAnswer, context=FactClassificationValidation(fact_count=len(input.facts), allowed_subjects=(input.companion_name, input.user_name, "both", "other")))

    def refine_narrative(self, input: NarrativeRenderInput) -> NarrativeRefinement:
        settings = self.session.renderer_settings_for(self.generation)
        with RendererCapture(self.session.capture_settings("memory.narrative"), RendererInput("memory.narrative", input, settings)) as capture:
            rendered = render_narrative(input, settings)
            capture.save_output(rendered)
        selected_ids = tuple(item.record_id for item in rendered.evidence if item.included)
        return NarrativeRefinement(self.json(rendered, NarrativeAnswer), selected_ids)


__all__ = ["ModelCalls", "ReconsolidationOutputError", "extract_object"]
