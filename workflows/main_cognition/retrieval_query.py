from __future__ import annotations

from dataclasses import dataclass
from threading import Event
from pm.constants import Sampler
from pm.context_render.turns import Role, Turn
from pm.context_render.prompt import RenderSettings, RenderedPrompt, require_rendered_prompt, RendererCapture, RendererInput
from pm.llm.stage_model import LazyStageModel
from pm.llm.session import ModelSession
from pm.llm.generation import GenerationSettings
from pm.llm.structured import complete_json


from pm.memory.retrieval import RetrievalQueryRewrite as QueryRewrite


@dataclass(frozen=True, slots=True)
class QueryRenderInput:
    text: str


def render_retrieval_query(input: QueryRenderInput, settings: RenderSettings) -> RenderedPrompt:
    turns = (Turn(Role.System, SYSTEM_PROMPT), Turn(Role.User, input.text))
    return require_rendered_prompt("retrieval query", turns, settings)


class QueryRewriter:
    def __init__(self, model: LazyStageModel) -> None:
        self.model = model

    def rewrite(self, text: str, cancel: Event) -> QueryRewrite:
        session = ModelSession(self.model, cancel)
        input = QueryRenderInput(text)
        settings = session.render_settings()
        with RendererCapture(session.capture_settings("retrieval query"),
                             RendererInput("retrieval query", input, settings)) as capture:
            prompt = render_retrieval_query(input, settings)
            capture.save_output(prompt)
        return complete_json(session, prompt.turns, QueryRewrite, settings=GenerationSettings(temperature=Sampler.TempInstruct, enable_thinking=False))


SYSTEM_PROMPT = 'Rewrite this message as a short memory search query. Search for original user reports or concrete executed events, not earlier answers to recall requests. Remove boilerplate about remembering, recorded history, and supporting conversations. Retain actual topics, outcomes, corrections, and dates. For an unspecified failed method, search for a practical attempt that did not work and its reported outcome. Never invent a specific topic. Also choose evidence_use: event for factual recall, preferences, or past events; utterance for what someone said; general for other requests or uncertainty. A request to evaluate newly reported advice is general, even when it mentions a familiar topic. What do you actually know about that claim asks for supported knowledge, not a past conversation. Use event only when the user asks to recall an event, retained preference, name, or plan; use utterance when recalling what someone previously said. Return only JSON with query and evidence_use.'
