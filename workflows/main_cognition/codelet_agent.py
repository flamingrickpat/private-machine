"""Agent mode of a codelet: one pm-coder session with scoped memory tools.

The host grants :func:`build_codelet_agent` as the ``codelet_agent`` tool of
each main cognition run. The codelets phase calls it from a worker thread
with the kernel context and an :class:`AgentCodeletJob`. The session works
as follows:

- A :class:`~pm.memory.agent_tools.MemoryReadSession` over the frozen grant
  of the run serves the memory tools through a loopback MCP server. The
  session never writes memory.
- A second loopback server offers ``mind_view``: the codelet's view of the
  CSM, with record ids.
- The skill ``resources/codelet_agent.md`` and the shared story prefix open
  the prompt. The codelet instruction and its retrieval plan
  (``data_aquisation``) follow.
- Every cited id must be a line of the view, a record that the cycle
  already holds (``known_ids``), or a record that the memory tools read
  (``memory.accessed``).

Invalid final JSON receives repair turns without another tool session.
An uninspected cited id or a session deadline starts a
fresh read-only session with the latest reason until cancellation. Provider
and cancellation errors propagate.

Sessions of one run take turns. They share the in-memory Bash machine of
the run, and that machine is not documented as safe for parallel shells.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import replace, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from fastmcp import FastMCP
from pm_workflows.drivers import PmCoderDriver
from pm_workflows.python_role import RoleContext

from pm.constants import CODELET_AGENT_TIMEOUT_SECONDS
from pm.host.mcp_listeners import serve_services
from pm.llm.multimodal import ImageAttachment, agent_images
from pm.llm.prompt_log import log_directory
from pm.memory.agent_tools import MemoryReadSession, MemoryToolConfig, MemoryView
from pm.memory.bash_view import MemoryBashEnvironment
from pm.memory.store import MemoryStore
from pm.records.media import IMAGE_TAG, RecordImage
from pm.context_render.budget import TokenEstimator, evidence_excerpt
from pm.context_render.images import ImageCatalog
from pm.context_render.prompt import RenderSettings, RenderedPrompt, RendererCapture, RendererInput, require_rendered_prompt
from pm.context_render.turns import Role, Turn, Content, Media, MediaKind

from workflows.main_cognition.csm_view import record_view
from workflows.main_cognition.schemas import AgentCodeletAnswer, AgentCodeletJob

if TYPE_CHECKING:
    from pm.memory.embeddings import EmbeddingService

SKILL = Path(__file__).resolve().parent / "resources" / "codelet_agent.md"

@dataclass(frozen=True, slots=True)
class CodeletAgentRenderInput:
    job: AgentCodeletJob
    skill: str
    problems: tuple[str, ...]
    images: tuple[RecordImage, ...]


def prompt_text(input: CodeletAgentRenderInput) -> str:
    job = input.job
    plan = "\n".join(f"- {item}" for item in job.retrieval_plan) or "- (no plan; use your judgment)"
    text = f"{job.story}\n\n{job.view}\n\n{job.instruction}\n\nRetrieval plan:\n{plan}"
    if input.problems:
        text += "\n\nEarlier attempts of this task failed:\n" + "\n".join(f"- {item}" for item in input.problems)
    catalog = ImageCatalog(input.images)
    parts: list[str] = []
    cursor = 0
    for match in IMAGE_TAG.finditer(text):
        parts.append(text[cursor:match.start()])
        image = catalog.find_image(match.group(1))
        if image is None:
            raise ValueError(f"Native codelet references missing image {match.group(1)}.")
        parts.append(image.alt_label())
        cursor = match.end()
    parts.append(text[cursor:])
    return "".join(parts)


def prompt_images(input: CodeletAgentRenderInput) -> list[ImageAttachment]:
    job = input.job
    return agent_images(f"{job.story}\n\n{job.view}\n\n{job.instruction}", ImageCatalog(input.images))


def render_codelet_agent(input: CodeletAgentRenderInput, settings: RenderSettings) -> RenderedPrompt:
    estimator = TokenEstimator(estimate_factor=settings.estimate_factor)
    job = input.job.model_copy()
    selected = replace(input, job=job)
    text = prompt_text(selected)
    turns = [Turn(Role.System, input.skill), Turn(Role.User, text)]
    visible_ids = tuple(job.view_ids)
    if estimator.estimate(turns) > settings.input_ceiling:
        job.view = "Use mind_view to read the complete current view with its record IDs."
        visible_ids = ()
        retained = settings.input_ceiling
        while True:
            job.story = evidence_excerpt(input.job.story, retained)
            job.instruction = evidence_excerpt(input.job.instruction, retained)
            job.retrieval_plan = [evidence_excerpt("\n".join(input.job.retrieval_plan), retained)]
            selected = replace(selected, problems=tuple(evidence_excerpt(item, retained) for item in input.problems))
            turns[1] = Turn(Role.User, prompt_text(selected))
            if estimator.estimate(turns) <= settings.input_ceiling or retained == 0:
                break
            retained //= 2
    for index, image in enumerate(prompt_images(selected)):
        turn = Turn(Role.User, Content(Media(MediaKind.Image, image.media_type, data=image.data, name=f"codelet-image-{index}")))
        if estimator.estimate([*turns, turn]) <= settings.input_ceiling:
            turns.append(turn)
    prompt = require_rendered_prompt("native codelet initial prompt", tuple(turns), settings)
    return replace(prompt, visible_source_ids=visible_ids)


def fitted_codelet_images(prompt: RenderedPrompt) -> list[ImageAttachment]:
    images: list[ImageAttachment] = []
    for turn in prompt.turns[2:]:
        for part in turn.content.parts:
            if not isinstance(part, Media) or part.kind != MediaKind.Image or part.data is None:
                raise ValueError("The fitted codelet attachment must be an image with captured bytes.")
            images.append(ImageAttachment(part.data, part.mime_type))
    return images


@dataclass(slots=True)
class CodeletMindView:
    text: str
    inspected: bool = False

    def read(self) -> str:
        self.inspected = True
        return self.text


@dataclass(slots=True)
class CodeletAgent:
    ghost: MemoryStore
    environment: MemoryBashEnvironment
    embeddings: EmbeddingService | None
    cancel: threading.Event
    turn: threading.Lock = field(default_factory=threading.Lock, init=False)
    ids: frozenset[int] = field(init=False)
    scopes: frozenset[tuple[int, str]] = field(init=False)

    def __post_init__(self) -> None:
        self.ids = frozenset(self.environment.records)
        self.scopes = frozenset((record.scope.reality_id, record.scope.frame.value)
                                for record in self.environment.records.values())

    def execute(self, context: RoleContext, job: AgentCodeletJob) -> AgentCodeletAnswer:
        """Run sessions until one returns a valid answer. See the module docstring."""
        runtime = context.runtime
        assert runtime is not None
        if runtime.classification_session is None:
            raise RuntimeError("Agent codelets need workflow models for their pm-coder sessions.")
        problems: list[str] = []
        with self.turn:
            attempt = 0
            while True:
                runtime.check_cancelled()
                attempt += 1
                if problems:
                    logging.getLogger(__name__).warning("Codelet %s retry %d: %s", job.codelet, attempt, problems[-1])
                    problems = problems[-1:]
                memory = MemoryReadSession(MemoryToolConfig(self.ghost, MemoryView(self.ids, self.scopes),
                    semantic=self.embeddings is not None, lexical=True, graph=True,
                    embeddings=self.embeddings, cancel=self.cancel))
                mind = FastMCP("codelet_mind")

                mind_view = CodeletMindView(job.view)
                mind.tool(name="mind_view")(mind_view.read)
                session = runtime.classification_session("Default", False, runtime.stop_event)
                with memory, session as model, serve_services({"memory": memory.server(), "mind": mind}) as servers:
                    config = context.kernel_data / f"codelet-{job.codelet}-mcp.json"
                    config.write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")
                    driver = PmCoderDriver(log_root=log_directory() / "agents",
                                           timeout_seconds=CODELET_AGENT_TIMEOUT_SECONDS)
                    images: list[RecordImage] = []
                    texts = f"{job.story}\n{job.view}\n{job.instruction}"
                    for image_id in sorted({match.group(1) for match in IMAGE_TAG.finditer(texts)}):
                        image = self.ghost.find_image(image_id)
                        if image is None:
                            raise ValueError(f"Native codelet references missing image {image_id}.")
                        images.append(image.model_copy(deep=True))
                    from pm.llm.role_model import role_model
                    model_session = role_model(runtime)
                    settings = model_session.render_settings()
                    render_input = CodeletAgentRenderInput(job, SKILL.read_text(encoding="utf-8"), tuple(problems), tuple(images))
                    with RendererCapture(model_session.capture_settings("native_codelet"), RendererInput("native_codelet", render_input, settings)) as capture:
                        rendered = render_codelet_agent(render_input, settings)
                        capture.save_output(rendered)
                    try:
                        result = driver.run_session(
                            run_id=f"{context.task_id}-{job.codelet}", attempt=attempt, skill=str(SKILL),
                            prompt=rendered.turns[1].content.text, work_dir=context.workspace,
                            images=fitted_codelet_images(rendered),
                            runtime=replace(runtime, agent_model=replace(model, enable_thinking=True),
                                            mcp_servers=servers),
                            mcp_config=config,
                            trace_file=context.kernel_data / f"codelet-{job.codelet}-{attempt}.jsonl")
                    except TimeoutError as error:
                        problems.append(f"attempt {attempt}: {error}")
                        continue
                    runtime.check_cancelled()
                    from pm.llm.structured import repair_agent_result
                    answer = repair_agent_result(result, AgentCodeletAnswer, model,
                        instruction=rendered.turns[1].content.text, cancel=self.cancel)
                    permitted = set(memory.accessed) | set(rendered.visible_source_ids)
                    if mind_view.inspected:
                        permitted.update(job.view_ids)
                    uninspected = sorted(set(answer.evidence_ids) - permitted)
                    if uninspected:
                        problems.append(f"attempt {attempt}: evidence_ids {uninspected} were neither in the mind "
                                        "view nor read with the memory tools. Cite only records that you read.")
                        continue
                    views = [record_view(memory.snapshot.get(key)) for key in dict.fromkeys(answer.evidence_ids)
                             if key in memory.records]
                    completed = answer.model_copy()
                    completed.records = views
                    completed.session_ref = result.session_ref
                    return completed



__all__ = ["CodeletAgent", "SKILL", "prompt_images", "prompt_text"]
