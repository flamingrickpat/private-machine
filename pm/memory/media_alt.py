"""Alt text for record images, written by the multimodal model before embedding.

The alt text is the text form of an image. Memory consolidation,
reconsolidation, embeddings, and every prompt beyond the last-N image rule
read it instead of the pixels. It is an unconscious, neutral description: it
says what the image shows, and it is not the companion's opinion.

:func:`complete_alt_text` asks the model once for each image without alt
text. The call shows the image itself. The result is stored in every record
that holds the image, with ``alt_text_source="model"``.

:class:`MediaLibrary` connects the memory store and the model. The
application installs it as the process-wide media lookup
(:func:`pm.records.media.install_media_lookup`). The embedding service then
completes alt text before it embeds a text that holds image tags.
"""
from __future__ import annotations

from pm.llm.stage_model import LazyStageModel
from pm.llm.session import ModelSession
from pm.llm.generation import GenerationSettings
from pm.context_render.prompt import require_rendered_prompt, RenderSettings, RenderedPrompt, RendererCapture, RendererInput
from pm.context_render.turns import Turn, Role, Content, Media, MediaKind
from dataclasses import dataclass
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pm.memory.store import MemoryStore

import asyncio
from collections.abc import Sequence
from pathlib import Path
from threading import Event, Lock

from pydantic import BaseModel

from pm.llm.multimodal import image_bytes_for_prompt

from pm.constants import ALT_TEXT_MAX_CHARS, ALT_TEXT_TEMPERATURE
from pm.records.media import IMAGE_TAG, RecordImage
from pm.utils import yaml_util


class AltTextPrompts(BaseModel):
    """The prompt texts of the alt-text call (``media_alt.yaml``)."""

    system_prompt: str
    instruction: str
    repair: str


def alt_text_prompts() -> AltTextPrompts:
    """Load the alt-text prompts from ``media_alt.yaml`` beside this module."""
    path = Path(__file__).with_name("media_alt.yaml")
    return AltTextPrompts.model_validate(yaml_util.safe_load(path.read_text(encoding="utf-8")))


@dataclass(frozen=True, slots=True)
class AltTextRenderInput:
    system_prompt: str
    instruction: str
    image_bytes: bytes
    repair_template: str
    problem: str = ""
    rejected_output: str = ""


def render_alt_text(input: AltTextRenderInput, settings: RenderSettings) -> RenderedPrompt:
    converted, mime_type = image_bytes_for_prompt(input.image_bytes)
    content = Content(input.instruction.strip()) + Content(Media(MediaKind.Image, mime_type, data=converted))
    turns = [Turn(Role.System, input.system_prompt.strip()), Turn(Role.User, content)]
    if input.problem:
        turns.extend((Turn(Role.Assistant, input.rejected_output),
            Turn(Role.User, input.repair_template.strip().format(problem=input.problem))))
    return require_rendered_prompt("image description", tuple(turns), settings)


def clean_alt_text(raw: str) -> str:
    """Return the accepted alt text, or raise ValueError for invalid output.

    Valid output is one or two short sentences of plain text: not empty, at
    most :data:`pm.constants.ALT_TEXT_MAX_CHARS` characters, and without an
    image tag. Surrounding quotes and a leading ``Alt text:`` label go.
    """
    text = " ".join(raw.split()).strip().strip('"').strip()
    for label in ("alt text:", "alt-text:", "description:"):
        if text.lower().startswith(label):
            text = text[len(label):].strip()
    if not text:
        raise ValueError("The alt text is empty.")
    if len(text) > ALT_TEXT_MAX_CHARS:
        raise ValueError(f"The alt text has {len(text)} characters; the limit is {ALT_TEXT_MAX_CHARS}.")
    if IMAGE_TAG.search(text) or "[image" in text.lower():
        raise ValueError("The alt text holds an image tag.")
    return text


def describe_image(image: RecordImage, complete: LazyStageModel, *, cancel: Event | None = None,
                   prompts: AltTextPrompts | None = None) -> str:
    """Return a model alt text for one image. This call blocks.

    ``complete`` is a synchronous text completion (``LazyStageModel.complete``).
    Provider errors (ModelError) and invalid output are retried, up to
    :data:`pm.constants.ALT_TEXT_ATTEMPTS` calls; then the last error
    propagates. Invalid output gets a repair turn that names the problem.
    Cancellation propagates from the client.
    """
    prompts = prompts or alt_text_prompts()
    image_bytes = image.data()
    session = ModelSession(complete, cancel)
    settings = session.render_settings()
    problem = ""
    rejected_output = ""
    while True:
        input = AltTextRenderInput(prompts.system_prompt, prompts.instruction, image_bytes,
            prompts.repair, problem, rejected_output)
        with RendererCapture(session.capture_settings("memory.image_description"),
                RendererInput("memory.image_description", input, settings)) as capture:
            rendered = render_alt_text(input, settings)
            capture.save_output(rendered)
        raw = session.complete(rendered.turns, settings=GenerationSettings(temperature=ALT_TEXT_TEMPERATURE, enable_thinking=False))
        try:
            return clean_alt_text(raw)
        except ValueError as error:
            problem = str(error)
            rejected_output = raw


def complete_alt_text(store: MemoryStore, image_ids: Sequence[str], complete: LazyStageModel, *,
                      cancel: Event | None = None) -> dict[str, str]:
    """Write a model alt text for each listed image that has none. This call blocks.

    ``store`` is the :class:`~pm.memory.store.MemoryStore`. An id that no
    accepted record holds is skipped: it has nothing to describe yet. The
    alt text goes into every record that holds the image
    (``MemoryStore.annotate_alt_text``). Returns the new alt texts by id.
    """
    written: dict[str, str] = {}
    for image_id in dict.fromkeys(image_ids):
        image = store.find_image(image_id)
        if image is None or image.alt_text:
            continue
        text = describe_image(image, complete, cancel=cancel)
        store.annotate_alt_text(image_id, text, "model")
        written[image_id] = text
    return written


class MediaLibrary:
    """The application's media lookup: images from the store, alt text from the model.

    ``complete`` is a synchronous text completion; it is used only when an
    image needs an alt text. One lock per image id makes sure that two
    threads that embed the same new image ask the model only once.
    """

    def __init__(self, store: MemoryStore, complete: LazyStageModel) -> None:
        self.store = store
        self.complete = complete
        self._locks: dict[str, Lock] = {}
        self._guard = Lock()

    def find_image(self, image_id: str) -> RecordImage | None:
        """Return the image from the newest accepted record that holds it, or None."""
        return self.store.find_image(image_id)

    def _complete_one(self, image_id: str, cancel: Event | None) -> None:
        with self._guard:
            lock = self._locks.setdefault(image_id, Lock())
        with lock:
            complete_alt_text(self.store, [image_id], self.complete, cancel=cancel)

    async def complete_alt_text(self, image_ids: Sequence[str], *, cancel: Event | None = None) -> None:
        """Complete missing alt texts in a worker thread, so the caller's event loop keeps running.

        The model call runs its own event loop (``LazyStageModel.complete``),
        which cannot start inside the caller's loop. Errors propagate.
        """
        for image_id in dict.fromkeys(image_ids):
            image = self.store.find_image(image_id)
            if image is not None and not image.alt_text:
                await asyncio.to_thread(self._complete_one, image_id, cancel)


__all__ = ["AltTextPrompts", "MediaLibrary", "render_alt_text", "AltTextRenderInput", "alt_text_prompts", "clean_alt_text",
           "complete_alt_text", "describe_image"]
