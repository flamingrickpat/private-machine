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

import asyncio
import base64
from collections.abc import Callable, Sequence
from pathlib import Path
from threading import Event, Lock

from pydantic import BaseModel

from pm.llm.multimodal import _to_png

from pm.constants import ALT_TEXT_ATTEMPTS, ALT_TEXT_MAX_CHARS, ALT_TEXT_TEMPERATURE
from pm.llm.http_client import ModelError
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


def alt_text_messages(image: RecordImage, prompts: AltTextPrompts) -> list[dict]:
    """Return the system message and one user message with the instruction and the image.

    The messages use the OpenAI wire shape. Image bytes are converted to PNG
    when the original format (WebP, BMP, etc.) is not natively supported by
    llama.cpp's stb_image loader.
    """
    raw = image.data()
    converted, mime_type = _to_png(raw)
    data_url = f"data:{mime_type};base64,{base64.b64encode(converted).decode('ascii')}"
    return [{"role": "system", "content": prompts.system_prompt.strip()},
            {"role": "user", "content": [{"type": "text", "text": prompts.instruction.strip()},
                                         {"type": "image_url", "image_url": {"url": data_url}}]}]


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


def describe_image(image: RecordImage, complete: Callable[..., str], *, cancel: Event | None = None,
                   prompts: AltTextPrompts | None = None) -> str:
    """Return a model alt text for one image. This call blocks.

    ``complete`` is a synchronous text completion (``LazyStageModel.complete``).
    Provider errors (ModelError) and invalid output are retried, up to
    :data:`pm.constants.ALT_TEXT_ATTEMPTS` calls; then the last error
    propagates. Invalid output gets a repair turn that names the problem.
    Cancellation propagates from the client.
    """
    prompts = prompts or alt_text_prompts()
    original = alt_text_messages(image, prompts)
    turns = original
    last: Exception | None = None
    for _ in range(ALT_TEXT_ATTEMPTS):
        try:
            raw = complete(turns, cancel=cancel, temperature=ALT_TEXT_TEMPERATURE, enable_thinking=False)
        except ModelError as error:
            last = error
            turns = original
            continue
        try:
            return clean_alt_text(raw)
        except ValueError as error:
            last = error
            turns = [*original, {"role": "assistant", "content": raw},
                     {"role": "user", "content": prompts.repair.strip().format(problem=str(error))}]
    raise last


def complete_alt_text(store, image_ids: Sequence[str], complete: Callable[..., str], *,
                      cancel: Event | None = None) -> dict[str, str]:
    """Write a model alt text for each listed image that has none. This call blocks.

    ``store`` is the :class:`~pm.memory.store.MemoryStore`. An id that no
    accepted record holds is skipped: it has nothing to describe yet. The
    alt text goes into every record that holds the image
    (``MemoryStore.annotate_alt_text``). Returns the new alt texts by id.
    """
    written = {}
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

    def __init__(self, store, complete: Callable[..., str]) -> None:
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


__all__ = ["AltTextPrompts", "MediaLibrary", "alt_text_messages", "alt_text_prompts", "clean_alt_text",
           "complete_alt_text", "describe_image"]
