"""The one place where image tags in prompt text become OpenAI image parts.

Workflows render records to text and keep each ``[image:<id>]`` tag in that
text. :func:`encode_messages` runs at the provider boundary (``ChatClient``
and context measurement) and resolves the tags:

- Only the last :data:`pm.constants.PROMPT_MAX_IMAGES` distinct images of
  the whole prompt become real ``image_url`` parts. The count runs from the
  end of the last message backwards, across all messages, so the newest
  images are the ones the model sees.
- A chosen image is sent once, at its first tag. A later tag of the same
  image becomes ``[image: the same image as shown earlier]``. The story
  prefix comes first, so it keeps the same bytes in every call of a cycle.
- Every tag of an older image becomes ``[image: <alt text>]``. A tag whose image has
  no alt text yet becomes ``[image: no description yet]``.
- A message without tags stays exactly as before: its content is a plain
  string, so text-only prompts keep their bytes and their provider cache.

The OpenAI chat API accepts image input only on user turns. A selected tag in
a system, assistant, or tool turn therefore keeps its tag text as a
reference, and its image part goes to the end of the nearest earlier user
turn, after the line ``[image:<id>] (shown later in this conversation):``.
Without an earlier user turn, the tag becomes alt text. Example: the dialogue
workflow puts the transcript in an assistant prefill, so the user's photo
appears at the end of the context turn just before it.

Images that callers already pass as :class:`~pm.context_render.turns.Media`
parts (for example the alt-text call) pass through unchanged and do not count
toward the limit.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from pm.constants import PROMPT_MAX_IMAGES
from pm.context_render.turns import Content, Media, MediaKind, Role, Text, Turn
from pm.llm.prompt_encoding import OpenAIChatEncoder, TurnInput, normalize_turns
from pm.records.media import IMAGE_TAG, MISSING_IMAGE, RecordImage, find_image

Resolver = Callable[[str], RecordImage | None]
#: Text of a later tag of an image that the prompt already shows.
SAME_IMAGE = "the same image as shown earlier"


def _to_png(raw: bytes) -> tuple[bytes, str]:
    """Convert a WebP image to PNG; return unchanged data for native formats.

    llama.cpp's stb_image loader cannot decode WebP or BMP. JPEG and PNG are
    decoded natively. GIF also works natively but we convert to avoid the
    slightly larger files and palette overhead that some encoders produce.

    Returns ``(bytes, mime_type)`` — always ``PNG`` when a conversion was
    needed so the caller knows what format is actually in the payload.
    """
    from io import BytesIO as _BI  # noqa: N813
    from PIL import Image as _PILImage  # noqa: N813

    # WebP — not supported by stb_image. Decode with Pillow and encode PNG.
    if raw[:4] == b"RIFF" and len(raw) >= 12 and raw[8:12] == b"WEBP":
        img = _PILImage.open(_BI(raw))
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        buf = _BI()
        img.save(buf, format="PNG")
        return buf.getvalue(), "image/png"
    # BMP — also unsupported by stb_image.
    if raw[:2] == b"BM":
        img = _PILImage.open(_BI(raw))
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        buf = _BI()
        img.save(buf, format="PNG")
        return buf.getvalue(), "image/png"
    # GIF — supported natively but convert to avoid palette issues.
    if raw[:6] in (b"GIF87a", b"GIF89a"):
        img = _PILImage.open(_BI(raw))
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        buf = _BI()
        img.save(buf, format="PNG")
        return buf.getvalue(), "image/png"
    # PNG and JPEG are natively supported — return unchanged.
    return raw, "image/png" if raw[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"


@dataclass(frozen=True)
class _Tag:
    """One tag occurrence: its turn, its text part, and its span in that part."""

    turn: int
    part: int
    start: int
    end: int
    image_id: str


def _tags(turns: list[Turn]) -> list[_Tag]:
    return [_Tag(turn_index, part_index, match.start(), match.end(), match.group(1))
            for turn_index, turn in enumerate(turns)
            for part_index, part in enumerate(turn.content.parts) if isinstance(part, Text)
            for match in IMAGE_TAG.finditer(part.text)]


def _media(image: RecordImage) -> Media:
    raw = image.data()
    converted, mime_type = _to_png(raw)
    return Media(MediaKind.Image, mime_type, data=converted, name=image.id)


def resolve_turns(messages: Sequence[TurnInput], resolve: Resolver = find_image,
                  limit: int = PROMPT_MAX_IMAGES) -> list[Turn]:
    """Return the turns with every image tag resolved to an image part or alt text.

    ``resolve`` maps an image id to its image; the default uses the
    application's installed media lookup. A tag that no record holds becomes
    ``[image: unavailable]`` and does not count toward ``limit``. Turns
    without tags are returned unchanged (the same objects).
    """
    turns = normalize_turns(messages)
    found = _tags(turns)
    if not found:
        return turns
    images = {tag.image_id: resolve(tag.image_id) for tag in found}
    resolvable = [tag for tag in found if images[tag.image_id] is not None]
    # The last ``limit`` distinct images, counted from the end of the prompt.
    chosen: list[str] = []
    for tag in reversed(resolvable):
        if len(chosen) < limit and tag.image_id not in chosen:
            chosen.append(tag.image_id)
    user_turns = [index for index, turn in enumerate(turns) if turn.role is Role.User]
    # Each chosen image is shown once, at its first tag that can carry it.
    # The shared story prefix comes first, so it keeps the same bytes in
    # every call. Later tags of the same image refer back to it. The image
    # of a non-user turn moves to the nearest earlier user turn.
    selected: set[_Tag] = set()
    repeats: set[_Tag] = set()
    moved: dict[int, list[RecordImage]] = {}
    for tag in resolvable:
        if tag.image_id not in chosen:
            continue
        if any(item.image_id == tag.image_id for item in selected):
            repeats.add(tag)
            continue
        if turns[tag.turn].role is Role.User:
            selected.add(tag)
            continue
        earlier = [index for index in user_turns if index < tag.turn]
        if earlier:
            moved.setdefault(earlier[-1], []).append(images[tag.image_id])
            selected.add(tag)
    by_part: dict[tuple[int, int], list[_Tag]] = {}
    for tag in found:
        by_part.setdefault((tag.turn, tag.part), []).append(tag)
    result = []
    for turn_index, turn in enumerate(turns):
        if not any(key[0] == turn_index for key in by_part) and turn_index not in moved:
            result.append(turn)
            continue
        parts: list = []
        for part_index, part in enumerate(turn.content.parts):
            tags = by_part.get((turn_index, part_index))
            if not tags:
                parts.append(part)
                continue
            cursor = 0
            for tag in tags:
                parts.append(Text(part.text[cursor:tag.start]))
                image = images[tag.image_id]
                if image is None:
                    parts.append(Text(f"[image: {MISSING_IMAGE}]"))
                elif tag in repeats:
                    parts.append(Text(f"[image: {SAME_IMAGE}]"))
                elif tag not in selected:
                    parts.append(Text(image.alt_label()))
                elif turn.role is Role.User:
                    parts.append(_media(image))
                else:
                    parts.append(Text(part.text[tag.start:tag.end]))
                cursor = tag.end
            parts.append(Text(part.text[cursor:]))
        for image in moved.get(turn_index, []):
            parts.append(Text(f"\n{image.tag} (shown later in this conversation):\n"))
            parts.append(_media(image))
        merged = Content((), notes=turn.content.notes)
        for part in parts:
            if not isinstance(part, Text) or part.text:
                merged += Content(part)
        result.append(Turn(turn.role, merged, turn.tool_calls, turn.tool_call_id, turn.name, turn.provider_state))
    return result


def encode_messages(messages: Sequence[TurnInput], resolve: Resolver = find_image,
                    limit: int = PROMPT_MAX_IMAGES) -> dict[str, Any]:
    """Return the OpenAI-compatible ``{"messages": [...]}`` body for these turns.

    Tags resolve as :func:`resolve_turns` describes. The result is the
    provider boundary, so it is a plain dict. Invalid media raise ValueError
    before any network call.
    """
    return OpenAIChatEncoder().encode(resolve_turns(messages, resolve, limit))


def prompt_image_count(messages: Sequence[TurnInput], resolve: Resolver = find_image,
                       limit: int = PROMPT_MAX_IMAGES) -> int:
    """Return how many image parts the encoded prompt holds, including direct Media parts."""
    return sum(isinstance(part, Media) for turn in resolve_turns(messages, resolve, limit)
               for part in turn.content.parts)


__all__ = ["encode_messages", "prompt_image_count", "resolve_turns"]
