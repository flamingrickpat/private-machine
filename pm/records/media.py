"""Images inside records: the typed metadata shape, the content tag, and the lookup hook.

A record keeps its readable text in ``content``. Each image in that text is a
tag, ``[image:<id>]``. The image bytes live in the record's ``metadata``
under one key (:data:`MEDIA_KEY`), as a :class:`RecordMedia` document with
base64 data, a digest, the size, and an alt text.

Image ids are content addressed: ``img-`` and the first 16 hex digits of the
SHA-256 of the bytes. The same bytes therefore get the same id in every
record, and a tag can travel through any renderer (story prefix, dialogue
transcript, tails, summaries) without a record id. The prompt encoder and the
embedding path resolve a tag through the installed :class:`MediaLookup`, which
the application connects to its memory store. See
``docs/subsystems/multimodal-records.md``.

Write and read the metadata only through :func:`attach_images` and
:func:`record_media`. Other code does not read the raw metadata key.
"""
from __future__ import annotations

import base64
import hashlib
import re
import struct
from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from datetime import datetime, timezone
from threading import Event
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from pm.constants import MEDIA_MAX_IMAGE_BYTES

#: Metadata key that holds the :class:`RecordMedia` document of a record.
MEDIA_KEY = "media"
MEDIA_SCHEMA = "pm.record-media.v1"
#: MIME types that the record store and the prompt encoder accept.
IMAGE_TYPES = ("image/png", "image/jpeg", "image/webp", "image/gif")
#: One tag in record content. Group 1 is the image id.
IMAGE_TAG = re.compile(r"\[image:([A-Za-z0-9_.-]+)\]")
#: Text that replaces a tag whose image has no alt text yet.
NO_ALT_TEXT = "no description yet"
#: Text that replaces a tag whose image no record holds.
MISSING_IMAGE = "unavailable"


class MediaProvenance(BaseModel):
    """Where an image came from.

    ``source`` is ``user_upload`` (web chat), ``tool`` (an MCP tool result,
    for example a Minecraft screenshot), or ``import``. ``server`` names the
    MCP server and ``tool`` the tool that produced the image. ``run_id`` and
    ``artifact_path`` name the managed run and the file in its outbox.
    """

    source: Literal["user_upload", "tool", "import", ""] = ""
    run_id: str = ""
    artifact_path: str = ""
    server: str = ""
    tool: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class RecordImage(BaseModel):
    """One image with its bytes as base64, its digest, and its alt text.

    ``sha256`` and ``size_bytes`` describe the decoded bytes. ``alt_text`` is
    a short factual description; ``alt_text_source`` says who wrote it:
    ``model`` (:func:`pm.memory.media_alt.complete_alt_text`), ``user``, or
    ``tool``. An empty alt text means that the description is still due.
    """

    id: str = ""
    media_type: str = ""
    data_b64: str = Field(default="", repr=False)
    sha256: str = ""
    size_bytes: int = 0
    width: int | None = None
    height: int | None = None
    alt_text: str = ""
    alt_text_source: Literal["model", "user", "tool", ""] = ""
    provenance: MediaProvenance = Field(default_factory=MediaProvenance)

    @property
    def tag(self) -> str:
        """Return the content tag of this image."""
        return image_tag(self.id)

    def data(self) -> bytes:
        """Return the decoded image bytes."""
        return base64.b64decode(self.data_b64, validate=True)

    def data_url(self) -> str:
        """Return a ``data:`` URL for an OpenAI ``image_url`` part."""
        return f"data:{self.media_type};base64,{self.data_b64}"

    def alt_label(self) -> str:
        """Return the text form of this image: ``[image: <alt text>]``."""
        return f"[image: {self.alt_text or NO_ALT_TEXT}]"


class RecordMedia(BaseModel):
    """The media document of one record, in tag order."""

    schema_id: str = MEDIA_SCHEMA
    images: list[RecordImage] = Field(default_factory=list)


def image_tag(image_id: str) -> str:
    """Return the content tag ``[image:<id>]``."""
    return f"[image:{image_id}]"


def image_ids(text: str) -> list[str]:
    """Return the image ids of all tags in ``text``, in order, with repeats."""
    return IMAGE_TAG.findall(text)


def sniff_image_type(data: bytes) -> str:
    """Return the MIME type from the magic bytes. Unknown data raises ValueError."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    raise ValueError("The data is not a PNG, JPEG, WebP, or GIF image.")


def image_size(data: bytes, media_type: str) -> tuple[int | None, int | None]:
    """Read width and height from the image header, without decoding pixels.

    Returns ``(None, None)`` when the header form is not one of the simple
    forms below (for example a JPEG without an early SOF marker).
    """
    if media_type == "image/png" and len(data) >= 24 and data[12:16] == b"IHDR":
        return struct.unpack(">II", data[16:24])
    if media_type == "image/gif" and len(data) >= 10:
        return struct.unpack("<HH", data[6:10])
    if media_type == "image/webp" and len(data) >= 30:
        chunk = data[12:16]
        if chunk == b"VP8X":
            return (1 + int.from_bytes(data[24:27], "little"), 1 + int.from_bytes(data[27:30], "little"))
        if chunk == b"VP8 " and data[23:26] == b"\x9d\x01\x2a":
            width, height = struct.unpack("<HH", data[26:30])
            return width & 0x3FFF, height & 0x3FFF
        if chunk == b"VP8L" and data[20] == 0x2F:
            bits = int.from_bytes(data[21:25], "little")
            return 1 + (bits & 0x3FFF), 1 + ((bits >> 14) & 0x3FFF)
        return None, None
    if media_type == "image/jpeg":
        index = 2
        while index + 9 < len(data) and data[index] == 0xFF:
            marker, length = data[index + 1], int.from_bytes(data[index + 2:index + 4], "big")
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                height, width = struct.unpack(">HH", data[index + 5:index + 9])
                return width, height
            index += 2 + length
    return None, None


def image_from_bytes(data: bytes, media_type: str | None = None, *, alt_text: str = "",
                     alt_text_source: str = "", provenance: MediaProvenance | None = None) -> RecordImage:
    """Validate image bytes and return a :class:`RecordImage`.

    The magic bytes set the MIME type. A supplied ``media_type`` that
    disagrees raises ValueError, as do empty data, unknown formats, and data
    above :data:`pm.constants.MEDIA_MAX_IMAGE_BYTES`. A non-empty
    ``alt_text`` without ``alt_text_source`` counts as ``user``.
    """
    if not data:
        raise ValueError("The image is empty.")
    if len(data) > MEDIA_MAX_IMAGE_BYTES:
        raise ValueError(f"The image has {len(data)} bytes; the limit is {MEDIA_MAX_IMAGE_BYTES}.")
    sniffed = sniff_image_type(data)
    if media_type is not None and media_type.lower() != sniffed:
        raise ValueError(f"The image bytes are {sniffed}, not {media_type}.")
    digest = hashlib.sha256(data).hexdigest()
    width, height = image_size(data, sniffed)
    return RecordImage(
        id="img-" + digest[:16], media_type=sniffed, data_b64=base64.b64encode(data).decode("ascii"),
        sha256=digest, size_bytes=len(data), width=width, height=height,
        alt_text=alt_text.strip(), alt_text_source=alt_text_source or ("user" if alt_text.strip() else ""),
        provenance=provenance or MediaProvenance(),
    )


def image_from_base64(data_b64: str, media_type: str | None = None, **fields) -> RecordImage:
    """Decode strict base64 and validate it like :func:`image_from_bytes`."""
    return image_from_bytes(base64.b64decode(data_b64, validate=True), media_type, **fields)


def attach_images(metadata: MutableMapping[str, object], images: Sequence[RecordImage]) -> list[str]:
    """Add images to a record's metadata and return their content tags.

    Images keep their order. An image whose id the record already holds is
    not stored twice, but its tag is still returned. The caller appends the
    tags to the record content.
    """
    media = RecordMedia.model_validate(metadata[MEDIA_KEY]) if MEDIA_KEY in metadata else RecordMedia()
    known = {image.id for image in media.images}
    for image in images:
        if image.id not in known:
            media.images.append(image)
            known.add(image.id)
    metadata[MEDIA_KEY] = media.model_dump(mode="json")
    return [image.tag for image in images]


def metadata_media(metadata: Mapping[str, object]) -> RecordMedia:
    """Return the media document of a metadata mapping; empty without media."""
    return RecordMedia.model_validate(metadata[MEDIA_KEY]) if MEDIA_KEY in metadata else RecordMedia()


def record_media(record) -> RecordMedia:
    """Return the typed media document of a record (an object with ``metadata``)."""
    return metadata_media(record.metadata)


def with_tags(text: str, tags: Iterable[str]) -> str:
    """Return ``text`` with each tag on its own line after it. Image-only text is the tags alone."""
    lines = [text.rstrip(), *tags] if text.strip() else list(tags)
    return "\n".join(lines)


def merge_alt_text(metadata: MutableMapping[str, object], current: Mapping[str, object]) -> None:
    """Keep alt texts from ``current`` that a stale replacement would drop.

    The store writes alt text after a record is accepted. A write batch that
    read the record before that and replaces it later carries an empty alt
    text. This keeps the newer description for each image with the same id.
    """
    if MEDIA_KEY not in metadata or MEDIA_KEY not in current:
        return
    known = {image.id: image for image in metadata_media(current).images if image.alt_text}
    media = metadata_media(metadata)
    changed = False
    for image in media.images:
        if not image.alt_text and image.id in known:
            image.alt_text, image.alt_text_source = known[image.id].alt_text, known[image.id].alt_text_source
            changed = True
    if changed:
        metadata[MEDIA_KEY] = media.model_dump(mode="json")


class MediaLookup(Protocol):
    """Resolve image ids to images, and complete missing alt text.

    :class:`pm.memory.media_alt.MediaLibrary` implements this over the
    memory store. ``complete_alt_text`` writes the descriptions into every
    record that holds the image. It returns after they are stored.
    """

    def find_image(self, image_id: str) -> RecordImage | None: ...

    async def complete_alt_text(self, image_ids: Sequence[str], *, cancel: Event | None = None) -> None: ...


_LOOKUP: MediaLookup | None = None


def install_media_lookup(lookup: MediaLookup | None) -> None:
    """Set the process-wide media lookup. ``None`` removes it.

    The application installs one lookup after it opens its memory store. Every
    chat client and embedding service in the process then resolves tags the
    same way, without wiring the store through each constructor. One process
    runs one application; a second install replaces the first.
    """
    global _LOOKUP
    _LOOKUP = lookup


def media_lookup() -> MediaLookup | None:
    """Return the installed media lookup, or None when no application installed one."""
    return _LOOKUP


def find_image(image_id: str) -> RecordImage | None:
    """Resolve one id through the installed lookup. Without a lookup, return None."""
    return _LOOKUP.find_image(image_id) if _LOOKUP is not None else None


def alt_text_view(text: str, resolve: Callable[[str], RecordImage | None] = find_image) -> str:
    """Return ``text`` with every tag replaced by ``[image: <alt text>]``.

    Memory consolidation, embedding text, and other text-only consumers use
    this form. The base64 data never enters the result. An unknown id becomes
    ``[image: unavailable]``.
    """
    def replace(match: re.Match) -> str:
        image = resolve(match.group(1))
        return image.alt_label() if image is not None else f"[image: {MISSING_IMAGE}]"
    return IMAGE_TAG.sub(replace, text)


__all__ = [
    "IMAGE_TAG", "IMAGE_TYPES", "MEDIA_KEY", "MEDIA_SCHEMA", "MISSING_IMAGE", "NO_ALT_TEXT",
    "MediaLookup", "MediaProvenance", "RecordImage", "RecordMedia",
    "alt_text_view", "attach_images", "find_image", "image_from_base64", "image_from_bytes",
    "image_ids", "image_size", "image_tag", "install_media_lookup", "media_lookup", "merge_alt_text",
    "metadata_media", "record_media", "sniff_image_type", "with_tags",
]
