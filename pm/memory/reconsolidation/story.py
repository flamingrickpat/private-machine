"""Story lines and token counts for the reconsolidation prompts.

The prompts show records as story lines in time order, without record ids.
A numbered story adds ``[n]`` markers, so a fact can cite the lines that
state it. The caller passes the time format (``RECONSOLIDATION_TIME_FORMAT``).
Image tags in record content always become alt text here
(``[image: <alt text>]``), never image parts: memory is built from the
description of an image, so a memory reads "The user says: ... [image: a grey
cat on a sofa]". A prepared unit carries any new captions to publication.
Token counts use tiktoken (``o200k_base``), like
``pm.llm.context_window.estimate_tokens``. They are estimates for budgets.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import tiktoken

from pm.records.enums import FeatureType
from pm.records.media import RecordImage, IMAGE_TAG
from pm.records.memory_records import Feature, MemoryClusterRecord

@dataclass(frozen=True, slots=True)
class ImageCatalog:
    images: tuple[RecordImage, ...] = ()

    def alt_text(self, text: str) -> str:
        parts: list[str] = []
        position = 0
        for match in IMAGE_TAG.finditer(text):
            image_id = match.group(1)
            image = next((item for item in self.images if item.id == image_id), None)
            if image is None:
                raise ValueError(f"Image {image_id} is absent from the captured reconstruction input.")
            parts.extend((text[position:match.start()], image.alt_label()))
            position = match.end()
        parts.append(text[position:])
        return "".join(parts)

_THOUGHTS = {FeatureType.Thought, FeatureType.ExternalThought, FeatureType.InnerMonologue}


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    """Return the shared tokenizer of the story prompts."""
    return tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    """Return the estimated token count of ``text``."""
    return len(_encoding().encode(text, disallowed_special=()))


def feature_text(feature: Feature, images: ImageCatalog = ImageCatalog()) -> str:
    """Return the content of a feature with every image tag replaced by its alt text."""
    return images.alt_text(feature.content)


def feature_line(feature: Feature, timestamp_format: str, images: ImageCatalog = ImageCatalog()) -> str:
    """Return one feature as a dated story line, with images as alt text.

    Speech is quoted. A thought is marked as a thought, so a summary never
    reads it as speech. A state change stays in the observer's perspective.
    """
    when = feature.timestamp_world_begin.strftime(timestamp_format)
    content = feature_text(feature, images)
    if feature.feature_type == FeatureType.Dialogue:
        return f'{when} {feature.source} says: "{content}"'
    if feature.feature_type in _THOUGHTS:
        return f"{when} {feature.source} thinks: *{content}*"
    if feature.feature_type == FeatureType.SubjectiveExperience:
        return f"{when} {feature.source} experiences: {content}"
    if feature.feature_type == FeatureType.StateUpdate:
        return f"{when} {feature.source}'s internal state and decisions: {content}"
    return f"{when} (event from {feature.source}) {content}"


def render_story(features: Sequence[Feature], timestamp_format: str, *, numbered: bool = False,
                 images: ImageCatalog = ImageCatalog()) -> str:
    """Return the features as story lines, one per line, in the given order."""
    lines = [feature_line(feature, timestamp_format, images) for feature in features]
    if numbered:
        lines = [f"[{index}] {line}" for index, line in enumerate(lines, 1)]
    return "\n".join(lines)


def render_summaries(clusters: Sequence[MemoryClusterRecord], timestamp_format: str) -> str:
    """Return earlier summaries as dated blocks, in the given order."""
    return "\n\n".join(
        f"From {cluster.timestamp_world_begin.strftime(timestamp_format)} to "
        f"{cluster.timestamp_world_end.strftime(timestamp_format)}:\n{cluster.content}"
        for cluster in clusters)


__all__ = ["count_tokens", "feature_line", "feature_text", "render_story", "render_summaries"]
