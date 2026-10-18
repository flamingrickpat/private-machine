"""Story lines and token counts for the reconsolidation prompts.

The prompts show records as story lines in time order, without record ids.
A numbered story adds ``[n]`` markers, so a fact can cite the lines that
state it. The caller passes the time format (``RECONSOLIDATION_TIME_FORMAT``).
Token counts use tiktoken (``o200k_base``), like
``pm.llm.context_window.estimate_tokens``. They are estimates for budgets.
"""
from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import tiktoken

from pm.records.enums import FeatureType
from pm.records.memory_records import Feature, MemoryClusterRecord

_THOUGHTS = {FeatureType.Thought, FeatureType.ExternalThought, FeatureType.InnerMonologue}


@lru_cache(maxsize=1)
def _encoding():
    return tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    """Return the estimated token count of ``text``."""
    return len(_encoding().encode(text, disallowed_special=()))


def feature_line(feature: Feature, timestamp_format: str) -> str:
    """Return one feature as a dated story line.

    Speech is quoted. A thought is marked as a thought, so a summary never
    reads it as speech. An event names its source.
    """
    when = feature.timestamp_world_begin.strftime(timestamp_format)
    if feature.feature_type == FeatureType.Dialogue:
        return f'{when} {feature.source} says: "{feature.content}"'
    if feature.feature_type in _THOUGHTS:
        return f"{when} {feature.source} thinks: *{feature.content}*"
    if feature.feature_type == FeatureType.SubjectiveExperience:
        return f"{when} {feature.source} experiences: {feature.content}"
    return f"{when} (event from {feature.source}) {feature.content}"


def render_story(features: Sequence[Feature], timestamp_format: str, *, numbered: bool = False) -> str:
    """Return the features as story lines, one per line, in the given order."""
    lines = [feature_line(feature, timestamp_format) for feature in features]
    if numbered:
        lines = [f"[{index}] {line}" for index, line in enumerate(lines, 1)]
    return "\n".join(lines)


def render_summaries(clusters: Sequence[MemoryClusterRecord], timestamp_format: str) -> str:
    """Return earlier summaries as dated blocks, in the given order."""
    return "\n\n".join(
        f"From {cluster.timestamp_world_begin.strftime(timestamp_format)} to "
        f"{cluster.timestamp_world_end.strftime(timestamp_format)}:\n{cluster.content}"
        for cluster in clusters)


__all__ = ["count_tokens", "feature_line", "render_story", "render_summaries"]
