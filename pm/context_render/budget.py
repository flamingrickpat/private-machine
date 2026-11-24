"""Select rendered context by workflow salience within a conservative ceiling."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil

import tiktoken
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from pm.context_render.turns import Content, Text, Turn
from pm.constants import PROMPT_IMAGE_TOKENS


def evidence_excerpt(text: str, retained_characters: int) -> str:
    """Render a marked excerpt of one source, keeping the opening and most recent text.

    Callers select the source and keep its identity separately. This is a display
    representation of evidence. It must never be used to shorten instructions,
    schemas, executable tool arguments or the original stored record.
    """
    if len(text) <= retained_characters:
        return text
    first = retained_characters // 2
    last = retained_characters - first
    omitted = len(text) - retained_characters
    ending = text[-last:] if last else ""
    return f"{text[:first]}\n[{omitted} source characters omitted]\n{ending}"


class RequiredContextTooLarge(ValueError):
    """Required instructions and evidence cannot fit the configured input ceiling."""


class TokenEstimator(BaseModel):
    """Count locally with a deliberate margin, without inventing a serving capacity."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    estimate_factor: float = Field(default=1.5, ge=1, description="Safety margin over the larger local tokenizer estimate.")

    def estimate(self, turns: Sequence[Turn]) -> int:
        """Count rendered text, chat framing, and known media allowances locally.

        The result is an estimate, not the provider's exact tokenizer count.
        Unsupported binary modalities need an explicit planning estimate.
        """
        total = 32
        for turn in turns:
            text_parts = [turn.role.value, *turn.content.notes,
                          *(part.text for part in turn.content.parts if isinstance(part, Text)),
                          str(turn.tool_calls), turn.tool_call_id, turn.name]
            state = turn.provider_state
            if state is not None:
                if state.reasoning is not None:
                    text_parts.append(state.reasoning)
                if state.reasoning_content is not None:
                    text_parts.append(state.reasoning_content)
                if state.reasoning_details_json is not None:
                    text_parts.append(state.reasoning_details_json)
            text = "\n".join(text_parts)
            total += 16 + self.text_tokens(text)
            for part in turn.content.parts:
                if not isinstance(part, Text):
                    from pm.context_render.turns import Media, MediaKind, VideoFrames
                    if isinstance(part, Media) and part.kind is MediaKind.Image:
                        total += PROMPT_IMAGE_TOKENS
                    elif isinstance(part, VideoFrames):
                        total += len(part.frames) * PROMPT_IMAGE_TOKENS
                    else:
                        raise TypeError("This media type needs an explicit token planning estimate.")
        return ceil(total * self.estimate_factor)

    def text_tokens(self, text: str) -> int:
        encoders = (tiktoken.get_encoding("o200k_base"), tiktoken.get_encoding("cl100k_base"))
        return max(len(encoder.encode(text, disallowed_special=())) for encoder in encoders)

    def measurement_label(self, turns: Sequence[Turn]) -> str:
        """Name the estimator and explicit media allowance in execution receipts."""
        from pm.context_render.turns import Media, MediaKind, VideoFrames
        images = any(isinstance(part, VideoFrames) or isinstance(part, Media) and part.kind is MediaKind.Image
                     for turn in turns for part in turn.content.parts)
        label = f"estimated.tiktoken.conservative.max_o200k_cl100k.x{self.estimate_factor:g}"
        return label + ".images_by_constant" if images else label


class StoryTokenEstimator(TokenEstimator):
    _counts: dict[str, int] = PrivateAttr(default_factory=dict[str, int])

    def text_tokens(self, text: str) -> int:
        cached = self._counts.get(text)
        if cached is None:
            cached = super().text_tokens(text)
            self._counts[text] = cached
        return cached


class ContextBudget(TokenEstimator):
    """Reserve output space and apply the input ceiling to rendered evidence."""

    context_window: int = Field(gt=0, description="Usable serving capacity, including any configured cap.")
    input_fraction: float = Field(default=0.8, gt=0, lt=1, description="Input ceiling; unused space stays unused.")

    @property
    def input_ceiling(self) -> int:
        """Return the largest estimated input that this budget admits."""
        return int(self.context_window * self.input_fraction)

    def require_fit(self, turns: Sequence[Turn]) -> int:
        """Return the estimate or fail before model I/O. Never clip required text."""
        tokens = self.estimate(turns)
        if tokens > self.input_ceiling:
            raise RequiredContextTooLarge(f"Required prompt uses an estimated {tokens} tokens; input ceiling is {self.input_ceiling}.")
        return tokens


@dataclass(frozen=True)
class ContextGroup:
    """An indivisible rendered evidence group with a workflow-owned salience score."""

    name: str
    content: Content
    salience: float
    required: bool = False
    source_ids: tuple[int, ...] = ()
    episode_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class PromptSelection:
    """The actual selected prompt and explicit omissions for its receipt."""

    turns: tuple[Turn, ...]
    included: tuple[str, ...]
    omitted: tuple[str, ...]
    estimated_tokens: int
    source_ids: tuple[int, ...]
    episode_ids: tuple[int, ...]

    def receipt(self) -> SelectionReceipt:
        """Keep a typed execution receipt without serializing the prompt a second time."""
        return SelectionReceipt(included=self.included, omitted=self.omitted,
                                estimated_tokens=self.estimated_tokens, source_ids=self.source_ids, episode_ids=self.episode_ids)


class SelectionReceipt(BaseModel):
    """Persisted decisions made by the shared context selector."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    included: tuple[str, ...]
    omitted: tuple[str, ...]
    estimated_tokens: int
    source_ids: tuple[int, ...]
    episode_ids: tuple[int, ...]
