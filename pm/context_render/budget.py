"""Select rendered context by workflow salience within a conservative ceiling."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import ceil

import tiktoken
from pydantic import BaseModel, ConfigDict, Field

from pm.context_render.turns import Content, Role, Text, Turn
from pm.constants import PROMPT_IMAGE_TOKENS


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
        encoders = (tiktoken.get_encoding("o200k_base"), tiktoken.get_encoding("cl100k_base"))
        for turn in turns:
            text = "\n".join([turn.role.value, *turn.content.notes,
                              *(part.text for part in turn.content.parts if isinstance(part, Text)),
                              str(turn.tool_calls), turn.tool_call_id, turn.name])
            total += 16 + max(len(encoder.encode(text, disallowed_special=())) for encoder in encoders)
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

    def measurement_label(self, turns: Sequence[Turn]) -> str:
        """Name the estimator and explicit media allowance in execution receipts."""
        from pm.context_render.turns import Media, MediaKind, VideoFrames
        images = any(isinstance(part, VideoFrames) or isinstance(part, Media) and part.kind is MediaKind.Image
                     for turn in turns for part in turn.content.parts)
        label = f"estimated.tiktoken.conservative.max_o200k_cl100k.x{self.estimate_factor:g}"
        return label + ".images_by_constant" if images else label


class ContextBudget(TokenEstimator):
    """Reserve output space and apply the input ceiling to rendered evidence."""

    context_window: int = Field(gt=0, description="Usable serving capacity, including any configured cap.")
    input_fraction: float = Field(default=0.8, gt=0, le=0.8, description="Input ceiling; unused space stays unused.")

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


def select_context(instructions: Sequence[Turn], groups: Sequence[ContextGroup],
                   budget: ContextBudget) -> PromptSelection:
    """Keep required groups and add optional groups in descending salience order.

    A group must include its citations and required support. The caller supplies
    domain salience; this function performs no model call or memory query.
    """
    def assemble(chosen: Sequence[ContextGroup], omitted: Sequence[str]) -> tuple[Turn, ...]:
        """Render the current selection, including the omission receipt."""
        evidence = Content("")
        for group in chosen:
            evidence += f"\n{group.name}:\n"
            evidence += group.content
        if omitted:
            evidence += "\nOmitted optional context groups: " + ", ".join(omitted)
        return (*instructions, Turn(Role.User, evidence))

    return select_rendered_context(groups, budget.input_ceiling, assemble, estimator=budget)


def select_rendered_context(
    groups: Sequence[ContextGroup],
    input_ceiling: int,
    assemble: Callable[[Sequence[ContextGroup], Sequence[str]], Sequence[Turn]],
    *,
    estimator: TokenEstimator,
) -> PromptSelection:
    """Select optional groups against the caller's complete rendered prompt.

    The renderer preserves workflow-specific role ordering and includes required
    instructions, citations, omission notices, and any repair turns. Required
    content must fit before optional groups compete by their existing salience.
    """
    if input_ceiling <= 0:
        raise ValueError("The rendered prompt input ceiling must be positive.")
    chosen = [group for group in groups if group.required]
    optional = sorted((group for group in groups if not group.required), key=lambda group: group.salience, reverse=True)
    omitted: list[str] = []

    def require_fit() -> int:
        tokens = estimator.estimate(assemble(chosen, omitted))
        if tokens > input_ceiling:
            raise RequiredContextTooLarge(
                f"Required prompt uses an estimated {tokens} tokens; input ceiling is {input_ceiling}."
            )
        return tokens

    require_fit()
    # Reserve the omission receipt before deciding which optional groups fit.
    omitted.extend(group.name for group in optional)
    require_fit()
    for group in optional:
        chosen.append(group)
        omitted.remove(group.name)
        if estimator.estimate(assemble(chosen, omitted)) > input_ceiling:
            chosen.pop()
            omitted.append(group.name)
    turns = tuple(assemble(chosen, omitted))
    selected_sources: set[int] = set()
    selected_episodes: set[int] = set()
    for group in chosen:
        selected_sources.update(group.source_ids)
        selected_episodes.update(group.episode_ids)
    source_ids = tuple(sorted(selected_sources))
    episode_ids = tuple(sorted(selected_episodes))
    return PromptSelection(turns, tuple(group.name for group in chosen), tuple(omitted), require_fit(), source_ids, episode_ids)
