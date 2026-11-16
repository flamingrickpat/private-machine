"""Fit optional evidence and batch complete tasks within conservative context ceilings."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from math import ceil
from datetime import datetime, timezone
from random import Random

import tiktoken
from pydantic import BaseModel, ConfigDict, Field

from pm.context_render.turns import Content, ContentPart, ContextTag, Role, Text, Turn
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
    input_fraction: float = Field(default=0.8, gt=0, lt=1, description="Input ceiling; unused space stays unused.")
    reserved_tokens: int = Field(default=0, ge=0, description="Input tokens occupied by wire schemas outside the rendered turns.")

    @property
    def input_ceiling(self) -> int:
        """Return the largest estimated input that this budget admits."""
        return int(self.context_window * self.input_fraction) - self.reserved_tokens

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
    recency: float = 0


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
    """Keep required groups and rank optional evidence by recency, salience, and randomness.

    A group must include its citations and required support. The caller supplies
    domain salience; this function performs no model call or memory query.
    """
    def assemble(chosen: Sequence[ContextGroup], omitted: Sequence[str]) -> tuple[Turn, ...]:
        """Render the current selection, including the omission receipt."""
        evidence = Content("")
        for group in chosen:
            evidence += f"\n{group.name}:\n"
            evidence += optional_content(group.content, ContextTag(group.name, group.salience,
                record_id=group.source_ids[0] if group.source_ids else None,
                support_ids=group.source_ids, required=group.required, recency=group.recency))
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
    A complete prompt that fits needs no per-group fitting pass.
    """
    if input_ceiling <= 0:
        raise ValueError("The rendered prompt input ceiling must be positive.")
    chosen = [group for group in groups if group.required]
    optional = rank_context([group for group in groups if not group.required])
    omitted: list[str] = []

    def require_fit() -> int:
        tokens = estimator.estimate(assemble(chosen, omitted))
        if tokens > input_ceiling:
            raise RequiredContextTooLarge(
                f"Required prompt uses an estimated {tokens} tokens; input ceiling is {input_ceiling}."
            )
        return tokens

    complete = [*chosen, *optional]
    if estimator.estimate(assemble(complete, ())) <= input_ceiling:
        # Large remote windows often hold the entire retrieval result. Avoid
        # rebuilding and tokenizing thousands of successively larger prompts.
        chosen = complete
        omitted.clear()
    else:
        require_fit()
        # Reserve the omission receipt before optional groups compete.
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


def optional_text(text: str, tag: ContextTag) -> Content:
    """Attach selection policy without changing the text sent when everything fits."""
    return Content(Text(text, tag))


def context_recency(occurred_at: datetime | str | None) -> float:
    """Return UTC seconds for relative recency ranking; missing chronology has no signal."""
    if occurred_at is None:
        return 0
    when = datetime.fromisoformat(occurred_at) if isinstance(occurred_at, str) else occurred_at
    return when.replace(tzinfo=timezone.utc).timestamp() if when.tzinfo is None else when.timestamp()


def rank_context(groups: Sequence[ContextGroup], *, random: Random | None = None) -> list[ContextGroup]:
    """Combine 40% recency, 40% salience, and 20% fresh randomness.

    Normalize each supplied signal across the competing groups. Draw once per
    selection, so candidate fitting cannot reshuffle a measured prompt.
    Required groups never compete here. A seeded generator supports small
    deterministic checks without replacing any external service.
    """
    if not groups:
        return []
    rng = random if random is not None else Random()
    minimum_salience = min(group.salience for group in groups)
    salience_range = max(group.salience for group in groups) - minimum_salience
    chronological = sorted({group.recency for group in groups})
    recency_scores = {value: position / (len(chronological) - 1) if len(chronological) > 1 else 0
                      for position, value in enumerate(chronological)}
    scores: dict[str, float] = {}
    for group in groups:
        salience = (group.salience - minimum_salience) / salience_range if salience_range else 0
        recency = recency_scores[group.recency]
        scores[group.name] = .4 * recency + .4 * salience + .2 * rng.random()
    return sorted(groups, key=lambda group: scores[group.name], reverse=True)


def optional_content(content: Content, tag: ContextTag) -> Content:
    """Mark a rendered text group; attachments and their references stay required."""
    if any(not isinstance(part, Text) for part in content.parts):
        return content
    return Content(tuple(replace(part, context=tag) for part in content.parts if isinstance(part, Text)),
                   notes=content.notes)


def fit_turns(turns: Sequence[Turn], budget: ContextBudget, *, include_optional: bool = True) -> PromptSelection:
    """Fit complete turns by annotated groups; unannotated text and media stay required.

    Owners mark whole records or context sections before composing their prompts.
    Support records present in the prompt follow their selected dependent records.
    The adapter preserves role order, tool fields, media, and required repair turns.
    """
    tags: dict[str, ContextTag] = {}
    for turn in turns:
        for part in turn.content.parts:
            if isinstance(part, Text) and part.context is not None:
                tag = part.context
                previous = tags.setdefault(tag.name, tag)
                if previous != tag:
                    raise ValueError(f"Conflicting context policy for {tag.name}.")

    by_id: dict[int, list[ContextTag]] = {}
    for tag in tags.values():
        if tag.record_id is not None:
            by_id.setdefault(tag.record_id, []).append(tag)

    def support_ids(tag: ContextTag) -> tuple[int, ...]:
        selected = set(tag.support_ids)
        if tag.record_id is not None:
            selected.add(tag.record_id)
        pending = list(selected)
        while pending:
            for source in by_id.get(pending.pop(), ()):
                for record_id in source.support_ids:
                    if record_id not in selected:
                        selected.add(record_id)
                        pending.append(record_id)
        return tuple(sorted(selected))

    groups = [ContextGroup(tag.name, Content(""), tag.salience, required=tag.required,
                           source_ids=support_ids(tag), recency=tag.recency) for tag in tags.values()]

    def assemble(chosen: Sequence[ContextGroup], omitted: Sequence[str]) -> Sequence[Turn]:
        names = {group.name for group in chosen}
        ids = {record_id for group in chosen for record_id in group.source_ids}
        rendered: list[Turn] = []
        for turn in turns:
            parts: list[ContentPart] = []
            for part in turn.content.parts:
                if isinstance(part, Text) and part.context is not None:
                    tag = part.context
                    if tag.name not in names and tag.record_id not in ids:
                        continue
                parts.append(part)
            if not parts and turn.content.parts and not turn.content.notes and not turn.tool_calls and not turn.tool_call_id:
                continue
            rendered.append(replace(turn, content=Content(tuple(parts), notes=turn.content.notes)))
        if omitted:
            # A short count cannot consume the space saved by omitting many records.
            index = next((index for index in reversed(range(len(rendered))) if rendered[index].role is Role.User), 0)
            last = rendered[index]
            rendered[index] = replace(last, content=last.content +
                                  f"\n({len(omitted)} optional context groups omitted; use only the evidence shown.)")
        return rendered

    selectable = groups if include_optional else [group for group in groups if group.required]
    return select_rendered_context(selectable, budget.input_ceiling, assemble, estimator=budget)


def batch_prompt(context: Sequence[Turn], groups: Sequence[ContextGroup], budget: ContextBudget,
                 instruction: Callable[[Sequence[ContextGroup]], Content]) -> tuple[PromptSelection, ...]:
    """Process every independent item once under one fixed, fitted context.

    Reserve required context and the largest indivisible task first. Split
    remaining space equally between optional common context and more task
    items, so rich context does not force one model call per candidate.
    Then pack all items into complete prompts. No task is optional.
    Callers validate each batch's IDs and merge results before changing state.
    A single task that cannot fit with mandatory context raises visibly.
    """
    if not groups:
        return ()
    names = [group.name for group in groups]
    if len(names) != len(set(names)):
        raise ValueError("Batch task names must be unique.")
    note = "\nEvaluate every listed item. If multiple batches are needed, the caller combines their results before deciding."

    def task(chosen: Sequence[ContextGroup]) -> Turn:
        return Turn(Role.User, instruction(chosen) + note)

    # A task payload must remain complete after the generic inference fitter.
    # Freeze annotations on both the shared context and the task instructions.
    def freeze(turn: Turn) -> Turn:
        parts = tuple(replace(part, context=None) if isinstance(part, Text) else part for part in turn.content.parts)
        return replace(turn, content=Content(parts, notes=turn.content.notes))

    entire = [*context, task(groups)]
    if budget.estimate(entire) <= budget.input_ceiling:
        shared = tuple(freeze(turn) for turn in context)
    else:
        largest = max(budget.estimate([task([group])]) - budget.estimate([]) for group in groups) + 1
        required = fit_turns(context, budget, include_optional=False)
        discretionary = budget.input_ceiling - required.estimated_tokens - largest
        if discretionary < 0:
            raise RequiredContextTooLarge(
                f"Required shared context uses {required.estimated_tokens} tokens and one indivisible task needs {largest}; "
                f"input ceiling is {budget.input_ceiling}.")
        context_allowance = required.estimated_tokens + discretionary // 2
        reserved = budget.reserved_tokens + budget.input_ceiling - context_allowance
        context_budget = budget.model_copy(update={"reserved_tokens": reserved})
        selected = fit_turns(context, context_budget)
        shared = tuple(freeze(turn) for turn in selected.turns)

    def render(chosen: Sequence[ContextGroup]) -> tuple[Turn, ...]:
        return (*shared, freeze(task(chosen)))

    def selection(chosen: Sequence[ContextGroup]) -> PromptSelection:
        turns = render(chosen)
        tokens = budget.require_fit(turns)
        return PromptSelection(turns, tuple(group.name for group in chosen), (), tokens,
            tuple(sorted({key for group in chosen for key in group.source_ids})),
            tuple(sorted({key for group in chosen for key in group.episode_ids})))

    if budget.estimate(render(groups)) <= budget.input_ceiling:
        return (selection(groups),)
    batches: list[PromptSelection] = []
    chosen: list[ContextGroup] = []
    for group in rank_context(groups):
        if budget.estimate(render([*chosen, group])) > budget.input_ceiling:
            if chosen:
                batches.append(selection(chosen))
                chosen = []
            budget.require_fit(render([group]))
        chosen.append(group)
    if chosen:
        batches.append(selection(chosen))
    covered = [name for batch in batches for name in batch.included]
    if sorted(covered) != sorted(names):
        raise AssertionError("Prompt batches lost or repeated a task.")
    return tuple(batches)
