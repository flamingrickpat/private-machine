"""Interleave dialogue and artifact blocks into alternating prompt turns.

The Story prompt uses a time-ordered layout:

  system     story-writer instruction + architecture description
  user       opening directive + character card + time
  assistant  dialogue block (both speakers)
  user       artifact block (memories, facts, narratives)
  assistant  dialogue block
  user       artifact block
  ...
  user       GWT state + feelings + action + continue directive

Role rule: every dialogue line (both speakers, and the companion's own
earlier thoughts) is an assistant turn. Every derived artifact (memories,
summaries, facts, narratives, private state) is a user turn. Adjacent
same-role runs merge into one turn, so the result alternates strictly.

This module is pure. It reads no store, calls no model, and embeds no text.
The caller supplies rendered text lines with timestamps and lane names.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Literal, Sequence, TypedDict

from pm.context_render.turns import Role, Turn
from pm.memory.retrieval import utc_time
from workflows.public_speech.public_context import DialogueTurn

_EPOCH = datetime(2000, 1, 1)


@dataclass(frozen=True)
class DialogueLine:
    """One rendered dialogue line with its world time and record id."""

    timestamp: datetime | None
    record_id: int
    text: str


@dataclass(frozen=True)
class ArtifactLine:
    """One derived memory line with its world time, lane, and record id."""

    timestamp: datetime | None
    record_id: int
    lane: str
    text: str


@dataclass(frozen=True, slots=True)
class StoryItem:
    """One prompt item before, at, or after the accepted main snapshot."""

    role: Role
    timestamp: datetime | None
    record_id: int
    text: str
    section: Literal[0, 1, 2]


@dataclass(order=True, frozen=True, slots=True)
class StorySequencePosition:
    section: int
    timestamp_missing: bool
    timestamp: datetime
    record_id: int
    position: int
    item: StoryItem = field(compare=False)


@dataclass(slots=True)
class StoryBlock:
    """Adjacent story lines sharing one chat role."""

    role: Role
    lines: list[str]


class DialogueTemplates(TypedDict, total=False):
    """Optional speech, thought, and day templates from a workflow prompt file."""

    speaker_line: str
    thought_line: str
    day_marker: str


def dialogue_lines(turns: Sequence[DialogueTurn], prompts: DialogueTemplates) -> list[DialogueLine]:
    """Return one renderable line per dialogue turn, with day markers attached.

    A day marker belongs to the first turn of that day, so a block boundary
    never separates a marker from its line.
    """
    speech = prompts.get("speaker_line", '{speaker} says: "{text}"')
    thought = prompts.get("thought_line", "{speaker} thinks: *{text}*")
    marker = prompts.get("day_marker", "*({date})*")
    lines: list[DialogueLine] = []
    day: date | None = None
    for turn in turns:
        ts = utc_time(turn.timestamp) if turn.timestamp else None
        prefix = ""
        if ts and (day is None or ts.date() != day):
            day = ts.date()
            prefix = marker.format(date=f"{ts.day} {ts.strftime('%B %Y')}") + "\n"
        template = thought if turn.kind == "thought" else speech
        text = prefix + template.format(speaker=turn.speaker, text=turn.text)
        lines.append(DialogueLine(
            timestamp=ts,
            record_id=turn.knoxel_id or 0,
            text=text,
        ))
    return lines


def render_interleaved_story(
    *,
    system_text: str,
    opening_text: str,
    dialogue: Sequence[DialogueLine],
    artifacts: Sequence[ArtifactLine],
    directive_text: str | None,
    guidance_frontier: int | None = None,
) -> list[Turn]:
    """Return the interleaved Story turns.

    Order: system, opening user turn, then dialogue and artifact items in
    world-time order grouped into one turn per uninterrupted run of the same
    role, then the directive appended to the last user turn (or its own user
    turn when the last block is assistant).

    ``None`` leaves the final transcript block open for natural continuation.
    An explicitly supplied empty directive is an internal contract error.
    """
    if directive_text is not None and not directive_text.strip():
        raise ValueError("The directive text must not be empty.")

    # Source identity anchors the main frame even when an imported record has
    # an old world timestamp. Time alone cannot decide which view saw a line.
    items: list[StoryItem] = []
    for line in dialogue:
        section: Literal[0, 1, 2] = 0
        if guidance_frontier is not None and line.record_id > guidance_frontier:
            section = 2
        items.append(StoryItem(Role.Assistant, line.timestamp, line.record_id, line.text, section))
    for line in artifacts:
        section = 0
        if guidance_frontier is not None:
            if line.lane == "main_guidance":
                section = 1
            elif line.record_id > guidance_frontier:
                section = 2
        items.append(StoryItem(Role.User, line.timestamp, line.record_id, line.text, section))
    ordered = [StorySequencePosition(item.section, item.timestamp is None, item.timestamp or _EPOCH,
                 item.record_id, index, item) for index, item in enumerate(items)]
    ordered.sort()
    items = [position.item for position in ordered]

    blocks: list[StoryBlock] = []
    for item in items:
        if blocks and blocks[-1].role is item.role:
            blocks[-1].lines.append(item.text)
        else:
            blocks.append(StoryBlock(item.role, [item.text]))

    turns: list[Turn] = [Turn(Role.System, system_text)]

    # The first turn after system is always user: the opening plus any
    # artifacts that sort before the first dialogue line.
    if blocks and blocks[0].role is Role.User:
        leading = "\n\n".join(blocks.pop(0).lines)
        turns.append(Turn(Role.User, opening_text + "\n\n" + leading))
    else:
        turns.append(Turn(Role.User, opening_text))

    for block in blocks:
        turns.append(Turn(block.role, "\n".join(block.lines)))

    # Append directive to the last user turn, or add its own user turn.
    if directive_text is None:
        return turns
    if turns[-1].role is Role.User:
        turns[-1] = Turn(Role.User, turns[-1].content.text + "\n\n" + directive_text)
    else:
        turns.append(Turn(Role.User, directive_text))

    return turns


__all__ = [
    "ArtifactLine",
    "DialogueLine",
    "dialogue_lines",
    "render_interleaved_story",
]
