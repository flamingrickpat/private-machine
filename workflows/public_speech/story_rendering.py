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

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

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


def dialogue_lines(turns: Sequence[DialogueTurn], prompts: dict) -> list[DialogueLine]:
    """Return one renderable line per dialogue turn, with day markers attached.

    A day marker belongs to the first turn of that day, so a block boundary
    never separates a marker from its line.
    """
    speech = prompts.get("speaker_line", '{speaker} says: "{text}"')
    thought = prompts.get("thought_line", "{speaker} thinks: *{text}*")
    marker = prompts.get("day_marker", "*({date})*")
    lines: list[DialogueLine] = []
    day: datetime | None = None
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
    directive_text: str,
) -> list[Turn]:
    """Return the interleaved Story turns.

    Order: system, opening user turn, then dialogue and artifact items in
    world-time order grouped into one turn per uninterrupted run of the same
    role, then the directive appended to the last user turn (or its own user
    turn when the last block is assistant).

    Raises ValueError when directive_text is empty.
    """
    if not directive_text.strip():
        raise ValueError("The directive text must not be empty.")

    items: list[tuple[Role, datetime | None, int, str]] = []
    for line in dialogue:
        items.append((Role.Assistant, line.timestamp, line.record_id, line.text))
    for line in artifacts:
        items.append((Role.User, line.timestamp, line.record_id, line.text))
    items.sort(key=lambda item: (item[1] is None, item[1] or _EPOCH, item[2]))

    blocks: list[tuple[Role, list[str]]] = []
    for role, _ts, _rid, text in items:
        if blocks and blocks[-1][0] is role:
            blocks[-1][1].append(text)
        else:
            blocks.append((role, [text]))

    turns: list[Turn] = [Turn(Role.System, system_text)]

    # The first turn after system is always user: the opening plus any
    # artifacts that sort before the first dialogue line.
    if blocks and blocks[0][0] is Role.User:
        leading = "\n\n".join(blocks.pop(0)[1])
        turns.append(Turn(Role.User, opening_text + "\n\n" + leading))
    else:
        turns.append(Turn(Role.User, opening_text))

    for role, lines in blocks:
        turns.append(Turn(role, "\n".join(lines)))

    # Append directive to the last user turn, or add its own user turn.
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
