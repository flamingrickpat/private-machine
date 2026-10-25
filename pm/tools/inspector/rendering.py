"""Render one record, one shard, and one cycle as the model reads them.

The cycle turns workspace shards and the CSM into story text. This module
calls the same renderers that the cycle calls
(:mod:`pm.context_render.story`), so the text here is the text the model got.
On top of that text the module measures what a maintainer looks for when they
hunt for waste:

- the token cost of every line, and the framing cost around it;
- the same content more than once in one block or in one story;
- a record that sits in several shards of one cycle at once;
- a record that appears both as an item and in the tail of its shard;
- the token count that the cycle stored on an item, next to the count that
  the line costs now.

The prompt words live in ``workflows/main_cognition/phases/prompts``. This
module reads two of those YAML files as data, because the inspector must show
the operator's current words rather than a copy of them. It imports nothing
from ``workflows``. A missing story prompt file raises ``FileNotFoundError``,
so a moved file is loud instead of quietly showing stale labels.

Every renderer takes a :class:`pm.tools.inspector.model.MemoryIndex`. Nothing
here writes to the store, calls a model, or embeds text.
"""
from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from pm.context_render import ContextKind, render_context
from pm.context_render.story import (
    ShardLabels, fill_story_names, record_tokens, render_story, shard_block_text, shard_focus, shard_name,
    shard_titles, story_entity_names, story_line, story_shards, story_tokens,
)
from pm.context_render.turns import format_content
from pm.paths import REPO_ROOT
from pm.records.workspace import GlobalWorkspace, ShardKey, WorkspaceShard
from pm.tools.inspector.model import MemoryIndex
from pm.utils import yaml_util

#: The prompt file with the story template, the shard template, and the shard
#: block labels. Read as data, never imported.
STORY_PROMPTS = REPO_ROOT / "workflows" / "main_cognition" / "phases" / "prompts" / "story.yaml"
#: The prompt file with the focus text of each shard kind.
ATTENTION_PROMPTS = REPO_ROOT / "workflows" / "main_cognition" / "phases" / "prompts" / "attention.yaml"


class LineRender(BaseModel):
    """One record as one story line, with what that line costs."""

    record_id: int
    line: str
    tokens: int
    content_tokens: int
    kind: str = ""
    source: str = ""
    content_chars: int = 0
    in_checkpoint: bool = True


class ItemRender(LineRender):
    """One workspace item: its line, and what the cycle recorded about it."""

    entered_tick: int = 0
    activation_at_entry: float = 0.0
    structure_id: int | None = None
    reason: str = ""
    required_by: str = ""
    stored_tokens: int = 0


class ShardRender(BaseModel):
    """One shard of one cycle, rendered as the story block that it becomes.

    ``items`` keeps the shard order (the order attention chose). The rendered
    text inside the block is in world-time order, because that is what
    ``render_story`` does. ``framing_tokens`` is what the block costs beyond
    its own lines: title, tail label, the ``BEGIN``/``END`` markers, and the
    blank lines.
    """

    key: ShardKey
    name: str
    title: str
    tail_label: str
    focus: str
    in_story: bool = True
    items: list[ItemRender] = Field(default_factory=list)
    tail: list[LineRender] = Field(default_factory=list)
    block_text: str = ""
    items_text: str = ""
    tail_text: str = ""
    items_tokens: int = 0
    tail_tokens: int = 0
    block_tokens: int = 0
    framing_tokens: int = 0
    stored_item_tokens: int = 0
    stored_tail_tokens: int = 0
    in_items_and_tail: list[int] = Field(default_factory=list)
    duplicate_content: list[list[int]] = Field(default_factory=list)
    token_drift: dict[str, int] = Field(default_factory=dict)


class TextRender(BaseModel):
    """One rendered text with its size."""

    text: str = ""
    tokens: int = 0
    lines: int = 0
    record_ids: list[int] = Field(default_factory=list)


class CsmItemRender(BaseModel):
    """One CSM entry, with the line cost of its record and where it also sits."""

    record_id: int
    activation: float = 0.0
    base_activation: float = 0.0
    entered_tick: int = 0
    last_boost_tick: int = 0
    ticks_present: int = 0
    endorsers: list[str] = Field(default_factory=list)
    merged_ids: list[int] = Field(default_factory=list)
    shard_hint: ShardKey | None = None
    tokens: int = 0
    in_workspace: bool = False


class StoryRender(BaseModel):
    """The reconstructed story prefix of one cycle.

    Two inputs of the live prefix do not survive in a stored cycle: the
    feeling in words, and the candidate set that the retrieve phase had
    picked. ``notes`` names every substitution, so a reader never takes a
    reconstruction for a recording.
    """

    text: str = ""
    tokens: int = 0
    shards: list[ShardKey] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    other_characters: str = ""


class TickRender(BaseModel):
    """One cycle: its shards, its CSM, its new stimuli, and the waste flags.

    ``shared_records`` maps a record id to the shards that hold it, for the
    records that more than one shard holds in this cycle. ``totals`` holds the
    token sums, and ``duplicate_content`` lists the lines that repeat.
    ``token_estimate`` and ``llm_calls`` are what the cycle recorded for
    itself, so a reader can compare them with the sums recomputed here.
    """

    tick_id: int
    timestamp: datetime | None = None
    slot_id: str = ""
    turn_summary: str = ""
    experience_id: int | None = None
    decision_id: int | None = None
    token_estimate: int = 0
    llm_calls: int = 0
    groups: list[dict] = Field(default_factory=list)
    shards: list[ShardRender] = Field(default_factory=list)
    csm: list[CsmItemRender] = Field(default_factory=list)
    csm_text: TextRender = Field(default_factory=TextRender)
    new_items: TextRender = Field(default_factory=TextRender)
    story: StoryRender = Field(default_factory=StoryRender)
    totals: dict[str, int] = Field(default_factory=dict)
    shared_records: dict[str, list[str]] = Field(default_factory=dict)
    duplicate_content: list[dict] = Field(default_factory=list)


class RecordRender(BaseModel):
    """One record as the cycle renders it, and as typed context.

    ``story_line`` is the line that the CSM and the shards show.
    ``context_text`` is the same record through ``render_context`` of
    :mod:`pm.context_render`, which is how records reach a tool-argument
    prompt. The two renderings differ on purpose: the story line is short
    prose, and the typed context is a field table. Both cost tokens, so both
    are measured.
    """

    record_id: int
    kind: str = ""
    record_class: str = ""
    source: str = ""
    category: str = ""
    feature_type: str = ""
    timestamp: datetime | None = None
    content: str = ""
    story_line: str = ""
    story_line_with_id: str = ""
    line_tokens: int = 0
    content_tokens: int = 0
    framing_tokens: int = 0
    line_chars: int = 0
    context_text: str = ""
    context_tokens: int = 0
    in_shards: int = 0


def _load_labels(path: Path) -> ShardLabels:
    """Return the ``shard_labels`` block of one prompt file, or empty labels."""
    data = yaml_util.safe_load(path.read_text(encoding="utf-8")) or {}
    return ShardLabels.model_validate(data.get("shard_labels") or {})


class StoryPrompts:
    """The prompt texts that decide how a shard block and a story look.

    ``shard_template``, the story template, and the block labels come from
    ``story.yaml``. The focus labels come from ``attention.yaml``, because the
    attention phase owns those words. A label that neither file sets stays
    empty, and :meth:`describe` lists it instead of inventing a phrase.
    """

    def __init__(self, *, story_path: Path = STORY_PROMPTS, attention_path: Path = ATTENTION_PROMPTS) -> None:
        if not story_path.exists():
            raise FileNotFoundError(f"Story prompts are missing: {story_path}")
        story = yaml_util.safe_load(story_path.read_text(encoding="utf-8")) or {}
        self.story_path = story_path
        self.attention_path = attention_path
        self.system_prompt = story.get("system_prompt", "")
        self.story_template = story.get("story_template", "")
        self.story_ack = story.get("story_ack", "")
        self.shard_template = story.get("shard_template", "")
        self.labels = _load_labels(story_path)
        self.focus_labels = _load_labels(attention_path) if attention_path.exists() else ShardLabels()

    def describe(self) -> dict:
        """Return the prompt texts and labels, so the reader sees the words in use."""
        return {
            "story_prompts": str(self.story_path),
            "attention_prompts": str(self.attention_path),
            "system_prompt": self.system_prompt,
            "story_template": self.story_template,
            "story_ack": self.story_ack,
            "shard_template": self.shard_template,
            "labels": self.labels.model_dump(),
            "focus_labels": {name: value for name, value in self.focus_labels.model_dump().items()
                             if name.endswith("_focus")},
            "missing_labels": sorted(name for name, value in self.labels.model_dump().items() if not value),
        }


def _has_shard(workspace: GlobalWorkspace, key: ShardKey) -> bool:
    """Return True when this workspace holds the shard of ``key``.

    A cycle creates a shard when the first stimulus of an entity or a workflow
    arrives. An old checkpoint can name a group whose shard is gone, so the
    inspector checks before it asks, because ``GlobalWorkspace.shard`` raises
    ``KeyError`` on a missing shard.
    """
    if key.kind == "self":
        return True
    if key.kind == "entity":
        return key.ref.isdigit() and int(key.ref) in workspace.entity_shards
    return key.ref in workspace.workflow_shards


def _duplicates(rows: list[LineRender]) -> tuple[list[list[int]], int, list[dict]]:
    """Return the repeated-content groups, their total waste, and one row per group.

    Two records whose lines hold the same content give the model the same
    words twice. The cheapest copy counts; the rest count as waste. The
    comparison ignores case and repeated spaces, because those differ only in
    how the record was written, not in what the model reads.
    """
    by_line: dict[str, list[LineRender]] = {}
    for row in rows:
        by_line.setdefault(" ".join(row.line.split()).casefold(), []).append(row)
    groups, wasted, details = [], 0, []
    for group in by_line.values():
        if len(group) < 2:
            continue
        group_waste = sum(row.tokens for row in group) - min(row.tokens for row in group)
        groups.append([row.record_id for row in group])
        wasted += group_waste
        details.append({
            "line": group[0].line,
            "record_ids": [row.record_id for row in group],
            "tokens": sum(row.tokens for row in group),
            "wasted_tokens": group_waste,
            "identical_text": len({row.line for row in group}) == 1,
        })
    return groups, wasted, details


class StoryRenderer:
    """Render records, shards, and cycles of one checkpoint as story text.

    Build one renderer per inspector run. It holds the index and the prompt
    texts, and caches nothing but the record views that the index already
    keeps. One cycle render costs a few hundred token counts, which is fast
    enough to redo per request and keeps the report honest when the reader
    switches cycles.
    """

    def __init__(self, index: MemoryIndex, prompts: StoryPrompts | None = None) -> None:
        self.index = index
        self.prompts = prompts or StoryPrompts()
        self.companion = index.companion_name

    # ------------------------------------------------------------------ names

    def name_of(self, key: ShardKey) -> str:
        """Return the story name of one shard key, with the names of the checkpoint."""
        return shard_name(key, user_entity_id=self.index.user_entity_id, user_name=self.index.user_name,
                          entity_names=self.index.entity_names)

    # ---------------------------------------------------------------- records

    def line(self, record_id: int, *, with_id: bool = False) -> LineRender:
        """Return one record as one story line, with its token cost."""
        view = self.index.view(record_id)
        text = story_line(view, with_id=with_id)
        return LineRender(
            record_id=record_id, line=text, tokens=story_tokens(text),
            content_tokens=story_tokens(view.content), kind=view.kind or view.record_class,
            source=view.source, content_chars=len(view.content or ""),
            in_checkpoint=self.index.label(record_id).in_checkpoint)

    def lines(self, record_ids: Iterable[int]) -> list[LineRender]:
        """Return the story lines of many records, in the order given."""
        return [self.line(record_id) for record_id in record_ids]

    def record(self, record_id: int) -> RecordRender:
        """Return one record with both of its renderings and their token costs.

        A record that the checkpoint does not hold renders its stub line and
        gets no typed context, because there is no record to render.
        """
        view = self.index.view(record_id)
        label = self.index.label(record_id)
        line = story_line(view, with_id=False)
        with_id = story_line(view, with_id=True)
        record = self.index.records.get(record_id)
        context_text = render_context([record], ContextKind.All).text if record is not None else ""
        content_tokens = story_tokens(view.content)
        return RecordRender(
            record_id=record_id, kind=view.kind or label.kind, record_class=view.record_class or label.record_class,
            source=view.source, category=view.category, feature_type=view.feature_type,
            timestamp=view.timestamp, content=view.content,
            story_line=line, story_line_with_id=with_id,
            line_tokens=story_tokens(line), content_tokens=content_tokens,
            framing_tokens=story_tokens(line) - content_tokens, line_chars=len(line),
            context_text=context_text, context_tokens=story_tokens(context_text),
            in_shards=len(self.index.shard_hits(record_id)))

    # ----------------------------------------------------------------- shards

    def shard(self, key: ShardKey, shard: WorkspaceShard, *, in_story: bool = True) -> ShardRender:
        """Render one workspace shard as its story block, with the cost of each line.

        ``in_story`` marks a shard that the shared story prefix of this cycle
        showed. A shard can exist and still stay out of the story, when its
        group carried no stimulus in this cycle.
        """
        name = self.name_of(key)
        title, tail_label = shard_titles(key, name, self.companion, self.prompts.labels)
        item_ids = shard.item_ids()
        views = self.index.views([*item_ids, *shard.tail_ids])
        block = shard_block_text(key, name, self.companion, views, item_ids, shard.tail_ids,
                                 template=self.prompts.shard_template, labels=self.prompts.labels)
        items = []
        for item in shard.items:
            items.append(ItemRender(**self.line(item.record_id).model_dump(),
                                    entered_tick=item.entered_tick, activation_at_entry=item.activation_at_entry,
                                    structure_id=item.structure_id, reason=item.reason,
                                    required_by=item.required_by, stored_tokens=item.tokens))
        tail = self.lines(shard.tail_ids)
        items_text = render_story(views, item_ids, with_id=False)
        tail_text = render_story(views, shard.tail_ids, with_id=False)
        groups, _, _ = _duplicates([*items, *tail])
        drift = {str(item.record_id): item.tokens - self.line(item.record_id).tokens
                 for item in shard.items if item.tokens}
        return ShardRender(
            key=key, name=name, title=title, tail_label=tail_label,
            focus=shard_focus(key, name, self.companion, self.prompts.focus_labels), in_story=in_story,
            items=items, tail=tail, block_text=block, items_text=items_text, tail_text=tail_text,
            items_tokens=story_tokens(items_text), tail_tokens=story_tokens(tail_text),
            block_tokens=story_tokens(block),
            framing_tokens=story_tokens(block) - story_tokens(items_text) - story_tokens(tail_text),
            stored_item_tokens=sum(item.tokens for item in shard.items),
            stored_tail_tokens=sum(record_tokens(views, record_id) for record_id in shard.tail_ids),
            in_items_and_tail=[record_id for record_id in item_ids if record_id in set(shard.tail_ids)],
            duplicate_content=groups,
            token_drift={record_id: delta for record_id, delta in drift.items() if delta})

    # ------------------------------------------------------------------ cycle

    def tick(self, tick_id: int) -> TickRender:
        """Render one cycle: its shards, its CSM, its new stimuli, and the waste flags.

        Raises ``KeyError`` when the checkpoint holds no cycle state for
        ``tick_id``. A cycle with a GWT turn and no state has no workspace to
        render, so the caller shows the turn instead.
        """
        state = self.index.states[tick_id]
        turn = self.index.turns.get(tick_id)
        trace = turn.trace if turn is not None else None
        workspace = state.workspace
        group_keys = [group.key for group in state.stimulus_groups]
        shown = {shard.key for shard in story_shards(
            workspace, [key for key in group_keys if _has_shard(workspace, key)])}
        shards = [self.shard(shard.key, shard, in_story=shard.key in shown)
                  for shard in self.index.shards(state)]
        csm_rows = self.csm(tick_id)
        for row in csm_rows:
            row.in_workspace = any(row.record_id == held.record_id for shard in shards
                                   for held in [*shard.items, *shard.tail])
        csm_text = self._text([row.record_id for row in csm_rows])
        new_items = self._text(state.claimed_ids())
        story = self.story(tick_id)
        all_lines = [row for shard in shards for row in [*shard.items, *shard.tail]]
        _, wasted, details = _duplicates(all_lines)
        return TickRender(
            tick_id=tick_id, timestamp=state.timestamp,
            slot_id=turn.slot_id if turn is not None else "(no GWT turn)",
            turn_summary=turn.summary if turn is not None else "",
            experience_id=turn.experience_id if turn is not None else None,
            decision_id=turn.decision_id if turn is not None else None,
            token_estimate=(trace.input_tokens_estimate + trace.output_tokens_estimate) if trace else 0,
            llm_calls=trace.llm_calls if trace else 0,
            groups=[{"kind": group.key.kind, "ref": group.key.ref, "stimulus_ids": group.stimulus_ids,
                     "category": group.category, "insignificant": group.insignificant}
                    for group in state.stimulus_groups],
            shards=shards, csm=csm_rows, csm_text=csm_text, new_items=new_items, story=story,
            totals={
                "shards": len(shards),
                "shards_in_story": len(shown),
                "item_lines": len(all_lines),
                "story_tokens": story.tokens,
                "block_tokens": sum(shard.block_tokens for shard in shards),
                "content_tokens": sum(row.content_tokens for row in all_lines),
                "framing_tokens": sum(shard.framing_tokens for shard in shards),
                "csm_tokens": csm_text.tokens,
                "csm_only_records": len([row for row in csm_rows if not row.in_workspace]),
                "new_tokens": new_items.tokens,
                "duplicate_wasted_tokens": wasted,
                "stored_item_tokens": sum(shard.stored_item_tokens for shard in shards),
            },
            shared_records=self._shared(shards),
            duplicate_content=details,
        )

    def _shared(self, shards: list[ShardRender]) -> dict[str, list[str]]:
        """Return the records that more than one shard of this cycle holds."""
        holders: dict[int, list[str]] = {}
        for shard in shards:
            label = f"{shard.key.kind}:{shard.key.ref or '-'}"
            held = dict.fromkeys([row.record_id for row in shard.items] + [row.record_id for row in shard.tail])
            for record_id in held:
                holders.setdefault(record_id, []).append(label)
        return {str(record_id): labels for record_id, labels in holders.items() if len(labels) > 1}

    def csm(self, tick_id: int) -> list[CsmItemRender]:
        """Return the CSM entries of one cycle with the line cost of each record."""
        state = self.index.states.get(tick_id)
        if state is None:
            return []
        views = self.index.views(state.csm.items)
        return [CsmItemRender(
            record_id=record_id, activation=item.activation, base_activation=item.base_activation,
            entered_tick=item.entered_tick, last_boost_tick=item.last_boost_tick,
            ticks_present=item.ticks_present, endorsers=list(item.endorsers), merged_ids=list(item.merged_ids),
            shard_hint=item.shard_hint, tokens=record_tokens(views, record_id))
            for record_id, item in state.csm.items.items()]

    def story(self, tick_id: int) -> StoryRender:
        """Rebuild the shared story prefix of one cycle as far as the checkpoint allows.

        The rebuild uses the same template, labels, shard selection, and story
        line renderers as the cycle. Two slots cannot be read back from a
        stored cycle, and ``notes`` names each substitution:

        - the feeling in words: the live cycle reads ``affect.felt``, which no
          record keeps, so the inspector shows the summary of the last
          broadcast of the cycle. It is the same kind of sentence, and it can
          be worded differently;
        - the new events: the live prefix takes them from the cycle request,
          which is not stored. The inspector takes the stimulus ids of the
          stored stimulus groups, which is what intake made of them;
        - the character card: the checkpoint carries the card of the persona
          files at load time, which can be newer than the cycle.

        This is the whole-moment prefix, so no tail record is hidden. A
        per-group prefix of the live cycle leaves out the experience records
        of the other groups; no checkpoint stores that choice.
        """
        state = self.index.states[tick_id]
        group_keys = [group.key for group in state.stimulus_groups]
        shown = story_shards(state.workspace, [key for key in group_keys
                                               if _has_shard(state.workspace, key)])
        blocks = [self.shard(shard.key, shard) for shard in shown]
        others = story_entity_names(group_keys, user_entity_id=self.index.user_entity_id,
                                    user_name=self.index.user_name, entity_names=self.index.entity_names)
        feeling = state.broadcasts[-1].summary if state.broadcasts else ""
        card = self.index.store.ghost_config.character_card_story
        new_ids = state.claimed_ids()
        notes = [
            "feeling: the summary of the last broadcast of this cycle. The live feeling text is not stored.",
            "new events: the stimulus ids of the stored groups. The live prefix reads the cycle request, which is not stored.",
            f"character card: the card that the persona files hold now ({len(card or '')} chars).",
            "shards: the stored workspace of this cycle, with every tail record. The per-group prefix of the live cycle hides the experience records of the other groups.",
        ]
        text = ""
        if self.prompts.story_template:
            filled = fill_story_names(self.prompts.story_template, self.companion, others)
            text = format_content(filled,
                                  character_card=(card or "").strip() or "(no character card)",
                                  shards="\n\n".join(block.block_text for block in blocks),
                                  feeling=feeling or "(no feeling recorded)",
                                  new=render_story(self.index.views(new_ids), new_ids, with_id=False)
                                  or "(nothing new happens)").text.strip()
        return StoryRender(text=text, tokens=story_tokens(text), shards=[shard.key for shard in shown],
                           notes=notes, other_characters=others)

    def _text(self, record_ids: Iterable[int]) -> TextRender:
        """Return one rendered text, its size, and its ids."""
        ids = list(dict.fromkeys(record_ids))
        text = render_story(self.index.views(ids), ids, with_id=False) if ids else ""
        return TextRender(text=text, tokens=story_tokens(text), lines=len(text.splitlines()), record_ids=ids)


__all__ = ["ATTENTION_PROMPTS", "CsmItemRender", "LineRender", "RecordRender", "STORY_PROMPTS", "ShardRender",
           "StoryPrompts", "StoryRender", "StoryRenderer", "TextRender", "TickRender"]
