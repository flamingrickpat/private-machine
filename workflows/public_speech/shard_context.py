"""Build the context of one public reply from the workspace shards of its cycle.

Main cognition chose the reply in one cycle. The action request carries the
shards of that cycle: the self shard, the shard of the partner entity, and
the shard of a workflow group. This module turns them into the reply
context. It does not retrieve memory again, so the reply reads the same
conscious content that the decision read.

- The verbatim transcript is the tail of the partner shard (the dialogue
  with that entity) and the private thoughts of the self tail that belong
  to that exchange, in world-time order. A thought belongs to the exchange
  when it is not older than the oldest dialogue line of the transcript.
  At most ``DIALOGUE_THOUGHT_LIMIT`` thoughts are shown, the newest ones.
  The self tail holds 6k tokens of thoughts. Without this limit, old
  thoughts from earlier days crowd out the dialogue and bring back old
  wording into the reply.
- The story text renders each shard with
  :func:`workflows.main_cognition.story_prefix.shard_block`: the same words
  as the story prefix of the cycle. The transcript records are left out of
  it, because the prompt shows them verbatim.
- The evidence lists every shard record that the story text or the
  transcript shows. The speech review and the publication guard read it.

The functions read a frozen memory store view and make no model call.
"""
from __future__ import annotations

from pm.constants import DIALOGUE_THOUGHT_LIMIT
from pm.memory.snapshot import Snapshot
from pm.memory.retrieval import candidate
from pm.records.enums import FeatureType
from pm.records.memory_records import Feature

from workflows.main_cognition.csm_view import record_view
from workflows.main_cognition.phases.model import phase_prompts
from workflows.main_cognition.schemas import ActionRequest, RecordView
from workflows.main_cognition.story_prefix import shard_block
from workflows.public_speech.context_planning import PlannedContext
from workflows.public_speech.public_context import DialogueTurn, EvidenceItem

#: Feature types of the self tail that the Story transcript shows as thoughts.
THOUGHT_TYPES = frozenset({FeatureType.Thought, FeatureType.InnerMonologue})
#: The text between two rendered shard blocks.
BLOCK_SEPARATOR = chr(10) * 2


def _turn(record: Feature) -> DialogueTurn:
    kind = "thought" if record.feature_type in THOUGHT_TYPES else "speech"
    return DialogueTurn(speaker=record.source or "unknown", text=" ".join(record.content.split()),
                        knoxel_id=record.id, timestamp=record.timestamp_world_begin.isoformat(), kind=kind)


def exchange_thoughts(dialogue: list[Feature], thoughts: list[Feature]) -> list[Feature]:
    """Return the thoughts that belong to the exchange of ``dialogue``, oldest first.

    A thought belongs to the exchange when it is not older than the oldest
    line of ``dialogue``. Of these, the newest ``DIALOGUE_THOUGHT_LIMIT``
    remain. Without dialogue, only the newest thoughts remain.
    """
    ordered = sorted(thoughts, key=lambda record: (record.timestamp_world_begin, record.id))
    if dialogue:
        start = min(record.timestamp_world_begin for record in dialogue)
        ordered = [record for record in ordered if record.timestamp_world_begin >= start]
    return ordered[-DIALOGUE_THOUGHT_LIMIT:]


def _views(view: Snapshot, request: ActionRequest, late_ids: list[int] = ()) -> dict[int, RecordView]:
    """Return a record view of every item and tail record of the request shards.

    ``late_ids`` are partner records that arrived after the cycle committed.
    They get a view too, so the transcript can show them.
    """
    return {record_id: record_view(view.get(record_id))
            for item in request.shards
            for record_id in [*item.shard.item_ids(), *item.shard.tail_ids, *late_ids]}


def shard_story(view: Snapshot, request: ActionRequest, companion_name: str) -> str:
    """Render every shard of ``request`` with its items and its full tail, as the cycle saw it.

    Deliberate mental work reads this text. It is the shard part of the story
    prefix of the cycle that chose the action.
    """
    records = _views(view, request)
    prompts = phase_prompts("story")
    blocks = [shard_block(prompts, item.shard.key, item.name, companion_name, records,
                          item.shard.item_ids(), item.shard.tail_ids) for item in request.shards]
    return BLOCK_SEPARATOR.join(blocks)


def shard_context(view: Snapshot, request: ActionRequest, companion_name: str,
                  late_ids: list[int] = ()) -> PlannedContext:
    """Return the reply context of ``request`` from its shards.

    ``view`` is a frozen memory store view that holds every shard record.
    The shards hold committed records only, so a missing record is a bug
    and raises KeyError. The current stimuli are not in any tail.
    Empty history remains in evidence but supplies no transcript turn.
    This function does not change accepted records or call a model.

    ``late_ids`` are partner dialogue records that arrived after the cycle
    chose the action. The caller reads them from the live store, because the
    frozen shard tail stops at that cycle. Without them the reply writer
    never sees the newest line, and it answers an older one. A late record
    joins the transcript and the evidence of the partner shard. It does not
    join the shard story text, which stays the words of the deciding cycle.
    """
    records = _views(view, request, late_ids)
    dialogue: list[Feature] = []
    all_thoughts: list[Feature] = []
    for item in request.shards:
        for record_id in item.shard.tail_ids:
            record = view.get(record_id)
            if item.shard.key.kind == "entity" and record.feature_type == FeatureType.Dialogue:
                dialogue.append(record)
            elif item.shard.key.kind == "self" and record.feature_type in THOUGHT_TYPES:
                all_thoughts.append(record)
    # A late line belongs to the partner shard, never to the self shard.
    partner_shard = next((item for item in request.shards if item.shard.key.kind == "entity"), None)
    late: list[Feature] = []
    if partner_shard is not None:
        late = [view.get(record_id) for record_id in late_ids
                if view.get(record_id).feature_type == FeatureType.Dialogue]
        dialogue += late
    # Empty records can exist in imported or previously accepted history.
    # Filter before exchange selection so they cannot displace useful thoughts.
    empty = {record.id for record in [*dialogue, *all_thoughts] if not record.content.strip()}
    spoken = [record for record in dialogue if record.id not in empty]
    thoughts = exchange_thoughts(spoken, [record for record in all_thoughts if record.id not in empty])
    transcript = [*spoken, *thoughts]
    # The story text leaves out every verbatim line. It also leaves out the
    # thoughts that the transcript drops, so they do not come back there.
    verbatim = {record.id for record in [*dialogue, *all_thoughts]}
    dropped = {record.id for record in all_thoughts} - {record.id for record in thoughts} - empty
    shown_tail = {index: [record_id for record_id in item.shard.tail_ids if record_id not in verbatim]
                  for index, item in enumerate(request.shards)}
    moved = {index: len(shown_tail[index]) < len(item.shard.tail_ids) for index, item in enumerate(request.shards)}
    prompts = phase_prompts("story")
    blocks = [shard_block(prompts, item.shard.key, item.name, companion_name, records,
                          item.shard.item_ids(), shown_tail[index], verbatim_tail=moved[index])
              for index, item in enumerate(request.shards)]
    transcript.sort(key=lambda record: (record.timestamp_world_begin, record.id))
    evidence = []
    for item in request.shards:
        ids = [*item.shard.item_ids(), *item.shard.tail_ids]
        if item is partner_shard:
            ids += [record.id for record in late]
        for record_id in dict.fromkeys(ids):
            if record_id in dropped:
                continue
            # A shard retains the exact version selected by cognition. Its
            # successor can already be the current query head.
            entry = EvidenceItem.model_validate(candidate(view, record_id, view.get(record_id)))
            entry.lane = f"shard:{item.name}"
            evidence.append(entry)
    trace = {"source": "shards",
             "shards": [{"name": item.name, "key": item.shard.key.model_dump(mode="json"),
                         "items": len(item.shard.items), "tail": len(item.shard.tail_ids)}
                        for item in request.shards],
             "transcript": len(transcript), "empty_transcript_ids": sorted(empty),
             "late_ids": sorted(record.id for record in late)}
    return PlannedContext(dialogue=[_turn(record) for record in transcript],
                          evidence=evidence, story=BLOCK_SEPARATOR.join(blocks), trace=trace)


__all__ = ["THOUGHT_TYPES", "exchange_thoughts", "shard_context", "shard_story"]
