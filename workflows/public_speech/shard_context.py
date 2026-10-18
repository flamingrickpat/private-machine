"""Build the context of one public reply from the workspace shards of its cycle.

Main cognition chose the reply in one cycle. The action request carries the
shards of that cycle: the self shard, the shard of the partner entity, and
the shard of a workflow group. This module turns them into the reply
context. It does not retrieve memory again, so the reply reads the same
conscious content that the decision read.

- The verbatim transcript is the tail of the partner shard (the dialogue
  with that entity) and the companion's private thoughts in the self tail,
  in world-time order.
- The story text renders each shard with
  :func:`workflows.main_cognition.story_prefix.shard_block`: the same words
  as the story prefix of the cycle. The transcript records are left out of
  it, because the prompt shows them verbatim.
- The evidence lists every shard record that the story text or the
  transcript shows. The speech review and the publication guard read it.

The functions read a frozen memory store view and make no model call.
"""
from __future__ import annotations

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


def _views(view, request: ActionRequest) -> dict[int, RecordView]:
    """Return a record view of every item and tail record of the request shards."""
    return {record_id: record_view(view.get(record_id))
            for item in request.shards for record_id in [*item.shard.item_ids(), *item.shard.tail_ids]}


def shard_story(view, request: ActionRequest, companion_name: str) -> str:
    """Render every shard of ``request`` with its items and its full tail, as the cycle saw it.

    Deliberate mental work reads this text. It is the shard part of the story
    prefix of the cycle that chose the action.
    """
    records = _views(view, request)
    prompts = phase_prompts("story")
    blocks = [shard_block(prompts, item.shard.key, item.name, companion_name, records,
                          item.shard.item_ids(), item.shard.tail_ids) for item in request.shards]
    return BLOCK_SEPARATOR.join(blocks)


def shard_context(view, request: ActionRequest, companion_name: str) -> PlannedContext:
    """Return the reply context of ``request`` from its shards.

    ``view`` is a frozen memory store view that holds every shard record.
    The shards hold committed records only, so a missing record is a bug
    and raises KeyError. The current stimuli are not in any tail.
    """
    records = _views(view, request)
    transcript: list[Feature] = []
    shown_tail: dict[int, list[int]] = {}
    moved: dict[int, bool] = {}
    for index, item in enumerate(request.shards):
        shard = item.shard
        tail = []
        for record_id in shard.tail_ids:
            record = view.get(record_id)
            verbatim = (record.feature_type == FeatureType.Dialogue if shard.key.kind == "entity"
                        else shard.key.kind == "self" and record.feature_type in THOUGHT_TYPES)
            if verbatim:
                transcript.append(record)
            else:
                tail.append(record_id)
        shown_tail[index] = tail
        moved[index] = len(tail) < len(shard.tail_ids)
    prompts = phase_prompts("story")
    blocks = [shard_block(prompts, item.shard.key, item.name, companion_name, records,
                          item.shard.item_ids(), shown_tail[index], verbatim_tail=moved[index])
              for index, item in enumerate(request.shards)]
    transcript.sort(key=lambda record: (record.timestamp_world_begin, record.id))
    serialized = view.serialize_knoxels()
    evidence = []
    for item in request.shards:
        for record_id in dict.fromkeys([*item.shard.item_ids(), *item.shard.tail_ids]):
            entry = EvidenceItem.model_validate(candidate(view, record_id, serialized[record_id]))
            entry.lane = f"shard:{item.name}"
            evidence.append(entry)
    trace = {"source": "shards",
             "shards": [{"name": item.name, "key": item.shard.key.model_dump(mode="json"),
                         "items": len(item.shard.items), "tail": len(item.shard.tail_ids)}
                        for item in request.shards],
             "transcript": len(transcript)}
    return PlannedContext(dialogue=[_turn(record) for record in transcript],
                          evidence=evidence, story=BLOCK_SEPARATOR.join(blocks), trace=trace)


__all__ = ["THOUGHT_TYPES", "shard_context", "shard_story"]
