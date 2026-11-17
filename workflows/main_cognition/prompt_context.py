"""Select shard records through the complete-prompt context renderer.

StoryPrefix owns the story frame. This adapter adds record priorities without
changing workspace state. The common fitter selects whole annotated lines.
Entity context, dialogue, leading narratives, and private thoughts supply
importance signals. Selection mixes those signals with recency and randomness.
Four newest partner lines remain required.
"""
from pm.context_render.budget import context_recency, optional_text
from datetime import datetime
from pm.context_render.story import RecordView, shard_titles, story_line
from pm.context_render.turns import Content, ContextTag, format_content
from pm.records.workspace import WorkspaceShard
from workflows.main_cognition.model_json import PhasePrompts


def shard_content(shard: WorkspaceShard, records: dict[int, RecordView], prompts: PhasePrompts,
                  name: str, companion: str, tail_ids: list[int]) -> Content:
    """Render a shard with stable record priorities and original chronology."""
    required_ids: set[int] = set(tail_ids[-4:]) if shard.key.kind == "entity" else set()
    ranked_items = sorted(shard.items, key=lambda item: item.activation_at_entry)
    maximum_activation = max((item.activation_at_entry for item in ranked_items), default=0)
    narratives = [item.record_id for item in reversed(ranked_items)
                  if records[item.record_id].record_class == "Narrative"][:3]
    priorities: dict[int, float] = {}
    for item in ranked_items:
        if item.required_by == "entity_context":
            tier = 5
        elif item.record_id in narratives:
            tier = 3
        else:
            tier = 1
        activation = item.activation_at_entry / maximum_activation if maximum_activation else 0
        priorities[item.record_id] = tier + .9 * activation
    for record_id in tail_ids:
        tier = 4 if shard.key.kind == "entity" else 2
        priorities[record_id] = max(priorities.get(record_id, 0), float(tier))

    def lines(record_ids: list[int], empty: str) -> Content:
        views = [records[record_id] for record_id in dict.fromkeys(record_ids)]
        timed = sorted((view for view in views if view.timestamp is not None), key=lambda view: view.timestamp or datetime.min)
        untimed = [view for view in views if view.timestamp is None]
        result = Content("")
        for position, view in enumerate([*timed, *untimed]):
            tag = ContextTag(f"shard:{shard.key.kind}:{shard.key.ref}:{view.record_id}", priorities[view.record_id],
                             record_id=view.record_id, required=view.record_id in required_ids,
                             recency=context_recency(view.timestamp), reducible=True)
            result += optional_text(("\n" if position else "") + story_line(view, with_id=False), tag)
        return result if views else Content(empty)

    title, tail_label = shard_titles(shard.key, name, companion, prompts.shard_labels)
    return format_content(prompts.shard_template, title=title, tail_label=tail_label,
                          items=lines(shard.item_ids(), prompts.shard_labels.empty_items),
                          tail=lines(tail_ids, prompts.shard_labels.empty_tail)).strip()
