"""The shared story prefix of every story-framed call of one cycle.

Every story-framed call starts with the same three turns (plan section
"Prompt frame"):

1. The shared system prompt: an expert story writer who simulates the mind
   of a character.
2. The story: the character card, then the self shard, then the shard of
   each current stimulus group. Each shard shows its items and then its tail,
   each in world-time order. Then the feeling in words and the new stimuli.
3. The acknowledgement of the story.

The instruction of the call comes last. The prefix reads only the cycle
state, so all calls that read the same workspace get the same bytes and the
provider can reuse its cache. The triage, appraise, retrieve, codelets,
structures, and attention phases read the workspace before attention changes
it. The experience, goals, decide, and route phases build the prefix again
and read the updated shards. The experience phase adds the subjective
experience to the self tail, so the later phases read it too.

The prompt texts are in ``phases/prompts/story.yaml``. The prefix holds no
record ids and no embeddings.
"""
from __future__ import annotations

from pm.context_render.story import (
    RecordView, shard_block_text, shard_name, story_entity_names, story_shards,
)
from pm.context_render.turns import Role, Turn, format_content
from pm.records.workspace import SELF_SHARD, ShardKey, WorkspaceShard

from workflows.main_cognition.codelet_registry import fill
from workflows.main_cognition.csm_view import render_story
from workflows.main_cognition.cycle_work import CycleRequest, CycleWork
from workflows.main_cognition.model_json import PhasePrompts
from workflows.main_cognition.phases.model import phase_prompts


def shard_label(key: ShardKey, request: CycleRequest) -> str:
    """Return the story name of a shard: ``self``, the entity name, or the workflow name.

    The host supplies the names of the group entities in
    ``request.entity_names``. The user is named by ``request.user_name``.
    An entity without a known name is ``entity <id>``. The naming itself
    lives in :func:`pm.context_render.story.shard_name`, which the memory
    inspector calls with the names of the checkpoint.
    """
    return shard_name(key, user_entity_id=request.user_entity_id, user_name=request.user_name,
                      entity_names=request.entity_names)


def resolve_target(value: str, targets: dict[str, ShardKey]) -> tuple[str, str]:
    """Return the shard label that ``value`` names, and the word it corrected.

    A model names a shard in its own words. It can write the partner name
    when that partner has no shard in this cycle. An exact label passes
    through. A label that differs only in case or spacing is corrected. Any
    other word becomes ``self``, because the self shard always exists and a
    percept of an absent partner still belongs there.

    The second value is the original word, or an empty string when no
    correction happened. A correction costs nothing. A rejected answer costs
    one full repair round with the whole story prefix again, and a real
    session burned about ninety of those in one hour for this field alone.
    """
    text = str(value or "").strip()
    if text in targets:
        return text, ""
    for label in targets:
        if label.casefold() == text.casefold():
            return label, text
    return "self", text


def story_entities(work: CycleWork) -> str:
    """Return the names of the other characters of this moment, for ``{entity_name}``.

    These are the entities of the current stimulus groups, in group order.
    Without an entity group, the other character is the user. The naming is
    :func:`pm.context_render.story.story_entity_names`.
    """
    return story_entity_names([group.key for group in work.state.stimulus_groups],
                              user_entity_id=work.request.user_entity_id, user_name=work.request.user_name,
                              entity_names=work.request.entity_names)


def shown_shards(work: CycleWork, group: ShardKey | None = None) -> list[WorkspaceShard]:
    """Return the shards that the prefix shows: the self shard, then the shard of each current group.

    The group shards keep the order of the stimulus groups. A self group
    adds no second self shard. With ``group``, only the self shard and the
    shard of that group remain: the calls for one group read only their own
    part of the story. The selection is
    :func:`pm.context_render.story.story_shards`, which the inspector shares.
    """
    return story_shards(work.state.workspace, [item.key for item in work.state.stimulus_groups], group=group)


def conscious_percepts(work: CycleWork) -> list[str]:
    """Return the story lines of the codelet percepts in the shown shards, in world-time order.

    The goals phase lists them as the companion's own conscious impulses.
    They are suggestions, not commands. The decide phase uses
    ``group_percepts`` of the decide module for one group.
    """
    ids = [item.record_id for shard in shown_shards(work) for item in shard.items
           if work.records[item.record_id].record_class == "CodeletPercept"]
    text = render_story(work.records, ids, with_id=False)
    return text.splitlines() if text else []


def shard_block(prompts: PhasePrompts, key: ShardKey, name: str, companion: str,
                records: dict[int, RecordView], item_ids: list[int], tail_ids: list[int],
                *, verbatim_tail: bool = False) -> str:
    """Render one shard as a story block: its title, its items, and its tail.

    ``name`` is the shard label (see :func:`shard_label`). This adapter only
    supplies the shard template and the shard labels of the story prompts.
    The layout is :func:`pm.context_render.story.shard_block_text`.

    The public dialogue workflows call this function too, so the reply reads
    the shard in the same words as the cycle that chose it. A reply shows the
    dialogue of a tail verbatim in its transcript. With ``verbatim_tail``, an
    empty ``tail_ids`` then says so, instead of saying that nothing happened
    yet.
    """
    return shard_block_text(key, name, companion, records, item_ids, tail_ids,
                            template=prompts.shard_template, labels=prompts.shard_labels,
                            verbatim_tail=verbatim_tail)


class StoryPrefix:
    """The three shared turns of one cycle and the texts that build them.

    ``turns`` are the system prompt, the story, and the acknowledgement.
    ``story_text`` is the same content as plain text, for agent sessions.
    ``targets`` maps each shard label that a call may name to its shard key:
    ``self`` and one label for each stimulus group.
    """

    def __init__(self, work: CycleWork, prompts: PhasePrompts | None = None,
                 group: ShardKey | None = None) -> None:
        """Build the prefix of the whole moment, or with ``group`` the prefix of one stimulus group.

        The prefix of one group shows the self shard and that group's shard,
        the group's stimuli as new events, and the group's entity as the
        other character. The self tail leaves out the experiences of this
        cycle that belong to the other groups (``ExperienceResult.hidden_for``).
        Without ``group``, every call of the cycle gets the same bytes.
        """
        request = work.request
        self.prompts = prompts or phase_prompts("story")
        self.companion = request.companion_name
        groups = [item for item in work.state.stimulus_groups if group is None or item.key == group]
        self.entity = story_entities(work) if group is None else (
            shard_label(group, request) if group.kind == "entity" else request.user_name)
        hidden = set(work.experience.hidden_for(group))
        blocks = [shard_block(self.prompts, shard.key, shard_label(shard.key, request), self.companion,
                              work.records, shard.item_ids(),
                              [record_id for record_id in shard.tail_ids if record_id not in hidden])
                  for shard in shown_shards(work, group)]
        new_ids = [record_id for item in groups for record_id in item.stimulus_ids] if group is not None else [
            item.knoxel_id for item in request.stimuli]
        system = self.fill(self.prompts.system_prompt).strip()
        story = format_content(self.fill(self.prompts.story_template),
            character_card=request.character_card.strip() or "(no character card)",
            shards="\n\n".join(blocks),
            feeling=work.affect.felt or work.state_summary or "(no feeling recorded)",
            new=render_story(work.records, new_ids, with_id=False) or "(nothing new happens)",
        ).text.strip()
        self.turns = [Turn(Role.System, system), Turn(Role.User, story),
                      Turn(Role.Assistant, self.fill(self.prompts.story_ack).strip())]
        self.story_text = f"{system}\n\n{story}"
        self.targets: dict[str, ShardKey] = {"self": SELF_SHARD}
        for item in groups:
            if item.key != SELF_SHARD:
                self.targets[shard_label(item.key, request)] = item.key

    def fill(self, text: str) -> str:
        """Replace the name placeholders ``{companion_name}`` and ``{entity_name}``."""
        return fill(text, self.companion, self.entity)

    def ask(self, instruction: str) -> list[Turn]:
        """Return the prefix and then ``instruction`` as the last user turn."""
        return [*self.turns, Turn(Role.User, instruction.strip())]

    def label(self, key: ShardKey) -> str:
        """Return the target label of ``key``. A key that is not a target raises StopIteration."""
        return next(label for label, target in self.targets.items() if target == key)


__all__ = ["StoryPrefix", "conscious_percepts", "resolve_target", "shard_block", "shard_label", "shown_shards",
          "story_entities"]
