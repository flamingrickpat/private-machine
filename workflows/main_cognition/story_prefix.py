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
provider can reuse its cache. The codelets, structures, and attention phases
read the workspace before attention changes it. The goals, experience, and
decide phases build the prefix again and read the updated shards.

The prompt texts are in ``phases/prompts/story.yaml``. The prefix holds no
record ids and no embeddings.
"""
from __future__ import annotations

from pm.context_render.turns import Role, Turn, format_content
from pm.records.workspace import SELF_SHARD, ShardKey, WorkspaceShard

from workflows.main_cognition.codelet_registry import fill
from workflows.main_cognition.csm_view import render_story
from workflows.main_cognition.cycle_work import CycleRequest, CycleWork
from workflows.main_cognition.model_json import PhasePrompts
from workflows.main_cognition.phases.model import phase_prompts


def shard_label(key: ShardKey, request: CycleRequest) -> str:
    """Return the story name of a shard: ``self``, the user name, an entity label, or the workflow name.

    Step 8 of A19F gives every entity its name. Until then an entity other
    than the user is ``entity <id>``.
    """
    if key.kind == "self":
        return "self"
    if key.kind == "entity":
        return request.user_name if key.ref == str(request.user_entity_id) else f"entity {key.ref}"
    return key.ref


def shown_shards(work: CycleWork) -> list[WorkspaceShard]:
    """Return the shards that the prefix shows: the self shard, then the shard of each current group.

    The group shards keep the order of the stimulus groups. A self group
    adds no second self shard.
    """
    workspace = work.state.workspace
    keys = list(dict.fromkeys(group.key for group in work.state.stimulus_groups if group.key != SELF_SHARD))
    return [workspace.self_shard, *(workspace.shard(key) for key in keys)]


def conscious_percepts(work: CycleWork) -> list[str]:
    """Return the story lines of the codelet percepts in the shown shards, in world-time order.

    The goals and decide phases list them as the companion's own conscious
    impulses. They are suggestions, not commands.
    """
    ids = [item.record_id for shard in shown_shards(work) for item in shard.items
           if work.records[item.record_id].record_class == "CodeletPercept"]
    text = render_story(work.records, ids, with_id=False)
    return text.splitlines() if text else []


class StoryPrefix:
    """The three shared turns of one cycle and the texts that build them.

    ``turns`` are the system prompt, the story, and the acknowledgement.
    ``story_text`` is the same content as plain text, for agent sessions.
    ``targets`` maps each shard label that a call may name to its shard key:
    ``self`` and one label for each stimulus group.
    """

    def __init__(self, work: CycleWork, prompts: PhasePrompts | None = None):
        request = work.request
        self.prompts = prompts or phase_prompts("story")
        self.companion = request.companion_name
        # Step 8 of A19F gives every group its own entity. Until then the
        # other character of every story is the user.
        self.entity = request.user_name
        labels = self.prompts.shard_labels
        blocks = []
        for shard in shown_shards(work):
            name = shard_label(shard.key, request)
            if shard.key.kind == "self":
                title, tail_label = labels.self_title, labels.self_tail
            elif shard.key.kind == "entity":
                title, tail_label = labels.entity_title, labels.entity_tail
            else:
                title, tail_label = labels.workflow_title, labels.workflow_tail
            values = {"companion_name": self.companion, "entity_name": name, "workflow_name": name}
            blocks.append(format_content(self.prompts.shard_template,
                title=title.format(**values), tail_label=tail_label.format(**values),
                items=render_story(work.records, shard.item_ids(), with_id=False) or labels.empty_items,
                tail=render_story(work.records, shard.tail_ids, with_id=False) or labels.empty_tail,
            ).text.strip())
        system = self.fill(self.prompts.system_prompt).strip()
        story = format_content(self.fill(self.prompts.story_template),
            character_card=request.character_card.strip() or "(no character card)",
            shards="\n\n".join(blocks),
            feeling=work.affect.felt or work.state_summary or "(no feeling recorded)",
            new=render_story(work.records, [item.knoxel_id for item in request.stimuli], with_id=False)
                or "(nothing new happens)",
        ).text.strip()
        self.turns = [Turn(Role.System, system), Turn(Role.User, story),
                      Turn(Role.Assistant, self.fill(self.prompts.story_ack).strip())]
        self.story_text = f"{system}\n\n{story}"
        self.targets: dict[str, ShardKey] = {"self": SELF_SHARD}
        for group in work.state.stimulus_groups:
            if group.key != SELF_SHARD:
                self.targets[shard_label(group.key, request)] = group.key

    def fill(self, text: str) -> str:
        """Replace the name placeholders ``{companion_name}`` and ``{entity_name}``."""
        return fill(text, self.companion, self.entity)

    def ask(self, instruction: str) -> list[Turn]:
        """Return the prefix and then ``instruction`` as the last user turn."""
        return [*self.turns, Turn(Role.User, instruction.strip())]

    def label(self, key: ShardKey) -> str:
        """Return the target label of ``key``. A key that is not a target raises StopIteration."""
        return next(label for label, target in self.targets.items() if target == key)


__all__ = ["StoryPrefix", "conscious_percepts", "shard_label", "shown_shards"]
