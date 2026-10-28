"""Experience phase: the subjective experience of this moment.

Port of ``_generate_subjective_experience`` from
``private-machine-head/pm/ghosts/ghost_lida.py`` (commit 7101933), with its
seed phrases (``_generate_state_seed_phrases``).

1. :func:`seed_phrases` turns the accepted mental state into short
   first-person phrases: the cognitive style, the dominant feeling, and the
   most pressing need. The phrase table is data in
   ``phases/prompts/experience.yaml``.
2. One story-framed call continues them. The call starts with the shared
   story prefix, then the instruction, then an assistant turn that opens
   with the seed phrases. So the model writes on in the companion's felt
   inner voice. The call is an unconscious story simulation; the experience
   speaks to no one.
3. The experience text is the seed phrases, the continuation, and the focus
   phrase for a message from an entity.
4. The text becomes a pending ``SubjectiveExperience`` record. It enters
   the tail of the self shard, so the goals, decide, and route calls read
   it in the shared prefix. The commit stores the record, links it from the
   broadcast and the turn record, and makes it the cycle summary.
5. When more than one stimulus group is significant, one more call runs for
   each of them. Each reads the prefix of its own group: the self shard and
   the group's shard only. Its text is the experience that the decision and
   the reply of that group read. So a reply to one entity does not mention
   the content of another group. The calls run in parallel.
"""
from __future__ import annotations

from pm_workflows.python_role import RoleContext
from pydantic import BaseModel, ConfigDict, Field

from pm.constants import WORKSPACE_TAIL_TOKENS
from pm.context_render.turns import Role, Turn, format_content
from pm.records.base import persisted_type_name
from pm.records.enums import FeatureType
from pm.records.memory_records import Feature
from pm.records.mental_state import FullMentalState
from pm.llm.concurrency import ordered_parallel_map
from pm.records.workspace import SELF_SHARD, ShardKey

from workflows.main_cognition.csm_view import record_tokens, render_story
from workflows.main_cognition.cycle_work import CycleWork
from workflows.main_cognition.model_json import load_yaml
from workflows.main_cognition.phases.model import PROMPTS_DIR, complete_prose
from workflows.inference_settings import DECISION_WORKERS
from workflows.main_cognition.schemas import ExperienceResult, GroupExperience, RecordView
from workflows.main_cognition.story_prefix import StoryPrefix, shard_label
from workflows.main_cognition.workspace_selection import select_tail

#: A tier is ``(threshold, word)``. The first tier whose threshold the intensity reaches gives the word.
Tier = tuple[float, str]


class SeedSide(BaseModel):
    """One side of a cognitive axis: the sentence template and its intensity tiers."""

    model_config = ConfigDict(extra="forbid")

    template: str = ""
    tiers: list[Tier] = Field(default_factory=list)


class CognitionSeed(BaseModel):
    """The seed phrases of one ``StateCognition`` axis.

    ``high`` reads the axis value. ``low`` reads ``low_origin - value``.
    A side that is None gives no phrase.
    """

    model_config = ConfigDict(extra="forbid")

    axis: str = ""
    low_origin: float = 0.0
    high: SeedSide | None = None
    low: SeedSide | None = None


class BipolarSeed(BaseModel):
    """A bipolar feeling axis: ``path`` is ``<state part>.<axis>``, with a name for each pole."""

    model_config = ConfigDict(extra="forbid")

    path: str = ""
    positive: str = ""
    negative: str = ""


class UnipolarSeed(BaseModel):
    """A unipolar feeling axis: ``path`` is ``<state part>.<axis>``, with its name in the phrase."""

    model_config = ConfigDict(extra="forbid")

    path: str = ""
    name: str = ""


class SeedTable(BaseModel):
    """The seed phrase table of ``experience.yaml``. See the comments in that file."""

    model_config = ConfigDict(extra="forbid")

    cognition: list[CognitionSeed] = Field(default_factory=list)
    emotion_threshold: float = 0.4
    bipolar: list[BipolarSeed] = Field(default_factory=list)
    unipolar: list[UnipolarSeed] = Field(default_factory=list)
    fear: UnipolarSeed = Field(default_factory=UnipolarSeed)
    emotion_tiers: list[Tier] = Field(default_factory=list)
    emotion_template: str = ""
    emotion_adverbs: dict[str, str] = Field(default_factory=dict)
    fear_template: str = ""
    drives: list[str] = Field(default_factory=list)
    need_tiers: list[Tier] = Field(default_factory=list)
    need_template: str = ""


class ExperiencePrompts(BaseModel):
    """The prompt file of the experience phase."""

    model_config = ConfigDict(extra="forbid")

    user_template: str = ""
    seed_tail: str = ""
    focus_phrase: str = ""
    seed_phrases: SeedTable = Field(default_factory=SeedTable)


def load_experience_prompts() -> ExperiencePrompts:
    """Read and validate ``phases/prompts/experience.yaml``. An invalid file raises."""
    return ExperiencePrompts.model_validate(load_yaml(PROMPTS_DIR / "experience.yaml"))


def tier_word(value: float, tiers: list[Tier]) -> str | None:
    """Return the word of the first tier that ``value`` reaches, or None."""
    return next((word for threshold, word in tiers if value >= threshold), None)


def axis_value(state: FullMentalState, path: str) -> float | None:
    """Return the value at ``<state part>.<axis>``, or None when the part is absent.

    Only ``state_relationship`` can be absent: a cycle without a partner
    has no relationship state.
    """
    part, axis = path.split(".")
    container = getattr(state, part)
    return None if container is None else float(getattr(container, axis))


def seed_phrases(state: FullMentalState, table: SeedTable) -> list[str]:
    """Return the seed phrases of ``state``. Port of ``_generate_state_seed_phrases``.

    1. Cognitive style: one phrase for each cognitive axis whose value
       reaches a tier on its high side, else on its low side.
    2. The dominant feeling: the strongest feeling above
       ``emotion_threshold``, with a gradual intensity word.
    3. The most pressing need, when its pressure reaches a need tier.
    """
    phrases: list[str] = []
    for seed in table.cognition:
        value = float(getattr(state.state_cognition, seed.axis))
        for side, intensity in ((seed.high, value), (seed.low, seed.low_origin - value)):
            word = tier_word(intensity, side.tiers) if side is not None else None
            if word is not None:
                phrases.append(side.template.format(word=word))
                break

    candidates: list[tuple[float, str, bool]] = []
    for seed in table.bipolar:
        value = axis_value(state, seed.path)
        if value is not None and abs(value) > table.emotion_threshold:
            candidates.append((abs(value), seed.positive if value > 0 else seed.negative, False))
    for seed in [*table.unipolar, table.fear]:
        value = axis_value(state, seed.path)
        if value > table.emotion_threshold:
            candidates.append((value, seed.name, seed is table.fear))
    if candidates:
        value, name, fear = max(candidates, key=lambda item: item[0])
        intensity = tier_word(value, table.emotion_tiers) or "a"
        if fear:
            phrases.append(table.fear_template.format(word=intensity.capitalize(), name=name))
        else:
            phrases.append(table.emotion_template.format(name=name, adverb=table.emotion_adverbs[intensity]))

    needs = state.state_needs.model_dump()
    pressures = [(value if name in table.drives else 1.0 - value, name) for name, value in needs.items()]
    pressure, name = max(pressures)
    word = tier_word(pressure, table.need_tiers)
    if word is not None:
        phrases.append(table.need_template.format(word=word, name=name.replace("_", " ")))
    return phrases


def seed_text(phrases: list[str], prompts: ExperiencePrompts) -> str:
    """Return the assistant prefill: the phrases and the seed tail, or empty text without phrases."""
    return f"{' '.join(phrases)}{prompts.seed_tail}" if phrases else ""


def focus_phrase(work: CycleWork, prompts: ExperiencePrompts, group: ShardKey | None = None) -> str:
    """Return the focus phrase for the newest message of an entity group, or empty text.

    The old code added it for a user message. The port drops its reply
    instruction ("make a concise, relevant reply"), because the experience
    does not plan the reply. With ``group``, only that group counts.
    """
    groups = [item for item in work.state.stimulus_groups
              if item.key.kind == "entity" and (group is None or item.key == group)]
    if not groups:
        return ""
    last = groups[-1]
    text = next(item.text for item in work.request.stimuli if item.knoxel_id == last.stimulus_ids[-1])
    return prompts.focus_phrase.format(entity_name=shard_label(last.key, work.request), message=text)


def render_entered(work: CycleWork, group: ShardKey | None = None) -> str:
    """Return the story lines of the items that entered a shard in this cycle, in world-time order.

    With ``group``, only the self shard and the shard of that group count.
    """
    entered = [item.record_id for change in work.attention.changes
               if group is None or change.key in (SELF_SHARD, group) for item in change.entered]
    return render_story(work.records, entered, with_id=False) or "(nothing new)"


def build_messages(work: CycleWork, prompts: ExperiencePrompts, prefix: StoryPrefix, seed: str,
                   group: ShardKey | None = None) -> list[Turn]:
    """Return the story prefix, the instruction, and the seed as an open assistant turn."""
    instruction = format_content(prefix.fill(prompts.user_template), entered=render_entered(work, group)).text
    turns = prefix.ask(instruction)
    return [*turns, Turn(Role.Assistant, seed)] if seed else turns


def continuation(raw: str, seed: str) -> str:
    """Return the model text after the seed, ready to append to the seed.

    A model that echoes the seed gets it cut off. The provider strips the
    leading space of a continuation, so a word that follows the seed
    directly gets one space in front.
    """
    stripped = seed.strip()
    text = raw[len(stripped):] if stripped and raw.startswith(stripped) else raw
    if seed and text and not seed[-1].isspace() and not text[0].isspace() and text[0] not in ".,;:!?'\"":
        return f" {text}"
    return text


def enter_self_tail(work: CycleWork, record_id: int) -> None:
    """Append the experience record to the self shard tail and to the self shard change.

    The tail keeps its token budget, so the oldest tail records can leave.
    """
    shard = work.state.workspace.self_shard
    exclude = {item.knoxel_id for item in work.request.stimuli}
    shard.tail_ids = select_tail([*shard.tail_ids, record_id], lambda key: record_tokens(work.records, key),
                                 WORKSPACE_TAIL_TOKENS, exclude)
    change = next(change for change in work.attention.changes if change.key == SELF_SHARD)
    change.tail_ids = list(shard.tail_ids)


def _record(work: CycleWork, text: str) -> int:
    """Add a pending ``SubjectiveExperience`` view to the cycle records and return its id."""
    request = work.request
    record_id = work.allocate_pending_id()
    work.records[record_id] = RecordView(
        record_id=record_id, kind=persisted_type_name(Feature),
        source=f"{request.companion_name} ({FeatureType.SubjectiveExperience.value})", content=text,
        timestamp=request.cycle_time, record_class="Feature", feature_type=FeatureType.SubjectiveExperience.value)
    return record_id


def experience_groups(work: CycleWork) -> list[ShardKey]:
    """Return the groups that get an experience of their own: none, or every significant group when there are several."""
    keys = [group.key for group in work.state.stimulus_groups if not group.insignificant]
    return keys if len(keys) > 1 else []


def run(context: RoleContext, work: CycleWork) -> CycleWork:
    prompts = load_experience_prompts()
    seed = seed_text(seed_phrases(work.state.mental_state, prompts.seed_phrases), prompts)
    prefix = StoryPrefix(work)
    raw = complete_prose(context, build_messages(work, prompts, prefix, seed))
    text = f"{seed}{continuation(raw, seed)}{focus_phrase(work, prompts)}".strip()
    record_id = _record(work, text)
    enter_self_tail(work, record_id)
    work.experience = ExperienceResult(text=text, seed=seed, record_id=record_id)

    keys = experience_groups(work)
    # The group entries exist before the calls, so the prefix of each group
    # hides the whole-moment experience (``hidden_for``). The group text then
    # does not repeat the content of the other groups.
    work.experience.groups = [GroupExperience(key=key, seed=seed) for key in keys]
    prefixes = [StoryPrefix(work, group=key) for key in keys]
    texts = ordered_parallel_map(
        lambda index: complete_prose(context, build_messages(work, prompts, prefixes[index], seed, keys[index])),
        list(range(len(keys))), workers=DECISION_WORKERS, cancel=context.runtime.stop_event)
    for item, raw_group in zip(work.experience.groups, texts, strict=True):
        item.text = f"{seed}{continuation(raw_group, seed)}{focus_phrase(work, prompts, item.key)}".strip()
        item.record_id = _record(work, item.text)
        enter_self_tail(work, item.record_id)
    work.receipt("experience", f"experience of {len(text)} characters, {len(seed)} seed characters; "
                               f"{len(keys)} group experiences")
    return work


__all__ = [
    "BipolarSeed",
    "CognitionSeed",
    "ExperiencePrompts",
    "SeedSide",
    "SeedTable",
    "UnipolarSeed",
    "axis_value",
    "build_messages",
    "continuation",
    "enter_self_tail",
    "experience_groups",
    "focus_phrase",
    "load_experience_prompts",
    "render_entered",
    "run",
    "seed_phrases",
    "seed_text",
    "tier_word",
]
