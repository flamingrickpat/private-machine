"""Freeze and render the self workspace and selected partner or workflow shards."""
from __future__ import annotations
from dataclasses import dataclass, field, replace
from pm.context_render.budget import TokenEstimator, evidence_excerpt
from pm.context_render.images import resolve_media, ImageCatalog
from pm.context_render.prompt import EvidenceDecision, RenderSettings, RenderedPrompt, require_rendered_prompt
from pm.context_render.turns import Content, Role, TemplateValues, Turn, format_content
from pm.records.media import RecordImage
from pm.context_render.story import RecordView, ShardLabels, shard_block_text, shard_name, story_shards
from pm.records.workspace import GlobalWorkspace, ShardKey


@dataclass(frozen=True, slots=True)
class ShardRenderInput:
    key: ShardKey
    name: str
    item_ids: tuple[int, ...]
    tail_ids: tuple[int, ...]
    activation: dict[int, float]  # pm-lint: allow PM004 -- keyed by entity or knoxel identity
    entity_context_ids: frozenset[int]


def select_shards(workspace: GlobalWorkspace, keys: list[ShardKey], *, hidden_tail: set[int],
                  user_id: int, user_name: str, entity_names: dict[int, str]) -> tuple[ShardRenderInput, ...]:  # pm-lint: allow PM004 -- keyed by entity or knoxel identity
    selected: list[ShardRenderInput] = []
    for shard in story_shards(workspace, keys):
        name = shard_name(shard.key, user_entity_id=user_id, user_name=user_name, entity_names=entity_names)
        weights = {item.record_id: item.activation_at_entry for item in shard.items}
        selected.append(ShardRenderInput(shard.key, name, tuple(shard.item_ids()),
            tuple(key for key in shard.tail_ids if key not in hidden_tail),
            weights,  # pm-lint: allow PM005 -- activation indexed by knoxel ID
            frozenset(item.record_id for item in shard.items if item.required_by == "entity_context")))
    return tuple(selected)


def render_shards(shards: tuple[ShardRenderInput, ...], records: dict[int, RecordView],  # pm-lint: allow PM004 -- keyed by entity or knoxel identity
                  selected: frozenset[int], companion: str, template: str, labels: ShardLabels) -> str:
    """Render only fitted evidence, retaining its source IDs, shard order and chronology."""
    displayed: dict[int, RecordView] = {}  # pm-lint: allow PM004 -- keyed by entity or knoxel identity
    for record_id in selected:
        source = records[record_id].model_copy()
        source.content = f"[source {record_id}] " + source.content
        displayed[record_id] = source
    blocks: list[str] = []
    for shard in shards:
        items = [key for key in shard.item_ids if key in selected]
        tails = [key for key in shard.tail_ids if key in selected and key not in items]
        blocks.append(shard_block_text(shard.key, shard.name, companion, displayed, items, tails,  # pm-lint: allow PM005 -- source views indexed by knoxel ID
                                      template=template, labels=labels))
    return "\n\n".join(blocks)


@dataclass(frozen=True, slots=True)
class ShardTemplateValues:
    character_card: Content | str
    shards: Content | str
    feeling: Content | str
    situational_guidance: Content | str
    new: Content | str

    def field_value(self, name: str) -> Content | str | int | float:
        match name:
            case "character_card":
                return self.character_card
            case "shards":
                return self.shards
            case "feeling":
                return self.feeling
            case "situational_guidance":
                return self.situational_guidance
            case "new":
                return self.new
            case _:
                raise KeyError(f"Unknown ShardTemplateValues field: {name}")


@dataclass(frozen=True, slots=True)
class ShardPromptInput:
    operation: str
    companion_name: str
    system: str
    story_template: str
    acknowledgement: str
    shard_template: str
    shard_labels: ShardLabels
    character: str
    feeling: str
    guidance: str
    new_events: str
    new_ids: frozenset[int]
    records: dict[int, RecordView]  # pm-lint: allow PM004 -- keyed by entity or knoxel identity
    shards: tuple[ShardRenderInput, ...]
    instruction: str = ""
    optional_ceiling: int = 0
    images: tuple[RecordImage, ...] = ()


@dataclass(frozen=True, slots=True)
class InstructionEvidence:
    """Only the named evidence slots can shrink. Instructions and output schemas stay literal."""

    values: TemplateValues
    evidence_fields: frozenset[str]
    retained_characters: int

    def field_value(self, name: str) -> Content | str | int | float:
        value = self.values.field_value(name)
        if name in self.evidence_fields and isinstance(value, str):
            return evidence_excerpt(value, self.retained_characters)
        return value


def fit_shard_instruction(template: str, values: TemplateValues, evidence_fields: frozenset[str],
                          images: tuple[RecordImage, ...], settings: RenderSettings) -> str:
    """Reserve room for the workspace when an action template repeats large source records."""
    instruction = format_content(template, values).text
    estimator = TokenEstimator(estimate_factor=settings.estimate_factor)
    retained = len(instruction)
    catalog = ImageCatalog(images)
    while retained:
        turns = resolve_media((Turn(Role.User, instruction),), catalog)
        if estimator.estimate(turns) <= settings.input_ceiling // 2:
            break
        retained //= 2
        instruction = format_content(template, InstructionEvidence(values, evidence_fields, retained)).text
    return instruction


@dataclass(frozen=True, slots=True, order=True)
class RankedStoryRecord:
    negative_priority: float
    record_id: int
    priority: float = field(compare=False)


def shard_prompt_turns(input: ShardPromptInput, selected: frozenset[int]) -> tuple[Turn, ...]:
    shards = render_shards(input.shards, input.records, selected, input.companion_name,
                           input.shard_template, input.shard_labels)
    story = format_content(input.story_template, ShardTemplateValues(
        character_card=input.character,
        shards=shards,
        feeling=input.feeling,
        situational_guidance=input.guidance,
        new=input.new_events)).text.strip()
    turns = (Turn(Role.System, input.system), Turn(Role.User, story), Turn(Role.Assistant, input.acknowledgement))
    if input.instruction:
        turns = (*turns, Turn(Role.User, input.instruction))
    return resolve_media(turns, ImageCatalog(input.images))


def fit_story_frame(input: ShardPromptInput, settings: RenderSettings) -> ShardPromptInput:
    """Fit the frame before admitting shard records. Keep phase instructions intact."""
    estimator = TokenEstimator(estimate_factor=settings.estimate_factor)
    retained = max(len(input.character), len(input.feeling), len(input.guidance), len(input.new_events))
    frame = input
    while retained and estimator.estimate(shard_prompt_turns(frame, frozenset())) > settings.input_ceiling:
        retained //= 2
        frame = replace(input, character=evidence_excerpt(input.character, retained),
                        feeling=evidence_excerpt(input.feeling, retained),
                        guidance=evidence_excerpt(input.guidance, retained),
                        new_events=evidence_excerpt(input.new_events, retained))
    return frame


def render_shard_prompt(input: ShardPromptInput, settings: RenderSettings) -> RenderedPrompt:
    estimator = TokenEstimator(estimate_factor=settings.estimate_factor)
    input = fit_story_frame(input, settings)
    required: set[int] = set()
    priority: dict[int, float] = {}  # pm-lint: allow PM004 -- keyed by entity or knoxel identity
    for shard in input.shards:
        if shard.key.kind == "entity":
            required.update(shard.tail_ids[-2:])
        for position, record_id in enumerate(shard.tail_ids):
            if record_id in input.new_ids:
                continue
            weight = (5 if shard.key.kind == "entity" else 3) + (position + 1) / max(1, len(shard.tail_ids))
            priority[record_id] = max(priority.get(record_id, 0), weight)
        for record_id in shard.item_ids:
            if record_id in input.new_ids:
                continue
            weight = 2 + shard.activation[record_id]
            if record_id in shard.entity_context_ids:
                weight += 2
            priority[record_id] = max(priority.get(record_id, 0), weight)
    required.difference_update(input.new_ids)
    selected: frozenset[int] = frozenset(required)
    limits = shard_prompt_turns(input, selected)
    excerpted: set[int] = set()
    if required and estimator.estimate(limits) > settings.input_ceiling:
        # Preserve both speakers' recent words when whole pinned records would displace the exchange.
        originals = input.records
        retained = max(len(originals[key].content) for key in required)
        while retained and estimator.estimate(limits) > settings.input_ceiling:
            retained //= 2
            records = originals.copy()
            for key in required:
                if len(originals[key].content) > retained:
                    excerpted.add(key)
                    record = originals[key].model_copy()
                    record.content = evidence_excerpt(record.content, retained)
                    records[key] = record
            input = replace(input, records=records)
            limits = shard_prompt_turns(input, selected)
    relax_pins = bool(required) and estimator.estimate(limits) > settings.input_ceiling
    if relax_pins:
        selected = selected.difference(required)
        limits = shard_prompt_turns(input, selected)
    require_rendered_prompt(input.operation, limits, settings)
    ranked = sorted(RankedStoryRecord(-score, record_id, score) for record_id, score in priority.items())
    if relax_pins:
        ranked = [item for item in ranked if item.record_id in required] + [
            item for item in ranked if item.record_id not in required]
    decisions = [EvidenceDecision(record_id, priority.get(record_id, 0), True,
                                  "latest partner dialogue excerpt" if record_id in excerpted else "latest partner dialogue")
                 for record_id in sorted(selected)]
    seen_texts = {input.records[record_id].content.strip() for record_id in selected | set(input.new_ids)}
    for ranked_record in ranked:
        record_id = ranked_record.record_id
        if record_id in selected:
            continue
        text = input.records[record_id].content.strip()
        if text in seen_texts:
            decisions.append(EvidenceDecision(record_id, ranked_record.priority, False, "duplicate source content"))
            continue
        trial = selected | {record_id}
        limit = min(settings.input_ceiling, input.optional_ceiling or settings.input_ceiling)
        included = estimator.estimate(shard_prompt_turns(input, trial)) <= limit
        if included:
            selected = trial
            seen_texts.add(text)
        decisions.append(EvidenceDecision(record_id, ranked_record.priority, included,
                                          "workspace salience" if included else "workspace budget"))
    groups = ("workspace evidence pins relaxed",) if relax_pins or excerpted else ()
    rendered = require_rendered_prompt(input.operation, shard_prompt_turns(input, selected), settings,
                                      tuple(decisions), selected_groups=groups)
    return replace(rendered, required_source_ids=tuple(sorted((required & selected) - excerpted)))
