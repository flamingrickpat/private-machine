from __future__ import annotations

from workflows.main_cognition.prompt_values import CognitionStoryTemplateValues

from dataclasses import dataclass, field, replace

from pm.context_render.budget import TokenEstimator
from pm.context_render.images import ImageCatalog, resolve_media
from pm.records.media import RecordImage
from pm.context_render.prompt import EvidenceDecision, RenderedPrompt, RenderSettings, require_rendered_prompt
from pm.context_render.story import RecordView, ShardLabels, shard_block_text
from pm.context_render.turns import Role, Turn, format_content
from pm.records.workspace import ShardKey
from workflows.main_cognition.schemas import AppraisalView, GoalsResult, ManagedRunView, MemoryCandidate, ReflectionGuidance
from workflows.public_speech.situational_guidance import SituationSource, SituationalGuidance


@dataclass(frozen=True, slots=True, order=True)
class SituationEvidence:
    negative_priority: float
    record_id: int
    text: str = field(compare=False)


def situation_source_text(source: SituationSource) -> str:
    return (f"[{source.record_id}; {source.occurred_at_utc.isoformat(timespec='minutes')}; {source.source}; "
            f"{source.kind}; reality {source.scope.reality_id}/{source.scope.frame.value}] {source.text}")


def render_cognition_guidance(situations: list[SituationalGuidance], settings: RenderSettings) -> str:
    required = ["Partner statements are attributed evidence. Interpretations remain uncertain. "
                "Silence does not establish a new state. Delivered speech remains delivered."]
    optional: list[SituationEvidence] = []
    shown_ids: set[int] = set()
    for situation in situations:
        required.append(f"World time: {situation.as_of_utc.isoformat(timespec='minutes')} UTC.")
        partner = situation.partner
        if partner is not None:
            required.append(f"Partner: {partner.name}; entity {partner.entity_id}; {partner.entity_class.value}.")
            for source in partner.recent_dialogue[-2:]:
                if source.record_id not in shown_ids:
                    required.append(situation_source_text(source))
                    shown_ids.add(source.record_id)
            for source in partner.context:
                optional.append(SituationEvidence(-2, source.record_id, situation_source_text(source)))
            for context in partner.entity_contexts:
                for row in context.claims:
                    claim = row.claim
                    status = f"{row.applicability}; {'contested' if row.contested else 'unchallenged'}"
                    text = (f"[claim {claim.claim_id}; observer {context.key.observer_entity_id}; {claim.epistemic_kind}; "
                            f"{status}; support {claim.confidence:.3f}; sources {[item.source.record_id for item in claim.evidence]}] {claim.statement}")
                    priority = 5 if claim.interaction is not None else 3
                    if row.applicability in {"expired", "future"}:
                        priority = 1
                    source_id = claim.evidence[0].source.record_id
                    optional.append(SituationEvidence(-priority - claim.confidence, source_id, text))
        for move in situation.recent_moves[-2:]:
            required.append(f"Recent purpose ({move.status}): {situation_source_text(move.action)}")
            shown_ids.add(move.action.record_id)
            if move.utterance is not None and move.utterance.record_id not in shown_ids:
                required.append("Actual speech: " + situation_source_text(move.utterance))
                shown_ids.add(move.utterance.record_id)
        for source in situation.intentions:
            optional.append(SituationEvidence(-4, source.record_id, "Accepted intention: " + situation_source_text(source)))
    estimator = TokenEstimator(estimate_factor=settings.estimate_factor)
    budget = settings.input_ceiling // 5
    selected = list(required)
    selected_texts = set(required)
    for row in sorted(optional):
        if row.text in selected_texts:
            continue
        text = "\n".join([*selected, row.text])
        if estimator.estimate((Turn(Role.User, text),)) <= budget:
            selected.append(row.text)
            selected_texts.add(row.text)
    return "\n".join(selected)


def render_managed_runs(runs: list[ManagedRunView]) -> str:
    return "\n".join(f"[{run.run_id}; {run.workflow_name}; {run.state}] {run.instructions or ''}" for run in runs) or "(none)"


def render_goal_changes(goals: GoalsResult) -> str:
    return "\n".join(f"{change.change}: {change.goal} (priority {change.priority:.3f}; {change.reason})"
                     for change in goals.changes) or "(none)"


def render_appraisal(appraisal: AppraisalView) -> str:
    axes = ", ".join(f"{name}={value:.3f}" for name, value in appraisal.appraisal.items())
    return f"Judgment: {appraisal.feeling}\nFeeling: {appraisal.felt_state}\nAppraisal: {axes}\nReason: {appraisal.reasoning}"


def render_reflection_guidance(guidance: list[ReflectionGuidance]) -> str:
    return "\n".join(f"[experiment {item.experiment_id}; {item.status}; scope {item.scope}] {item.instruction}"
                     for item in guidance)


def memory_candidate_text(candidate: MemoryCandidate) -> str:
    return (f"[{candidate.knoxel_id}; revision {candidate.revision}; {candidate.memory_kind}; "
            f"{candidate.evidence_role}; {'historical' if candidate.historical else 'current'}] {candidate.text}")


@dataclass(frozen=True, slots=True, order=True)
class RankedMemoryCandidate:
    negative_score: float
    record_id: int
    candidate: MemoryCandidate = field(compare=False)


def select_recall_candidates(candidates: list[MemoryCandidate], settings: RenderSettings) -> list[MemoryCandidate]:
    estimator = TokenEstimator(estimate_factor=settings.estimate_factor)
    budget = settings.input_ceiling // 4
    selected: list[MemoryCandidate] = []
    texts: list[str] = []
    seen: set[str] = set()
    ranked = sorted(RankedMemoryCandidate(-candidate.score, candidate.knoxel_id, candidate) for candidate in candidates)
    for item in ranked:
        candidate = item.candidate
        if candidate.text.strip() in seen:
            continue
        text = memory_candidate_text(candidate)
        trial = "\n".join([*texts, text])
        if estimator.estimate((Turn(Role.User, trial),)) > budget:
            continue
        selected.append(candidate)
        texts.append(text)
        seen.add(candidate.text.strip())
    return selected


@dataclass(frozen=True, slots=True)
class CognitionShardInput:
    key: ShardKey
    name: str
    item_ids: tuple[int, ...]
    tail_ids: tuple[int, ...]
    activation: dict[int, float]
    entity_context_ids: frozenset[int]


@dataclass(frozen=True, slots=True)
class CognitionStoryInput:
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
    records: dict[int, RecordView]
    shards: tuple[CognitionShardInput, ...]
    situations: tuple[SituationalGuidance, ...] = ()
    instruction: str = ""
    optional_ceiling: int = 0
    images: tuple[RecordImage, ...] = ()


@dataclass(frozen=True, slots=True, order=True)
class RankedStoryRecord:
    negative_priority: float
    record_id: int
    priority: float = field(compare=False)


def cognition_story_turns(input: CognitionStoryInput, selected: frozenset[int]) -> tuple[Turn, ...]:
    blocks: list[str] = []
    displayed_records: dict[int, RecordView] = {}
    for record_id in selected:
        source = input.records[record_id].model_copy()
        source.content = f"[source {record_id}] " + source.content
        displayed_records[record_id] = source
    for shard in input.shards:
        items = [record_id for record_id in shard.item_ids if record_id in selected]
        tails = [record_id for record_id in shard.tail_ids if record_id in selected and record_id not in items]
        blocks.append(shard_block_text(shard.key, shard.name, input.companion_name, displayed_records,
            items, tails, template=input.shard_template, labels=input.shard_labels))
    story = format_content(input.story_template, CognitionStoryTemplateValues(
        character_card=input.character,
        shards="\n\n".join(blocks),
        feeling=input.feeling,
        situational_guidance=input.guidance,
        new=input.new_events)).text.strip()
    turns = (Turn(Role.System, input.system), Turn(Role.User, story), Turn(Role.Assistant, input.acknowledgement))
    if input.instruction:
        turns = (*turns, Turn(Role.User, input.instruction))
    return resolve_media(turns, ImageCatalog(input.images))


def render_cognition_story(input: CognitionStoryInput, settings: RenderSettings) -> RenderedPrompt:
    if input.situations:
        input = replace(input, guidance=render_cognition_guidance(list(input.situations), settings), situations=())
    required: set[int] = set()
    priority: dict[int, float] = {}
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
    selected = frozenset(required)
    require_rendered_prompt(input.operation, cognition_story_turns(input, selected), settings)
    estimator = TokenEstimator(estimate_factor=settings.estimate_factor)
    ranked = sorted(RankedStoryRecord(-score, record_id, score) for record_id, score in priority.items())
    decisions = [EvidenceDecision(record_id, priority.get(record_id, 0), True, "latest partner dialogue")
                 for record_id in sorted(required)]
    seen_texts = {input.records[record_id].content.strip() for record_id in required | set(input.new_ids)}
    for ranked_record in ranked:
        record_id = ranked_record.record_id
        if record_id in required:
            continue
        text = input.records[record_id].content.strip()
        if text in seen_texts:
            decisions.append(EvidenceDecision(record_id, ranked_record.priority, False, "duplicate source content"))
            continue
        trial = selected | {record_id}
        limit = min(settings.input_ceiling, input.optional_ceiling or settings.input_ceiling)
        included = estimator.estimate(cognition_story_turns(input, trial)) <= limit
        if included:
            selected = trial
            seen_texts.add(text)
        decisions.append(EvidenceDecision(record_id, ranked_record.priority, included,
                                          "workspace salience" if included else "workspace budget"))
    return require_rendered_prompt(input.operation, cognition_story_turns(input, selected), settings, tuple(decisions))
