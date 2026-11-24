from __future__ import annotations


from dataclasses import dataclass, field, replace

from pm.context_render.budget import TokenEstimator
from pm.records.media import RecordImage
from pm.context_render.prompt import RenderedPrompt, RenderSettings
from pm.context_render.story import RecordView, ShardLabels
from pm.context_render.shards import ShardRenderInput as CognitionShardInput, ShardPromptInput, shard_prompt_turns, render_shard_prompt
from pm.context_render.turns import Role, Turn
from workflows.main_cognition.schemas import AppraisalView, GoalsResult, ManagedRunView, MemoryCandidate, ReflectionGuidance
from workflows.public_speech.situational_guidance import SituationSource, SituationalGuidance


@dataclass(frozen=True, slots=True, order=True)
class SituationEvidence:
    negative_priority: float
    tie_break: int
    text: str = field(compare=False)


def situation_source_text(source: SituationSource) -> str:
    return (f"[{source.record_id}; {source.occurred_at_utc.isoformat(timespec='minutes')}; {source.source}; "
            f"{source.kind}; reality {source.scope.reality_id}/{source.scope.frame.value}] {source.text}")


def render_cognition_guidance(situations: list[SituationalGuidance], settings: RenderSettings, *,
                              reserved_story_tokens: int = 0) -> str:
    required = ["Partner statements are attributed evidence. Interpretations remain uncertain. "
                "Silence does not establish a new state. Delivered speech remains delivered."]
    optional: list[SituationEvidence] = []
    pinned: list[SituationEvidence] = []
    shown_ids: set[int] = set()
    for situation in situations:
        required.append(f"World time: {situation.as_of_utc.isoformat(timespec='minutes')} UTC.")
        partner = situation.partner
        if partner is not None:
            required.append(f"Partner: {partner.name}; entity {partner.entity_id}; {partner.entity_class.value}.")
            for source in partner.recent_dialogue[-2:]:
                if source.record_id not in shown_ids:
                    text = situation_source_text(source)
                    required.append(text)
                    pinned.append(SituationEvidence(-6, -source.record_id, text))
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
            text = f"Recent purpose ({move.status}): {situation_source_text(move.action)}"
            required.append(text)
            pinned.append(SituationEvidence(-5, -move.action.record_id, text))
            shown_ids.add(move.action.record_id)
            if move.utterance is not None and move.utterance.record_id not in shown_ids:
                text = "Actual speech: " + situation_source_text(move.utterance)
                required.append(text)
                pinned.append(SituationEvidence(-6, -move.utterance.record_id, text))
                shown_ids.add(move.utterance.record_id)
        for source in situation.intentions:
            optional.append(SituationEvidence(-4, source.record_id, "Accepted intention: " + situation_source_text(source)))
    estimator = TokenEstimator(estimate_factor=settings.estimate_factor)
    budget = min(settings.input_ceiling // 5, settings.input_ceiling - reserved_story_tokens)
    selected = list(required)
    if estimator.estimate((Turn(Role.User, "\n".join(selected)),)) > budget:
        pinned_texts = {row.text for row in pinned}
        selected = [text for text in required if text not in pinned_texts]
        optional.extend(pinned)
    selected_texts = set(selected)
    for row in sorted(optional):
        if row.text in selected_texts:
            continue
        text = "\n".join([*selected, row.text])
        if estimator.estimate((Turn(Role.User, text),)) <= budget:
            selected.append(row.text)
            selected_texts.add(row.text)
    return "\n".join(selected)


def render_managed_runs(runs: list[ManagedRunView]) -> str:
    lines: list[str] = []
    for run in runs:
        lines.append(f"[{run.run_id}; {run.workflow_name}; {run.state}] {run.instructions or ''}")
        if run.revision:
            lines.append(f"Instruction revision {run.revision}, worker has consumed {run.applied_revision}.")
            if run.original_instructions and run.original_instructions != run.instructions:
                lines.append("Original assignment and site: " + run.original_instructions)
    return "\n".join(lines) or "(none)"


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
    """Render one recall candidate with its selectable id on its own marker line.

    The id leads as ``candidate_id=<integer>`` instead of a bare bracketed
    ``[15377; ...]``. The story prefix, the state of mind, and the body of a
    memory all print record ids in brackets, and a candidate body can cite
    other records. A distinct marker keeps the one number the model may copy
    apart from every id it may not, which is what
    :class:`workflows.main_cognition.schemas.ContextResult` refuses otherwise.
    """
    status = "historical" if candidate.historical else "current"
    marker = (f"candidate_id={candidate.knoxel_id} | revision {candidate.revision} | {candidate.memory_kind}"
              f" | {candidate.evidence_role} | {status}")
    return f"{marker}\n{candidate.text}"


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


def shard_prompt(input: CognitionStoryInput) -> ShardPromptInput:
    return ShardPromptInput(input.operation, input.companion_name, input.system, input.story_template,
        input.acknowledgement, input.shard_template, input.shard_labels, input.character, input.feeling,
        input.guidance, input.new_events, input.new_ids, input.records, input.shards, input.instruction,
        input.optional_ceiling, input.images)


def cognition_story_turns(input: CognitionStoryInput, selected: frozenset[int]) -> tuple[Turn, ...]:
    return shard_prompt_turns(shard_prompt(input), selected)


def fit_story_frame(input: CognitionStoryInput, settings: RenderSettings) -> CognitionStoryInput:
    from pm.context_render.shards import fit_story_frame as fit_shard_frame
    fitted = fit_shard_frame(shard_prompt(input), settings)
    return replace(input, character=fitted.character, feeling=fitted.feeling, guidance=fitted.guidance,
                   new_events=fitted.new_events)


def render_cognition_story(input: CognitionStoryInput, settings: RenderSettings) -> RenderedPrompt:
    prompt = shard_prompt(input)
    if input.situations:
        estimator = TokenEstimator(estimate_factor=settings.estimate_factor)
        mandatory = shard_prompt_turns(replace(prompt, guidance=""), frozenset())
        guidance = render_cognition_guidance(list(input.situations), settings,
                                             reserved_story_tokens=estimator.estimate(mandatory))
        prompt = replace(prompt, guidance=guidance)
    return render_shard_prompt(prompt, settings)
