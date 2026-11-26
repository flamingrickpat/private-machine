from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from typing import TYPE_CHECKING

from pm.context_render.budget import TokenEstimator, evidence_excerpt
from pm.context_render.prompt import RenderedPrompt, RenderSettings, EvidenceDecision, require_rendered_prompt
from pm.context_render.turns import Turn, Role
from pm.memory.reconsolidation.schemas import CanonicalFactOption, FactClaimOption

if TYPE_CHECKING:
    from pm.memory.media_alt import AltTextRenderInput
    from pm.context_render.correction import JsonCorrectionInput


@dataclass(frozen=True, slots=True)
class SceneSummaryInput:
    system: str
    template: str
    intro: str
    story: str
    companion_name: str
    user_name: str


def render_scene_summary(input: SceneSummaryInput, settings: RenderSettings) -> RenderedPrompt:
    user = input.template.format(intro=input.intro, story=input.story,
        companion_name=input.companion_name, user_name=input.user_name)
    return require_rendered_prompt("memory scene summary", (Turn(Role.System, input.system), Turn(Role.User, user)), settings)


@dataclass(frozen=True, slots=True)
class TemporalSummaryInput:
    scene: SceneSummaryInput
    start: str
    end: str


def render_temporal_summary(input: TemporalSummaryInput, settings: RenderSettings) -> RenderedPrompt:
    scene = input.scene
    user = scene.template.format(intro=scene.intro, story=scene.story, start=input.start, end=input.end,
        companion_name=scene.companion_name, user_name=scene.user_name)
    return require_rendered_prompt("memory temporal summary", (Turn(Role.System, scene.system), Turn(Role.User, user)), settings)


@dataclass(frozen=True, slots=True)
class RecordedStateInput:
    scene: SceneSummaryInput
    recorded: str


def render_recorded_state(input: RecordedStateInput, settings: RenderSettings) -> RenderedPrompt:
    scene = input.scene
    user = scene.template.format(intro=scene.intro, story=scene.story, recorded=input.recorded,
        companion_name=scene.companion_name, user_name=scene.user_name)
    return require_rendered_prompt("memory recorded state", (Turn(Role.System, scene.system), Turn(Role.User, user)), settings)


@dataclass(frozen=True, slots=True)
class FactExtractionInput:
    scene: SceneSummaryInput


def render_fact_extraction(input: FactExtractionInput, settings: RenderSettings) -> RenderedPrompt:
    scene = input.scene
    user = scene.template.format(intro=scene.intro, story=scene.story,
        companion_name=scene.companion_name, user_name=scene.user_name)
    return require_rendered_prompt("memory fact extraction", (Turn(Role.System, scene.system), Turn(Role.User, user)), settings)


@dataclass(frozen=True, slots=True)
class FactClassificationInput:
    system: str
    template: str
    intro: str
    facts: tuple[str, ...]
    categories: tuple[str, ...]
    companion_name: str
    user_name: str
    subjects: tuple[FactSubject, ...] = ()
    claim_options: tuple[tuple[FactClaimOption, ...], ...] = ()
    canonical_options: tuple[tuple[CanonicalFactOption, ...], ...] = ()
    source_times: tuple[datetime, ...] = ()


@dataclass(frozen=True, slots=True)
class FactSubject:
    """One identity visible in the original episode, including peer speakers."""

    identity: str
    label: str


def render_fact_classification(input: FactClassificationInput, settings: RenderSettings) -> RenderedPrompt:
    entries: list[str] = []
    for index, fact in enumerate(input.facts, 1):
        timestamp = f" (source {input.source_times[index - 1]:%Y-%m-%d %H:%M})" if input.source_times else ""
        entries.append(f"{index}. {fact}{timestamp}")
        if input.claim_options:
            for option in input.claim_options[index - 1]:
                entries.append(f"   Candidate {option.claim_id} about {option.about}: {option.statement}")
        if input.canonical_options:
            for prior in input.canonical_options[index - 1]:
                entries.append(f"   Filed memory ID {prior.head_id} about {prior.about} ({prior.record_type.value}, "
                               f"supported through {prior.supported_through:%Y-%m-%d %H:%M}): {prior.statement}")
    listing = "\n".join(entries)
    subjects = ", ".join(f"{item.identity} ({item.label})" for item in input.subjects)
    user = input.template.format(intro=input.intro, facts=listing, categories=", ".join(input.categories),
        subjects=subjects, companion_name=input.companion_name, user_name=input.user_name)
    return require_rendered_prompt("memory fact classification", (Turn(Role.System, input.system), Turn(Role.User, user)), settings)


@dataclass(frozen=True, slots=True)
class NarrativeEvidence:
    record_id: int
    text: str


@dataclass(frozen=True, slots=True)
class NarrativeRenderInput:
    system: str
    template: str
    intro: str
    card: str
    label: str
    target: str
    previous: str
    goal: str
    companion_name: str
    user_name: str
    evidence: tuple[NarrativeEvidence, ...]


def render_narrative(input: NarrativeRenderInput, settings: RenderSettings) -> RenderedPrompt:
    estimator = TokenEstimator(estimate_factor=settings.estimate_factor)
    original = input
    retained = max(len(input.intro), len(input.card), len(input.previous), len(input.target), len(input.goal))
    while True:
        body = input.template.format(intro=input.intro, card=input.card, label=input.label, target=input.target,
            previous=input.previous, story="", goal=input.goal, companion_name=input.companion_name, user_name=input.user_name)
        turns = (Turn(Role.System, input.system), Turn(Role.User, body))
        if estimator.estimate(turns) <= settings.input_ceiling // 2 or retained == 0:
            break
        retained //= 2
        input = replace(original, intro=evidence_excerpt(original.intro, retained), card=evidence_excerpt(original.card, retained),
            previous=evidence_excerpt(original.previous, retained), goal=evidence_excerpt(original.goal, retained),
            target=evidence_excerpt(original.target, retained))
    selected: list[str] = []
    decisions: list[EvidenceDecision] = []
    exhausted = False
    for item in input.evidence:
        body = input.template.format(intro=input.intro, card=input.card, label=input.label, target=input.target,
            previous=input.previous, story="\n".join([*selected, item.text]), goal=input.goal,
            companion_name=input.companion_name, user_name=input.user_name)
        trial = (turns[0], Turn(Role.User, body))
        admitted = not exhausted and estimator.estimate(trial) <= settings.input_ceiling
        if admitted:
            selected.append(item.text)
            turns = trial
        else:
            exhausted = True
        decisions.append(EvidenceDecision(item.record_id, 1, admitted,
            "chronological unprocessed narrative evidence" if admitted else "deferred to next narrative version"))
    if input.evidence and not selected:
        first = input.evidence[0]
        retained = len(first.text)
        while True:
            excerpt = evidence_excerpt(first.text, retained)
            body = input.template.format(intro=input.intro, card=input.card, label=input.label, target=input.target,
                previous=input.previous, story=excerpt, goal=input.goal,
                companion_name=input.companion_name, user_name=input.user_name)
            turns = (turns[0], Turn(Role.User, body))
            if estimator.estimate(turns) <= settings.input_ceiling or retained == 0:
                decisions[0] = EvidenceDecision(first.record_id, 1, True, "bounded first narrative source")
                break
            retained //= 2
    return require_rendered_prompt("memory narrative", turns, settings, tuple(decisions))


def replay_memory_capture(input: SceneSummaryInput | TemporalSummaryInput | RecordedStateInput |
                          FactExtractionInput | FactClassificationInput | NarrativeRenderInput |
                          AltTextRenderInput | JsonCorrectionInput, settings: RenderSettings) -> RenderedPrompt:
    from pm.memory.media_alt import AltTextRenderInput, render_alt_text
    from pm.context_render.correction import JsonCorrectionInput, render_json_correction
    match input:
        case SceneSummaryInput():
            return render_scene_summary(input, settings)
        case TemporalSummaryInput():
            return render_temporal_summary(input, settings)
        case RecordedStateInput():
            return render_recorded_state(input, settings)
        case FactExtractionInput():
            return render_fact_extraction(input, settings)
        case FactClassificationInput():
            return render_fact_classification(input, settings)
        case NarrativeRenderInput():
            return render_narrative(input, settings)
        case AltTextRenderInput():
            return render_alt_text(input, settings)
        case JsonCorrectionInput():
            return render_json_correction(input, settings)
