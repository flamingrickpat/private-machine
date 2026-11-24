from __future__ import annotations

from dataclasses import dataclass
from pm.context_render.turns import Content


@dataclass(frozen=True, slots=True)
class ExpectationCheckTemplateValues:
    expectations: Content | str
    stimuli: Content | str

    def field_value(self, name: str) -> Content | str | int | float:
        match name:
            case "expectations":
                return self.expectations
            case "stimuli":
                return self.stimuli
            case _:
                raise KeyError(f"Unknown ExpectationCheckTemplateValues field: {name}")


@dataclass(frozen=True, slots=True)
class AnticipationTemplateValues:
    decision: Content | str
    action: Content | str
    target: Content | str
    count: int

    def field_value(self, name: str) -> Content | str | int | float:
        match name:
            case "decision":
                return self.decision
            case "action":
                return self.action
            case "target":
                return self.target
            case "count":
                return self.count
            case _:
                raise KeyError(f"Unknown AnticipationTemplateValues field: {name}")


@dataclass(frozen=True, slots=True)
class AppraisalTemplateValues:
    interval_review: Content | str
    state_summary: Content | str
    active_goals: Content | str
    triage: Content | str
    expectations: Content | str

    def field_value(self, name: str) -> Content | str | int | float:
        match name:
            case "interval_review":
                return self.interval_review
            case "state_summary":
                return self.state_summary
            case "active_goals":
                return self.active_goals
            case "triage":
                return self.triage
            case "expectations":
                return self.expectations
            case _:
                raise KeyError(f"Unknown AppraisalTemplateValues field: {name}")


@dataclass(frozen=True, slots=True)
class AttentionTemplateValues:
    attention_narrative: Content | str
    mental_state: Content | str
    shard_focus: Content | str
    structures: Content | str
    aperture: Content | str
    winners: Content | str
    count: Content | str

    def field_value(self, name: str) -> Content | str | int | float:
        match name:
            case "attention_narrative":
                return self.attention_narrative
            case "mental_state":
                return self.mental_state
            case "shard_focus":
                return self.shard_focus
            case "structures":
                return self.structures
            case "aperture":
                return self.aperture
            case "winners":
                return self.winners
            case "count":
                return self.count
            case _:
                raise KeyError(f"Unknown AttentionTemplateValues field: {name}")


@dataclass(frozen=True, slots=True)
class CodeletViewTemplateValues:
    view: Content | str

    def field_value(self, name: str) -> Content | str | int | float:
        match name:
            case "view":
                return self.view
            case _:
                raise KeyError(f"Unknown CodeletViewTemplateValues field: {name}")


@dataclass(frozen=True, slots=True)
class CodeletJobTemplateValues:
    codelet_name: Content | str
    description: Content | str
    instructions: Content | str

    def field_value(self, name: str) -> Content | str | int | float:
        match name:
            case "codelet_name":
                return self.codelet_name
            case "description":
                return self.description
            case "instructions":
                return self.instructions
            case _:
                raise KeyError(f"Unknown CodeletJobTemplateValues field: {name}")


@dataclass(frozen=True, slots=True)
class CodeletConversionTemplateValues:
    schema: Content | str
    targets: Content | str

    def field_value(self, name: str) -> Content | str | int | float:
        match name:
            case "schema":
                return self.schema
            case "targets":
                return self.targets
            case _:
                raise KeyError(f"Unknown CodeletConversionTemplateValues field: {name}")


@dataclass(frozen=True, slots=True)
class CodeletSelectionTemplateValues:
    candidates: Content | str
    echoes: Content | str

    def field_value(self, name: str) -> Content | str | int | float:
        match name:
            case "candidates":
                return self.candidates
            case "echoes":
                return self.echoes
            case _:
                raise KeyError(f"Unknown CodeletSelectionTemplateValues field: {name}")


@dataclass(frozen=True, slots=True)
class DecisionTemplateValues:
    part: Content | str
    stimuli: Content | str
    affect: Content | str
    regulation_words: Content | str
    experience: Content | str
    impulses: Content | str
    goals: Content | str
    managed_runs: Content | str
    variants: int

    def field_value(self, name: str) -> Content | str | int | float:
        match name:
            case "part":
                return self.part
            case "stimuli":
                return self.stimuli
            case "affect":
                return self.affect
            case "regulation_words":
                return self.regulation_words
            case "experience":
                return self.experience
            case "impulses":
                return self.impulses
            case "goals":
                return self.goals
            case "managed_runs":
                return self.managed_runs
            case "variants":
                return self.variants
            case _:
                raise KeyError(f"Unknown DecisionTemplateValues field: {name}")


@dataclass(frozen=True, slots=True)
class InhibitionTemplateValues:
    companion_name: Content | str
    character: Content | str
    situation: Content | str
    stimuli: Content | str
    proposals: Content | str
    managed_runs: Content | str

    def field_value(self, name: str) -> Content | str | int | float:
        match name:
            case "companion_name":
                return self.companion_name
            case "character":
                return self.character
            case "situation":
                return self.situation
            case "stimuli":
                return self.stimuli
            case "proposals":
                return self.proposals
            case "managed_runs":
                return self.managed_runs
            case _:
                raise KeyError(f"Unknown InhibitionTemplateValues field: {name}")


@dataclass(frozen=True, slots=True)
class ExperienceTemplateValues:
    entered: Content | str

    def field_value(self, name: str) -> Content | str | int | float:
        match name:
            case "entered":
                return self.entered
            case _:
                raise KeyError(f"Unknown ExperienceTemplateValues field: {name}")


@dataclass(frozen=True, slots=True)
class GoalsTemplateValues:
    interval_review: Content | str
    affect: Content | str
    regulation_words: Content | str
    current_goals: Content | str
    appraisal: Content | str
    context: Content | str
    managed_runs: Content | str
    inner_proposals: Content | str
    deliberate_proposals: Content | str

    def field_value(self, name: str) -> Content | str | int | float:
        match name:
            case "interval_review":
                return self.interval_review
            case "affect":
                return self.affect
            case "regulation_words":
                return self.regulation_words
            case "current_goals":
                return self.current_goals
            case "appraisal":
                return self.appraisal
            case "context":
                return self.context
            case "managed_runs":
                return self.managed_runs
            case "inner_proposals":
                return self.inner_proposals
            case "deliberate_proposals":
                return self.deliberate_proposals
            case _:
                raise KeyError(f"Unknown GoalsTemplateValues field: {name}")


@dataclass(frozen=True, slots=True)
class RetrievalTemplateValues:
    state_summary: Content | str
    appraisal_summary: Content | str
    active_goals: Content | str
    memory_candidates: Content | str
    guidance: Content | str

    def field_value(self, name: str) -> Content | str | int | float:
        match name:
            case "state_summary":
                return self.state_summary
            case "appraisal_summary":
                return self.appraisal_summary
            case "active_goals":
                return self.active_goals
            case "memory_candidates":
                return self.memory_candidates
            case "guidance":
                return self.guidance
            case _:
                raise KeyError(f"Unknown RetrievalTemplateValues field: {name}")


@dataclass(frozen=True, slots=True)
class RouteTemplateValues:
    part: Content | str
    decision: Content | str
    stimuli: Content | str
    memories: Content | str
    goals: Content | str
    managed_runs: Content | str
    capabilities: Content | str
    workflows: Content | str
    wait_default_seconds: float

    def field_value(self, name: str) -> Content | str | int | float:
        match name:
            case "part":
                return self.part
            case "decision":
                return self.decision
            case "stimuli":
                return self.stimuli
            case "memories":
                return self.memories
            case "goals":
                return self.goals
            case "managed_runs":
                return self.managed_runs
            case "capabilities":
                return self.capabilities
            case "workflows":
                return self.workflows
            case "wait_default_seconds":
                return self.wait_default_seconds
            case _:
                raise KeyError(f"Unknown RouteTemplateValues field: {name}")


@dataclass(frozen=True, slots=True)
class StructureTemplateValues:
    view: Content | str
    broadcasts: Content | str
    max_structures: Content | str
    targets: Content | str

    def field_value(self, name: str) -> Content | str | int | float:
        match name:
            case "view":
                return self.view
            case "broadcasts":
                return self.broadcasts
            case "max_structures":
                return self.max_structures
            case "targets":
                return self.targets
            case _:
                raise KeyError(f"Unknown StructureTemplateValues field: {name}")


@dataclass(frozen=True, slots=True)
class TriageTemplateValues:
    state_summary: Content | str
    active_goals: Content | str
    stimuli: Content | str
    budget_max: Content | str
    budget_default: Content | str

    def field_value(self, name: str) -> Content | str | int | float:
        match name:
            case "state_summary":
                return self.state_summary
            case "active_goals":
                return self.active_goals
            case "stimuli":
                return self.stimuli
            case "budget_max":
                return self.budget_max
            case "budget_default":
                return self.budget_default
            case _:
                raise KeyError(f"Unknown TriageTemplateValues field: {name}")
