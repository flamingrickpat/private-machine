from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from pm.context_render.budget import TokenEstimator, evidence_excerpt
from pm.context_render.prompt import RenderSettings, RenderedPrompt, require_rendered_prompt
from pm.context_render.turns import Role, Turn
from workflows.cognitive_control.schemas import FocusProposal
from workflows.main_cognition.schemas import AppraisalView, Stimulus


class FocusEvidence(BaseModel):
    appraisal: AppraisalView = Field(default_factory=AppraisalView)
    codelet_notes: list[str] = Field(default_factory=list[str])
    goals: list[str] = Field(default_factory=list[str])
    stimuli: list[Stimulus] = Field(default_factory=list[Stimulus])


@dataclass(frozen=True, slots=True)
class FocusRenderInput:
    prefix: tuple[Turn, ...]
    instruction_template: str
    proposal: FocusProposal
    evidence: FocusEvidence


def render_focus(input: FocusRenderInput, settings: RenderSettings) -> RenderedPrompt:
    proposal = input.proposal
    proposed = (f"Focus: {proposal.instruction}\nPromote records: {proposal.promote_ids}\n"
                f"Reduce records: {proposal.reduce_ids}\nStrength: {proposal.strength:.3f}\n"
                f"Duration seconds: {proposal.seconds:g}\nTopics: {'; '.join(proposal.topic_hints)}\nReason: {proposal.reason}")
    evidence = input.evidence
    state = (f"Appraisal: {evidence.appraisal.feeling}\nAccepted feeling: {evidence.appraisal.felt_state}\n"
             "New events:\n" + "\n".join(f"[{item.knoxel_id}; {item.source}; {item.kind}] {item.text}" for item in evidence.stimuli))
    instruction = input.instruction_template.format(proposal=proposed, evidence=state)
    prefix = input.prefix
    turns = (*prefix, Turn(Role.User, instruction))
    estimator = TokenEstimator(estimate_factor=settings.estimate_factor)
    if estimator.estimate(turns) > settings.input_ceiling:
        # The proposal authorizes the exact IDs below. Drop cached story context before reducing its supporting events.
        prefix = ()
        turns = (Turn(Role.User, instruction),)
    original_state = state
    retained = len(original_state)
    while retained and estimator.estimate(turns) > settings.input_ceiling:
        retained //= 2
        state = evidence_excerpt(original_state, retained)
        turns = (Turn(Role.User, input.instruction_template.format(proposal=proposed, evidence=state)),)
    require_rendered_prompt("cognitive focus", turns, settings)
    selected: list[str] = []
    for goal in evidence.goals:
        line = "Active goal: " + goal
        trial = state + "\n" + "\n".join([*selected, line])
        trial_turns = (*prefix, Turn(Role.User, input.instruction_template.format(proposal=proposed, evidence=trial)))
        if estimator.estimate(trial_turns) <= settings.input_ceiling:
            selected.append(line)
            turns = trial_turns
    seen: set[str] = set()
    for note in evidence.codelet_notes:
        if not note.strip() or note.strip() in seen:
            continue
        seen.add(note.strip())
        line = "Competing impulse: " + note
        trial = state + "\n" + "\n".join([*selected, line])
        trial_turns = (*prefix, Turn(Role.User, input.instruction_template.format(proposal=proposed, evidence=trial)))
        if estimator.estimate(trial_turns) <= settings.input_ceiling:
            selected.append(line)
            turns = trial_turns
    return require_rendered_prompt("cognitive focus", turns, settings)
