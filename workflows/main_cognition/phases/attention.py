"""Attention phase: score the codelet proposals and run the competition.

The phase works in two steps:

1. Scoring (model). The model gives four 0..10 integer scores for each
   proposal, in batches of ``score_batch_size``. A batch that misses ids gets
   one repair request for only the missing ids. A proposal that still has no
   score is ``rejected`` as ``unscored``.
2. Selection (pure Python). ``attention_competition.compete`` chooses the
   winners and gives every nomination a status and a reason.

Fresh deliberate focus proposals are accepted or rejected here, before the
competition, and bias it within bounds. Target ids come only from proposals.
The codelets phase limited them to records of the CSM and their percepts.
The id guard here repeats that check at the phase boundary.

A codelet whose nomination wins gets a higher base-level activation (LIDA
attentional learning). Step 6 of A19F replaces this phase with attention
into workspace shards, and then a percept that enters a shard does this.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from pm.constants import THREAT_APPRAISAL_THRESHOLD
from pm.context_render.turns import Role, Turn, format_content
from pm.llm.concurrency import ordered_parallel_map

from workflows.cognitive_control.attention import attention_bias, resolve_focus
from workflows.inference_settings import ATTENTION_WORKERS
from workflows.main_cognition.attention_competition import compete
from workflows.main_cognition.codelet_registry import reinforce
from workflows.main_cognition.cycle_work import CycleWork
from workflows.main_cognition.model_json import PhasePrompts, StageOutputError
from workflows.main_cognition.phases.model import complete, phase_prompts
from workflows.main_cognition.schemas import (
    AppraisalView, AttentionNomination, AttentionResult, CodeletProposal, NominationScore,
    NominationScores, Stimulus,
)


class FocusEvidence(BaseModel):
    """The cycle context that the focus decision reads beside each proposal."""

    appraisal: AppraisalView = Field(default_factory=AppraisalView)
    codelet_notes: list[str] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)
    stimuli: list[Stimulus] = Field(default_factory=list)


def candidate_ids(work: CycleWork) -> set[int]:
    """Return the ids that attention may select.

    These are the selected memories, the stimuli, the active goals, and every
    record that the CSM of this cycle can name, new percepts included.
    """
    ids = set(work.context.candidate_ids) | set(work.records)
    ids |= {item.knoxel_id for item in work.request.stimuli}
    ids |= {goal.knoxel_id for goal in work.request.active_goal_records}
    return ids


def ensure_candidate_ids(result: AttentionResult, allowed: set[int]) -> None:
    """Reject any id that was not in the supplied candidate list."""
    for nomination in result.nominations:
        for target_id in nomination.target_ids:
            if target_id not in allowed:
                raise StageOutputError(
                    f"Attention nominated target id {target_id} that was not a candidate."
                )
    for selected_id in result.selected_ids:
        if selected_id not in allowed:
            raise StageOutputError(
                f"Attention selected id {selected_id} that was not a candidate."
            )


def render_items(proposals: list[CodeletProposal], prompts: PhasePrompts) -> str:
    labels = prompts.reality_labels
    lines = [f'- {item.nomination_id} ({labels[item.reality] if item.reality in labels else item.reality}): "{item.text}"'
             for item in proposals]
    return "\n".join(lines) or "(none)"


def build_messages(work: CycleWork, prompts: PhasePrompts, proposals: list[CodeletProposal]) -> list[Turn]:
    companion = work.request.companion_name
    user = work.request.user_name
    system = prompts.system_prompt.format(companion_name=companion, user_name=user)
    user_message = format_content(prompts.user_template,
        companion_name=companion, user_name=user,
        affect=work.affect.felt,
        active_goals="\n".join(f"- {goal}" for goal in work.request.active_goals) or "(none)",
        items=render_items(proposals, prompts),
    )
    return [Turn(Role.System, system.strip()), Turn(Role.User, user_message.strip())]


def score_batch(context, work: CycleWork, prompts: PhasePrompts,
                batch: list[CodeletProposal]) -> dict[str, NominationScore]:
    """Return the scores of one batch by nomination id.

    Unknown ids in the model answer are ignored. Missing ids get one repair
    request that lists only those items. Scoring calls get no grounding turn.
    """
    wanted = {item.nomination_id for item in batch}
    messages = build_messages(work, prompts, batch)
    result = complete(context, work, messages, NominationScores)
    scores = {item.id: item for item in result.scores if item.id in wanted}
    missing = [item for item in batch if item.nomination_id not in scores]
    if missing:
        repair = [*messages, {"role": "assistant", "content": result.model_dump_json()},
                  {"role": "user", "content": prompts.repair_template.format(
                      missing=", ".join(item.nomination_id for item in missing),
                      items=render_items(missing, prompts))}]
        retry = complete(context, work, repair, NominationScores)
        for item in retry.scores:
            if item.id in wanted and item.id not in scores:
                scores[item.id] = item
    return scores


def nominations_from(proposals: list[CodeletProposal], scores: dict[str, NominationScore]) -> list[AttentionNomination]:
    """Merge proposals and their 0..10 scores into 0..1 nominations."""
    nominations = []
    for proposal in proposals:
        nomination = AttentionNomination(
            target_ids=list(proposal.target_ids), attention_kind=proposal.proposal_kind,
            nomination_id=proposal.nomination_id, codelet=proposal.codelet, family=proposal.family,
            origin=proposal.origin, reality=proposal.reality, text=proposal.text, cause=list(proposal.cause),
            activation=proposal.activation, source_revisions=dict(proposal.source_revisions),
            topic=proposal.topic,
        )
        if proposal.nomination_id not in scores:
            nomination.status, nomination.reason, nomination.score = "rejected", "unscored", 0.0
        else:
            score = scores[proposal.nomination_id]
            nomination.urgency = round(score.urgency / 10.0, 2)
            nomination.novelty = round(score.novelty / 10.0, 2)
            nomination.goal_relevance = round(score.goal_relevance / 10.0, 2)
            nomination.reportability = round(score.reportability / 10.0, 2)
            nomination.reasoning = score.why
        nominations.append(nomination)
    return nominations


def run(context, work: CycleWork) -> CycleWork:
    prompts = phase_prompts("attention")
    proposals = work.codelets.proposals
    policy = work.codelets.policy
    scores: dict[str, NominationScore] = {}
    batches = [proposals[start:start + policy.score_batch_size]
               for start in range(0, len(proposals), policy.score_batch_size)]
    results = ordered_parallel_map(lambda batch: score_batch(context, work, prompts, batch), batches,
                                   workers=ATTENTION_WORKERS, cancel=context.runtime.stop_event)
    for result in results:
        scores.update(result)
    danger = work.appraisal.general().bodily_threat >= THREAT_APPRAISAL_THRESHOLD
    evidence = FocusEvidence(appraisal=work.appraisal_view(), codelet_notes=[item.text for item in proposals],
                             goals=work.request.active_goals, stimuli=work.request.stimuli)
    focus = resolve_focus(context, work.request.active_focus, work.request.focus_proposals, evidence)
    nominations = nominations_from(proposals, scores)
    result = compete(nominations, policy, danger=danger,
                     biases=attention_bias(nominations, focus, danger=danger))
    result.competition.focus = focus
    result.competition.scoring_calls = len(batches)
    ensure_candidate_ids(result, candidate_ids(work))
    for name in dict.fromkeys(item.codelet for item in result.nominations if item.status == "selected"):
        reinforce(work.state.codelets.runtime(name))
    work.attention = result
    winners = sum(1 for item in result.nominations if item.status == "selected")
    work.receipt("attention", f"scored {len(scores)} of {len(proposals)} nominations and selected {winners}")
    return work


__all__ = [
    "FocusEvidence",
    "build_messages",
    "candidate_ids",
    "ensure_candidate_ids",
    "nominations_from",
    "render_items",
    "run",
    "score_batch",
]
