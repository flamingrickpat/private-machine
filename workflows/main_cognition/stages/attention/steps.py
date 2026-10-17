"""Attention role.

Entry point: ``run(context)``. Since A15 the role works in two steps:

1. Scoring (model). The codelets stage supplies typed proposals. The role
   asks the model for four 0..10 integer scores per proposal, in batches of
   ``score_batch_size``. A batch that misses ids gets one repair request for
   only the missing ids. A proposal that still has no score is ``rejected``
   as ``unscored``.
2. Selection (pure Python). ``attention_competition.compete`` chooses the
   winners and gives every nomination a status and a reason.

The role writes :class:`AttentionResult` to ``outbox/report.yaml``. It never
writes memory. Target ids come only from proposals, and the codelets stage
already limited them to supplied stimuli, memories, and goals; the id guard
here repeats that check at the stage boundary.
"""
from __future__ import annotations

from pm.constants import Sampler
from pm.llm.concurrency import ordered_parallel_map
from workflows.inference_settings import ATTENTION_WORKERS

from pathlib import Path
from pm.context_render.turns import Role, Turn, format_content

from workflows.main_cognition.attention_competition import compete
from workflows.cognitive_control.attention import attention_bias, resolve_focus
from workflows.main_cognition.codelet_schedule import AttentionPolicy, in_danger
from workflows.main_cognition.schemas import (
    AttentionNomination, AttentionResult, CodeletProposal, NominationScores,
)
from workflows.main_cognition.stage_support import (
    StageOutputError, complete_json, load_yaml, read_request, write_report,
)

WORKFLOW_DIR = Path(__file__).resolve().parent


def load_prompts(path: Path | None = None) -> dict:
    return load_yaml(path or WORKFLOW_DIR / "prompts.yaml")


def candidate_ids(candidates: list[dict]) -> set[int]:
    return {int(candidate["knoxel_id"]) for candidate in candidates}


def ensure_candidate_ids(result: AttentionResult, allowed: set[int]) -> None:
    """Reject any id that was not in the supplied candidate list."""
    allowed = {int(candidate_id) for candidate_id in allowed}
    for nomination in result.nominations:
        for target_id in nomination.target_ids:
            if int(target_id) not in allowed:
                raise StageOutputError(
                    f"Attention nominated target id {target_id} that was not a candidate."
                )
    for selected_id in result.selected_ids:
        if int(selected_id) not in allowed:
            raise StageOutputError(
                f"Attention selected id {selected_id} that was not a candidate."
            )


def proposals_from(request: dict) -> list[CodeletProposal]:
    return [CodeletProposal.model_validate(item)
            for item in (request.get("codelets") or {}).get("proposals", [])]


def render_items(proposals: list[CodeletProposal], prompts: dict) -> str:
    labels = prompts["reality_labels"]
    lines = [f'- {item.nomination_id} ({labels.get(item.reality, item.reality)}): "{item.text}"'
             for item in proposals]
    return "\n".join(lines) or "(none)"


def build_messages(request: dict, prompts: dict, proposals: list[CodeletProposal]) -> list[Turn]:
    companion = request.get("companion_name", "Spamuel")
    user = request.get("user_name", "Jesse")
    system = prompts["system_prompt"].format(companion_name=companion, user_name=user)
    affect = request.get("affect") or {}
    user_message = format_content(prompts["user_template"],
        companion_name=companion, user_name=user,
        affect=affect.get("summary") or request.get("state_summary") or "(calm, nothing strong)",
        active_goals="\n".join(f"- {goal}" for goal in request.get("active_goals", [])) or "(none)",
        items=render_items(proposals, prompts),
    )
    return [Turn(Role.System, system.strip()), Turn(Role.User, user_message.strip())]


def score_batch(context, request: dict, prompts: dict, batch: list[CodeletProposal]) -> dict[str, dict]:
    """Return ``{nomination_id: score dict}`` for one batch.

    Unknown ids in the model answer are ignored. Missing ids get one repair
    request that lists only those items.
    """
    wanted = {item.nomination_id for item in batch}
    messages = build_messages(request, prompts, batch)
    result = complete_json(context, messages, NominationScores, temperature=Sampler.TempInstruct,
                           enable_thinking=False)
    scores = {item.id: item.model_dump() for item in result.scores if item.id in wanted}
    missing = [item for item in batch if item.nomination_id not in scores]
    if missing:
        repair = [*messages, {"role": "assistant", "content": result.model_dump_json()},
                  {"role": "user", "content": prompts["repair_template"].format(
                      missing=", ".join(item.nomination_id for item in missing),
                      items=render_items(missing, prompts))}]
        retry = complete_json(context, repair, NominationScores, temperature=Sampler.TempInstruct,
                              enable_thinking=False)
        for item in retry.scores:
            if item.id in wanted and item.id not in scores:
                scores[item.id] = item.model_dump()
    return scores


def nominations_from(proposals: list[CodeletProposal], scores: dict[str, dict]) -> list[AttentionNomination]:
    """Merge proposals and their 0..10 scores into 0..1 nominations."""
    nominations = []
    for proposal in proposals:
        score = scores.get(proposal.nomination_id)
        nomination = AttentionNomination(
            target_ids=list(proposal.target_ids), attention_kind=proposal.proposal_kind,
            nomination_id=proposal.nomination_id, codelet=proposal.codelet, family=proposal.family,
            origin=proposal.origin, reality=proposal.reality, text=proposal.text, cause=list(proposal.cause),
            activation=proposal.activation, source_revisions=dict(proposal.source_revisions),
            topic=proposal.topic,
        )
        if score is None:
            nomination.status, nomination.reason, nomination.score = "rejected", "unscored", 0.0
        else:
            for axis in ("urgency", "novelty", "goal_relevance", "reportability"):
                setattr(nomination, axis, round(int(score[axis]) / 10.0, 2))
            nomination.reasoning = str(score.get("why", ""))
        nominations.append(nomination)
    return nominations


def run(context) -> dict:
    request = read_request(context.runtime.environment)
    prompts = load_prompts()
    proposals = proposals_from(request)
    policy = AttentionPolicy.model_validate((request.get("codelets") or {}).get("policy") or {})
    scores: dict[str, dict] = {}
    batches = [proposals[start:start + policy.score_batch_size]
               for start in range(0, len(proposals), policy.score_batch_size)]
    results = ordered_parallel_map(lambda batch: score_batch(context, request, prompts, batch), batches,
                                   workers=ATTENTION_WORKERS, cancel=context.runtime.stop_event)
    for result in results:
        scores.update(result)
    danger = in_danger((request.get("codelets") or {}).get("signals") or {})
    focus = resolve_focus(context, request)
    nominations = nominations_from(proposals, scores)
    result = compete(nominations, policy, danger=danger,
                     biases=attention_bias(nominations, focus, danger=danger))
    result.competition["focus"] = focus
    result.competition["scoring_calls"] = -(-len(proposals) // policy.score_batch_size) if proposals else 0
    allowed = candidate_ids(list(request.get("candidates", [])))
    allowed |= {int(goal["knoxel_id"]) for goal in request.get("active_goal_records", [])}
    ensure_candidate_ids(result, allowed)
    write_report(context.runtime.environment, result)
    winners = sum(1 for item in result.nominations if item.status == "selected")
    return {
        "status": "done",
        "summary": f"scored {len(scores)} of {len(proposals)} nominations and selected {winners}",
    }


__all__ = [
    "build_messages",
    "candidate_ids",
    "ensure_candidate_ids",
    "load_prompts",
    "nominations_from",
    "proposals_from",
    "render_items",
    "run",
    "score_batch",
]
