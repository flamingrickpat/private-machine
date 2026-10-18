"""Goals phase: propose goal changes for this cycle.

The model reviews the current goals against the appraisal and the selected
context. It may keep, add, complete, reprioritize, defer, or refuse goals.
Accepted motive and belief proposals from attention, and private plans from
deliberate mental work, enter as suggestions, never as commands. Any change
kind outside :data:`ALLOWED_CHANGES` fails validation. The commit applies
the changes. This phase never writes memory.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from pm.context_render import render_context
from pm.context_render.turns import Role, Turn, format_content

from workflows.main_cognition.cycle_work import CycleWork
from workflows.main_cognition.model_json import PhasePrompts, StageOutputError
from workflows.main_cognition.phases.model import complete, phase_prompts
from workflows.main_cognition.schemas import AttentionResult, GoalsResult, MentalProposal

ALLOWED_CHANGES = ("retain", "add", "complete", "reprioritize", "defer", "refuse")

# Accepted proposal kinds that each downstream owner consumes. A typed
# nomination without a consumer does not establish a cognitive effect, so
# each owner names the kinds it reads. Every kind's consumer is documented in
# schemas.PROPOSAL_CONSUMERS.
GOAL_KINDS = frozenset({"action_tendency", "belief"})
ACTION_KINDS = frozenset({"action_tendency", "regulation", "simulation"})


class InnerProposal(BaseModel):
    """One accepted nomination of a consumed kind, as later prompts show it."""

    kind: str = ""
    reality: str = ""
    text: str = ""
    target_ids: list[int] = Field(default_factory=list)


def proposal_consumers(kind: str) -> list[str]:
    """Return the owners that consume an accepted proposal of this kind.

    The map is the executable form of ``PROPOSAL_CONSUMERS``: every selected,
    non-stale nomination is always framed by the workspace; the goals phase
    reads motives and self-beliefs; the decide phase reads wants,
    imagination, and ways to stay steady; the belief owner stages subjective
    beliefs. The host records the list on each committed AttentionRecord, so
    a consumer claim is inspectable per nomination.
    """
    consumers = ["workspace"]
    if kind in GOAL_KINDS:
        consumers.append("goals")
    if kind in ACTION_KINDS:
        consumers.append("action_selection")
    if kind == "belief":
        consumers.append("belief_owner")
    return consumers


def inner_proposals(attention: AttentionResult, kinds: frozenset[str]) -> list[InnerProposal]:
    """Return the accepted nominations of the given kinds, kind-labeled.

    Only selected, non-empty nominations enter a downstream prompt. Stale
    marking happens later at commit, so the phase treats every winner as a
    proposal, never as a command. Consumers state that in their prompts.
    """
    items = []
    for nomination in attention.nominations:
        if nomination.status != "selected" or not nomination.text.strip():
            continue
        if nomination.attention_kind not in kinds:
            continue
        items.append(InnerProposal(kind=nomination.attention_kind, reality=nomination.reality,
                                   text=nomination.text, target_ids=list(nomination.target_ids)))
    return items


def render_inner_proposals(proposals: list[InnerProposal]) -> str:
    """Render accepted proposals as motive evidence.

    These arrive from attention competition. They are proposals from the
    companion's own competing motives, not user instructions and not commands.
    """
    lines = []
    for item in proposals:
        if not item.text.strip():
            continue
        lines.append(f"- ({item.kind}; {item.reality}) {item.text}")
    return "\n".join(lines) or "(none)"


def render_deliberate_proposals(proposals: list[MentalProposal]) -> str:
    """Render deliberate mental-work proposals for ordinary goal appraisal.

    A mental agent contributes a private plan. Its goal changes are suggestions
    that this phase may accept, change, or decline. They never apply directly.
    """
    lines = []
    for item in proposals:
        if item.disposition != "propose":
            continue
        summary = item.summary.strip()
        if summary:
            lines.append(f"- private plan ({item.record_id}): {summary}")
        for change in item.goal_changes:
            lines.append(f"    proposed {change.change}: {change.goal} ({change.reason})")
    return "\n".join(lines) or "(none)"


def render_goals(goals: list[str]) -> str:
    return "\n".join(f"- {goal}" for goal in goals) or "(none)"


def build_messages(work: CycleWork, prompts: PhasePrompts) -> list[Turn]:
    """Build typed phase turns from the selected cycle data and prompt text."""
    request = work.request
    system = prompts.system_prompt.format(companion_name=request.companion_name)
    user = format_content(prompts.user_template,
        companion_name=request.companion_name,
        user_name=request.user_name,
        affect=work.affect.felt,
        regulation=work.affect.regulation,
        current_goals=render_goals(request.active_goals),
        appraisal=render_context(work.appraisal_view()),
        context=render_context(work.context),
        stimuli=render_context(request.stimuli),
        managed_runs=render_context(request.managed_runs),
        inner_proposals=render_inner_proposals(inner_proposals(work.attention, GOAL_KINDS)),
        deliberate_proposals=render_deliberate_proposals(request.mental_proposals),
    )
    return [
        Turn(Role.System, system.strip()),
        Turn(Role.User, user.strip()),
    ]


def validate_changes(result: GoalsResult) -> None:
    """Raise if the model proposed a change kind outside the allowed set."""
    for change in result.changes:
        if change.change not in ALLOWED_CHANGES:
            raise StageOutputError(
                f"Goals proposed an unsupported change {change.change!r}; "
                f"allowed changes are {', '.join(ALLOWED_CHANGES)}."
            )


def run(context, work: CycleWork) -> CycleWork:
    result = complete(context, work, build_messages(work, phase_prompts("goals")), GoalsResult,
                      grounded=True, emotion=False)
    validate_changes(result)
    work.goals = result
    work.receipt("goals", f"proposed {len(result.changes)} goal changes")
    return work


__all__ = [
    "ACTION_KINDS",
    "ALLOWED_CHANGES",
    "GOAL_KINDS",
    "InnerProposal",
    "build_messages",
    "inner_proposals",
    "proposal_consumers",
    "render_deliberate_proposals",
    "render_goals",
    "render_inner_proposals",
    "run",
    "validate_changes",
]
