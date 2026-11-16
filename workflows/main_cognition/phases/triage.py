"""Triage phase: rate each stimulus and each stimulus group.

The call is story-framed (A19F step 7): it starts with the shared story
prefix and ends with the triage instruction. The model classifies each
stimulus by relevance, urgency, and input kind, and sets the codelet budget
of the cycle. This decides how much effort the rest of the cycle spends.
The phase then gives each stimulus group the highest category of its
stimuli. A group whose stimuli are all ``Insignificant`` is marked
insignificant and gets no decision. The phase never writes memory.

Triage must stay fast, so its prompt never plans for more than
:data:`TRIAGE_CONTEXT_WINDOW_TOKENS` of serving context, even when the
configured model window is larger. :func:`fit_messages` keeps the story
frame, the new events, and the newest dialogue lines of each partner.
Other shown workspace records and retrieved evidence compete by
:class:`TriageContextTier` through the shared selector in
:mod:`pm.context_render.budget`: accepted entity context first, then the
latest dialogue, the strongest narratives, the latest thoughts, other
workspace records, and retrieved evidence last. A prompt that already fits
keeps the exact bytes of the unbounded prompt. Early input triage
(:mod:`workflows.input_triage.steps`) uses the same fitting.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Literal, cast

from pm_workflows.python_role import RoleContext

from pm.constants import CODELET_BUDGET_DEFAULT, CODELET_BUDGET_MAX
from pm.context_render.budget import ContextBudget, ContextGroup, PromptSelection, select_rendered_context
from pm.context_render.turns import Content, Turn, format_content
from pm.llm.contracts import ContextMeasurer, ModelTool, WorkflowCompletion
from pm.llm.structured import context_measure
from pm.records.workspace import WorkspaceShard

from workflows.main_cognition.cycle_work import CycleWork
from workflows.main_cognition.model_json import PhasePrompts
from workflows.main_cognition.phases.intake import group_key, group_stimuli
from workflows.main_cognition.phases.model import complete, grounding_turn, phase_prompts
from workflows.main_cognition.schemas import TRIAGE_CATEGORIES, TriageResult
from workflows.main_cognition.story_prefix import StoryPrefix, shown_shards

#: Triage categories from the least to the most important.
CATEGORIES = TRIAGE_CATEGORIES

#: Largest serving context, in tokens, that one triage prompt may plan for.
#: The configured input fraction of this window is the triage input ceiling.
#: A smaller configured model window wins.
TRIAGE_CONTEXT_WINDOW_TOKENS = 32_768

#: Newest dialogue lines of each partner tail that every triage prompt keeps.
TRIAGE_REQUIRED_DIALOGUE_LINES = 4

#: Narratives with the highest activation that rank above the latest thoughts.
#: The other narratives compete as ordinary workspace records.
TRIAGE_LEADING_NARRATIVES = 3


class TriageContextTier(IntEnum):
    """Importance class of one optional triage record. A higher tier enters the prompt first.

    Inside one tier, newer tail lines and higher activation or retrieval
    scores enter first.
    """

    RetrievedEvidence = 0
    WorkspaceRecord = 1
    LatestThought = 2
    LeadingNarrative = 3
    LatestDialogue = 4
    EntityContext = 5


@dataclass(frozen=True, slots=True)
class TriageCandidate:
    """One optional record of the triage prompt and where the prompt shows it."""

    record_id: int
    place: Literal["workspace", "evidence"]
    tier: TriageContextTier


@dataclass(frozen=True, slots=True)
class TriageContext:
    """The selectable context of one triage prompt.

    ``required_ids`` are workspace records that every triage prompt shows.
    ``candidates`` maps each optional context group name to its record.
    """

    required_ids: frozenset[int]
    candidates: dict[str, TriageCandidate]
    groups: tuple[ContextGroup, ...]


@dataclass(frozen=True, slots=True)
class TriagePrompt:
    """One fitted triage prompt and the selection that produced it."""

    turns: list[Turn]
    selection: PromptSelection
    input_ceiling: int

    def summary(self) -> str:
        """Describe the fitting for the phase receipt."""
        optional_shown = len(self.selection.included) - 1  # Without the required group.
        return (f"{optional_shown} optional records shown, {len(self.selection.omitted)} omitted, "
                f"estimated {self.selection.estimated_tokens} of {self.input_ceiling} input tokens")


def render_stimuli(work: CycleWork, prefix: StoryPrefix) -> str:
    """Render each stimulus with its id, source, kind, text, and the story part of its group."""
    lines = []
    for stimulus in work.request.stimuli:
        part = prefix.label(group_key(stimulus, work.request))
        lines.append(f"- id={stimulus.knoxel_id} source={stimulus.source} "
                     f"kind={stimulus.kind} text={stimulus.text!r} part={part}")
    return "\n".join(lines) or "(none)"


def build_messages(work: CycleWork, prompts: PhasePrompts, prefix: StoryPrefix) -> list[Turn]:
    """Return the shared story prefix and then the triage instruction as the last turn."""
    request = work.request
    user = format_content(prefix.fill(prompts.user_template),
        state_summary=work.state_summary or "(no feeling recorded yet)",
        active_goals="; ".join(request.active_goals) or "(nothing in particular)",
        stimuli=render_stimuli(work, prefix),
        budget_max=CODELET_BUDGET_MAX,
        budget_default=CODELET_BUDGET_DEFAULT,
    )
    return prefix.ask(user.text)


def rate_groups(work: CycleWork) -> None:
    """Copy the triage categories onto the stimulus groups of the state.

    A group gets the highest category of its stimuli. A group without a
    rated stimulus keeps an empty category and stays significant.
    """
    for group in work.state.stimulus_groups:
        categories = [item.category for item in work.triage.items if item.knoxel_id in group.stimulus_ids]
        if not categories:
            continue
        ranks = [CATEGORIES.index(name) for name in categories]
        group.category = CATEGORIES[max(ranks)]
        group.insignificant = max(ranks) == 0


def triage_budget(context: RoleContext) -> ContextBudget:
    """Return the triage input budget: min(32k, serving context) times the configured input fraction."""
    assert context.runtime is not None, "Triage requires an active workflow runtime."
    tools = context.runtime.tools
    if ModelTool.MeasureContext in tools:
        measure: ContextMeasurer | None = cast(ContextMeasurer, tools[ModelTool.MeasureContext])
    else:
        measure = context_measure(cast(WorkflowCompletion, tools[ModelTool.Complete]))
    if measure is None:
        raise ValueError("Triage requires context measurement from the selected model.")
    capacity = measure([], cancel=context.runtime.stop_event)
    window = min(TRIAGE_CONTEXT_WINDOW_TOKENS, capacity["context_window"])
    return ContextBudget(context_window=window, input_fraction=capacity["input_fraction"])


def _tier_salience(tier: TriageContextTier, position: int, count: int) -> float:
    """Return the salience of the record at ``position`` of ``count`` ascending records of one tier.

    The value stays below the next tier, so every record of a higher tier
    enters the prompt first.
    """
    return tier + 0.9 * (position + 1) / count


def _offer(offers: dict[int, tuple[TriageContextTier, float]], record_id: int,
           tier: TriageContextTier, salience: float) -> None:
    """Keep the strongest offer of a record that the story shows in more than one place."""
    if record_id in offers and offers[record_id][1] >= salience:
        return
    offers[record_id] = (tier, salience)


def _shard_offers(work: CycleWork, shard: WorkspaceShard, required_ids: set[int],
                  offers: dict[int, tuple[TriageContextTier, float]]) -> None:
    """Rank the tail and the items of one shown shard for the triage prompt."""
    # ``tail_ids`` is oldest first, so a later position is a newer line.
    tail_ids = list(dict.fromkeys(shard.tail_ids))
    if shard.key.kind == "entity":
        tail_tier = TriageContextTier.LatestDialogue
        required_ids.update(tail_ids[-TRIAGE_REQUIRED_DIALOGUE_LINES:])
    else:
        tail_tier = TriageContextTier.LatestThought
    for position, record_id in enumerate(tail_ids):
        _offer(offers, record_id, tail_tier, _tier_salience(tail_tier, position, len(tail_ids)))

    items_by_activation = sorted(shard.items, key=lambda item: item.activation_at_entry)
    strongest_narratives = [item.record_id for item in reversed(items_by_activation)
                            if work.records[item.record_id].record_class == "Narrative"]
    leading_narratives = set(strongest_narratives[:TRIAGE_LEADING_NARRATIVES])
    for position, item in enumerate(items_by_activation):
        if item.required_by == "entity_context":
            tier = TriageContextTier.EntityContext
        elif item.record_id in leading_narratives:
            tier = TriageContextTier.LeadingNarrative
        else:
            tier = TriageContextTier.WorkspaceRecord
        _offer(offers, item.record_id, tier, _tier_salience(tier, position, len(items_by_activation)))


def triage_context(work: CycleWork) -> TriageContext:
    """Return the required records and the ranked optional groups of one triage prompt.

    The new events are always part of the prompt and need no group. The
    newest dialogue lines of each partner shard are required. Every other
    shown workspace record and every retrieved evidence row is optional.
    """
    required_ids: set[int] = set()
    offers: dict[int, tuple[TriageContextTier, float]] = {}
    for shard in shown_shards(work):
        _shard_offers(work, shard, required_ids, offers)

    candidates: dict[str, TriageCandidate] = {}
    groups = [ContextGroup(name="required triage context", content=Content(""), salience=0, required=True,
                           source_ids=tuple(sorted(required_ids)))]
    for record_id, (tier, salience) in offers.items():
        if record_id in required_ids:
            continue
        name = f"{tier.name}:{record_id}"
        candidates[name] = TriageCandidate(record_id=record_id, place="workspace", tier=tier)
        groups.append(ContextGroup(name=name, content=Content(""), salience=salience, source_ids=(record_id,)))

    public = work.request.public_context
    if public is not None:
        stimulus_texts = {item.knoxel_id: item.text for item in work.request.stimuli}
        # grounding_turn drops the evidence rows that repeat a current stimulus word for word.
        rows = [row for row in public.evidence
                if row.knoxel_id not in stimulus_texts or row.text != stimulus_texts[row.knoxel_id]]
        rows_by_score = sorted(rows, key=lambda row: row.score)
        evidence_tier = TriageContextTier.RetrievedEvidence
        for position, row in enumerate(rows_by_score):
            name = f"{evidence_tier.name}:{row.knoxel_id}"
            candidates[name] = TriageCandidate(record_id=row.knoxel_id, place="evidence", tier=evidence_tier)
            groups.append(ContextGroup(name=name, content=Content(""),
                                       salience=_tier_salience(evidence_tier, position, len(rows_by_score)),
                                       source_ids=(row.knoxel_id,)))
    return TriageContext(required_ids=frozenset(required_ids), candidates=candidates, groups=tuple(groups))


def selected_work(work: CycleWork, workspace_ids: set[int], evidence_ids: set[int]) -> CycleWork:
    """Return a shallow view of ``work`` whose shards and evidence hold only the selected records.

    The view exists only to render the prompt. The phase classifies with,
    and returns, the complete ``work``. Shards that the story does not show
    are emptied too; no prompt reads them.
    """
    def trimmed(shard: WorkspaceShard) -> WorkspaceShard:
        items = [item for item in shard.items if item.record_id in workspace_ids]
        tail_ids = [record_id for record_id in shard.tail_ids if record_id in workspace_ids]
        return shard.model_copy(update={"items": items, "tail_ids": tail_ids})

    workspace = work.state.workspace
    entity_shards = {key: trimmed(shard) for key, shard in workspace.entity_shards.items()}
    workflow_shards = {key: trimmed(shard) for key, shard in workspace.workflow_shards.items()}
    selected_workspace = workspace.model_copy(update={"self_shard": trimmed(workspace.self_shard),
                                                      "entity_shards": entity_shards,
                                                      "workflow_shards": workflow_shards})
    request = work.request
    public = request.public_context
    if public is not None:
        evidence = [row for row in public.evidence if row.knoxel_id in evidence_ids]
        # With evidence, grounding shows no memory list. Omitted evidence must not bring it back.
        memories = [] if public.evidence else public.memories
        selected_public = public.model_copy(update={"evidence": evidence, "memories": memories})
        request = request.model_copy(update={"public_context": selected_public})
    state = work.state.model_copy(update={"workspace": selected_workspace})
    return work.model_copy(update={"request": request, "state": state})


def omission_notice(omitted: Sequence[str]) -> str:
    """Tell the model, in story words, that triage left out less important records."""
    if not omitted:
        return ""
    return (f"\n\n(For a quick judgment, {len(omitted)} older or less important story records were left out. "
            "Judge from what is shown.)")


def fit_messages(context: RoleContext, work: CycleWork, *, appended: Sequence[Turn] = ()) -> TriagePrompt:
    """Return the complete grounded triage prompt within the triage input budget.

    The prompt is the story prefix, the private grounding turn, the triage
    instruction, and then ``appended`` (early triage adds its entity
    extraction turn). Required content that does not fit raises
    ``RequiredContextTooLarge``; nothing is clipped. The selection receipt is
    written to ``scratch/triage-context-selection.json``.
    """
    assert context.runtime is not None, "Triage requires an active workflow runtime."
    prompt = fit_prompt(work, triage_budget(context), appended=appended)
    context.runtime.environment.write_text("scratch/triage-context-selection.json",
                                           prompt.selection.receipt().model_dump_json(indent=2))
    return prompt


def fit_prompt(work: CycleWork, budget: ContextBudget, *, appended: Sequence[Turn] = ()) -> TriagePrompt:
    """Select the triage context for ``budget`` and render the prompt. No model or file I/O happens here."""
    selectable = triage_context(work)
    prompts = phase_prompts("triage")

    def assemble(chosen: Sequence[ContextGroup], omitted: Sequence[str]) -> list[Turn]:
        workspace_ids = set(selectable.required_ids)
        evidence_ids: set[int] = set()
        for group in chosen:
            if group.required:
                continue
            candidate = selectable.candidates[group.name]
            if candidate.place == "workspace":
                workspace_ids.add(candidate.record_id)
            else:
                evidence_ids.add(candidate.record_id)
        shown = selected_work(work, workspace_ids, evidence_ids)
        prefix = StoryPrefix(shown)
        turns = build_messages(shown, prompts, prefix)
        notice = omission_notice(omitted)
        if notice:
            instruction = turns[-1]
            turns[-1] = Turn(instruction.role, instruction.content + notice)
        # The same position as the grounding insert of ``complete``, so a fitting prompt keeps its bytes.
        grounding = grounding_turn(shown, emotion=True)
        if grounding is not None:
            turns.insert(len(prefix.turns), grounding)
        return [*turns, *appended]

    selection = select_rendered_context(selectable.groups, budget.input_ceiling, assemble, estimator=budget)
    return TriagePrompt(turns=list(selection.turns), selection=selection, input_ceiling=budget.input_ceiling)


def run(context: RoleContext, work: CycleWork) -> CycleWork:
    """Reuse accepted input triage and classify only the remaining stimuli."""
    early = work.request.early_triage
    classified_ids = {item.knoxel_id for item in early.items} if early is not None else set()
    remaining = [item for item in work.request.stimuli if item.knoxel_id not in classified_ids]
    if remaining:
        pending = work.model_copy(deep=True)
        pending.request.stimuli = remaining
        pending.state.stimulus_groups = group_stimuli(pending.request)
        prompt = fit_messages(context, pending)
        work.receipt("triage-context", prompt.summary())
        result = complete(context, pending, prompt.turns, TriageResult,
            validation_context={"stimulus_ids": [item.knoxel_id for item in remaining]})
        if early is not None:
            result = TriageResult(items=[*early.items, *result.items],
                                  codelet_budget=max(early.codelet_budget, result.codelet_budget))
    else:
        if early is None:
            raise ValueError("Triage requires stimuli or an accepted early assessment.")
        result = early
    work.triage = result
    rate_groups(work)
    work.receipt("triage", f"triaged {len(result.items)} stimuli")
    return work


__all__ = ["CATEGORIES", "TRIAGE_CONTEXT_WINDOW_TOKENS", "TriageContextTier", "TriagePrompt", "build_messages",
           "fit_messages", "fit_prompt", "rate_groups", "render_stimuli", "run", "triage_budget", "triage_context"]
