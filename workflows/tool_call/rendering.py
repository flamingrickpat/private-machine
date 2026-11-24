from __future__ import annotations

from dataclasses import dataclass
from operator import attrgetter
import re
from pm.context_render.budget import TokenEstimator, evidence_excerpt
from pm.context_render.images import ImageCatalog, resolve_media
from pm.context_render.prompt import RenderSettings, RenderedPrompt, EvidenceDecision, require_rendered_prompt
from pm.context_render.records import render_record
from pm.context_render.turns import Turn, Role
from workflows.tool_call.schemas import ToolCallRequest, ToolContextEvidence


@dataclass(frozen=True, slots=True)
class ToolSessionRenderInput:
    request: ToolCallRequest
    instruction: str
    server_names: tuple[str, ...]
    work_root: str
    skill_text: str


@dataclass(frozen=True, slots=True)
class RankedToolEvidence:
    evidence: ToolContextEvidence
    score: float


def render_tool_session(input: ToolSessionRenderInput, settings: RenderSettings) -> RenderedPrompt:
    request = input.request
    context = request.tool_call_context
    source = context.source_action
    estimator = TokenEstimator(estimate_factor=settings.estimate_factor)
    source_count = len(source.stimuli) if source is not None else 0
    selected_count = source_count
    retained = max(len(context.character_card), len(input.instruction), len(request.tool_goal),
                   len(context.original_assignment), len(context.current_assignment))
    if source is not None:
        retained = max(retained, len(source.affect), max((len(item.text) for item in source.stimuli), default=0))
    decisions: list[EvidenceDecision] = []
    seen: set[int] = set()
    reduced = False
    while True:
        lines = [f"You are {context.companion_name}. Carry out your selected decision with your own tools.",
            "Character:\n" + evidence_excerpt(context.character_card.strip(), retained),
            "Selected decision:\n" + evidence_excerpt(input.instruction, retained)]
        if request.tool_goal:
            lines.append("Concrete tool goal: " + evidence_excerpt(request.tool_goal, retained))
        if context.related_run_id:
            lines.append("Related worker " + context.related_run_id + ". Original assignment and site:\n"
                         + evidence_excerpt(context.original_assignment, retained))
            if context.current_assignment != context.original_assignment:
                lines.append("Current assignment:\n" + evidence_excerpt(context.current_assignment, retained))
        if reduced and input.work_root:
            lines.append("Source detail was abbreviated. Check inbox/request.yaml before an external action "
                         "if its goal or constraints are incomplete. Report uncertainty if you cannot establish them.")
        decisions.clear()
        seen.clear()
        if source is not None:
            lines.append(f"Decision action {source.action_schema_id}; source tick {source.tick_id}.")
            for index, stimulus in enumerate(source.stimuli):
                included = index >= source_count - selected_count
                if included:
                    lines.append(f"Current cause [{stimulus.knoxel_id}; {stimulus.kind}; {stimulus.source}]:\n"
                                 + evidence_excerpt(stimulus.text, retained))
                    seen.add(stimulus.knoxel_id)
                decisions.append(EvidenceDecision(stimulus.knoxel_id, 100, included,
                    "cause of selected tool goal" if included else "omitted cause retained in task input"))
            if source.affect:
                lines.append("Feeling when selected: " + evidence_excerpt(source.affect, retained))
        else:
            lines.append("This task starts from the operator's instruction and current accepted context.")
        lines.append("MCP servers: " + ", ".join(input.server_names) + ". Tool names use server_name prefixes.")
        lines.append("Ground external work in actual tool returns. Resolve a named site or object against its "
                     "original task and check it before acting. If its identity is ambiguous, inspect or report "
                     "the ambiguity. Keep the quantity collected separate from the total held. Report the "
                     "actual action, target and instrument. If a requested measurement is unavailable, say so. "
                     "A substitute experiment can establish only what its instrument actually measured.")
        if input.work_root:
            lines.append(f"Work folder: {input.work_root}. Place host artifacts directly in outbox/.")
        if input.work_root:
            lines.append("Return the terminal JSON object specified by the tool-session instructions.")
        turns = resolve_media((Turn(Role.System, input.skill_text), Turn(Role.User, "\n\n".join(lines))), ImageCatalog(context.images))
        if estimator.estimate(turns) <= settings.input_ceiling:
            break
        reduced = True
        if retained:
            retained //= 2
        elif selected_count:
            selected_count //= 2
        else:
            break
    required = require_rendered_prompt("tool goal session", turns, settings, tuple(decisions))
    remaining = settings.input_ceiling - required.estimated_tokens
    selected_groups: list[str] = []
    if source is not None and source.workspace:
        workspace = "Subjective experience when deciding:\n" + source.workspace
        trial = resolve_media((turns[0], Turn(Role.User, "\n\n".join([*lines, workspace]))), ImageCatalog(context.images))
        if estimator.estimate(trial) - required.estimated_tokens <= remaining * 0.2:
            lines.append(workspace)
            turns = trial
            selected_groups.append("decision experience")
    terms = set(re.findall(r"[a-zA-Z]{4,}", input.instruction.lower() + " " + request.tool_goal.lower()))
    ranked: list[RankedToolEvidence] = []
    if source is not None:
        for evidence in source.evidence:
            record = evidence.record
            overlap = len(terms.intersection(re.findall(r"[a-zA-Z]{4,}", record.text.lower())))
            recency = 1 / (1 + max(0, context.captured_tick_id - record.tick_id))
            score = overlap * 20 + evidence.activation * 5 + recency + int(evidence.in_recent_tail)
            ranked.append(RankedToolEvidence(evidence, score))
    ranked.sort(key=attrgetter("score"), reverse=True)
    source_ceiling = settings.input_ceiling - int(remaining * 0.2)
    for candidate in ranked:
        record = candidate.evidence.record
        if record.knoxel_id in seen:
            continue
        block = "Goal-relevant remembered source:\n" + render_record(record).text
        trial = resolve_media((turns[0], Turn(Role.User, "\n\n".join([*lines, block]))), ImageCatalog(context.images))
        included = estimator.estimate(trial) <= source_ceiling
        decisions.append(EvidenceDecision(record.knoxel_id, candidate.score, included,
            "goal overlap, accepted attention and recency" if included else "outside source evidence allocation"))
        if included:
            lines.append(block)
            turns = trial
            seen.add(record.knoxel_id)
            selected_groups.append(f"source {record.knoxel_id}")
    used_summaries: set[str] = set()
    history: list[str] = []
    for moment in reversed(context.gwt_turns):
        if not moment.summary.strip() or moment.summary in used_summaries:
            continue
        block = f"Earlier accepted moment [tick {moment.tick_id}]: {moment.summary}"
        candidate_history = [block, *history]
        trial = resolve_media((turns[0], Turn(Role.User, "\n\n".join([*lines, *candidate_history]))), ImageCatalog(context.images))
        if estimator.estimate(trial) <= settings.input_ceiling:
            history = candidate_history
            used_summaries.add(moment.summary)
            selected_groups.append(f"moment {moment.tick_id}")
    turns = resolve_media((turns[0], Turn(Role.User, "\n\n".join([*lines, *history]))), ImageCatalog(context.images))
    return require_rendered_prompt("tool goal session", turns, settings, tuple(decisions),
        selected_groups=tuple(selected_groups))
