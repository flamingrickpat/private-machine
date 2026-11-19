from __future__ import annotations

from dataclasses import dataclass
from operator import attrgetter
import re
from pm.context_render.budget import TokenEstimator
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
    lines = [f"You are {context.companion_name}. Carry out your selected decision with your own tools.",
        "Character:\n" + context.character_card.strip(), "Selected decision:\n" + input.instruction]
    if request.tool_goal:
        lines.append("Concrete tool goal: " + request.tool_goal)
    source = context.source_action
    decisions: list[EvidenceDecision] = []
    seen: set[int] = set()
    if source is not None:
        lines.append(f"Decision action {source.action_schema_id}; source tick {source.tick_id}.")
        for stimulus in source.stimuli:
            lines.append(f"Current cause [{stimulus.knoxel_id}; {stimulus.kind}; {stimulus.source}]:\n{stimulus.text}")
            seen.add(stimulus.knoxel_id)
            decisions.append(EvidenceDecision(stimulus.knoxel_id, 100, True, "current cause of selected tool goal"))
        if source.affect:
            lines.append("Feeling when selected: " + source.affect)
    else:
        lines.append("This task starts from the operator's instruction and current accepted context.")
    lines.append("MCP servers: " + ", ".join(input.server_names) + ". Tool names use server_name prefixes.")
    lines.append(f"Work folder: {input.work_root}. Place host artifacts directly in outbox/.")
    lines.append("Return the terminal JSON object specified by the tool-session instructions.")
    turns = resolve_media((Turn(Role.System, input.skill_text), Turn(Role.User, "\n\n".join(lines))), ImageCatalog(context.images))
    required = require_rendered_prompt("tool goal session", turns, settings, tuple(decisions))
    estimator = TokenEstimator(estimate_factor=settings.estimate_factor)
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
