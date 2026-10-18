"""Codelets phase: schedule and run the codelets of this cycle.

The phase:

1. loads the codelet catalogue (the request may name another catalogue file);
2. derives scheduler signals from the request and the accepted affect;
3. asks the pure scheduler which codelets run this cycle;
4. runs pure codelets in Python and each scheduled LLM codelet with one call;
5. records every proposal and every scheduling decision in ``CycleWork``.

A proposal only nominates. It never writes memory or starts an action. The
main module commits each proposal as an ``AttentionRecord`` after attention
competition. LLM codelets run in parallel. Their results join in catalogue
order, so proposal ids are stable. Exhausted repairs of one codelet answer
raise ``ModelFormatError``, and the host retries the whole cycle.

Step 5 of A19F replaces this phase with the codelet registry.
"""
from __future__ import annotations

import time
from collections.abc import Callable

from pydantic import BaseModel, Field

from pm.context_render.turns import Role, Turn, format_content
from pm.llm.concurrency import ordered_parallel_map
from pm.llm.context_window import estimate_tokens

from workflows.inference_settings import CODELET_WORKERS
from workflows.main_cognition.codelet_schedule import (
    CodeletDefinition, Decision, load_catalogue, request_signals, schedule, signal, state_signals,
)
from workflows.main_cognition.cycle_work import CycleWork
from workflows.main_cognition.model_json import PhasePrompts
from workflows.main_cognition.pathways import PATHWAY_SIGNAL_PREFIX, pathway_signals
from workflows.main_cognition.phases.model import complete, phase_prompts
from workflows.main_cognition.schemas import (
    CodeletAnswer, CodeletCost, CodeletProposal, CodeletRun, CodeletsResult, MemoryCandidate, Stimulus,
)

TEXT_LIMIT = 400
# Words, not numbers, describe a feeling inside proposal text. Numbers would
# invite the workspace to quote internal values.
INTENSITY_WORDS = ((0.7, "strong"), (0.5, "clear"), (0.0, "faint"))


class ProposalItem(BaseModel):
    """One raw codelet output before it becomes a typed proposal.

    An empty ``origin`` or ``reality`` takes the codelet default.
    """

    text: str = ""
    target_ids: list[int] = Field(default_factory=list)
    origin: str = ""
    reality: str = ""
    topic: str = ""


class RunCost(BaseModel):
    """Estimated cost of one codelet run. Pure codelets cost nothing."""

    calls: int = 0
    input_tokens_estimate: int = 0
    output_tokens_estimate: int = 0
    seconds: float = 0.0


PureCodelet = Callable[[CycleWork, CodeletDefinition, dict[str, float]], list[ProposalItem]]


def _clip(text: str, limit: int = TEXT_LIMIT) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def selected_memories(work: CycleWork) -> list[MemoryCandidate]:
    """Return the memories that the retrieve phase selected, in its order.

    Retrieval can return the current stimulus, or a record derived only from
    it, as a memory. Codelets exclude those records: the event is already a
    current event, and listing it under "memories" would make it look like
    something that happened before.
    """
    current = {item.knoxel_id for item in work.request.stimuli}
    by_id = {item.knoxel_id: item for item in work.request.memory_candidates}
    memories = []
    for key in dict.fromkeys(work.context.candidate_ids):
        if key not in by_id or key in current:
            continue
        item = by_id[key]
        sources = set(item.source_ids)
        if sources and sources <= current:
            continue
        memories.append(item)
    return memories


def source_revisions(work: CycleWork) -> dict[int, int]:
    """Map every knoxel id a codelet may read to the revision in this request."""
    revisions = dict(work.request.stimulus_revisions)
    for item in work.request.memory_candidates:
        revisions.setdefault(item.knoxel_id, item.revision)
    for goal in work.request.active_goal_records:
        revisions.setdefault(goal.knoxel_id, goal.revision)
    return revisions


def allowed_targets(work: CycleWork) -> set[int]:
    """Ids that a proposal may cite: stimuli, selected memories, and active goals."""
    ids = {item.knoxel_id for item in work.request.stimuli}
    ids |= {item.knoxel_id for item in selected_memories(work)}
    ids |= {goal.knoxel_id for goal in work.request.active_goal_records}
    return ids


def stimulus_label(stimulus: Stimulus, user_name: str) -> tuple[str, str, str]:
    """Return ``(label, origin, reality)`` for one stimulus.

    A thought stays labelled as the companion's own thought, never as an
    outside event.
    """
    kind = stimulus.kind
    if kind == "thought.created":
        return "I thought", "thought", "internal"
    if kind == "minecraft.world":
        return "Minecraft observation", "outside", "observed"
    if kind.startswith("workflow."):
        return "Task event", "outside", "observed"
    if kind == "system.lifecycle":
        return "System event", "outside", "observed"
    return f"{user_name} said", "outside", "observed"


def _intensity(value: float) -> str:
    return next(word for limit, word in INTENSITY_WORDS if value >= limit)


# ---- pure codelets -------------------------------------------------------

def direct_stimulus(work: CycleWork, definition: CodeletDefinition, signals: dict[str, float]) -> list[ProposalItem]:
    items = []
    for stimulus in work.request.stimuli:
        label, origin, reality = stimulus_label(stimulus, work.request.user_name)
        items.append(ProposalItem(text=f"{label}: {_clip(stimulus.text)}", target_ids=[stimulus.knoxel_id],
                                  origin=origin, reality=reality))
    return items


def context_recall(work: CycleWork, definition: CodeletDefinition, signals: dict[str, float]) -> list[ProposalItem]:
    return [ProposalItem(text=f"I remember: {_clip(item.text)}", target_ids=[item.knoxel_id])
            for item in selected_memories(work)]


def absence_return(work: CycleWork, definition: CodeletDefinition, signals: dict[str, float]) -> list[ProposalItem]:
    hours = signal(signals, "time.idle_hours")
    span = f"about {round(hours)} hours" if hours < 48 else f"about {round(hours / 24)} days"
    targets = [item.knoxel_id for item in work.request.stimuli if item.kind == "user.input"]
    return [ProposalItem(text=f"{work.request.user_name} is back after {span} without talking to me.",
                         target_ids=targets[:1])]


def goal_reminder(work: CycleWork, definition: CodeletDefinition, signals: dict[str, float]) -> list[ProposalItem]:
    """Nominate the first active goal. The module lists goals by salience."""
    goals = work.request.active_goal_records
    if not goals:
        return []
    goal = goals[0]
    return [ProposalItem(text=f"I still want to: {_clip(goal.text, 200)}", target_ids=[goal.knoxel_id])]


def lingering_feeling(work: CycleWork, definition: CodeletDefinition, signals: dict[str, float]) -> list[ProposalItem]:
    emotions = {key.removeprefix("emotion."): value for key, value in signals.items()
                if key.startswith("emotion.") and key not in {"emotion.strongest", "emotion.negative_strongest"}}
    if not emotions:
        return []
    name, value = max(emotions.items(), key=lambda item: item[1])
    return [ProposalItem(text=f"A {_intensity(value)} feeling of {name} is still with me.")]


def novelty_spotlight(work: CycleWork, definition: CodeletDefinition, signals: dict[str, float]) -> list[ProposalItem]:
    """Nominate the first stimulus again, labelled as new.

    The appraisal's novelty axis decides whether this codelet runs. The
    nomination adds a second, interpreted view of the same event; it never
    invents an event of its own.
    """
    for stimulus in work.request.stimuli:
        label, origin, reality = stimulus_label(stimulus, work.request.user_name)
        return [ProposalItem(text=f"Something new is happening: {_clip(stimulus.text)}",
                             target_ids=[stimulus.knoxel_id], origin=origin, reality=reality)]
    return []


def connection_lack(work: CycleWork, definition: CodeletDefinition, signals: dict[str, float]) -> list[ProposalItem]:
    """Name the low connection level.

    Loneliness has no dedicated emotion field, so this implemented need
    carries it instead of an invented label. The scheduler gate keeps this
    quiet unless the accepted connection level is genuinely low. A missing
    connection signal counts as full connection.
    """
    level = signals["needs.connection"] if "needs.connection" in signals else 1.0
    if level > 0.35:
        return []
    return [ProposalItem(text="There has been too little real back-and-forth. I miss the contact.",
                         origin="state", reality="internal")]


PURE_CODELETS: dict[str, PureCodelet] = {
    "direct_stimulus": direct_stimulus,
    "context_recall": context_recall,
    "absence_return": absence_return,
    "goal_reminder": goal_reminder,
    "lingering_feeling": lingering_feeling,
    "novelty_spotlight": novelty_spotlight,
    "connection_lack": connection_lack,
}


# ---- LLM codelets --------------------------------------------------------

def render_events(work: CycleWork) -> str:
    lines = []
    for stimulus in work.request.stimuli:
        label, _, _ = stimulus_label(stimulus, work.request.user_name)
        lines.append(f"- [id {stimulus.knoxel_id}] {label}: {_clip(stimulus.text)}")
    return "\n".join(lines) or "(no current events)"


def render_memories(work: CycleWork) -> str:
    lines = []
    for item in selected_memories(work):
        when = f" (from {item.timestamp})" if item.timestamp else ""
        lines.append(f"- [id {item.knoxel_id}]{when} {_clip(item.text)}")
    return "\n".join(lines) or "(no memories)"


def render_goals(work: CycleWork) -> str:
    lines = [f"- [id {goal.knoxel_id}] {_clip(goal.text, 200)}" for goal in work.request.active_goal_records]
    return "\n".join(lines) or "(no active goals)"


def build_messages(work: CycleWork, definition: CodeletDefinition, prompts: PhasePrompts) -> list[Turn]:
    companion = work.request.companion_name
    user = work.request.user_name
    instructions = definition.instructions.strip().format(companion_name=companion, user_name=user)
    system = prompts.system_prompt.format(companion_name=companion, user_name=user)
    user_message = format_content(prompts.user_template,
        companion_name=companion, user_name=user, codelet_name=definition.name,
        affect=work.affect.felt,
        events=render_events(work), memories=render_memories(work), goals=render_goals(work),
    )
    job = prompts.job_template.format(codelet_name=definition.name, instructions=instructions,
                                      reality_rule=prompts.reality_rules[definition.reality])
    # Every codelet shares the exact prefix before its concrete job.
    return [Turn(Role.System, system.strip()), Turn(Role.User, user_message.strip()), Turn(Role.User, job.strip())]


def run_llm_codelet(context, work: CycleWork, definition: CodeletDefinition,
                    prompts: PhasePrompts) -> tuple[list[ProposalItem], RunCost]:
    """Make one model call and return ``(proposal items, cost)``.

    Codelet calls get no grounding turn. Each is one small question for a
    small model. Later phases read the grounding before public speech.
    """
    messages = build_messages(work, definition, prompts)
    started = time.perf_counter()
    answer = complete(context, work, messages, CodeletAnswer)
    input_tokens, _ = estimate_tokens("", messages)
    output_tokens, _ = estimate_tokens("", [Turn(Role.Assistant, answer.model_dump_json())])
    cost = RunCost(calls=1, input_tokens_estimate=input_tokens, output_tokens_estimate=output_tokens,
                   seconds=round(time.perf_counter() - started, 3))
    if not answer.fires or not answer.text.strip():
        return [], cost
    return [ProposalItem(text=_clip(answer.text), topic=answer.topic.strip(), target_ids=list(answer.target_ids))], cost


# ---- phase ---------------------------------------------------------------

def make_proposal(item: ProposalItem, decision: Decision, nomination_id: str, *, allowed: set[int],
                  revisions: dict[int, int], thought_only: bool) -> CodeletProposal:
    """Turn one codelet output item into a typed proposal.

    Targets outside ``allowed`` move to ``dropped_target_ids``. A cycle that
    only a private thought started cannot produce an outside-origin or
    state-origin proposal. This keeps rumination distinguishable from events.
    """
    definition = decision.definition
    targets = list(item.target_ids)
    kept = [key for key in dict.fromkeys(targets) if key in allowed]
    origin = item.origin or definition.origin
    if thought_only and origin in {"outside", "state"}:
        origin = "thought"
    topic = item.topic.strip()
    if definition.proposal != "belief":
        # Only a belief proposal carries a topic. A model that answers a
        # different codelet with a topic cannot smuggle a belief key in.
        topic = ""
    return CodeletProposal(
        nomination_id=nomination_id, codelet=definition.name, family=definition.family,
        proposal_kind=definition.proposal, origin=origin,
        reality=item.reality or definition.reality, text=item.text,
        target_ids=kept, cause=list(decision.cause), activation=decision.activation,
        source_revisions={str(key): revisions[key] for key in kept if key in revisions},
        dropped_target_ids=[key for key in targets if key not in allowed],
        topic=topic,
    )


def collect(context, work: CycleWork, prompts: PhasePrompts) -> CodeletsResult:
    """Schedule, run, and record codelets for one cycle.

    Raises ``ValueError`` when the catalogue names a pure codelet that has no
    Python function. That is a configuration error, so it stops the cycle.
    """
    request = work.request
    loaded = load_catalogue(request.codelet_catalogue or None)
    missing = [item.name for item in loaded.catalogue.codelets
               if item.kind == "pure" and item.name not in PURE_CODELETS]
    if missing:
        raise ValueError(f"Pure codelets without a Python function: {', '.join(missing)}.")
    loop = request.attention_loop
    pathways = loaded.catalogue.pathway_map()
    precursors = request.pathway_precursors
    latest_tick = request.pathway_latest_tick or max((item.tick_id for item in precursors), default=0)
    # Pathway activation comes from committed precursor records, derived fresh
    # by the main module on every cycle. A pathway without a fresh precursor
    # stays at 0, which closes every gate and boost on its signal.
    pathway = pathway_signals(precursors, pathways, latest_tick=latest_tick)
    # Codelets read the accepted mental state of this cycle, not the previous
    # state, so an emotion that this event just caused can already activate them.
    events = request_signals(
        request.stimuli,
        active_goal_count=len(request.active_goal_records or request.active_goals),
        selected_memory_count=len(work.context.candidate_ids),
        seconds_since_user_input=request.seconds_since_user_input,
        regulation=work.affect.regulation,
        appraisal=work.appraisal,
    )
    signals = {**events, **state_signals(work.state.mental_state), **pathway}
    decisions, policy = schedule(loaded, signals, request.codelet_history, loop=loop)
    allowed = allowed_targets(work)
    revisions = source_revisions(work)
    thought_only = bool(signal(signals, "stimulus.thought")) and not (
        signal(signals, "stimulus.user") or signal(signals, "stimulus.workflow") or signal(signals, "stimulus.system"))
    proposals: list[CodeletProposal] = []
    runs: list[CodeletRun] = []

    def execute(decision: Decision) -> tuple[list[ProposalItem], RunCost]:
        definition = decision.definition
        if definition.kind == "pure":
            return PURE_CODELETS[definition.name](work, definition, signals), RunCost()
        return run_llm_codelet(context, work, definition, prompts)

    results = iter(ordered_parallel_map(execute, [item for item in decisions if item.run],
                                       workers=CODELET_WORKERS, cancel=context.runtime.stop_event))
    for decision in decisions:
        definition = decision.definition
        run = CodeletRun(codelet=definition.name, kind=definition.kind, family=definition.family,
                         status="skipped", reason=decision.reason, activation=decision.activation,
                         cause=decision.cause)
        runs.append(run)
        if not decision.run:
            continue
        # Assign IDs only after joining results in catalogue order.
        items, cost = next(results)
        run.calls = cost.calls
        run.input_tokens_estimate = cost.input_tokens_estimate
        run.output_tokens_estimate = cost.output_tokens_estimate
        run.seconds = cost.seconds
        for item in items:
            proposal = make_proposal(item, decision, f"n{len(proposals) + 1}", allowed=allowed,
                                     revisions=revisions, thought_only=thought_only)
            proposals.append(proposal)
            run.proposal_ids.append(proposal.nomination_id)
        run.status = "ran" if run.proposal_ids else "silent"
        run.reason = "scheduled"
    cost = CodeletCost(
        llm_calls=sum(run.calls for run in runs),
        input_tokens_estimate=sum(run.input_tokens_estimate for run in runs),
        output_tokens_estimate=sum(run.output_tokens_estimate for run in runs),
        seconds=round(sum(run.seconds for run in runs), 3),
    )
    return CodeletsResult(proposals=proposals, runs=runs, catalogue_version=loaded.catalogue.version,
                          catalogue_digest=loaded.digest, loop=loop, policy=policy,
                          signals=signals, cost=cost,
                          pathway_signals={key.removeprefix(PATHWAY_SIGNAL_PREFIX): value
                                           for key, value in pathway.items()})


def run(context, work: CycleWork) -> CycleWork:
    result = collect(context, work, phase_prompts("codelets"))
    work.codelets = result
    ran = [item.codelet for item in result.runs if item.status in {"ran", "silent"}]
    work.receipt("codelets", f"ran {len(ran)} codelets ({', '.join(ran) or 'none'}), "
                             f"{len(result.proposals)} proposals, {result.cost.llm_calls} model calls")
    return work


__all__ = [
    "PURE_CODELETS",
    "ProposalItem",
    "allowed_targets",
    "build_messages",
    "collect",
    "make_proposal",
    "render_events",
    "render_goals",
    "render_memories",
    "run",
    "run_llm_codelet",
    "selected_memories",
    "source_revisions",
    "stimulus_label",
]
