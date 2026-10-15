"""Codelets role.

Entry point: ``run(context)``. The role:

1. loads the codelet catalogue (the request may name another catalogue file);
2. derives scheduler signals from the request and the accepted affect;
3. asks the pure scheduler which codelets run this cycle;
4. runs pure codelets in Python and each scheduled LLM codelet with one call;
5. writes every proposal and every scheduling decision to ``outbox/report.yaml``.

A proposal only nominates. It never writes memory or starts an action. The
main module commits each proposal as an ``AttentionRecord`` after attention
competition. A failed LLM codelet is recorded as ``failed`` and does not stop
the cycle: one quiet part of the mind must not break the whole reply.
Transport errors (``ModelError``) still propagate, because every later stage
needs the same model.
"""
from __future__ import annotations

from pm.llm.concurrency import ordered_parallel_map
from workflows.inference_settings import CODELET_WORKERS

import time
from pathlib import Path
from pm.context_render.turns import Role, Turn, format_content

from pm.llm.context_window import estimate_tokens
from workflows.main_cognition.codelet_schedule import (
    CodeletDefinition, load_catalogue, request_signals, schedule,
)
from workflows.main_cognition.pathways import PATHWAY_SIGNAL_PREFIX, pathway_signals
from workflows.main_cognition.schemas import CodeletAnswer, CodeletProposal, CodeletRun, CodeletsResult
from workflows.main_cognition.stage_support import (
    StageOutputError, complete_json, load_yaml, read_request, write_report,
)

WORKFLOW_DIR = Path(__file__).resolve().parent
TEXT_LIMIT = 400
# Words, not numbers, describe a feeling inside proposal text. Numbers would
# invite the workspace to quote internal values.
INTENSITY_WORDS = ((0.7, "strong"), (0.5, "clear"), (0.0, "faint"))


def load_prompts(path: Path | None = None) -> dict:
    return load_yaml(path or WORKFLOW_DIR / "prompts.yaml")


def _clip(text: str, limit: int = TEXT_LIMIT) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _memories(request: dict) -> list[dict]:
    """Return the memories that the context stage selected, in its order.

    Retrieval can return the current stimulus, or a record derived only from
    it, as a memory. Codelets exclude those records: the event is already a
    current event, and listing it under "memories" would make it look like
    something that happened before.
    """
    current = {int(item["knoxel_id"]) for item in request.get("stimuli", [])}
    by_id = {int(item["knoxel_id"]): item for item in request.get("memory_candidates", [])}
    chosen = [int(key) for key in (request.get("context") or {}).get("candidate_ids", [])]
    memories = []
    for key in dict.fromkeys(chosen):
        item = by_id.get(key)
        sources = {int(source) for source in (item or {}).get("source_ids", [])}
        if item is None or key in current or (sources and sources <= current):
            continue
        memories.append(item)
    return memories


def source_revisions(request: dict) -> dict[int, int]:
    """Map every knoxel id a codelet may read to the revision in this request."""
    revisions = {int(key): int(value) for key, value in (request.get("stimulus_revisions") or {}).items()}
    for item in request.get("memory_candidates", []):
        revisions.setdefault(int(item["knoxel_id"]), int(item.get("revision", 1)))
    for goal in request.get("active_goal_records", []):
        revisions.setdefault(int(goal["knoxel_id"]), int(goal.get("revision", 1)))
    return revisions


def allowed_targets(request: dict) -> set[int]:
    """Ids that a proposal may cite: stimuli, selected memories, and active goals."""
    ids = {int(item["knoxel_id"]) for item in request.get("stimuli", [])}
    ids |= {int(item["knoxel_id"]) for item in _memories(request)}
    ids |= {int(goal["knoxel_id"]) for goal in request.get("active_goal_records", [])}
    return ids


def stimulus_label(stimulus: dict, user_name: str) -> tuple[str, str, str]:
    """Return ``(label, origin, reality)`` for one stimulus.

    A thought stays labelled as the companion's own thought, never as an
    outside event.
    """
    kind = str(stimulus.get("kind", ""))
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

def direct_stimulus(request: dict, definition: CodeletDefinition, signals: dict) -> list[dict]:
    user = request.get("user_name", "Jesse")
    items = []
    for stimulus in request.get("stimuli", []):
        label, origin, reality = stimulus_label(stimulus, user)
        items.append({"text": f"{label}: {_clip(stimulus.get('text', ''))}",
                      "target_ids": [int(stimulus["knoxel_id"])], "origin": origin, "reality": reality})
    return items


def context_recall(request: dict, definition: CodeletDefinition, signals: dict) -> list[dict]:
    return [{"text": f"I remember: {_clip(item.get('text', ''))}", "target_ids": [int(item["knoxel_id"])]}
            for item in _memories(request)]


def absence_return(request: dict, definition: CodeletDefinition, signals: dict) -> list[dict]:
    user = request.get("user_name", "Jesse")
    hours = float(signals.get("time.idle_hours", 0.0))
    span = f"about {round(hours)} hours" if hours < 48 else f"about {round(hours / 24)} days"
    targets = [int(item["knoxel_id"]) for item in request.get("stimuli", []) if item.get("kind") == "user.input"]
    return [{"text": f"{user} is back after {span} without talking to me.", "target_ids": targets[:1]}]


def goal_reminder(request: dict, definition: CodeletDefinition, signals: dict) -> list[dict]:
    """Nominate the first active goal. The module lists goals by salience."""
    goals = request.get("active_goal_records", [])
    if not goals:
        return []
    goal = goals[0]
    return [{"text": f"I still want to: {_clip(goal.get('text', ''), 200)}", "target_ids": [int(goal["knoxel_id"])]}]


def lingering_feeling(request: dict, definition: CodeletDefinition, signals: dict) -> list[dict]:
    emotions = {key.removeprefix("emotion."): value for key, value in signals.items()
                if key.startswith("emotion.") and key not in {"emotion.strongest", "emotion.negative_strongest"}}
    if not emotions:
        return []
    name, value = max(emotions.items(), key=lambda item: item[1])
    return [{"text": f"A {_intensity(value)} feeling of {name} is still with me.", "target_ids": []}]


def novelty_spotlight(request: dict, definition: CodeletDefinition, signals: dict) -> list[dict]:
    """Nominate the first stimulus again, labelled as new.

    The appraisal's novelty axis decides whether this codelet runs. The
    nomination adds a second, interpreted view of the same event; it never
    invents an event of its own.
    """
    user = request.get("user_name", "Jesse")
    for stimulus in request.get("stimuli", []):
        label, origin, reality = stimulus_label(stimulus, user)
        return [{"text": f"Something new is happening: {_clip(stimulus.get('text', ''))}",
                 "target_ids": [int(stimulus["knoxel_id"])], "origin": origin, "reality": reality}]
    return []


def connection_lack(request: dict, definition: CodeletDefinition, signals: dict) -> list[dict]:
    """Name the low connection level.

    Loneliness has no dedicated emotion field, so this implemented need
    carries it instead of an invented label. The scheduler gate keeps this
    quiet unless the accepted connection level is genuinely low.
    """
    level = float(signals.get("needs.connection", 1.0))
    if level > 0.35:
        return []
    return [{"text": "There has been too little real back-and-forth. I miss the contact.",
             "target_ids": [], "origin": "state", "reality": "internal"}]


PURE_CODELETS = {
    "direct_stimulus": direct_stimulus,
    "context_recall": context_recall,
    "absence_return": absence_return,
    "goal_reminder": goal_reminder,
    "lingering_feeling": lingering_feeling,
    "novelty_spotlight": novelty_spotlight,
    "connection_lack": connection_lack,
}


# ---- LLM codelets --------------------------------------------------------

def render_events(request: dict) -> str:
    user = request.get("user_name", "Jesse")
    lines = []
    for stimulus in request.get("stimuli", []):
        label, _, _ = stimulus_label(stimulus, user)
        lines.append(f"- [id {stimulus['knoxel_id']}] {label}: {_clip(stimulus.get('text', ''))}")
    return "\n".join(lines) or "(no current events)"


def render_memories(request: dict) -> str:
    lines = []
    for item in _memories(request):
        when = f" (from {item['timestamp']})" if item.get("timestamp") else ""
        lines.append(f"- [id {item['knoxel_id']}]{when} {_clip(item.get('text', ''))}")
    return "\n".join(lines) or "(no memories)"


def render_goals(request: dict) -> str:
    lines = [f"- [id {goal['knoxel_id']}] {_clip(goal.get('text', ''), 200)}"
             for goal in request.get("active_goal_records", [])]
    return "\n".join(lines) or "(no active goals)"


def build_messages(request: dict, definition: CodeletDefinition, prompts: dict) -> list[Turn]:
    companion = request.get("companion_name", "Spamuel")
    user = request.get("user_name", "Jesse")
    instructions = definition.instructions.strip().format(companion_name=companion, user_name=user)
    system = prompts["system_prompt"].format(companion_name=companion, user_name=user)
    affect = request.get("affect") or {}
    user_message = format_content(prompts["user_template"],
        companion_name=companion, user_name=user, codelet_name=definition.name,
        affect=affect.get("summary") or request.get("state_summary") or "(calm, nothing strong)",
        events=render_events(request), memories=render_memories(request), goals=render_goals(request),
    )
    job = prompts["job_template"].format(codelet_name=definition.name, instructions=instructions,
                                         reality_rule=prompts["reality_rules"][definition.reality])
    # Every codelet shares the exact prefix before its concrete job.
    return [Turn(Role.System, system.strip()), Turn(Role.User, user_message.strip()), Turn(Role.User, job.strip())]


def run_llm_codelet(context, request: dict, definition: CodeletDefinition, prompts: dict) -> tuple[list[dict], dict]:
    """Make one model call and return ``(proposal items, cost record)``.

    Raises ``StageOutputError`` when the model never returns a valid answer.
    """
    messages = build_messages(request, definition, prompts)
    started = time.perf_counter()
    answer = complete_json(context, messages, CodeletAnswer, temperature=0.4,
                           enable_thinking=False)
    input_tokens, _ = estimate_tokens("", messages)
    output_tokens, _ = estimate_tokens("", [Turn(Role.Assistant, answer.model_dump_json())])
    cost = {"calls": 1, "input_tokens_estimate": input_tokens, "output_tokens_estimate": output_tokens,
            "seconds": round(time.perf_counter() - started, 3)}
    if not answer.fires or not answer.text.strip():
        return [], cost
    return [{"text": _clip(answer.text), "topic": str(answer.topic or "").strip(),
             "target_ids": list(answer.target_ids)}], cost


# ---- role ----------------------------------------------------------------

def _proposal(item: dict, decision, nomination_id: str, *, allowed: set[int],
              revisions: dict[int, int], thought_only: bool) -> CodeletProposal:
    """Turn one codelet output item into a typed proposal.

    Targets outside ``allowed`` move to ``dropped_target_ids``. A cycle that
    only a private thought started cannot produce an outside-origin or
    state-origin proposal. This keeps rumination distinguishable from events.
    """
    definition = decision.definition
    targets = [int(key) for key in item.get("target_ids", [])]
    kept = [key for key in dict.fromkeys(targets) if key in allowed]
    origin = item.get("origin") or definition.origin
    if thought_only and origin in {"outside", "state"}:
        origin = "thought"
    topic = str(item.get("topic", "") or "").strip()
    if definition.proposal != "belief":
        # Only a belief proposal carries a topic. A model that answers a
        # different codelet with a topic cannot smuggle a belief key in.
        topic = ""
    return CodeletProposal(
        nomination_id=nomination_id, codelet=definition.name, family=definition.family,
        proposal_kind=definition.proposal, origin=origin,
        reality=item.get("reality") or definition.reality, text=str(item.get("text", "")),
        target_ids=kept, cause=list(decision.cause), activation=decision.activation,
        source_revisions={str(key): revisions[key] for key in kept if key in revisions},
        dropped_target_ids=[key for key in targets if key not in allowed],
        topic=topic,
    )


def collect(context, request: dict, prompts: dict) -> CodeletsResult:
    """Schedule, run, and record codelets for one cycle.

    Raises ``StageOutputError`` when the catalogue names a pure codelet that
    has no Python function. That is a configuration error, so it stops the
    cycle instead of becoming a quiet per-cycle failure.
    """
    loaded = load_catalogue(request.get("codelet_catalogue") or None)
    missing = [item.name for item in loaded.catalogue.codelets
               if item.kind == "pure" and item.name not in PURE_CODELETS]
    if missing:
        raise StageOutputError(f"Pure codelets without a Python function: {', '.join(missing)}.")
    loop = str(request.get("attention_loop") or "main")
    pathways = loaded.catalogue.pathway_map()
    precursors = list(request.get("pathway_precursors") or [])
    latest_tick = int(request.get("pathway_latest_tick") or 0) or max(
        (int(item.get("tick_id", 0)) for item in precursors), default=0)
    # Pathway activation comes from committed precursor records, derived fresh
    # by the main module on every cycle. A pathway without a fresh precursor
    # stays at 0, which closes every gate and boost on its signal.
    pathway = pathway_signals(precursors, pathways, latest_tick=latest_tick)
    signals = {**request_signals(request), **(request.get("state_signals") or {}), **pathway}
    decisions, policy = schedule(loaded, signals, request.get("codelet_history") or {}, loop=loop)
    allowed = allowed_targets(request)
    revisions = source_revisions(request)
    thought_only = bool(signals.get("stimulus.thought")) and not (
        signals.get("stimulus.user") or signals.get("stimulus.workflow") or signals.get("stimulus.system"))
    proposals: list[CodeletProposal] = []
    runs: list[CodeletRun] = []

    def execute(decision):
        definition = decision.definition
        try:
            if definition.kind == "pure":
                return PURE_CODELETS[definition.name](request, definition, signals), {}, ""
            items, cost = run_llm_codelet(context, request, definition, prompts)
            return items, cost, ""
        except StageOutputError as error:
            return [], {}, str(error)[:300]

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
        items, cost, error = next(results)
        if error:
            run.status, run.reason = "failed", error
            continue
        for key, value in cost.items():
            setattr(run, key, value)
        for item in items:
            proposal = _proposal(item, decision, f"n{len(proposals) + 1}", allowed=allowed,
                                 revisions=revisions, thought_only=thought_only)
            proposals.append(proposal)
            run.proposal_ids.append(proposal.nomination_id)
        run.status = "ran" if run.proposal_ids else "silent"
        run.reason = "scheduled"
    cost = {
        "llm_calls": sum(run.calls for run in runs),
        "input_tokens_estimate": sum(run.input_tokens_estimate for run in runs),
        "output_tokens_estimate": sum(run.output_tokens_estimate for run in runs),
        "seconds": round(sum(run.seconds for run in runs), 3),
        "measurement": "estimated.tiktoken",
    }
    return CodeletsResult(proposals=proposals, runs=runs, catalogue_version=loaded.catalogue.version,
                          catalogue_digest=loaded.digest, loop=loop, policy=policy.model_dump(mode="json"),
                          signals=signals, cost=cost,
                          pathway_signals={key.removeprefix(PATHWAY_SIGNAL_PREFIX): value
                                           for key, value in pathway.items()})


def run(context) -> dict:
    request = read_request(context.runtime.environment)
    result = collect(context, request, load_prompts())
    write_report(context.runtime.environment, result)
    ran = [item.codelet for item in result.runs if item.status in {"ran", "silent"}]
    return {
        "status": "done",
        "summary": f"ran {len(ran)} codelets ({', '.join(ran) or 'none'}), "
                   f"{len(result.proposals)} proposals, {result.cost['llm_calls']} model calls",
    }


__all__ = [
    "PURE_CODELETS",
    "allowed_targets",
    "build_messages",
    "collect",
    "load_prompts",
    "render_events",
    "render_goals",
    "render_memories",
    "run",
    "run_llm_codelet",
    "source_revisions",
    "stimulus_label",
]
