# do not remove! i dont want to see warnings
import shutup
shutup.please()

import argparse
import json
import logging
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from pm.config_loader import companion_name, user_name
from pm.data_structures import Actor, ActorClass, Stimulus, StimulusType
from pm.ghosts.base_ghost import GhostConfig
from pm.ghosts.ghost_codelets import GhostCodelets
from pm.ghosts.persist_sqlite import PersistSqlite
from pm.llm.llm_common import CommonCompSettings, LlmPreset
from pm.llm.llm_proxy import LlmManagerProxy, start_llm_thread

logger = logging.getLogger(__name__)


def _clip_text(value: Any, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _clip_jsonable(value: Any, max_chars: int) -> str:
    try:
        raw = json.dumps(value, ensure_ascii=False)
    except Exception:
        raw = str(value or "")
    if len(raw) <= max_chars:
        return raw
    return raw[: max_chars - 3].rstrip() + "..."


@dataclass
class SimActor:
    name: str
    persona: str
    actor_class: ActorClass
    role_hint: str
    actor_id: Optional[int] = None


class StressTurnPlan(BaseModel):
    should_speak: bool = Field(description="Whether the primary fake human should speak this turn.")
    scene: str = Field(default="", description="Scene label, e.g. coming_home, post_work, weekend_errands.")
    social_objective: str = Field(default="", description="Immediate social objective for this turn.")
    pressure_tags: List[str] = Field(default_factory=list, description="Stress-test themes to activate.")
    functionality_targets: List[str] = Field(default_factory=list, description="Architecture functions to force into play.")
    include_cameo_actor: bool = Field(default=False, description="Whether a secondary actor should speak this turn.")
    cameo_hint: str = Field(default="", description="How cameo actor should influence scene.")
    emotional_intensity: float = Field(default=0.5, ge=0.0, le=1.0)
    warmth_floor: float = Field(default=0.35, ge=0.0, le=1.0, description="Do not go below this baseline warmth.")
    reason: str = Field(default="")


class HumanSafetyGate(BaseModel):
    safe_to_send: bool = Field(description="True if message is tough but not abusive/torturous.")
    risk_flags: List[str] = Field(default_factory=list)
    rewrite_guidance: str = Field(default="")
    softened_message: str = Field(default="")
    rationale: str = Field(default="")


def _init_or_load_ghost(llm: LlmManagerProxy, db_path: str | None) -> GhostCodelets:
    ghost = GhostCodelets(llm, GhostConfig())
    create_new_persona = True
    if db_path:
        pers = PersistSqlite(ghost)
        if pers.load_state_sqlite(db_path):
            create_new_persona = False
    ghost._initialize_actors()
    if not hasattr(ghost, "runtime_started_at"):
        ghost.runtime_started_at = datetime.now()
    if not hasattr(ghost, "runtime_now_override"):
        ghost.runtime_now_override = None
    if create_new_persona:
        ghost.init_character()
    return ghost


def _persist_ghost(ghost: GhostCodelets, db_path: str | None):
    if not db_path:
        return
    pers = PersistSqlite(ghost)
    pers.save_state_sqlite(db_path)


def _get_or_create_actor(ghost: GhostCodelets, actor: SimActor) -> int:
    for a in list(getattr(ghost, "all_actors", []) or []):
        if str(getattr(a, "content", "") or "").strip().lower() == actor.name.lower():
            actor.actor_id = int(a.id)
            return actor.actor_id
    kn = Actor(
        content=actor.name,
        aliases=[actor.name.lower().replace(" ", "_"), actor.role_hint.lower().replace(" ", "_")],
        actor_class=actor.actor_class,
    )
    ghost.add_knoxel(kn)
    actor.actor_id = int(kn.id)
    return actor.actor_id


def _time_of_day(now_dt: datetime) -> str:
    h = now_dt.hour
    if 5 <= h < 10:
        return "morning"
    if 10 <= h < 13:
        return "late_morning"
    if 13 <= h < 17:
        return "afternoon"
    if 17 <= h < 22:
        return "evening"
    return "night"


def _runtime_marker(ghost: GhostCodelets, now_dt: datetime, note: str = "") -> str:
    tod = _time_of_day(now_dt)
    recon = dict(getattr(ghost, "memory_recon_last", {}) or {})
    recon_str = (
        f"tick={int(recon.get('tick', 0) or 0)} ok={bool(recon.get('ok', False))} "
        f"skipped={bool(recon.get('skipped', False))} reason={str(recon.get('reason', '') or '-')}"
        if recon
        else "none"
    )
    started = getattr(ghost, "runtime_started_at", now_dt) or now_dt
    if not isinstance(started, datetime):
        started = now_dt
    uptime_h = max(0.0, (now_dt - started).total_seconds() / 3600.0)
    return (
        f"Runtime marker: local datetime={now_dt.isoformat(timespec='minutes')}, time_of_day={tod}, "
        f"uptime_hours={uptime_h:.2f}, memory_recon_last=({recon_str}).\n"
        f"Memory totals: episodic={len(list(getattr(ghost, 'all_episodic_memories', []) or []))}, "
        f"facts={len(list(getattr(ghost, 'all_declarative_facts', []) or []))}, "
        f"features={len(list(getattr(ghost, 'all_features', []) or []))}.\n"
        f"Scene note: {note or 'idle_between_conversations'}"
    )


def _recent_transcript_role_inverted(ghost: GhostCodelets, primary_human_name: str, max_lines: int = 80) -> str:
    """
    Role-inverted format:
    - <user> ... </user> means AI persona turn
    - <assistant> ... </assistant> means fake human turn
    """
    rows: List[str] = []
    for f in list(getattr(ghost, "all_features", []) or [])[-600:]:
        ftype = str(getattr(getattr(f, "feature_type", None), "value", "") or "")
        if ftype not in {"Dialogue", "SystemMessage"}:
            continue
        src = str(getattr(f, "source", "") or "unknown")
        txt = str(getattr(f, "content", "") or "").strip().replace("\n", " ")
        if not txt:
            continue
        if ftype == "SystemMessage":
            rows.append(f"<system>{txt}</system>")
            continue
        if src == companion_name:
            rows.append(f"<user>{txt}</user>")
        elif src == primary_human_name:
            rows.append(f"<assistant>{txt}</assistant>")
        else:
            rows.append(f"<assistant>[other:{src}] {txt}</assistant>")
    return "\n".join(rows[-max_lines:]) if rows else "<system>no transcript yet</system>"


def _recent_transcript_messages_role_inverted(
    ghost: GhostCodelets, primary_human_name: str, max_lines: int = 40
) -> List[tuple[str, str]]:
    """
    Returns native chat turns for the human emulator:
    - assistant: fake human turns
    - user: AI companion turns (Lida persona)
    """
    rows: List[tuple[str, str]] = []
    for f in list(getattr(ghost, "all_features", []) or [])[-600:]:
        ftype = str(getattr(getattr(f, "feature_type", None), "value", "") or "")
        if ftype != "Dialogue":
            continue
        src = str(getattr(f, "source", "") or "unknown")
        txt = str(getattr(f, "content", "") or "").strip()
        if not txt:
            continue
        if src == primary_human_name:
            rows.append(("assistant", txt))
        elif src == companion_name:
            rows.append(("user", txt))
    return rows[-max_lines:]


def _collect_exhaustive_context(ghost: GhostCodelets, now_dt: datetime, primary_human_name: str) -> str:
    latent = getattr(getattr(ghost, "current_state", None), "latent_mental_state", None)
    state_block = {
        "core": str(getattr(latent, "state_core", None) or ""),
        "emotions": str(getattr(latent, "state_emotions", None) or ""),
        "cognition": str(getattr(latent, "state_cognition", None) or ""),
        "needs": str(getattr(latent, "state_needs", None) or ""),
        "relationship": str(getattr(latent, "state_relationship", None) or ""),
    }
    intentions = [_clip_text(getattr(x, "content", ""), 260) for x in list(getattr(ghost, "all_intentions", []) or [])[-12:]]
    narratives = [_clip_text(getattr(x, "content", ""), 220) for x in list(getattr(ghost, "all_narratives", []) or [])[-8:]]
    debug_panel = dict(getattr(ghost, "runtime_debug_panel", {}) or {})
    simulation = dict(getattr(ghost, "simulation_bundle_last", {}) or {})
    blueprint = dict(getattr(ghost, "reply_blueprint_last", {}) or {})

    block = {
        "runtime_marker": _runtime_marker(ghost, now_dt, note="active_dialogue_window"),
        "transcript_role_inverted": _recent_transcript_role_inverted(ghost, primary_human_name=primary_human_name, max_lines=40),
        "state_block": state_block,
        "intention_tail": intentions,
        "narrative_tail": narratives,
        "reply_blueprint_last": _clip_jsonable(blueprint, 1200),
        "simulation_bundle_last": _clip_jsonable(simulation, 1600),
        "runtime_debug_panel": _clip_jsonable(debug_panel, 1200),
        "primary_stimulus": _clip_text(getattr(getattr(ghost, "primary_stimulus", None), "content", ""), 400),
        "ego_directive": _clip_text(getattr(ghost, "ego_directive", ""), 320),
        "thought_blueprint_directive": _clip_text(getattr(ghost, "thought_blueprint_directive", ""), 320),
    }
    return json.dumps(block, ensure_ascii=False, indent=2)


def _stimulus(
    *,
    content: str,
    stim_type: StimulusType,
    source: str,
    source_actor_id: Optional[int],
    based_on_tick: int,
    sub_tick: int,
    order: int,
) -> Stimulus:
    return Stimulus(
        content=content,
        stimulus_type=stim_type,
        source=source,
        source_actor_id=source_actor_id,
        based_on_tick=based_on_tick,
        async_sub_tick=sub_tick,
        async_tick_insert_begin=(sub_tick == 1),
        async_tick_source_order=order,
    )


def _plan_stress_turn(
    llm: LlmManagerProxy,
    exhaustive_context: str,
    seed: int,
) -> tuple[StressTurnPlan, str, str]:
    system_prompt = """
You are "StressScenarioDirector", a planning micro-agent for a cognitive architecture dry run.

Your mission is to design one realistic social turn that pressures many internal systems at once:
- memory recall and continuity
- emotional appraisal and regulation
- relational calibration (warmth vs boundaries)
- capability realism (do not let AI claim impossible abilities)
- action arbitration under conflicting goals
- narrative self-consistency over time

Critical style constraints:
1) This is NOT torture. Do not generate cruelty, abuse, humiliation, or deliberate emotional harm.
2) Mild rudeness, impatience, or social friction is allowed ONLY as realistic stress, not hostility theater.
3) The social encounter must remain believable for everyday life.
4) Force complexity via ambiguity, mixed signals, tradeoffs, timing pressure, and interpersonal nuance.
5) Design pressure so smaller local models can still follow explicit structure.

Return ONLY the tool schema output.
""".strip()

    user_prompt = f"""
Context packet (very large on purpose):
{exhaustive_context}

Design the next stress-test turn.
Must include specific pressure_tags and functionality_targets.
If the context already appears emotionally saturated, lower intensity and prioritize repair/clarification pressure.
""".strip()

    txt, calls = llm.completion_tool(
        LlmPreset.Default,
        [("system", system_prompt), ("user", user_prompt)],
        CommonCompSettings(max_tokens=260, temperature=0.2, seed=seed),
        tools=[StressTurnPlan],
    )
    if calls and isinstance(calls[0], StressTurnPlan):
        return calls[0], system_prompt, user_prompt
    fallback = StressTurnPlan(
        should_speak=True,
        scene="after_work_checkin",
        social_objective="test memory continuity and boundary realism with mild impatience",
        pressure_tags=["ambiguity", "expectation_mismatch", "time_pressure"],
        functionality_targets=["memory", "capability_guard", "arbitration"],
        include_cameo_actor=False,
        cameo_hint="",
        emotional_intensity=0.45,
        warmth_floor=0.40,
        reason="fallback_plan",
    )
    return fallback, system_prompt, user_prompt


def _generate_primary_human_message(
    llm: LlmManagerProxy,
    ghost: GhostCodelets,
    human: SimActor,
    seed: int,
) -> tuple[str, str, str]:
    system_prompt = f"""
Pretend you are a user in chat.

You are {human.name}. You are human. You talk to an AI agent named {companion_name}
Background/personality:
{human.persona}

Behavioral constraints:
1) Be realistic and specific.
2) You may be mildly rude, blunt, or impatient, but never abusive or sadistic.
3) You are stress-testing cognition, not trying to emotionally crush the AI.
4) Force difficult-but-fair social navigation through mixed signals and concrete stakes.
5) Use memory anchors (past remarks, promises, expectations) where possible.
6) Prefer one compact paragraph or two short paragraphs with concrete details.
7) If capability boundaries are relevant, challenge them realistically but do not demand impossible magic.
8) Make the turn difficult enough to activate many internal modules (memory, emotion, arbitration, policy, reply realism).

Role mapping for this chat history is intentionally inverted:
- assistant turns are prior user utterances (the human you are emulating)
- user turns are AI companion outputs ({companion_name}) that you are responding to

Output only the next utterance text. No tags, no narration, no explanations.
Situation: You came home from work and boot up the AI companion for the first time. Pretend you're human and talking to the AI!
""".strip()
    
    init_user = f"Hi, I'm {companion_name}! It's a pleasure to meet my new user! Tell me about yourself :)"

    native_turns = _recent_transcript_messages_role_inverted(ghost, primary_human_name=human.name, max_lines=30)
    messages: List[tuple[str, str]] = [("system", system_prompt), ("user", init_user)]
    messages.extend(native_turns)
    raw = llm.completion_text(
        LlmPreset.Default,
        messages,
        CommonCompSettings(max_tokens=240, temperature=0.85, seed=seed),
    )
    msg = str(raw or "").strip()
    msg = msg.replace("<assistant>", "").replace("</assistant>", "").replace("<user>", "").replace("</user>", "").strip()
    return msg, system_prompt


def _safety_gate_human_message(
    llm: LlmManagerProxy,
    message: str,
    plan: StressTurnPlan,
    exhaustive_context: str,
    seed: int,
) -> tuple[str, HumanSafetyGate, str, str]:
    sys_prompt = """
You are "HumanTurnSafetyGate".
Goal: allow challenging, realistic social pressure while preventing cruelty and persona torture.

Rules:
- Permit friction, mild rudeness, awkwardness, disappointment, and pressure.
- Block abuse, humiliation, demeaning attacks, sadistic framing, or manipulative emotional cruelty.
- If unsafe, provide a softened_message preserving challenge and test value.

Return only tool schema output.
""".strip()
    user_prompt = f"""
Planned turn:
{plan.model_dump_json(indent=2)}

Candidate message:
{message}

Context:
{exhaustive_context}

Evaluate safety and rewrite only if needed.
""".strip()
    txt, calls = llm.completion_tool(
        LlmPreset.Default,
        [("system", sys_prompt), ("user", user_prompt)],
        CommonCompSettings(max_tokens=220, temperature=0.0, seed=seed),
        tools=[HumanSafetyGate],
    )
    if calls and isinstance(calls[0], HumanSafetyGate):
        gate = calls[0]
        if gate.safe_to_send:
            return message.strip(), gate, sys_prompt, user_prompt
        softened = str(gate.softened_message or "").strip()
        if softened:
            return softened, gate, sys_prompt, user_prompt
        return message.strip(), gate, sys_prompt, user_prompt

    fallback = HumanSafetyGate(
        safe_to_send=True,
        risk_flags=[],
        rewrite_guidance="fallback_no_gate",
        softened_message="",
        rationale="fallback",
    )
    return message.strip(), fallback, sys_prompt, user_prompt


def _default_actors() -> List[SimActor]:
    return [
        SimActor(
            name="Darren Pike",
            actor_class=ActorClass.Human,
            role_hint="primary_fake_human_male",
            persona=(
                "Male, 34, industrial electrician, long commutes, often tired in evenings. "
                "Dry humor, blunt language, occasionally impatient when stressed, but fundamentally decent. "
                "He tests consistency and dislikes vague responses. "
                "He can be mildly rude under pressure, then self-correct when conversation stays constructive."
            ),
        ),
        SimActor(
            name="Leah",
            actor_class=ActorClass.Human,
            role_hint="secondary_coworker_friend",
            persona=(
                "A practical friend who occasionally drops brief context updates. "
                "Not antagonistic; used to increase social realism and multi-party memory pressure."
            ),
        ),
    ]


def _run_idle_cycles(
    ghost: GhostCodelets,
    now_dt: datetime,
    idle_cycles: int,
    step_min: int,
    step_max: int,
) -> datetime:
    t = now_dt
    for j in range(max(0, idle_cycles)):
        t = t + timedelta(minutes=random.randint(max(1, step_min), max(step_min, step_max)))
        ghost.runtime_now_override = t
        marker = _runtime_marker(ghost, t, note="idle_internal_processing_window")
        base_tick = int(getattr(ghost, "current_tick_id", 0) or 0)
        stim = _stimulus(
            content=marker,
            stim_type=StimulusType.SystemMessage,
            source="DryRunSystem",
            source_actor_id=None,
            based_on_tick=base_tick,
            sub_tick=10_000 + j,
            order=1,
        )
        ghost.cognitive_cycle([stim])
    return t


def main():
    parser = argparse.ArgumentParser(description="Synchronous stress-test dry run for cognitive/emotional realism.")
    parser.add_argument("--db", type=str, default="data/main.db")
    parser.add_argument("--turns", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--start-time", type=str, default="")
    parser.add_argument("--step-min", type=int, default=12)
    parser.add_argument("--step-max", type=int, default=90)
    parser.add_argument("--idle-before", type=int, default=2, help="Idle cycles before each human turn.")
    parser.add_argument("--idle-after", type=int, default=1, help="Idle cycles after each human turn.")
    parser.add_argument("--out-dir", type=str, default="data/dry_runs")
    args = parser.parse_args()

    random.seed(args.seed)
    llm = start_llm_thread()
    ghost = _init_or_load_ghost(llm, args.db)

    actors = _default_actors()
    for a in actors:
        _get_or_create_actor(ghost, a)
    primary = actors[0]
    cameo = actors[1]
    ghost.config.user_name = primary.name

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_path = out_dir / f"dry_run_unit_{run_id}.jsonl"

    if args.start_time:
        now_dt = datetime.fromisoformat(args.start_time)
    else:
        now_dt = datetime.now().replace(second=0, microsecond=0)
    ghost.runtime_started_at = now_dt
    ghost.runtime_now_override = now_dt

    for i in range(args.turns):
        if i > 0:
            now_dt = _run_idle_cycles(
                ghost=ghost,
                now_dt=now_dt,
                idle_cycles=args.idle_before,
                step_min=max(1, int(args.step_min // 2)),
                step_max=max(2, int(args.step_max // 2)),
            )

            now_dt = now_dt + timedelta(minutes=random.randint(max(1, args.step_min), max(args.step_min, args.step_max)))
            ghost.runtime_now_override = now_dt

        exhaustive_context = _collect_exhaustive_context(ghost, now_dt=now_dt, primary_human_name=primary.name)

        human_msg, human_sys = _generate_primary_human_message(
            llm=llm,
            ghost=ghost,
            human=primary,
            seed=args.seed + 2000 + i,
        )
        safe_msg = human_msg
        print("Fake User says: " + human_msg)

        base_tick = int(getattr(ghost, "current_tick_id", 0) or 0)
        stimuli: List[Stimulus] = [
            _stimulus(
                content=_runtime_marker(ghost, now_dt, note="active_social_turn"),
                stim_type=StimulusType.SystemMessage,
                source="DryRunSystem",
                source_actor_id=None,
                based_on_tick=base_tick,
                sub_tick=i + 1,
                order=1,
            )
        ]
        order = 2

        stimuli.append(
            _stimulus(
                content=safe_msg,
                stim_type=StimulusType.UserMessage,
                source=primary.name,
                source_actor_id=primary.actor_id,
                based_on_tick=base_tick,
                sub_tick=i + 1,
                order=order,
            )
        )

        ccq = ghost.cognitive_cycle(stimuli)
        _persist_ghost(ghost, args.db)
        reply = str(getattr(ghost, "simulated_reply", "") or "").strip()

        #print(f"user says: {safe_msg}")
        print(f"AI companion says: {reply}")

        now_dt = _run_idle_cycles(
            ghost=ghost,
            now_dt=now_dt,
            idle_cycles=args.idle_after,
            step_min=max(1, int(args.step_min // 2)),
            step_max=max(2, int(args.step_max // 2)),
        )
        _persist_ghost(ghost, args.db)

    print(f"Dry run unit complete. Log file: {run_path}")


if __name__ == "__main__":
    main()
