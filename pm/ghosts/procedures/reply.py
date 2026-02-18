import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pm.config_loader import QUOTE_START
from pm.data_structures import Feature, FeatureType, NarrativeTypes
from pm.ghosts.knoxel_trace import trace_substep
from pm.ghosts.persona_memory import collect_persona_signals
from pm.ghosts.agent_context import (
    AgentContextComposer,
    AgentContextConfig,
    AgentContextWeights,
    compute_agent_context_budget,
)
from pm.ghosts.capabilities import capability_context_text, enforce_reply_text_capability
from pm.ghosts.procedures.base import BaseProc, GhostProtocol
from pm.ghosts.llm_contracts import call_tool_with_contract
from pm.ghosts.schemas import ReplyRealityCheck
from pm.llm.llm_common import CommonCompSettings, LlmPreset
from pm.utils.token_utils import get_token_count

logger = logging.getLogger(__name__)


def _safe_excerpt(text: str, max_len: int = 220) -> str:
    t = str(text or "").replace("\n", " ").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 3].rstrip() + "..."


def _json_or_str(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=True)
    except Exception:
        return str(value)


class ReplyGenerationProc(BaseProc):
    """
    Story-writer reply phase.
    Consumes action schema + rich context and produces one final user-facing reply.
    """

    @staticmethod
    def run(ghost: GhostProtocol) -> Optional[str]:
        logger.info("Reply Agent: generating final user-facing reply from action schema...")

        behavior = getattr(ghost, "selected_action_schema", None)
        if behavior is None:
            behavior = trace_substep(
                ghost,
                "ReplyGenerationProc.find_latest_action_schema",
                "reply",
                ReplyGenerationProc._find_latest_action_schema,
            )
        if behavior is None:
            logger.info("Reply Agent: no action schema available; skipping.")
            return None

        context = trace_substep(
            ghost,
            "ReplyGenerationProc.build_reply_context",
            "reply",
            ReplyGenerationProc._build_reply_context,
        )

        system_prompt, seed_prompt, msgs = trace_substep(
            ghost,
            "ReplyGenerationProc.build_prompts",
            "reply",
            lambda g: ReplyGenerationProc._build_prompts(ghost, behavior, context),
        )

        user_request = str(getattr(getattr(ghost, "primary_stimulus", None), "content", "") or "")
        reply_text, realism_meta = trace_substep(
            ghost,
            "ReplyGenerationProc.generate_with_realism_triage_loop",
            "reply",
            lambda g: ReplyGenerationProc._generate_with_realism_triage_loop(
                ghost=ghost,
                behavior=behavior,
                context=context,
                msgs=msgs,
                user_request=user_request,
            ),
        )
        if not reply_text:
            logger.warning("Reply Agent: realism validator produced empty reply; using behavior fallback.")
            reply_text = ReplyGenerationProc._fallback_reply(behavior)

        ghost.simulated_reply = reply_text
        ghost.reply_agent_last = {
            "system_prompt_excerpt": _safe_excerpt(system_prompt, 180),
            "user_prompt_excerpt": _safe_excerpt(seed_prompt, 360),
            "reply_excerpt": _safe_excerpt(reply_text, 180),
            "tick": int(getattr(ghost, "current_tick_id", 0) or 0),
            "action_description": str(getattr(behavior, "action_description", "") or ""),
            "realism_meta": realism_meta or {},
        }

        source_name = str(getattr(getattr(ghost, "config", None), "companion_name", "") or "assistant")
        f = Feature(
            content=reply_text,
            feature_type=FeatureType.Dialogue,
            source=source_name,
            interlocus=1.0,
            causal=True,
            metadata={
                "source_phase": "ReplyGenerationProc",
                "action_description": str(getattr(behavior, "action_description", "") or ""),
                "action_internal_thought": str(getattr(behavior, "internal_thought", "") or ""),
                "realism_meta": realism_meta or {},
            },
        )
        ghost.add_knoxel(f)
        return reply_text

    @staticmethod
    def _generate_with_realism_triage_loop(
        *,
        ghost: GhostProtocol,
        behavior,
        context: Dict[str, Any],
        msgs: List[Tuple[str, str]],
        user_request: str,
    ) -> tuple[str, Dict[str, Any]]:
        max_retries = int(getattr(getattr(ghost, "config", None), "reply_realism_max_retries", 3) or 3)
        max_retries = max(0, min(6, max_retries))

        draft = ""
        final_reply = ""
        attempts: List[Dict[str, Any]] = []
        retry_msgs = list(msgs)

        for attempt in range(max_retries + 1):
            draft = str(
                trace_substep(
                    ghost,
                    "ReplyGenerationProc.call_completion_text",
                    "reply",
                    lambda g: ReplyGenerationProc._call_reply_llm(ghost, retry_msgs),
                )
                or ""
            ).strip()
            if not draft:
                break

            candidate, meta = ReplyGenerationProc._validate_reply_realism(
                ghost=ghost,
                reply_text=draft,
                context=context,
                user_request=user_request,
            )
            rejected_stuff = str((meta or {}).get("rejected_stuff", "") or "").strip()
            realism_ok = bool((meta or {}).get("realism_ok", True))
            attempts.append(
                {
                    "attempt": attempt + 1,
                    "realism_ok": realism_ok,
                    "rejected_stuff": rejected_stuff,
                    "issues": list((meta or {}).get("issues", []) or []),
                }
            )

            if realism_ok:
                final_reply = str(candidate or "").strip()
                return final_reply, {"attempts": attempts, "final_attempt": attempt + 1, "realism_meta": meta or {}}

            # Old behavior restored: feed rejected_stuff back as a user constraint turn.
            if rejected_stuff and attempt < max_retries:
                companion_name = str(getattr(getattr(ghost, "config", None), "companion_name", "") or "assistant")
                retry_msgs = ReplyGenerationProc._append_rejected_stuff_turn(
                    msgs=msgs,
                    companion_name=companion_name,
                    rejected_stuff=rejected_stuff,
                )
                continue

            final_reply = str(candidate or "").strip()
            return final_reply, {"attempts": attempts, "final_attempt": attempt + 1, "realism_meta": meta or {}}

        if not draft:
            fallback = ReplyGenerationProc._fallback_reply(behavior)
            return str(fallback or "").strip(), {"attempts": attempts, "fallback_reason": "empty_reply"}
        return str(draft or "").strip(), {"attempts": attempts, "fallback_reason": "loop_exhausted"}

    @staticmethod
    def _append_rejected_stuff_turn(
        *, msgs: List[Tuple[str, str]], companion_name: str, rejected_stuff: str
    ) -> List[Tuple[str, str]]:
        out = list(msgs)
        out.append(("user", f"Make sure {companion_name} does not say stuff like this: {rejected_stuff}"))
        out.append(("assistant", f"Ok, I won't. {companion_name} says: {QUOTE_START}"))
        return out

    @staticmethod
    def _find_latest_action_schema(ghost: GhostProtocol):
        for f in reversed(list(getattr(ghost, "all_features", []) or [])):
            if getattr(f, "feature_type", None) != FeatureType.Action:
                continue
            if str(getattr(f, "source", "") or "") != "ActionProc":
                continue
            try:
                from pm.ghosts.schemas import BehaviorOutput

                return BehaviorOutput.model_validate_json(str(getattr(f, "content", "") or ""))
            except Exception:
                continue
        return None

    @staticmethod
    def _build_reply_context(ghost: GhostProtocol) -> Dict[str, Any]:
        broadcast = getattr(ghost, "conscious_broadcast", None)
        broadcast_text = str(getattr(broadcast, "content", "") or "(none)")
        action_desc = str(getattr(getattr(ghost, "selected_action_schema", None), "action_description", "") or "")
        focus_text = "\n".join(
            [
                broadcast_text,
                action_desc,
                str(getattr(getattr(ghost, "primary_stimulus", None), "content", "") or ""),
                str(getattr(ghost, "ego_directive", "") or ""),
            ]
        )
        out_cap = int(getattr(ghost, "generation_token_cap", 0) or 320)
        prompt_budget = compute_agent_context_budget(
            ghost,
            output_tokens=out_cap,
            min_budget=720,
        )
        model_ctx = 0
        try:
            if hasattr(ghost, "llm") and hasattr(ghost.llm, "get_max_tokens"):
                model_ctx = int(ghost.llm.get_max_tokens(LlmPreset.Default))
        except Exception:
            model_ctx = 0
        packet = AgentContextComposer.build(
            ghost,
            focus_text=focus_text,
            config=AgentContextConfig(
                max_tokens=prompt_budget,
                weights=AgentContextWeights.from_config(
                    dict(getattr(getattr(ghost, "config", None), "agent_context_weights", {}) or {})
                ),
                latest_messages=int(getattr(getattr(ghost, "config", None), "agent_context_latest_messages", 40) or 40),
                workspace_items=int(getattr(getattr(ghost, "config", None), "agent_context_workspace_items", 48) or 48),
                min_section_tokens=int(getattr(getattr(ghost, "config", None), "agent_context_min_section_tokens", 96) or 96),
            ),
            purpose="reply_generation",
        )
        fact_lines = list(getattr(packet, "fact_lines", []) or [])

        # Static narrative context (slow-changing memory/persona summaries)
        companion_name = str(getattr(getattr(ghost, "config", None), "companion_name", "") or "assistant")
        user_name = str(getattr(getattr(ghost, "config", None), "user_name", "") or "user")
        get_narrative = getattr(ghost, "get_narrative", None)
        ai_bhv_narr = None
        ai_self_narr = None
        ai_rel_narr = None
        user_psy_narr = None
        if callable(get_narrative):
            try:
                ai_bhv_narr = get_narrative(NarrativeTypes.BehaviorActionSelection, companion_name)
            except Exception:
                ai_bhv_narr = None
            try:
                ai_self_narr = get_narrative(NarrativeTypes.SelfImage, companion_name)
            except Exception:
                ai_self_narr = None
            try:
                ai_rel_narr = get_narrative(NarrativeTypes.Relations, companion_name)
            except Exception:
                ai_rel_narr = None
            try:
                user_psy_narr = get_narrative(NarrativeTypes.PsychologicalAnalysis, user_name)
            except Exception:
                user_psy_narr = None

        latent = getattr(getattr(ghost, "current_state", None), "latent_mental_state", None)
        current_emotions = str(getattr(latent, "state_emotions", None) or "(unknown)")
        current_needs = str(getattr(latent, "state_needs", None) or "(unknown)")
        current_cognition = str(getattr(latent, "state_cognition", None) or "(unknown)")
        active_intentions = list(getattr(ghost, "all_intentions", []) or [])
        intention_summary = "; ".join([f"'{str(getattr(i, 'content', '') or '')}'" for i in active_intentions[-12:]]) if active_intentions else "None"
        runtime_ctx = ReplyGenerationProc._runtime_context(ghost)

        assistant_story = str(getattr(packet, "text", "") or "").strip() or "No prior history."

        return {
            "broadcast": broadcast_text,
            "assistant_story": assistant_story,
            "reply_blueprint": dict(getattr(ghost, "reply_blueprint_last", {}) or {}),
            "persona_quirks": collect_persona_signals(ghost, query=focus_text, limit=6),
            "static_character": str(getattr(getattr(ghost, "config", None), "universal_character_card", "") or "(none)"),
            "ai_behavior_narrative": str(getattr(ai_bhv_narr, "content", "") or "Default behavior."),
            "ai_self_narrative": str(getattr(ai_self_narr, "content", "") or "Default self-image."),
            "ai_relation_narrative": str(getattr(ai_rel_narr, "content", "") or "Default relationship view."),
            "user_psy_narrative": str(getattr(user_psy_narr, "content", "") or "Default user analysis."),
            "current_emotions": current_emotions,
            "current_needs": current_needs,
            "current_cognition": current_cognition,
            "intention_summary": intention_summary,
            "memory_notes": "\n".join(fact_lines) if fact_lines else "(none)",
            "prompt_budget": int(prompt_budget),
            "model_ctx": int(model_ctx),
            "out_cap": int(out_cap),
            "capability_context": capability_context_text(ghost),
            "runtime_context": runtime_ctx,
            "context_sections": dict(getattr(packet, "sections", {}) or {}),
            "context_token_usage": dict(getattr(packet, "token_usage", {}) or {}),
        }

    @staticmethod
    def _build_prompts(
        ghost: GhostProtocol, behavior, context: Dict[str, Any]
    ) -> Tuple[str, str, List[Tuple[str, str]]]:
        companion_name = str(getattr(getattr(ghost, "config", None), "companion_name", "") or "assistant")
        user_name = str(getattr(getattr(ghost, "config", None), "user_name", "") or "user")
        system_prompt = (
            f"You are an expert story writer. You write an endearing story about the interactions "
            f"between a next-gen AI companion {companion_name} and their user {user_name}.\n"
            f"The AI {companion_name} is very advanced and has a cognitive architecture based on LIDA with "
            "functional self awareness and all aspects of human cognition, emotion and needs.\n"
            "Hard boundary: never claim architecture capabilities that are not explicitly supported.\n"
            f"Use dialogue in {QUOTE_START}quotes{QUOTE_START}, narration in *italics*, and thoughts in (parentheses).\n"
            "Write one reply only, 2-3 paragraphs, with around 50% dialogue. "
            "Continue naturally from the assistant seed and do not reset context."
        )

        blueprint = dict(context.get("reply_blueprint", {}) or {})
        persona = "\n".join(context.get("persona_quirks", []) or []) or "(none)"
        context_usage = dict(context.get("context_token_usage", {}) or {})
        usage_text = ", ".join([f"{k}={int(v)}" for k, v in context_usage.items()]) or "n/a"
        user_prompt = (
            f"**{companion_name}'s Character:** {str(context.get('static_character', '') or '(none)')}\n"
            f"**Behavior Prior:** {str(context.get('ai_behavior_narrative', '') or 'Default behavior.')}\n"
            f"**Self Narrative Prior:** {str(context.get('ai_self_narrative', '') or 'Default self-image.')}\n"
            f"**Relation Prior:** {str(context.get('ai_relation_narrative', '') or 'Default relationship view.')}\n"
            f"**User Model Prior:** {str(context.get('user_psy_narrative', '') or 'Default user analysis.')}\n"
            f"**Current Emotional State:** {str(context.get('current_emotions', '') or '(unknown)')}\n"
            f"**Current Needs State:** {str(context.get('current_needs', '') or '(unknown)')}\n"
            f"**Current Cognitive State:** {str(context.get('current_cognition', '') or '(unknown)')}\n"
            f"**Active Internal Goals:** {str(context.get('intention_summary', '') or 'None')}\n"
            f"**Relevant Facts:** {str(context.get('memory_notes', '') or '(none)')}\n"
            f"**Action Intent:** {str(getattr(behavior, 'action_description', '') or '')}\n"
            f"**Expression Blueprint:** {str(blueprint.get('name', '') or '')}\n"
            f"**Persona Quirks:** {persona}\n"
            f"**Current Broadcast:** {str(context.get('broadcast', '') or '(none)')}\n"
            f"**Context Token Allocation:** {usage_text}"
            f"\n**Dynamic Runtime Context:** {str(context.get('runtime_context', '') or '(none)')}"
            f"\n**Architecture Capability Profile:**\n{str(context.get('capability_context', '') or '(none)')}"
        )

        assistant_seed = str(context.get("assistant_story", "") or "").strip()
        prefix = str(blueprint.get("generation_prefix", "") or "").strip()
        if prefix:
            assistant_seed = (assistant_seed + "\n" + prefix).strip()
        if assistant_seed:
            assistant_seed += f"\n{companion_name} says: {QUOTE_START}"
        else:
            assistant_seed = f"{companion_name} says: {QUOTE_START}"

        msgs: List[Tuple[str, str]] = [("system", system_prompt), ("user", user_prompt), ("assistant", assistant_seed)]
        return system_prompt, user_prompt, msgs

    @staticmethod
    def _build_assistant_story(
        *,
        ghost: GhostProtocol,
        story_knoxels: List[Any],
        workspace_knoxels: List[Any],
        max_tokens: int,
        fallback_story_text: str = "",
    ) -> str:
        lines: List[str] = []
        seen = set()
        for k in story_knoxels + workspace_knoxels:
            kid = int(getattr(k, "id", -1) or -1)
            sig = (type(k).__name__, kid, str(getattr(k, "content", "") or ""))
            if sig in seen:
                continue
            seen.add(sig)
            try:
                line = str(k.get_story_element(ghost))
            except Exception:
                line = str(getattr(k, "content", "") or "")
            line = line.strip()
            if line:
                lines.append(line)
        if not lines:
            raw_lines = [ln.strip() for ln in str(fallback_story_text or "").splitlines() if ln.strip()]
            lines.extend(raw_lines)
        if not lines:
            return "No prior history."

        # Keep newest-first selection under budget, then restore chronological order.
        selected_rev: List[str] = []
        used = 0
        for line in reversed(lines):
            tc = get_token_count(line)
            if used + tc > max_tokens:
                continue
            selected_rev.append(line)
            used += tc
        selected_rev.reverse()
        return "\n".join(selected_rev)

    @staticmethod
    def _call_reply_llm(ghost: GhostProtocol, msgs: List[Tuple[str, str]]) -> str:
        max_tokens = int(getattr(ghost, "generation_token_cap", 0) or 320)
        try:
            return ghost.llm.completion_text(
                preset=LlmPreset.Default,
                inp=msgs,
                comp_settings=CommonCompSettings(max_tokens=max_tokens, temperature=0.7),
                discard_thinks=True,
            )
        except TypeError:
            return ghost.llm.completion_text(
                preset=LlmPreset.Default,
                inp=msgs,
                discard_thinks=True,
            )

    @staticmethod
    def _runtime_context(ghost: GhostProtocol) -> str:
        now_dt = None
        raw_now = getattr(ghost, "runtime_now_override", None)
        if isinstance(raw_now, datetime):
            now_dt = raw_now
        elif raw_now:
            try:
                now_dt = datetime.fromisoformat(str(raw_now))
            except Exception:
                now_dt = None

        if now_dt is None:
            now_dt = getattr(getattr(ghost, "current_state", None), "timestamp", None) or datetime.now()

        if 5 <= now_dt.hour < 12:
            tod = "morning"
        elif 12 <= now_dt.hour < 17:
            tod = "afternoon"
        elif 17 <= now_dt.hour < 22:
            tod = "evening"
        else:
            tod = "night"

        started = None
        started_raw = getattr(ghost, "runtime_started_at", None)
        if isinstance(started_raw, datetime):
            started = started_raw
        elif started_raw:
            try:
                started = datetime.fromisoformat(str(started_raw))
            except Exception:
                started = None
        if started is None:
            feats = list(getattr(ghost, "all_features", []) or [])
            if feats:
                started = min([getattr(f, "timestamp_creation", now_dt) for f in feats])
            else:
                started = now_dt

        uptime_h = max(0.0, (now_dt - started).total_seconds() / 3600.0)
        recon = dict(getattr(ghost, "memory_recon_last", {}) or {})
        recon_text = "none"
        if recon:
            recon_text = (
                f"tick={int(recon.get('tick', 0) or 0)} "
                f"ok={bool(recon.get('ok', False))} "
                f"skipped={bool(recon.get('skipped', False))} "
                f"reason={str(recon.get('reason', '') or '-')}"
            )

        return (
            f"local_datetime={now_dt.isoformat(timespec='minutes')} "
            f"time_of_day={tod} "
            f"operation_uptime_hours={uptime_h:.2f} "
            f"memory_recon_last=({recon_text})"
        )

    @staticmethod
    def _validate_reply_realism(
        *,
        ghost: GhostProtocol,
        reply_text: str,
        context: Dict[str, Any],
        user_request: str,
    ) -> tuple[str, Dict[str, Any]]:
        original = str(reply_text or "").strip()
        if not original:
            return original, {}

        sys_prompt = (
            "You are a strict realism validator for final assistant replies. "
            "The reply must stay inside architecture and capability limits. "
            "If unrealistic, provide a corrected realistic reply."
        )
        user_prompt = (
            f"User request:\n{user_request}\n\n"
            f"Draft reply:\n{original}\n\n"
            f"Runtime context:\n{str(context.get('runtime_context', '') or '(none)')}\n\n"
            f"Capability profile:\n{str(context.get('capability_context', '') or '(none)')}\n\n"
            "Validate and correct if needed."
        )

        out, event = call_tool_with_contract(
            ghost,
            phase="reply_reality_check",
            schema=ReplyRealityCheck,
            system_prompt=sys_prompt,
            user_prompt=user_prompt,
            examples=None,
            preset=LlmPreset.Default,
            max_retries=1,
            max_output_tokens=280,
        )

        candidate = original
        meta: Dict[str, Any] = {"tool_event": event}
        if out is not None:
            realism_ok = bool(getattr(out, "realism_ok", True))
            meta["realism_ok"] = realism_ok
            meta["issues"] = list(getattr(out, "issues", []) or [])
            meta["rationale"] = str(getattr(out, "rationale", "") or "")
            meta["rejected_stuff"] = str(getattr(out, "rejected_stuff", "") or "")
            if not realism_ok:
                corrected = str(getattr(out, "corrected_reply", "") or "").strip()
                if corrected:
                    candidate = corrected
                    meta["corrected"] = True
                else:
                    meta["corrected"] = False
                    meta["fallback_reason"] = "empty_corrected_reply"
            else:
                meta["corrected"] = False

        guarded, cap_guard = enforce_reply_text_capability(ghost, candidate, user_request=user_request)
        if cap_guard:
            meta["capability_guard"] = cap_guard
            candidate = guarded

        return str(candidate or "").strip(), meta

    @staticmethod
    def _fallback_reply(behavior) -> str:
        speech = str(getattr(behavior, "speech", "") or "").strip()
        if speech:
            return speech
        action_desc = str(getattr(behavior, "action_description", "") or "").strip()
        if action_desc:
            return action_desc
        return "I need one moment to align context before answering."
