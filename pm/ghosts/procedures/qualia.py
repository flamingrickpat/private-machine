import logging
from typing import List, Optional, Tuple

from pm.data_structures import Feature, FeatureType, Narrative, NarrativeTypes
from pm.ghosts.llm_contracts import call_tool_with_contract
from pm.ghosts.knoxel_trace import trace_substep
from pm.ghosts.procedures.base import BaseProc, GhostProtocol
from pm.ghosts.prompts import (
    QUALIA_BIOGRAPHY_SYSTEM,
    QUALIA_BIOGRAPHY_USER,
    QUALIA_SELF_STORY_EXAMPLE_TURNS,
    QUALIA_SELF_STORY_SYSTEM,
    QUALIA_SELF_STORY_USER,
    QUALIA_THEORY_UPDATE_EXAMPLE_TURNS,
    QUALIA_THEORY_UPDATE_SYSTEM,
    QUALIA_THEORY_UPDATE_USER,
)
from pm.ghosts.schemas import QualiaSelfReflection, SelfTheoryUpdate
from pm.llm.llm_common import LlmPreset

logger = logging.getLogger(__name__)


_MAX_HISTORY = 120


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _safe_excerpt(text: str, max_len: int = 180) -> str:
    t = (text or "").replace("\n", " ").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 3].rstrip() + "..."


def _summarize_simulation(ghost: GhostProtocol) -> str:
    bundle = getattr(ghost, "simulation_bundle_last", None) or {}
    if not bundle:
        return "(none)"

    outcomes = bundle.get("outcomes", [])
    rows = []
    for o in outcomes[:3]:
        rows.append(
            f"{o.get('lens', '?')} util={float(o.get('utility', 0.0)):.2f} "
            f"pol={o.get('polarity', 'neutral')} sum={_safe_excerpt(o.get('summary', ''), 90)}"
        )
    return " | ".join(rows) if rows else "(none)"


def _self_model_summary(ghost: GhostProtocol) -> str:
    sm = getattr(ghost, "self_model", None)
    if sm is None:
        return "(no self model)"

    return (
        f"focus={_safe_excerpt(getattr(sm, 'current_focus', ''), 60)}; "
        f"quality={getattr(sm, 'attention_quality', 'stable')}; "
        f"confidence={float(getattr(sm, 'confidence', 0.5)):.2f}; "
        f"theory_conf={float(getattr(sm, 'theory_confidence', 0.5)):.2f}; "
        f"anchors={','.join(list(getattr(sm, 'style_anchors', []) or [])[:4]) or '-'}"
    )


def _state_snapshot(ghost: GhostProtocol) -> Tuple[float, float, float, float]:
    valence = 0.0
    arousal = 0.0
    interlocus = 0.0
    aperture = 0.0

    ms = getattr(getattr(ghost, "current_state", None), "latent_mental_state", None)
    if ms is None:
        return valence, arousal, interlocus, aperture

    core = getattr(ms, "state_core", None)
    cog = getattr(ms, "state_cognition", None)

    if core is not None:
        valence = float(getattr(core, "valence", 0.0) or 0.0)
        arousal = float(getattr(core, "arousal", 0.0) or 0.0)
    if cog is not None:
        interlocus = float(getattr(cog, "interlocus", 0.0) or 0.0)
        aperture = float(getattr(cog, "mental_aperture", 0.0) or 0.0)

    return valence, arousal, interlocus, aperture


def _deterministic_reflection(ghost: GhostProtocol, broadcast: str) -> QualiaSelfReflection:
    valence, arousal, interlocus, aperture = _state_snapshot(ghost)

    if arousal >= 0.75 and valence < -0.35:
        emo = "agitated"
        quality = "shifting"
        tension = 0.82
    elif arousal >= 0.75:
        emo = "amped"
        quality = "distracted"
        tension = 0.72
    elif valence < -0.55:
        emo = "irritated"
        quality = "narrow"
        tension = 0.68
    elif valence > 0.45:
        emo = "confident"
        quality = "stable"
        tension = 0.35
    else:
        emo = "neutral"
        quality = "stable"
        tension = 0.48

    if interlocus < -0.4:
        cause = "self-protection"
    elif interlocus > 0.4:
        cause = "social control"
    else:
        cause = "uncertainty management"

    consistency = _clamp(0.72 - abs(aperture) * 0.18 - tension * 0.22, 0.15, 0.95)
    confidence = _clamp(0.65 - tension * 0.25 + (consistency * 0.2), 0.10, 0.95)

    inner = (
        f"I keep circling '{_safe_excerpt(broadcast, 80)}'; "
        f"the tone in my head is {emo}, and I am steering for {cause}."
    )
    story = (
        "I am not trying to be agreeable, I am trying to stay coherent with my current drive and constraints."
    )

    return QualiaSelfReflection(
        inner_voice=inner,
        self_narrative=story,
        focus_target=_safe_excerpt(broadcast, 80),
        dominant_emotion=emo,
        tension=tension,
        self_consistency=consistency,
        confidence=confidence,
        hidden_causes=[cause],
        attention_bias=_clamp((0.08 if tension > 0.7 else 0.02) - max(0.0, valence) * 0.04, -0.3, 0.3),
    )


def _append_bounded(items: List, value, limit: int = _MAX_HISTORY) -> List:
    items.append(value)
    if len(items) > limit:
        return items[-limit:]
    return items


class QualiaProc(BaseProc):
    """
    Task 8 self-model loop:
    1) Self-story agent (in-character introspection, private)
    2) Psychologist-agent theory update (retain/new/discard)
    3) Deterministic fallback + retry hints for robustness
    """

    @staticmethod
    def run(ghost: GhostProtocol) -> None:
        trace_substep(ghost, "QualiaProc.generate_subjective_experience", "qualia_experience", QualiaProc.generate_subjective_experience)
        trace_substep(ghost, "QualiaProc.update_self_model", "qualia_model", QualiaProc.update_self_model)

    @staticmethod
    def generate_subjective_experience(ghost: GhostProtocol) -> None:
        logger.info("Qualia: generating structured self reflection...")

        broadcast = ""
        if getattr(ghost, "conscious_broadcast", None) is not None:
            broadcast = str(getattr(ghost.conscious_broadcast, "content", "") or "")
        if not broadcast:
            return

        valence, arousal, interlocus, aperture = _state_snapshot(ghost)
        companion = ghost.config.companion_name
        character_card = getattr(ghost.config, "universal_character_card", "") or ""
        ego_directive = str(getattr(ghost, "ego_directive", "") or "")

        sys_prompt = QUALIA_SELF_STORY_SYSTEM.format(companion_name=companion)
        usr_prompt = QUALIA_SELF_STORY_USER.format(
            character_card=character_card[:2500] or "(none)",
            broadcast=broadcast[:700],
            ego_directive=ego_directive[:700] or "(none)",
            simulation_snapshot=_summarize_simulation(ghost),
            valence=valence,
            arousal=arousal,
            interlocus=interlocus,
            aperture=aperture,
            self_model_summary=_self_model_summary(ghost),
        )

        reflection: Optional[QualiaSelfReflection] = None
        try:
            reflection, _ = trace_substep(
                ghost,
                "QualiaProc.call_tool_self_story",
                "qualia_experience",
                lambda g: call_tool_with_contract(
                ghost,
                phase="qualia_self_story",
                schema=QualiaSelfReflection,
                system_prompt=sys_prompt,
                user_prompt=usr_prompt,
                examples=QUALIA_SELF_STORY_EXAMPLE_TURNS,
                preset=LlmPreset.Default,
                max_retries=1,
                ),
            )
        except Exception as e:
            logger.warning("Qualia: self-story tool call failed: %s", e)

        if reflection is None:
            reflection = trace_substep(
                ghost, "QualiaProc._deterministic_reflection", "qualia_experience", _deterministic_reflection, broadcast
            )

        # Persist transparent trace.
        ghost.qualia_reflection_last_model = reflection
        ghost.qualia_reflection_last = reflection.model_dump()
        if not hasattr(ghost, "qualia_reflection_history"):
            ghost.qualia_reflection_history = []
        ghost.qualia_reflection_history = _append_bounded(
            list(getattr(ghost, "qualia_reflection_history", [])), reflection.model_dump()
        )

        # Update self model short-horizon runtime.
        if hasattr(ghost, "self_model") and ghost.self_model is not None:
            ghost.self_model.current_focus = _safe_excerpt(reflection.focus_target or broadcast, 100)
            ghost.self_model.attention_quality = "shifting" if reflection.tension > 0.7 else "stable"
            ghost.self_model.confidence = _clamp(reflection.confidence, 0.0, 1.0)
            ghost.self_model.last_hidden_causes = list(reflection.hidden_causes[:6])

        ghost.subjective_experience = reflection.inner_voice
        note = (
            f"{reflection.dominant_emotion} tension={reflection.tension:.2f} "
            f"consistency={reflection.self_consistency:.2f} "
            f"focus='{_safe_excerpt(reflection.focus_target, 50)}'"
        )

        if hasattr(ghost, "self_model") and ghost.self_model is not None:
            notes = list(getattr(ghost.self_model, "introspection_notes", []) or [])
            notes = _append_bounded(notes, note)
            ghost.self_model.introspection_notes = notes

        feat = Feature(
            content=(
                f"InnerVoice: {reflection.inner_voice} | narrative={reflection.self_narrative} "
                f"| hidden_causes={','.join(reflection.hidden_causes[:3]) or 'none'}"
            ),
            feature_type=FeatureType.SubjectiveExperience,
            source="QualiaSelfStory",
            interlocus=-1,
            causal=False,
            metadata=reflection.model_dump(),
        )
        ghost.add_knoxel(feat)

        narr = Narrative(
            content=reflection.inner_voice,
            narrative_type=NarrativeTypes.InnerMonologue,
            target_name=companion,
            tick_id=ghost.current_tick_id,
        )
        ghost.add_knoxel(narr)

    @staticmethod
    def _request_theory_update(
        ghost: GhostProtocol,
        reflection: QualiaSelfReflection,
        retry_hint: str,
    ) -> Optional[SelfTheoryUpdate]:
        companion = ghost.config.companion_name
        character_card = getattr(ghost.config, "universal_character_card", "") or ""
        existing = []
        notes = []
        if hasattr(ghost, "self_model") and ghost.self_model is not None:
            existing = list(getattr(ghost.self_model, "active_theories", []) or [])
            notes = list(getattr(ghost.self_model, "introspection_notes", []) or [])

        valence, arousal, interlocus, aperture = _state_snapshot(ghost)
        state_snapshot = (
            f"valence={valence:.2f}, arousal={arousal:.2f}, interlocus={interlocus:.2f}, aperture={aperture:.2f}; "
            f"ego_directive={_safe_excerpt(str(getattr(ghost, 'ego_directive', '') or ''), 120)}"
        )

        sys_prompt = QUALIA_THEORY_UPDATE_SYSTEM.format(companion_name=companion)
        usr_prompt = QUALIA_THEORY_UPDATE_USER.format(
            character_card=character_card[:2200] or "(none)",
            reflection=_safe_excerpt(reflection.model_dump_json(), 1800),
            existing_theories="\n".join(existing[:12]) or "(none)",
            recent_notes="\n".join(notes[-8:]) or "(none)",
            state_snapshot=state_snapshot,
            retry_hint=retry_hint or "(none)",
        )

        try:
            model, _ = call_tool_with_contract(
                ghost,
                phase="qualia_theory_update",
                schema=SelfTheoryUpdate,
                system_prompt=sys_prompt,
                user_prompt=usr_prompt,
                examples=QUALIA_THEORY_UPDATE_EXAMPLE_TURNS,
                preset=LlmPreset.Default,
                max_retries=1,
            )
            if model is not None:
                return model
        except Exception as e:
            logger.warning("Qualia: theory update tool call failed: %s", e)
        return None

    @staticmethod
    def _fallback_theory_update(ghost: GhostProtocol, reflection: QualiaSelfReflection) -> SelfTheoryUpdate:
        sm = getattr(ghost, "self_model", None)
        existing = list(getattr(sm, "active_theories", []) or []) if sm is not None else []

        candidate = f"When tension={reflection.tension:.2f}, behavior is steered by {','.join(reflection.hidden_causes[:2]) or 'uncertainty'}"
        retained = existing[:6]
        if candidate not in retained:
            retained.append(candidate)

        return SelfTheoryUpdate(
            revised_biography_line=(
                f"Tick {ghost.current_tick_id}: {reflection.dominant_emotion} while focusing on '{_safe_excerpt(reflection.focus_target, 70)}'."
            ),
            retained_theories=retained,
            new_theories=[candidate],
            discarded_theories=[],
            confidence=_clamp(reflection.confidence * 0.85, 0.2, 0.9),
            mismatch=_clamp(reflection.tension * (1.0 - reflection.self_consistency), 0.0, 1.0),
            rationale="Deterministic fallback from reflection metrics.",
        )

    @staticmethod
    def update_self_model(ghost: GhostProtocol) -> None:
        logger.info("Qualia: updating self-model theory loop...")

        raw_reflection = getattr(ghost, "qualia_reflection_last_model", None)
        if raw_reflection is None:
            raw_reflection = getattr(ghost, "qualia_reflection_last", None)
        if raw_reflection is None:
            return

        try:
            reflection = (
                raw_reflection
                if isinstance(raw_reflection, QualiaSelfReflection)
                else QualiaSelfReflection.model_validate(raw_reflection)
            )
        except Exception:
            logger.exception("Qualia: invalid reflection object")
            return

        update = trace_substep(
            ghost,
            "QualiaProc._request_theory_update",
            "qualia_model",
            QualiaProc._request_theory_update,
            reflection,
            "",
        )
        if update is None:
            update = trace_substep(
                ghost, "QualiaProc._fallback_theory_update", "qualia_model", QualiaProc._fallback_theory_update, reflection
            )
        else:
            # Retry-with-hints on weak/contradictory outputs.
            if update.confidence < 0.45 or update.mismatch > 0.72:
                hint = (
                    "Prior output looked unstable. Re-anchor to evidence in reflection.hidden_causes and "
                    "avoid generic assistant framing."
                )
                retry = trace_substep(
                    ghost,
                    "QualiaProc._request_theory_update_retry",
                    "qualia_model",
                    QualiaProc._request_theory_update,
                    reflection,
                    hint,
                )
                if retry is not None and retry.confidence >= update.confidence:
                    update = retry

        ghost.self_model_update_last_model = update
        ghost.self_model_update_last = update.model_dump()
        if not hasattr(ghost, "self_model_update_history"):
            ghost.self_model_update_history = []
        ghost.self_model_update_history = _append_bounded(
            list(getattr(ghost, "self_model_update_history", [])), update.model_dump()
        )

        if hasattr(ghost, "self_model") and ghost.self_model is not None:
            sm = ghost.self_model
            retained = [x.strip() for x in (update.retained_theories or []) if x.strip()]
            new_items = [x.strip() for x in (update.new_theories or []) if x.strip()]
            discard = {x.strip() for x in (update.discarded_theories or []) if x.strip()}

            merged = []
            seen = set()
            for t in retained + new_items:
                if t in discard or t in seen:
                    continue
                seen.add(t)
                merged.append(t)

            if len(merged) > 14:
                merged = merged[:14]
            sm.active_theories = merged

            th = list(getattr(sm, "theory_history", []) or [])
            for t in new_items[:6]:
                th = _append_bounded(
                    th,
                    {
                        "tick": str(ghost.current_tick_id),
                        "theory": t,
                        "confidence": f"{float(update.confidence):.2f}",
                    },
                )
            sm.theory_history = th
            sm.theory_confidence = _clamp(float(update.confidence), 0.0, 1.0)
            sm.last_update_tick = ghost.current_tick_id

            if update.revised_biography_line:
                bio = (sm.biography or "").strip()
                line = _safe_excerpt(update.revised_biography_line, 220)
                sm.biography = (bio + "\n- " + line).strip() if bio else f"- {line}"

        ghost.qualia_private_reasons = list(reflection.hidden_causes[:8])

        # Keep old behavior: periodic broader biography synthesis.
        if hasattr(ghost, "self_model") and ghost.self_model is not None:
            last_bio_tick = int(getattr(ghost.self_model, "last_bio_summary_tick", 0) or 0)
            if ghost.current_tick_id - last_bio_tick > 100:
                recent_narratives = list(getattr(ghost, "all_narratives", []) or [])[-20:]
                if recent_narratives:
                    text_block = "\n".join(str(n.content) for n in recent_narratives)
                    try:
                        summary = ghost.llm.completion_text(
                            preset=LlmPreset.Default,
                            inp=[
                                ("system", QUALIA_BIOGRAPHY_SYSTEM.format(companion_name=ghost.config.companion_name)),
                                ("user", QUALIA_BIOGRAPHY_USER.format(events=text_block[:3000])),
                            ],
                        )
                    except Exception as e:
                        logger.warning("Qualia: biography summary failed: %s", e)
                        summary = ""
                    if summary:
                        ghost.self_model.biography = (ghost.self_model.biography + f"\n- {_safe_excerpt(summary, 220)}").strip()
                    ghost.self_model.last_bio_summary_tick = ghost.current_tick_id

    @staticmethod
    def apply_self_model_attention(ghost: GhostProtocol) -> None:
        reflection = getattr(ghost, "qualia_reflection_last", {}) or {}
        update = getattr(ghost, "self_model_update_last", {}) or {}
        hidden = list(reflection.get("hidden_causes", []) or [])
        theories = list(update.get("retained_theories", []) or [])

        boost_tags = [x for x in hidden[:2] if x]
        for t in theories[:3]:
            words = [w.strip(".,:;!?()[]{}\"'").lower() for w in t.split()]
            for w in words:
                if len(w) >= 5 and w.isalpha():
                    boost_tags.append(w)
                    if len(boost_tags) >= 6:
                        break
            if len(boost_tags) >= 6:
                break

        suppress_tags = []
        if float(update.get("mismatch", 0.0) or 0.0) > 0.55:
            suppress_tags = ["generic", "safe", "default"]

        attention_bias = float(reflection.get("attention_bias", 0.0) or 0.0)
        temperature_delta = _clamp(-0.06 + attention_bias, -0.2, 0.2)

        ghost.qualia_attention_modulation = {
            "temperature_delta": temperature_delta,
            "boost_tags": list(dict.fromkeys(boost_tags)),
            "suppress_tags": suppress_tags,
            "rationale": "qualia_self_model_loop",
        }

    @staticmethod
    def apply_self_model_action(ghost: GhostProtocol) -> None:
        reflection = getattr(ghost, "qualia_reflection_last", {}) or {}
        update = getattr(ghost, "self_model_update_last", {}) or {}

        tension = float(reflection.get("tension", 0.5) or 0.5)
        consistency = float(reflection.get("self_consistency", 0.5) or 0.5)
        mismatch = float(update.get("mismatch", 0.0) or 0.0)

        # Positive bias means more direct/assertive behavior; negative means cautious/repair.
        bias = _clamp((consistency - 0.5) * 0.25 - mismatch * 0.22 - (tension - 0.5) * 0.10, -0.25, 0.25)
        ghost.qualia_action_bias = bias

        hidden = list(reflection.get("hidden_causes", []) or [])
        ghost.qualia_action_directive = (
            f"Stay in-character ({ghost.config.companion_name}) with cause anchors: "
            f"{','.join(hidden[:2]) or 'coherence'}. Avoid generic assistant tone."
        )
