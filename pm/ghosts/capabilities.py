import re
from typing import Dict, List, Optional, Tuple

from pm.ghosts.schemas import BehaviorOutput


def _normalize_text(text: str) -> str:
    t = (text or "").lower().strip()
    t = re.sub(r"\bvr\b", "virtual reality", t)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^a-z0-9\s]", "", t)
    return t


_CAPABILITY_STOPWORDS = {
    "can",
    "could",
    "able",
    "you",
    "your",
    "are",
    "have",
    "has",
    "do",
    "does",
    "please",
}


def _tokenize(text: str) -> List[str]:
    return [tok for tok in _normalize_text(text).split() if len(tok) > 2 and tok not in _CAPABILITY_STOPWORDS]


def _safe_list(value) -> List[str]:
    if not value:
        return []
    out = []
    for x in value:
        s = str(x or "").strip()
        if s:
            out.append(s)
    return out


def capability_profile(ghost) -> Dict[str, List[str]]:
    cfg = getattr(ghost, "config", None)
    supported = _safe_list(getattr(cfg, "supported_capabilities", []))
    unsupported = _safe_list(getattr(cfg, "unsupported_capabilities", []))
    notes = _safe_list(getattr(cfg, "capability_notes", []))
    return {
        "supported": supported,
        "unsupported": unsupported,
        "notes": notes,
    }


def capability_context_text(ghost) -> str:
    p = capability_profile(ghost)
    supported = p["supported"]
    unsupported = p["unsupported"]
    notes = p["notes"]
    lines: List[str] = []
    if supported:
        lines.append("Supported capabilities:")
        lines.extend([f"- {s}" for s in supported])
    if unsupported:
        lines.append("Unsupported capabilities:")
        lines.extend([f"- {s}" for s in unsupported])
    if notes:
        lines.append("Constraints:")
        lines.extend([f"- {n}" for n in notes])
    return "\n".join(lines) if lines else "No capability profile configured."


def _match_statement(text: str, statements: List[str]) -> Optional[str]:
    text_norm = _normalize_text(text)
    text_tokens = set(_tokenize(text))
    if not text_norm or not statements:
        return None

    best: Optional[Tuple[str, float]] = None
    for st in statements:
        st_norm = _normalize_text(st)
        st_tokens = set(_tokenize(st))
        if not st_norm:
            continue
        # direct substring match is strong and deterministic
        if st_norm in text_norm or text_norm in st_norm:
            return st
        # token overlap match for paraphrases
        if not st_tokens or not text_tokens:
            continue
        overlap = len(st_tokens & text_tokens) / max(1, min(len(st_tokens), len(text_tokens)))
        if overlap >= 0.30 and (best is None or overlap > best[1]):
            best = (st, overlap)
    return best[0] if best else None


def resolve_capability_question(user_text: str, ghost) -> Optional[str]:
    text = str(user_text or "").strip()
    if not text:
        return None
    low = _normalize_text(text)
    if not any(q in low for q in ["can you", "are you able", "could you", "do you have", "are you capable"]):
        return None

    p = capability_profile(ghost)
    unsupported_hit = _match_statement(text, p["unsupported"])
    if unsupported_hit:
        fallback = p["supported"][0] if p["supported"] else "I can communicate with you via text."
        return f"I can't do that in this architecture. {fallback}"

    supported_hit = _match_statement(text, p["supported"])
    if supported_hit:
        return f"Yes, within this architecture I can. {supported_hit}"

    return None


def enforce_behavior_capability(ghost, behavior: BehaviorOutput, user_request: str = "") -> Tuple[BehaviorOutput, Optional[Dict[str, str]]]:
    p = capability_profile(ghost)
    unsupported = p["unsupported"]
    supported = p["supported"]

    cap_answer = resolve_capability_question(user_request, ghost)
    if cap_answer:
        fixed = behavior.model_copy(
            update={
                "action_description": "Answer capability question with architecture-grounded constraints.",
                "speech": cap_answer,
                "internal_thought": (
                    f"{behavior.internal_thought} | capability_guard: answer explicit capability question from profile."
                ),
                "tool_call": None,
            }
        )
        return fixed, {"reason": "capability_question", "details": cap_answer}

    texts = [
        str(getattr(behavior, "action_description", "") or ""),
        str(getattr(behavior, "speech", "") or ""),
        str(getattr(behavior, "tool_call", "") or ""),
    ]
    joined = "\n".join(texts)
    hit = _match_statement(joined, unsupported)
    if not hit:
        return behavior, None

    fallback = supported[0] if supported else "I can only continue this interaction through text."
    guard_speech = (
        "I can't do that directly in this system. "
        f"{fallback} I can still help by planning it, describing it, or giving concrete steps."
    )
    fixed = behavior.model_copy(
        update={
            "action_description": "Acknowledge architecture limits and provide feasible text-only assistance.",
            "speech": guard_speech,
            "internal_thought": (
                f"{behavior.internal_thought} | capability_guard: blocked impossible action, switched to feasible text support."
            ),
            "tool_call": None,
        }
    )
    return fixed, {"reason": "unsupported_action", "details": hit}


def enforce_reply_text_capability(ghost, reply_text: str, user_request: str = "") -> Tuple[str, Optional[Dict[str, str]]]:
    text = str(reply_text or "").strip()
    if not text:
        return text, None

    p = capability_profile(ghost)
    unsupported = p["unsupported"]
    supported = p["supported"]

    # If user asked an explicit capability question, force an architecture-grounded answer.
    cap_answer = resolve_capability_question(user_request, ghost)
    if cap_answer:
        return cap_answer, {"reason": "capability_question", "details": cap_answer}

    hit = _match_statement(text, unsupported)
    if not hit:
        return text, None

    fallback = supported[0] if supported else "I can only continue this interaction through text."
    guarded = (
        "I can't do that directly in this architecture. "
        f"{fallback} I can still help by planning it or giving concrete text-only steps."
    )
    return guarded, {"reason": "unsupported_reply", "details": hit}
