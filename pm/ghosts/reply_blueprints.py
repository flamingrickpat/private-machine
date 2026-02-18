import logging
from typing import Any, Dict, List, Tuple

from pm.dialog import DialogActPool, ReplyModality

logger = logging.getLogger(__name__)

_MODALITY_HINTS: Dict[str, List[str]] = {
    "answer_concise": ["concise", "short", "brief", "quick", "direct"],
    "answer_elaborate": ["detailed", "depth", "elaborate", "deep", "thorough"],
    "answer_directive": ["steps", "plan", "implement", "guide", "order", "next"],
    "answer_clarification": ["clarify", "ambiguous", "missing", "unclear", "question"],
    "answer_reflective": ["reflect", "mirror", "understand", "you feel"],
    "answer_acknowledgement": ["acknowledge", "noted", "got it", "understood"],
    "answer_disagree": ["wrong", "incorrect", "disagree", "conflict", "nonsense"],
    "answer_validation": ["validation", "heard", "valid", "feelings", "emotion"],
    "answer_empathic": ["empathic", "sorry", "struggle", "hard", "upset"],
    "answer_humor": ["humor", "joke", "playful", "banter", "funny"],
    "answer_storytelling": ["story", "narrative", "example", "scene"],
    "answer_analogy": ["analogy", "metaphor", "like", "as if"],
    "answer_meta": ["how you think", "reasoning", "process", "meta", "why did you"],
    "answer_caution": ["risk", "unsafe", "danger", "harm", "caution"],
    "answer_boundary_set": ["cannot", "won't", "boundary", "policy", "not allowed"],
    "answer_socratic": ["why", "what if", "question back", "socratic"],
}


def _lower_blob(parts: List[str]) -> str:
    return " ".join(str(p or "") for p in parts).lower()


def _safe_text(x: Any) -> str:
    return str(x or "").strip()


def _modality_items(pool: DialogActPool) -> List[Tuple[str, ReplyModality]]:
    rows: List[Tuple[str, ReplyModality]] = []
    for field_name in type(pool).model_fields:
        item = getattr(pool, field_name, None)
        if isinstance(item, ReplyModality):
            rows.append((field_name, item))
    return rows


def choose_reply_blueprint(
    *,
    tick: int,
    behavior: Any,
    broadcast: str,
    sim_prediction: str = "",
    simulation_bundle: Dict[str, Any] | None = None,
    history: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """
    Pick a reply modality from DialogActPool with:
    - base bias from modality.bias
    - lexical hint matching against action/broadcast/simulation text
    - mild anti-repeat penalty for recent selections
    """
    pool = DialogActPool()
    rows = _modality_items(pool)
    history = list(history or [])

    text = _lower_blob(
        [
            getattr(behavior, "action_description", ""),
            getattr(behavior, "speech", ""),
            getattr(behavior, "internal_thought", ""),
            broadcast,
            sim_prediction,
            (simulation_bundle or {}).get("winner_lens", ""),
            ",".join((simulation_bundle or {}).get("consolidated_attractors", [])[:4]),
            ",".join((simulation_bundle or {}).get("consolidated_repellers", [])[:4]),
        ]
    )

    recent_names = [str(h.get("name", "")) for h in history[-4:]]
    scored: List[Tuple[float, str, ReplyModality, List[str]]] = []
    for name, mod in rows:
        score = float(mod.bias)
        matched: List[str] = []

        for hint in _MODALITY_HINTS.get(name, []):
            if hint in text:
                score += 0.08
                matched.append(hint)

        if name in recent_names:
            score -= 0.12

        # Stable tie-break nudging for diversity without randomness in tests.
        score += ((hash(f"{tick}:{name}") % 17) / 1000.0)
        scored.append((score, name, mod, matched))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_name, best_mod, matched = scored[0]
    second = scored[1][1] if len(scored) > 1 else ""
    result = {
        "name": best_name,
        "description": _safe_text(best_mod.description),
        "generation_prefix": _safe_text(best_mod.generation_prefix),
        "bias": float(best_mod.bias),
        "score": round(float(best_score), 4),
        "matched_hints": matched,
        "runner_up": second,
    }
    logger.debug("Reply blueprint selected tick=%s name=%s score=%.3f", tick, best_name, best_score)
    return result
