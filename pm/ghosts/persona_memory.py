import re
from typing import List, Tuple

from pm.data_structures import CauseEffectKnoxel, DeclarativeFactKnoxel


def _norm(text: str) -> str:
    t = str(text or "").lower().strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^a-z0-9\s]", "", t)
    return t


def _token_overlap(a: str, b: str) -> float:
    sa = set(_norm(a).split())
    sb = set(_norm(b).split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _excerpt(text: str, max_len: int = 180) -> str:
    t = str(text or "").replace("\n", " ").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 3].rstrip() + "..."


def collect_persona_signals(ghost, query: str = "", limit: int = 6) -> List[str]:
    """
    Retrieve persisted persona quirks from CauseEffect + Fact memory in ranked order.
    Ranking combines query similarity and recency.
    """
    query = str(query or "")
    rows: List[Tuple[float, str]] = []

    for k in list(getattr(ghost, "all_knoxels", {}).values()):
        if isinstance(k, CauseEffectKnoxel):
            cat = str(getattr(k, "category", "") or "")
            md = getattr(k, "metadata", {}) or {}
            if "persona" not in cat and not bool(md.get("persona_signal", False)):
                continue
            text = f"{k.situation} | {k.cause} -> {k.effect}"
            rec = min(1.0, max(0.0, (int(getattr(k, "tick_id", 0) or 0) + 1) / (int(getattr(ghost, 'current_tick_id', 1) or 1) + 1)))
            score = _token_overlap(query, text) * 0.75 + rec * 0.25
            rows.append((score, f"CE: {_excerpt(text, 220)}"))
        elif isinstance(k, DeclarativeFactKnoxel):
            md = getattr(k, "metadata", {}) or {}
            if not bool(md.get("persona_signal", False)):
                continue
            rec = min(1.0, max(0.0, (int(getattr(k, "tick_id", 0) or 0) + 1) / (int(getattr(ghost, 'current_tick_id', 1) or 1) + 1)))
            score = _token_overlap(query, str(getattr(k, "content", "") or "")) * 0.75 + rec * 0.25
            rows.append((score, f"FACT: {_excerpt(str(getattr(k, 'content', '') or ''), 220)}"))

    rows.sort(key=lambda x: x[0], reverse=True)
    out: List[str] = []
    seen = set()
    for _, text in rows:
        key = _norm(text)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= max(1, int(limit)):
            break
    return out
