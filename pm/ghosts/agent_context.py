from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import math
import re

from pm.data_structures import ClusterType, Feature, FeatureType, KnoxelBase, MemoryClusterKnoxel
from pm.ghosts.capabilities import capability_context_text
from pm.ghosts.persona_memory import collect_persona_signals
from pm.llm.llm_common import LlmPreset
from pm.utils.emb_utils import cosine_sim
from pm.utils.token_utils import get_token_count


def _norm(text: str) -> str:
    t = str(text or "").lower().strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^a-z0-9\s]", "", t)
    return t


def _excerpt(text: str, max_len: int = 240) -> str:
    t = str(text or "").replace("\n", " ").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 3].rstrip() + "..."


def _token_overlap(a: str, b: str) -> float:
    sa = set(_norm(a).split())
    sb = set(_norm(b).split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _safe_token_count(text: str) -> int:
    try:
        return max(0, int(get_token_count(str(text or "")) or 0))
    except Exception:
        return max(0, len(str(text or "").split()))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_embedding(value: Any) -> List[float]:
    if not isinstance(value, list):
        return []
    out: List[float] = []
    for x in value:
        try:
            out.append(float(x))
        except Exception:
            return []
    return out


def _sim_score(query_emb: List[float], item_emb: List[float], query_text: str, item_text: str) -> float:
    if query_emb and item_emb and len(query_emb) == len(item_emb):
        try:
            return max(0.0, float(cosine_sim(query_emb, item_emb)))
        except Exception:
            pass
    return _token_overlap(query_text, item_text)


def _fmt_dt(dt: Optional[datetime]) -> str:
    if not isinstance(dt, datetime):
        return "unknown"
    return dt.strftime("%Y-%m-%d %H:%M")


def _desired_temporal_level(start: Optional[datetime], end: Optional[datetime]) -> int:
    if not isinstance(start, datetime) or not isinstance(end, datetime):
        return 5
    span_h = max(0.01, (end - start).total_seconds() / 3600.0)
    if span_h <= 24:
        return 6
    if span_h <= 24 * 7:
        return 5
    if span_h <= 24 * 31:
        return 4
    if span_h <= 24 * 120:
        return 3
    if span_h <= 24 * 240:
        return 2
    return 1


def _temporal_label(level: int) -> str:
    mapping = {
        6: "time_of_day",
        5: "day",
        4: "week",
        3: "month",
        2: "season",
        1: "year",
    }
    return mapping.get(int(level), f"level_{int(level)}")


@dataclass(frozen=True)
class AgentContextWeights:
    workspace: float = 0.25
    latest: float = 0.30
    timeline: float = 0.30
    static: float = 0.15

    @staticmethod
    def from_config(raw: Dict[str, Any]) -> "AgentContextWeights":
        if not isinstance(raw, dict):
            return AgentContextWeights()
        return AgentContextWeights(
            workspace=max(0.0, _safe_float(raw.get("workspace", 0.25), 0.25)),
            latest=max(0.0, _safe_float(raw.get("latest", 0.30), 0.30)),
            timeline=max(0.0, _safe_float(raw.get("timeline", 0.30), 0.30)),
            static=max(0.0, _safe_float(raw.get("static", 0.15), 0.15)),
        )

    def as_dict(self) -> Dict[str, float]:
        return {
            "workspace": float(self.workspace),
            "latest": float(self.latest),
            "timeline": float(self.timeline),
            "static": float(self.static),
        }


@dataclass(frozen=True)
class AgentContextConfig:
    max_tokens: int
    weights: AgentContextWeights = field(default_factory=AgentContextWeights)
    latest_messages: int = 40
    workspace_items: int = 48
    min_section_tokens: int = 96


@dataclass
class AgentContextPacket:
    text: str
    sections: Dict[str, str]
    token_usage: Dict[str, int]
    selected_ids: List[int]
    dialogue_lines: List[str]
    fact_lines: List[str]
    debug: Dict[str, Any]


def compute_agent_context_budget(
    ghost,
    *,
    output_tokens: int,
    fallback_ctx: int = 2048,
    min_budget: int = 320,
) -> int:
    model_ctx = fallback_ctx
    try:
        model_ctx = int(ghost.llm.get_max_tokens(LlmPreset.Default))
    except Exception:
        model_ctx = fallback_ctx

    ratio = _safe_float(getattr(getattr(ghost, "config", None), "agent_context_target_ratio", 0.72), 0.72)
    ratio = max(0.15, min(0.95, ratio))
    margin = int(getattr(getattr(ghost, "config", None), "agent_context_safety_margin_tokens", 220) or 220)

    hard_cap = max(128, int(model_ctx - int(output_tokens) - margin))
    ratio_cap = max(128, int(model_ctx * ratio))
    budget = min(hard_cap, ratio_cap)
    budget = max(128, min(hard_cap, max(min_budget, budget)))
    return int(budget)


class AgentContextComposer:
    SECTION_ORDER = ["workspace", "latest", "timeline", "static"]

    @staticmethod
    def build(
        ghost,
        *,
        focus_text: str,
        config: AgentContextConfig,
        purpose: str = "general",
    ) -> AgentContextPacket:
        max_tokens = max(128, int(config.max_tokens))
        focus_text = str(focus_text or "").strip()
        focus_emb = AgentContextComposer._query_embedding(ghost, focus_text)
        weights = config.weights.as_dict()

        workspace_candidates = AgentContextComposer._workspace_candidates(ghost)
        latest_features = AgentContextComposer._latest_features(
            ghost, limit=max(1, int(config.latest_messages))
        )
        timeline_features = AgentContextComposer._timeline_features(ghost)
        temporal_clusters = AgentContextComposer._temporal_clusters(ghost)

        available = {
            "workspace": bool(workspace_candidates),
            "latest": bool(latest_features),
            "timeline": bool(timeline_features),
            "static": True,
        }
        budgets = AgentContextComposer._allocate_section_budgets(
            max_tokens=max_tokens,
            weights=weights,
            available=available,
            min_section_tokens=max(0, int(config.min_section_tokens)),
        )

        sections: Dict[str, str] = {}
        token_usage: Dict[str, int] = {}
        selected_ids: List[int] = []
        dialogue_lines: List[str] = []
        fact_lines: List[str] = []
        section_debug: Dict[str, Any] = {}

        for name in AgentContextComposer.SECTION_ORDER:
            budget = int(budgets.get(name, 0) or 0)
            if budget <= 0:
                continue
            if name == "workspace":
                text, ids, dbg = AgentContextComposer._render_workspace(
                    ghost=ghost,
                    focus_text=focus_text,
                    focus_emb=focus_emb,
                    budget=budget,
                    workspace_candidates=workspace_candidates,
                    max_items=max(1, int(config.workspace_items)),
                )
                if text:
                    sections[name] = text
                    token_usage[name] = _safe_token_count(text)
                    selected_ids.extend(ids)
                    section_debug[name] = dbg
            elif name == "latest":
                text, ids, dlg, dbg = AgentContextComposer._render_latest(
                    ghost=ghost,
                    budget=budget,
                    latest_features=latest_features,
                )
                if text:
                    sections[name] = text
                    token_usage[name] = _safe_token_count(text)
                    selected_ids.extend(ids)
                    dialogue_lines.extend(dlg)
                    section_debug[name] = dbg
            elif name == "timeline":
                text, ids, dbg = AgentContextComposer._render_timeline(
                    ghost=ghost,
                    focus_text=focus_text,
                    focus_emb=focus_emb,
                    budget=budget,
                    timeline_features=timeline_features,
                    temporal_clusters=temporal_clusters,
                )
                if text:
                    sections[name] = text
                    token_usage[name] = _safe_token_count(text)
                    selected_ids.extend(ids)
                    section_debug[name] = dbg
            elif name == "static":
                text, facts, dbg = AgentContextComposer._render_static(
                    ghost=ghost,
                    focus_text=focus_text,
                    focus_emb=focus_emb,
                    budget=budget,
                )
                if text:
                    sections[name] = text
                    token_usage[name] = _safe_token_count(text)
                    fact_lines.extend(facts)
                    section_debug[name] = dbg

        final_lines: List[str] = []
        for name in AgentContextComposer.SECTION_ORDER:
            section_text = str(sections.get(name, "") or "").strip()
            if not section_text:
                continue
            final_lines.append(f"[{name.upper()}]")
            final_lines.append(section_text)
        final_text = "\n".join(final_lines).strip()
        final_text = AgentContextComposer._fit_full_context(final_text, max_tokens)

        debug = {
            "purpose": str(purpose or ""),
            "focus_excerpt": _excerpt(focus_text, 220),
            "max_tokens": int(max_tokens),
            "weights": dict(weights),
            "budgets": dict(budgets),
            "token_usage": dict(token_usage),
            "sections_present": [k for k in AgentContextComposer.SECTION_ORDER if k in sections],
            "selected_count": len({int(x) for x in selected_ids if isinstance(x, int)}),
            "timeline_feature_count": len(timeline_features),
            "temporal_cluster_count": len(temporal_clusters),
            "section_debug": section_debug,
            "final_tokens": _safe_token_count(final_text),
        }

        AgentContextComposer._persist_debug(ghost, debug)
        return AgentContextPacket(
            text=final_text,
            sections=sections,
            token_usage=token_usage,
            selected_ids=sorted({int(x) for x in selected_ids if isinstance(x, int) and int(x) >= 0}),
            dialogue_lines=dialogue_lines,
            fact_lines=fact_lines,
            debug=debug,
        )

    @staticmethod
    def _fit_full_context(text: str, max_tokens: int) -> str:
        if _safe_token_count(text) <= max_tokens:
            return text
        lines = [ln for ln in str(text or "").splitlines() if ln.strip()]
        if not lines:
            return ""
        while lines and _safe_token_count("\n".join(lines)) > max_tokens:
            lines.pop(0)
        fitted = "\n".join(lines).strip()
        if _safe_token_count(fitted) <= max_tokens:
            return fitted
        tokens = max(1, _safe_token_count(fitted))
        ratio = max(0.05, float(max_tokens) / float(tokens))
        chars = max(32, int(len(fitted) * ratio))
        return _excerpt(fitted, chars)

    @staticmethod
    def _persist_debug(ghost, debug: Dict[str, Any]) -> None:
        ghost.agent_context_last = dict(debug)
        hist = list(getattr(ghost, "agent_context_history", []) or [])
        hist.append(dict(debug))
        if len(hist) > 100:
            hist = hist[-100:]
        ghost.agent_context_history = hist

        ghost.story_context_last = {
            "focus_excerpt": str(debug.get("focus_excerpt", "")),
            "selected_count": int(debug.get("selected_count", 0) or 0),
            "history_len": len(hist),
            "purpose": str(debug.get("purpose", "")),
            "max_tokens": int(debug.get("max_tokens", 0) or 0),
            "final_tokens": int(debug.get("final_tokens", 0) or 0),
            "section_tokens": dict(debug.get("token_usage", {}) or {}),
        }

    @staticmethod
    def _query_embedding(ghost, focus_text: str) -> List[float]:
        key = f"{int(getattr(ghost, 'current_tick_id', 0) or 0)}::{_norm(focus_text)[:320]}"
        cache = dict(getattr(ghost, "agent_context_embedding_cache", {}) or {})
        if key in cache:
            return list(cache[key] or [])
        emb: List[float] = []
        try:
            if hasattr(ghost, "llm") and hasattr(ghost.llm, "get_embedding"):
                emb = list(ghost.llm.get_embedding(focus_text) or [])
        except Exception:
            emb = []
        cache[key] = list(emb)
        if len(cache) > 24:
            keys = list(cache.keys())[-24:]
            cache = {k: cache[k] for k in keys}
        ghost.agent_context_embedding_cache = cache
        return emb

    @staticmethod
    def _allocate_section_budgets(
        *,
        max_tokens: int,
        weights: Dict[str, float],
        available: Dict[str, bool],
        min_section_tokens: int,
    ) -> Dict[str, int]:
        active = [s for s in AgentContextComposer.SECTION_ORDER if available.get(s, False)]
        if not active:
            return {}

        header_overhead = 10 * len(active)
        token_pool = max(64, int(max_tokens) - header_overhead)

        weight_sum = sum(max(0.0, float(weights.get(s, 0.0) or 0.0)) for s in active)
        if weight_sum <= 0.0:
            weight_sum = float(len(active))
            weights = {s: 1.0 for s in active}

        raw = {s: (max(0.0, float(weights.get(s, 0.0) or 0.0)) / weight_sum) * token_pool for s in active}
        budget = {s: int(math.floor(raw[s])) for s in active}
        remainder = token_pool - sum(budget.values())
        for s in sorted(active, key=lambda x: raw[x] - math.floor(raw[x]), reverse=True):
            if remainder <= 0:
                break
            budget[s] += 1
            remainder -= 1

        if min_section_tokens > 0:
            needy = [s for s in active if budget[s] < min_section_tokens]
            while needy:
                donors = [s for s in active if s not in needy and budget[s] > min_section_tokens]
                if not donors:
                    break
                for s in needy:
                    if budget[s] >= min_section_tokens:
                        continue
                    donor = max(donors, key=lambda d: budget[d])
                    move = min(min_section_tokens - budget[s], max(0, budget[donor] - min_section_tokens))
                    if move <= 0:
                        continue
                    budget[s] += move
                    budget[donor] -= move
                needy = [s for s in active if budget[s] < min_section_tokens]

        return budget

    @staticmethod
    def _workspace_candidates(ghost) -> List[KnoxelBase]:
        out: List[KnoxelBase] = []
        seen = set()

        def add_item(k: Any) -> None:
            if k is None:
                return
            kid = int(getattr(k, "id", -1) or -1)
            sig = (type(k).__name__, kid, str(getattr(k, "content", "") or ""))
            if sig in seen:
                return
            seen.add(sig)
            out.append(k)

        add_item(getattr(ghost, "conscious_broadcast", None))
        for k in list(getattr(ghost, "conscious_candidates", []) or []):
            add_item(k)
        for entry in list(getattr(ghost, "current_coalition", []) or []):
            try:
                kid = int(entry[0])
            except Exception:
                continue
            add_item(getattr(ghost, "get_knoxel_by_id", lambda _: None)(kid))
        return out

    @staticmethod
    def _latest_features(ghost, *, limit: int) -> List[Feature]:
        feats = [
            f
            for f in list(getattr(ghost, "all_features", []) or [])
            if isinstance(f, Feature)
            and getattr(f, "feature_type", None) in (FeatureType.Dialogue, FeatureType.SystemMessage)
        ]
        feats.sort(key=lambda x: (getattr(x, "timestamp_world_begin", datetime.min), int(getattr(x, "id", 0) or 0)))
        if limit <= 0:
            return feats
        return feats[-limit:]

    @staticmethod
    def _timeline_features(ghost) -> List[Feature]:
        feats = [
            f
            for f in list(getattr(ghost, "all_features", []) or [])
            if isinstance(f, Feature)
            and bool(getattr(f, "causal", False))
            and getattr(f, "feature_type", None) in (FeatureType.Dialogue, FeatureType.SystemMessage)
        ]
        feats.sort(key=lambda x: (getattr(x, "timestamp_world_begin", datetime.min), int(getattr(x, "id", 0) or 0)))
        if feats:
            return feats
        return AgentContextComposer._latest_features(ghost, limit=24)

    @staticmethod
    def _temporal_clusters(ghost) -> List[MemoryClusterKnoxel]:
        out: List[MemoryClusterKnoxel] = []
        for m in list(getattr(ghost, "all_episodic_memories", []) or []):
            if not isinstance(m, MemoryClusterKnoxel):
                continue
            if getattr(m, "cluster_type", None) != ClusterType.Temporal:
                continue
            if not str(getattr(m, "content", "") or "").strip():
                continue
            out.append(m)
        out.sort(key=lambda x: (getattr(x, "timestamp_world_begin", datetime.min), int(getattr(x, "id", 0) or 0)))
        return out

    @staticmethod
    def _render_workspace(
        *,
        ghost,
        focus_text: str,
        focus_emb: List[float],
        budget: int,
        workspace_candidates: List[KnoxelBase],
        max_items: int,
    ) -> Tuple[str, List[int], Dict[str, Any]]:
        csm_state = getattr(getattr(ghost, "csm_manager", None), "state", None)
        csm_items = dict(getattr(csm_state, "csm_item_states", {}) or {})
        tick_now = int(getattr(ghost, "current_tick_id", 0) or 0)

        ranked: List[Tuple[float, KnoxelBase]] = []
        for k in workspace_candidates:
            kid = int(getattr(k, "id", -1) or -1)
            sim = _sim_score(
                focus_emb,
                _safe_embedding(getattr(k, "embedding", []) or []),
                focus_text,
                str(getattr(k, "content", "") or ""),
            )
            salience = _safe_float(getattr(k, "incentive_salience", 0.0), 0.0)
            activation = _safe_float(getattr(csm_items.get(kid, None), "activation", 0.0), 0.0)
            kt = int(getattr(k, "tick_id", 0) or 0)
            dist = max(0, tick_now - kt)
            recency = 1.0 / (1.0 + math.log1p(dist))
            score = 0.50 * sim + 0.25 * activation + 0.15 * salience + 0.10 * recency
            ranked.append((score, k))
        ranked.sort(key=lambda x: x[0], reverse=True)

        lines: List[str] = []
        ids: List[int] = []
        used = 0
        for _score, k in ranked[: max(1, int(max_items))]:
            try:
                line = str(k.get_story_element(ghost))
            except Exception:
                line = str(getattr(k, "content", "") or "")
            line = line.strip()
            if not line:
                continue
            tc = _safe_token_count(line)
            if used + tc > budget:
                continue
            lines.append(line)
            used += tc
            kid = int(getattr(k, "id", -1) or -1)
            if kid >= 0:
                ids.append(kid)
        return "\n".join(lines), ids, {"budget": int(budget), "used": int(used), "candidate_count": len(ranked)}

    @staticmethod
    def _render_latest(
        *,
        ghost,
        budget: int,
        latest_features: List[Feature],
    ) -> Tuple[str, List[int], List[str], Dict[str, Any]]:
        chosen_rev: List[Feature] = []
        used = 0
        for f in reversed(latest_features):
            try:
                line = str(f.get_story_element(ghost))
            except Exception:
                line = str(getattr(f, "content", "") or "")
            tc = _safe_token_count(line)
            if used + tc > budget:
                continue
            chosen_rev.append(f)
            used += tc
        chosen = list(reversed(chosen_rev))

        lines: List[str] = []
        ids: List[int] = []
        dialogue_lines: List[str] = []
        for f in chosen:
            try:
                line = str(f.get_story_element(ghost))
            except Exception:
                line = str(getattr(f, "content", "") or "")
            if not line.strip():
                continue
            lines.append(line)
            fid = int(getattr(f, "id", -1) or -1)
            if fid >= 0:
                ids.append(fid)
            if getattr(f, "feature_type", None) == FeatureType.Dialogue:
                src = str(getattr(f, "source", "") or "unknown")
                dialogue_lines.append(f"{src}: {str(getattr(f, 'content', '') or '').strip()}")

        return (
            "\n".join(lines),
            ids,
            dialogue_lines,
            {"budget": int(budget), "used": int(used), "selected_count": len(chosen)},
        )

    @staticmethod
    def _cluster_overlap(cluster: MemoryClusterKnoxel, start: datetime, end: datetime) -> float:
        cb = getattr(cluster, "timestamp_world_begin", None)
        ce = getattr(cluster, "timestamp_world_end", None)
        if not isinstance(cb, datetime) or not isinstance(ce, datetime):
            return 0.0
        start_t = max(cb, start)
        end_t = min(ce, end)
        if end_t <= start_t:
            return 0.0
        span = max(1.0, (end - start).total_seconds())
        return max(0.0, min(1.0, (end_t - start_t).total_seconds() / span))

    @staticmethod
    def _summary_for_window(
        *,
        start: datetime,
        end: datetime,
        count: int,
        clusters: List[MemoryClusterKnoxel],
        focus_emb: List[float],
        focus_text: str,
    ) -> str:
        desired = _desired_temporal_level(start, end)
        best: Optional[Tuple[float, MemoryClusterKnoxel]] = None
        for c in clusters:
            overlap = AgentContextComposer._cluster_overlap(c, start, end)
            if overlap <= 0.0:
                continue
            rel = _sim_score(
                focus_emb,
                _safe_embedding(getattr(c, "embedding", []) or []),
                focus_text,
                str(getattr(c, "content", "") or ""),
            )
            level_fit = 1.0 - min(1.0, abs(int(getattr(c, "level", desired) or desired) - desired) / 6.0)
            score = 0.45 * overlap + 0.35 * rel + 0.20 * level_fit
            if best is None or score > best[0]:
                best = (score, c)

        if best is not None:
            c = best[1]
            level = _temporal_label(int(getattr(c, "level", desired) or desired))
            summary = _excerpt(str(getattr(c, "content", "") or ""), 260)
            return f"{_fmt_dt(start)}..{_fmt_dt(end)} [{level}] {summary}"

        level = _temporal_label(desired)
        return (
            f"{_fmt_dt(start)}..{_fmt_dt(end)} [{level}] "
            f"{max(1, int(count))} lower-salience events compressed."
        )

    @staticmethod
    def _render_timeline(
        *,
        ghost,
        focus_text: str,
        focus_emb: List[float],
        budget: int,
        timeline_features: List[Feature],
        temporal_clusters: List[MemoryClusterKnoxel],
    ) -> Tuple[str, List[int], Dict[str, Any]]:
        if not timeline_features:
            return "", [], {"budget": int(budget), "used": 0, "segment_count": 0}

        tick_now = int(getattr(ghost, "current_tick_id", 0) or 0)
        scored: List[Tuple[float, Feature]] = []
        for f in timeline_features:
            sim = _sim_score(
                focus_emb,
                _safe_embedding(getattr(f, "embedding", []) or []),
                focus_text,
                str(getattr(f, "content", "") or ""),
            )
            salience = _safe_float(getattr(f, "incentive_salience", 0.0), 0.0)
            kt = int(getattr(f, "tick_id", 0) or 0)
            dist = max(0, tick_now - kt)
            recency = 1.0 / (1.0 + math.log1p(dist))
            score = 0.65 * sim + 0.20 * recency + 0.15 * salience
            scored.append((score, f))
        scored.sort(key=lambda x: x[0], reverse=True)

        important_count = max(4, min(32, int(len(timeline_features) * 0.14)))
        important_ids = {int(getattr(f, "id", -1) or -1) for _, f in scored[:important_count]}
        for f in timeline_features[-4:]:
            important_ids.add(int(getattr(f, "id", -1) or -1))

        feature_scores = {int(getattr(f, "id", -1) or -1): score for score, f in scored}

        segments: List[Dict[str, Any]] = []
        run: List[Feature] = []

        def flush_run() -> None:
            if not run:
                return
            start = getattr(run[0], "timestamp_world_begin", None)
            end = getattr(run[-1], "timestamp_world_end", None) or getattr(run[-1], "timestamp_world_begin", None)
            line = AgentContextComposer._summary_for_window(
                start=start if isinstance(start, datetime) else datetime.min,
                end=end if isinstance(end, datetime) else datetime.min,
                count=len(run),
                clusters=temporal_clusters,
                focus_emb=focus_emb,
                focus_text=focus_text,
            )
            segments.append(
                {
                    "kind": "summary",
                    "line": line,
                    "tokens": _safe_token_count(line),
                    "start": start,
                    "end": end,
                    "importance": 0.0,
                    "ids": [],
                    "count": len(run),
                }
            )
            run.clear()

        for f in timeline_features:
            fid = int(getattr(f, "id", -1) or -1)
            if fid in important_ids:
                flush_run()
                try:
                    line = str(f.get_story_element(ghost))
                except Exception:
                    line = str(getattr(f, "content", "") or "")
                segments.append(
                    {
                        "kind": "raw",
                        "line": line.strip(),
                        "tokens": _safe_token_count(line),
                        "start": getattr(f, "timestamp_world_begin", None),
                        "end": getattr(f, "timestamp_world_end", None) or getattr(f, "timestamp_world_begin", None),
                        "importance": float(feature_scores.get(fid, 0.0)),
                        "ids": [fid] if fid >= 0 else [],
                        "count": 1,
                    }
                )
            else:
                run.append(f)
        flush_run()

        def total_tokens() -> int:
            return sum(int(seg.get("tokens", 0) or 0) for seg in segments)

        while len(segments) > 1 and total_tokens() > budget:
            a = segments.pop(0)
            b = segments.pop(0)
            start = a.get("start", None) or b.get("start", None)
            end = b.get("end", None) or a.get("end", None)
            count = int(a.get("count", 0) or 0) + int(b.get("count", 0) or 0)
            if not isinstance(start, datetime):
                start = datetime.min
            if not isinstance(end, datetime):
                end = start
            merged_line = AgentContextComposer._summary_for_window(
                start=start,
                end=end,
                count=count,
                clusters=temporal_clusters,
                focus_emb=focus_emb,
                focus_text=focus_text,
            )
            merged = {
                "kind": "summary",
                "line": merged_line,
                "tokens": _safe_token_count(merged_line),
                "start": start,
                "end": end,
                "importance": min(_safe_float(a.get("importance", 0.0), 0.0), _safe_float(b.get("importance", 0.0), 0.0)),
                "ids": [],
                "count": count,
            }
            segments.insert(0, merged)

        while len(segments) > 1 and total_tokens() > budget:
            if str(segments[0].get("kind", "")) == "summary":
                segments.pop(0)
            else:
                break

        lines: List[str] = []
        ids: List[int] = []
        used = 0
        for seg in segments:
            line = str(seg.get("line", "") or "").strip()
            if not line:
                continue
            tc = _safe_token_count(line)
            if used + tc > budget:
                continue
            used += tc
            lines.append(line)
            ids.extend([int(x) for x in list(seg.get("ids", []) or []) if int(x) >= 0])

        return "\n".join(lines), ids, {"budget": int(budget), "used": int(used), "segment_count": len(segments)}

    @staticmethod
    def _render_static(
        *,
        ghost,
        focus_text: str,
        focus_emb: List[float],
        budget: int,
    ) -> Tuple[str, List[str], Dict[str, Any]]:
        lines: List[str] = []
        fact_lines: List[str] = []
        used = 0

        def try_add(line: str) -> bool:
            nonlocal used
            line = str(line or "").strip()
            if not line:
                return False
            tc = _safe_token_count(line)
            if used + tc > budget:
                return False
            lines.append(line)
            used += tc
            return True

        card = str(getattr(getattr(ghost, "config", None), "universal_character_card", "") or "").strip()
        if card:
            card_lines = [ln.strip() for ln in card.splitlines() if ln.strip()]
            if card_lines:
                try_add("Character card:")
                for ln in card_lines:
                    if not try_add(f"- {ln}"):
                        break

        persona_signals = collect_persona_signals(ghost, query=focus_text, limit=6)
        if persona_signals:
            try_add("Persona signals:")
            for sig in persona_signals:
                if not try_add(f"- {sig}"):
                    break

        facts = []
        for f in list(getattr(ghost, "all_declarative_facts", []) or []):
            td = _safe_float(getattr(f, "time_dependent", 1.0), 1.0)
            if td > 0.35:
                continue
            rel = _sim_score(
                focus_emb,
                _safe_embedding(getattr(f, "embedding", []) or []),
                focus_text,
                str(getattr(f, "content", "") or ""),
            )
            facts.append((rel, str(getattr(f, "content", "") or "")))
        facts.sort(key=lambda x: x[0], reverse=True)
        if facts:
            try_add("Time-invariant facts:")
            for _, text in facts[:8]:
                row = f"- {_excerpt(text, 220)}"
                if try_add(row):
                    fact_lines.append(row)

        narratives = []
        for n in list(getattr(ghost, "all_narratives", []) or []):
            content = str(getattr(n, "content", "") or "").strip()
            if not content:
                continue
            rel = _sim_score(
                focus_emb,
                _safe_embedding(getattr(n, "embedding", []) or []),
                focus_text,
                content,
            )
            ntype = str(getattr(getattr(n, "narrative_type", None), "value", "") or "")
            target = str(getattr(n, "target_name", "") or "")
            narratives.append((rel, f"Narrative[{ntype}/{target}]: {_excerpt(content, 240)}"))
        narratives.sort(key=lambda x: x[0], reverse=True)
        if narratives:
            try_add("Relevant narratives:")
            for _, row in narratives[:6]:
                if not try_add(f"- {row}"):
                    break

        cap_text = str(capability_context_text(ghost) or "").strip()
        if cap_text:
            cap_lines = [ln.strip() for ln in cap_text.splitlines() if ln.strip()]
            if cap_lines:
                try_add("Capability profile:")
                for ln in cap_lines:
                    if not try_add(f"- {ln.lstrip('-').strip()}"):
                        break

        return "\n".join(lines), fact_lines, {"budget": int(budget), "used": int(used), "line_count": len(lines)}
