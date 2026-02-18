import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from pm.data_structures import Feature, FeatureType, KnoxelBase
from pm.utils.emb_utils import cosine_sim
from pm.utils.token_utils import get_token_count


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


def _excerpt(text: str, max_len: int = 220) -> str:
    t = str(text or "").replace("\n", " ").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 3].rstrip() + "..."


@dataclass(frozen=True)
class StorySelectionConfig:
    # Ranking weights
    w_cosine: float = 0.34
    w_recency: float = 0.16
    w_salience: float = 0.12
    w_emotion: float = 0.12
    w_interlocus: float = 0.08
    w_intentions: float = 0.10
    w_goals: float = 0.08
    w_explicit_recall: float = 0.95

    # Selection budgets
    max_items: int = 96
    max_tokens: int = 1400
    min_score: float = 0.08
    max_entries_per_tick: int = 24
    render_excerpt_chars: int = 220
    include_omission_summaries: bool = True

    # Type priors
    type_boosts: Dict[str, float] = field(
        default_factory=lambda: {
            "Feature:Dialogue": 0.11,
            "Feature:CodeletPercept": 0.08,
            "Feature:CodeletOutput": 0.08,
            "Feature:Thought": 0.07,
            "Feature:MetaInsight": 0.06,
            "DeclarativeFactKnoxel": 0.09,
            "CauseEffectKnoxel": 0.10,
            "Action": 0.07,
        }
    )


@dataclass
class StoryContextPacket:
    story_text: str
    continuation_seed: str
    selected_ids: List[int]
    dialogue_lines: List[str]
    fact_lines: List[str]
    percept_lines: List[str]
    debug_rows: List[Dict[str, Any]]


class StoryContextBuilder:
    """
    Unified context selector for story-writing and simulation threads.
    Scores knoxels using multi-factor salience and returns a token-bounded packet.
    """

    @staticmethod
    def build(
        ghost,
        *,
        focus_text: str,
        config: Optional[StorySelectionConfig] = None,
    ) -> StoryContextPacket:
        cfg = config or StorySelectionConfig()
        query_emb = StoryContextBuilder._query_embedding(ghost, focus_text)
        candidates = StoryContextBuilder._candidates(ghost)

        rows: List[Tuple[float, KnoxelBase, Dict[str, Any]]] = []
        forced_ids: set[int] = set()
        for k in candidates:
            score, breakdown = StoryContextBuilder._score_knoxel(ghost, k, focus_text, query_emb, cfg)
            if bool(breakdown.get("explicit_recall_active", False)):
                forced_ids.add(int(getattr(k, "id", -1) or -1))
            if score < cfg.min_score:
                continue
            rows.append((score, k, breakdown))

        rows.sort(key=lambda x: x[0], reverse=True)

        selected: List[KnoxelBase] = []
        selected_ids: List[int] = []
        debug_rows: List[Dict[str, Any]] = []
        token_budget = 0

        # First pass: force explicit recall knoxels into selection when active.
        for score, k, breakdown in rows:
            kid = int(getattr(k, "id", -1) or -1)
            if kid not in forced_ids:
                continue
            if len(selected) >= cfg.max_items:
                break
            try:
                story_el = k.get_story_element(ghost)
            except Exception:
                story_el = str(getattr(k, "content", "") or "")
            if not str(story_el or "").strip():
                continue
            tc = get_token_count(story_el)
            if token_budget + tc > cfg.max_tokens:
                continue
            selected.append(k)
            selected_ids.append(kid)
            token_budget += tc
            debug_rows.append(
                {
                    "id": kid,
                    "type": type(k).__name__,
                    "score": round(float(score), 4),
                    "token_cost": int(tc),
                    "forced": True,
                    "content_excerpt": _excerpt(str(getattr(k, "content", "") or ""), 140),
                    **breakdown,
                }
            )

        # Second pass: normal ranking fill.
        for score, k, breakdown in rows:
            if len(selected) >= cfg.max_items:
                break
            kid = int(getattr(k, "id", -1) or -1)
            if kid in selected_ids:
                continue
            try:
                story_el = k.get_story_element(ghost)
            except Exception:
                story_el = str(getattr(k, "content", "") or "")
            if not str(story_el or "").strip():
                continue
            tc = get_token_count(story_el)
            if token_budget + tc > cfg.max_tokens:
                continue
            selected.append(k)
            selected_ids.append(kid)
            token_budget += tc
            debug_rows.append(
                {
                    "id": kid,
                    "type": type(k).__name__,
                    "score": round(float(score), 4),
                    "token_cost": int(tc),
                    "forced": False,
                    "content_excerpt": _excerpt(str(getattr(k, "content", "") or ""), 140),
                    **breakdown,
                }
            )

        selected.sort(key=lambda x: (int(getattr(x, "tick_id", 0) or 0), int(getattr(x, "id", 0) or 0)))
        story_parts: List[str] = []
        dialogue_lines: List[str] = []
        fact_lines: List[str] = []
        percept_lines: List[str] = []

        user_name = str(getattr(getattr(ghost, "config", None), "user_name", "") or "user")
        companion_name = str(getattr(getattr(ghost, "config", None), "companion_name", "") or "assistant")
        for k in selected:
            try:
                line = k.get_story_element(ghost)
            except Exception:
                line = str(getattr(k, "content", "") or "")
            story_parts.append(line)
            if isinstance(k, Feature) and k.feature_type == FeatureType.Dialogue:
                src = str(getattr(k, "source", "") or "")
                role = "assistant"
                if src == user_name:
                    role = "user"
                elif src not in (companion_name, ""):
                    role = src
                dialogue_lines.append(f"{role}: {str(getattr(k, 'content', '') or '').strip()}")
            if isinstance(k, Feature) and k.feature_type in (FeatureType.CodeletPercept, FeatureType.CodeletOutput):
                percept_lines.append(f"- {str(getattr(k, 'source', '') or '')}: {_excerpt(str(getattr(k, 'content', '') or ''), 180)}")
            if type(k).__name__ in ("DeclarativeFactKnoxel", "CauseEffectKnoxel"):
                fact_lines.append(f"- {_excerpt(str(getattr(k, 'content', '') or ''), 200)}")

        packet = StoryContextPacket(
            story_text="\n".join(story_parts) if story_parts else "(no selected story context)",
            continuation_seed=StoryContextBuilder._render_continuation_seed(ghost, selected, focus_text, candidates, cfg),
            selected_ids=selected_ids,
            dialogue_lines=dialogue_lines,
            fact_lines=fact_lines,
            percept_lines=percept_lines,
            debug_rows=debug_rows,
        )
        StoryContextBuilder._persist_debug(ghost, focus_text, cfg, packet)
        return packet

    @staticmethod
    def _candidates(ghost) -> List[KnoxelBase]:
        values = list(getattr(ghost, "all_knoxels", {}).values())
        if not values:
            values = []
            values.extend(list(getattr(ghost, "all_features", []) or []))
            values.extend(list(getattr(ghost, "all_declarative_facts", []) or []))
            values.extend(list(getattr(ghost, "all_intentions", []) or []))
        out: List[KnoxelBase] = []
        seen = set()
        for k in values:
            kid = int(getattr(k, "id", -1) or -1)
            sig = (type(k).__name__, kid, str(getattr(k, "content", "") or ""))
            if sig in seen:
                continue
            seen.add(sig)
            if not str(getattr(k, "content", "") or "").strip():
                continue
            out.append(k)
        return out

    @staticmethod
    def _query_embedding(ghost, focus_text: str) -> List[float]:
        key = f"{int(getattr(ghost, 'current_tick_id', 0) or 0)}::{_norm(focus_text)[:320]}"
        cache = dict(getattr(ghost, "story_context_embedding_cache", {}) or {})
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
        ghost.story_context_embedding_cache = cache
        return emb

    @staticmethod
    def _score_knoxel(
        ghost,
        k: KnoxelBase,
        focus_text: str,
        query_emb: List[float],
        cfg: StorySelectionConfig,
    ) -> Tuple[float, Dict[str, Any]]:
        k_emb = list(getattr(k, "embedding", []) or [])
        cosine = float(cosine_sim(query_emb, k_emb)) if query_emb and k_emb else 0.0

        tick_now = int(getattr(ghost, "current_tick_id", 0) or 0)
        kt = int(getattr(k, "tick_id", 0) or 0)
        dist = max(0, tick_now - kt)
        recency = 1.0 / (1.0 + math.log1p(dist))

        salience = 0.0
        if isinstance(k, Feature):
            salience = float(getattr(k, "incentive_salience", 0.0) or 0.0)
        if type(k).__name__ == "DeclarativeFactKnoxel":
            salience = max(salience, float(getattr(k, "importance", 0.0) or 0.0))

        emotion = 0.0
        state_val = 0.0
        try:
            state_val = float(
                getattr(getattr(getattr(ghost, "current_state", None), "latent_mental_state", None), "state_core", None).valence
            )
        except Exception:
            state_val = 0.0
        if isinstance(k, Feature):
            kv = getattr(k, "affective_valence", None)
            if kv is not None:
                emotion = 1.0 - min(1.0, abs(float(kv) - state_val) / 2.0)

        interlocus_align = 0.0
        try:
            cur_inter = float(
                getattr(getattr(getattr(ghost, "current_state", None), "latent_mental_state", None), "state_cognition", None).interlocus
            )
            if isinstance(k, Feature):
                ki = float(getattr(k, "interlocus", 0.0) or 0.0)
                interlocus_align = 1.0 - min(1.0, abs(cur_inter - ki) / 2.0)
        except Exception:
            interlocus_align = 0.0

        intentions_text = " ".join(
            str(getattr(x, "content", "") or "")
            for x in list(getattr(ghost, "all_intentions", []) or [])[-10:]
        )
        intent_rel = _token_overlap(str(getattr(k, "content", "") or ""), intentions_text)

        goal_text = " ".join(
            [
                str(getattr(getattr(ghost, "selected_action_schema", None), "action_description", "") or ""),
                str(getattr(ghost, "thought_blueprint_directive", "") or ""),
                str(getattr(getattr(ghost, "primary_stimulus", None), "content", "") or ""),
                focus_text,
            ]
        )
        goal_rel = _token_overlap(str(getattr(k, "content", "") or ""), goal_text)
        explicit_recall = StoryContextBuilder._explicit_recall_score(ghost, k)

        type_key = type(k).__name__
        if isinstance(k, Feature):
            type_key = f"Feature:{k.feature_type.value}"
        type_boost = float((cfg.type_boosts or {}).get(type_key, 0.0) or 0.0)

        score = (
            cfg.w_cosine * cosine
            + cfg.w_recency * recency
            + cfg.w_salience * salience
            + cfg.w_emotion * emotion
            + cfg.w_interlocus * interlocus_align
            + cfg.w_intentions * intent_rel
            + cfg.w_goals * goal_rel
            + cfg.w_explicit_recall * explicit_recall
            + type_boost
        )
        breakdown = {
            "cosine": round(cosine, 4),
            "recency": round(recency, 4),
            "salience": round(salience, 4),
            "emotion": round(emotion, 4),
            "interlocus": round(interlocus_align, 4),
            "intent_rel": round(intent_rel, 4),
            "goal_rel": round(goal_rel, 4),
            "explicit_recall": round(explicit_recall, 4),
            "explicit_recall_active": bool(explicit_recall > 0.0),
            "type_boost": round(type_boost, 4),
        }
        return float(score), breakdown

    @staticmethod
    def _persist_debug(ghost, focus_text: str, cfg: StorySelectionConfig, packet: StoryContextPacket) -> None:
        payload = {
            "tick": int(getattr(ghost, "current_tick_id", 0) or 0),
            "focus_excerpt": _excerpt(focus_text, 240),
            "config": {
                "weights": {
                    "cosine": cfg.w_cosine,
                    "recency": cfg.w_recency,
                    "salience": cfg.w_salience,
                    "emotion": cfg.w_emotion,
                    "interlocus": cfg.w_interlocus,
                    "intentions": cfg.w_intentions,
                    "goals": cfg.w_goals,
                },
                "max_items": cfg.max_items,
                "max_tokens": cfg.max_tokens,
                "min_score": cfg.min_score,
                "w_explicit_recall": cfg.w_explicit_recall,
                "max_entries_per_tick": cfg.max_entries_per_tick,
                "render_excerpt_chars": cfg.render_excerpt_chars,
                "include_omission_summaries": cfg.include_omission_summaries,
            },
            "selected_count": len(packet.selected_ids),
            "selected_ids": list(packet.selected_ids),
            "top_rows": list(packet.debug_rows[:40]),
        }
        ghost.story_context_last = payload
        hist = list(getattr(ghost, "story_context_history", []) or [])
        hist.append(payload)
        if len(hist) > 80:
            hist = hist[-80:]
        ghost.story_context_history = hist

    @staticmethod
    def _explicit_recall_score(ghost, k: KnoxelBase) -> float:
        tick = int(getattr(ghost, "current_tick_id", 0) or 0)
        md = dict(getattr(k, "metadata", {}) or {})
        score = 0.0

        # Backward-compatible single tick markers.
        single_tick = md.get("explicit_recall_at_tick", None)
        if single_tick is not None:
            try:
                if int(single_tick) == tick:
                    score = max(score, 1.0)
            except Exception:
                pass

        # Repeated tick list markers.
        tick_list = md.get("explicit_recall_ticks", [])
        if isinstance(tick_list, list):
            for t in tick_list:
                try:
                    if int(t) == tick:
                        score = max(score, 1.0)
                        break
                except Exception:
                    continue

        # Structured windows: [{"at_tick": 100, "until_tick": 110, "weight": 1.0}, ...]
        plans = md.get("explicit_recall", [])
        if isinstance(plans, list):
            for plan in plans:
                if not isinstance(plan, dict):
                    continue
                try:
                    at_tick = int(plan.get("at_tick", tick + 1))
                    until_tick = int(plan.get("until_tick", at_tick))
                    weight = float(plan.get("weight", 1.0) or 1.0)
                    if at_tick <= tick <= until_tick:
                        score = max(score, min(1.2, max(0.0, weight)))
                except Exception:
                    continue
        return score

    @staticmethod
    def _render_continuation_seed(
        ghost,
        selected: List[KnoxelBase],
        focus_text: str,
        all_candidates: List[KnoxelBase],
        cfg: StorySelectionConfig,
    ) -> str:
        companion = str(getattr(getattr(ghost, "config", None), "companion_name", "") or "assistant")
        user = str(getattr(getattr(ghost, "config", None), "user_name", "") or "user")
        selected_ids = {int(getattr(k, "id", -1) or -1) for k in selected}
        omitted = max(0, len(all_candidates) - len(selected))

        lines: List[str] = []
        lines.append(
            f"Story so far: This is the lived timeline between {companion} and {user}, told in natural chronological flow."
        )
        lines.append(
            f"Current focus: {_excerpt(focus_text, 220)}"
        )

        if omitted > 0 and bool(cfg.include_omission_summaries):
            lines.append(
                f"Between key moments, {omitted} lower-salience events passed and are compressed into background continuity."
            )

        by_tick: Dict[int, List[KnoxelBase]] = {}
        for k in selected:
            tick = int(getattr(k, "tick_id", 0) or 0)
            by_tick.setdefault(tick, []).append(k)

        for tick in sorted(by_tick.keys()):
            entries = sorted(by_tick[tick], key=lambda x: int(getattr(x, "id", 0) or 0))
            prose_bits: List[str] = []
            mental_bits: List[str] = []
            for k in entries[: max(1, int(cfg.max_entries_per_tick or 1))]:
                if isinstance(k, Feature) and k.feature_type == FeatureType.Dialogue:
                    src = str(getattr(k, "source", "") or "")
                    content = str(getattr(k, "content", "") or "").strip()
                    prose_bits.append(f'{src} said "{content}"')
                else:
                    prose_bits.append(_excerpt(str(getattr(k, "content", "") or ""), int(cfg.render_excerpt_chars or 220)))

                if isinstance(k, Feature):
                    av = getattr(k, "affective_valence", None)
                    il = getattr(k, "interlocus", None)
                    if av is not None or il is not None:
                        parts = []
                        if av is not None:
                            parts.append(f"valence={float(av):.2f}")
                        if il is not None:
                            parts.append(f"interlocus={float(il):.2f}")
                        if parts:
                            mental_bits.append(", ".join(parts))

            if prose_bits:
                lines.append(f"At tick {tick}, " + "; ".join(prose_bits) + ".")
            if mental_bits:
                lines.append(
                    "This phase affected the persona's inner state through "
                    + "; ".join(mental_bits[:4])
                    + "."
                )

        # Mention one compact summary of omitted selected-adjacent items for flow.
        non_selected_dialogue = [
            k for k in all_candidates
            if isinstance(k, Feature)
            and k.feature_type == FeatureType.Dialogue
            and int(getattr(k, "id", -1) or -1) not in selected_ids
        ]
        if non_selected_dialogue and bool(cfg.include_omission_summaries):
            lines.append(
                f"Other conversation fragments happened in between and are summarized rather than replayed verbatim ({len(non_selected_dialogue)} snippets)."
            )

        blueprint = dict(getattr(ghost, "reply_blueprint_last", {}) or {})
        action_desc = str(getattr(getattr(ghost, "selected_action_schema", None), "action_description", "") or "")
        if action_desc:
            lines.append(f"In the present moment, {companion} intends to: {action_desc}.")
        if blueprint:
            lines.append(
                f"Current expression blueprint: {str(blueprint.get('name', '') or '')}."
            )
        seed_prefix = str(blueprint.get("generation_prefix", "") or "").strip()
        if seed_prefix:
            lines.append(seed_prefix)
        lines.append(f"{companion} says: \"")
        return "\n".join(lines)
