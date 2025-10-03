# memory_query_engine.py
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, Set

import numpy as np

# --- Import your project types/utilities (present in your repo) ---
from pm.llm.llm_common import LlmPreset, CommonCompSettings
from pm.config_loader import companion_name, user_name, timestamp_format
from pm.utils.token_utils import get_token_count

# Knoxel/ghost types (all optional/duck-typed—no strict imports required)
# They exist in pm.data_structures / pm.ghosts.base_ghost in your project.
# We only rely on attributes/methods used below.
# from pm.data_structures import FeatureType, KnoxelBase
# from pm.ghosts.base_ghost import BaseGhost


# ---------------------------
# Utility helpers
# ---------------------------

def _cosine_sim(a: Sequence[float], b: Sequence[float]) -> float:
    """Safe cosine similarity in pure numpy (returns 0.0 if degenerate)."""
    if not a or not b:
        return 0.0
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return default


def _knoxel_text(knoxel: Any, ghost: Any) -> str:
    """Prefer a story element if available; otherwise fall back to content/str."""
    try:
        if hasattr(knoxel, "get_story_element"):
            return knoxel.get_story_element(ghost)  # many of your knoxels implement this
    except Exception:
        pass
    c = _safe_getattr(knoxel, "content", None)
    return str(knoxel) if c is None else str(c)


def _now() -> datetime:
    try:
        # Your world timestamps already exist; only used for relative fallback.
        return datetime.now()
    except Exception:
        return datetime.utcnow()


# ---------------------------
# Config for the engine
# ---------------------------

@dataclass
class MQEConfig:
    # Retrieval fanouts
    per_question_top_k: int = 24          # initial semantic candidates per question
    global_pool_cap: int = 256            # limit the global deduped pool before scoring
    # Scoring weights
    w_semantic: float = 0.55
    w_emotion: float = 0.25
    w_recency: float = 0.10
    w_importance: float = 0.07
    w_coverage_bonus: float = 0.03        # reward items that match multiple questions
    # Packing
    pack_token_budget: int = 4000         # ~4k tokens for snippets (synthesis budget separate)
    min_snippet_len_chars: int = 20
    # Synthesis
    synthesis_max_tokens: int = 900       # leave headroom for model; adjust if needed
    synthesis_temperature: float = 0.4
    # Misc
    diversity_by_type: bool = True        # try to mix feature/fact/episodic/etc.
    max_per_type_ratio: float = 0.6       # no more than 60% of budget from a single type
    paraphrase_per_q: int = 3
    min_pool_after_first_pass: int = 60     # if fewer than this after pass 1, try paraphrases
    expansion_temperature: float = 0.2      # cooler = tighter paraphrases


# ---------------------------
# Main Engine
# ---------------------------

class MemoryQueryEngine:
    """
    Multi-hop, agentic RAG over Ghost knoxels to supply codelets with
    grounded knowledge and IDs of contributing knoxels.

    Usage:
        mqe = MemoryQueryEngine(ghost, llm)
        text, source_ids = mqe.get_knowledge(
            latest_context, stimulus, codelet_purpose, questions, max_tokens
        )
    """

    def __init__(self, ghost: Any, llm: Any, config: Optional[MQEConfig] = None):
        self.ghost = ghost
        self.llm = llm
        self.cfg = config or MQEConfig()

    # ---------------------------
    # Public API
    # ---------------------------

    def get_knowledge(
        self,
        latest_context: str,
        stimulus: str,
        codelet_purpose: str,
        questions: List[str],
        max_tokens: int
    ) -> Tuple[str, List[int]]:
        """
        1) Generates sub-questions (multi-hop) per question (fast).
        2) Retrieves & ranks knoxels by semantic + affective proximity, recency, and importance.
        3) Packs ~4k tokens of top snippets.
        4) Calls LLM to synthesize a grounded, compact answer tailored to codelet purpose.
        5) Returns (final_answer, contributing_knoxel_ids).
        """
        if not questions:
            return ("", [])

        # Step 0: compute a tunable “emotion vs. fact” slider (0=factual, 1=affective)
        affect_bias = self._get_affect_vs_fact_weight()

        # Step 1: optional light multi-hop expansion for each question
        expanded = self._expand_questions(latest_context, stimulus, codelet_purpose, questions)

        # Step 2: retrieve candidates per (sub)question
        pool: Dict[int, Dict[str, Any]] = {}  # knoxel_id -> { 'k': knoxel, 'score': float, 'hits': set(q_idx), 'type': str }
        for q_idx, q in enumerate(expanded):
            q_emb = self.llm.get_embedding(q)
            per_q = self._topk_for_query(q_emb, top_k=self.cfg.per_question_top_k)
            for k in per_q:
                kid = _safe_getattr(k, "id", None)
                if kid is None:
                    continue
                if kid not in pool:
                    pool[kid] = {"k": k, "score": 0.0, "hits": set(), "type": k.__class__.__name__}
                # mark question coverage; score added in next step
                pool[kid]["hits"].add(q_idx)

        # keep pool manageable
        if len(pool) > self.cfg.global_pool_cap:
            # quick prune by raw semantic to a blended centroid (cheap heuristic)
            centroid = self._centroid_embedding([self.llm.get_embedding(q) for q in expanded])
            scored = []
            for meta in pool.values():
                k = meta["k"]
                sim = _cosine_sim(_safe_getattr(k, "embedding", []) or [], centroid)
                scored.append((sim, k))
            scored.sort(key=lambda x: x[0], reverse=True)
            keep_ids: Set[int] = set()
            for _, k in scored[: self.cfg.global_pool_cap]:
                keep_ids.add(_safe_getattr(k, "id"))
            pool = {kid: meta for kid, meta in pool.items() if kid in keep_ids}

        # Step 3: compute blended scores
        for kid, meta in pool.items():
            k = meta["k"]
            sem = self._semantic_to_queries(k, expanded)
            emo = self._affective_proximity(k)  # 0..1
            rec = self._recency(k)              # 0..1
            imp = self._importance(k)           # 0..1
            cov = min(1.0, len(meta["hits"]) / max(1.0, len(expanded)))  # 0..1

            # Blend semantic and affect by slider
            sem_aff = (1.0 - affect_bias) * sem + affect_bias * emo

            score = (
                self.cfg.w_semantic * sem_aff +
                self.cfg.w_recency * rec +
                self.cfg.w_importance * imp +
                self.cfg.w_coverage_bonus * cov
            )
            meta["score"] = float(score)

        # Step 4: pack snippets within ~4k tokens with optional diversity constraint
        ordered = sorted(pool.values(), key=lambda m: m["score"], reverse=True)
        packed, src_ids = self._pack_snippets(ordered, self.cfg.pack_token_budget)

        # Step 5: synthesize final answer for the codelet
        final = self._synthesize(
            latest_context=latest_context,
            stimulus=stimulus,
            codelet_purpose=codelet_purpose,
            questions=questions,
            snippets=packed,
            token_budget=max_tokens or self.cfg.synthesis_max_tokens
        )
        return final, src_ids

    # ---------------------------
    # Multi-hop expansion
    # ---------------------------
    # --- Replace _expand_questions with this richer version: ---
    def _expand_questions(self, latest_context: str, stimulus: str, purpose: str, questions: List[str]) -> List[str]:
        """
        Produce multi-hop expansions AND paraphrase/bridge variants.
        Returns a deduped list: originals + subs + paraphrases.
        """
        base_prompt = f"""
    You are a retrieval planner and query rephraser for an agentic RAG system.

    Context (latest): {latest_context[:1000]}
    Stimulus: {stimulus[:500]}
    Purpose: {purpose[:500]}

    For EACH question, do three things:
    1) SUBS (0-2): concrete follow-ups that unlock hidden facts across memory.
    2) PARAS (1-{self.cfg.paraphrase_per_q}): tight paraphrases (≤ 15 words) that preserve meaning but vary wording.
    3) BRIDGES (0-2): add a missing bridge noun phrase or alias (synonym/entity/role) that would connect related knoxels.

    Output strict JSON:
    {{
      "items": [
         {{"root":"<original>",
           "subs":["<sub1>","<sub2>"],
           "paras":["<para1>", "..."],
           "bridges":["<alias or NP>", "..."]
         }},
         ...
      ]
    }}
        """.strip()

        msgs = [
            ("system", "Be precise. Output only valid JSON. Keep items short and grounded."),
            ("user", base_prompt + "\n\nQUESTIONS:\n" + "\n".join(f"- {q}" for q in questions))
        ]
        txt = self.llm.completion_text(
            LlmPreset.Fast,
            msgs,
            comp_settings=CommonCompSettings(
                temperature=self.cfg.expansion_temperature, max_tokens=700
            ),
        ) or "{}"

        import json
        expanded: List[str] = []
        try:
            data = json.loads(txt)
            for it in data.get("items", []):
                root = (it.get("root") or "").strip()
                if root:
                    expanded.append(root)
                for s in (it.get("subs") or [])[:2]:
                    s = (s or "").strip()
                    if s: expanded.append(s)
                for p in (it.get("paras") or [])[: self.cfg.paraphrase_per_q]:
                    p = (p or "").strip()
                    if p: expanded.append(p)
                for b in (it.get("bridges") or [])[:2]:
                    b = (b or "").strip()
                    if b: expanded.append(f"{root} — bridge: {b}")
        except Exception:
            # Fallback: at least return originals
            expanded = list(questions)

        # dedupe while preserving order
        seen, out = set(), []
        for q in expanded:
            if q and q not in seen:
                out.append(q)
                seen.add(q)
        # safety cap to avoid explosion
        return out[: 5 * max(1, len(questions))]

    # --- Add this small helper to run a recall backoff in get_knowledge(): ---
    def _maybe_paraphrase_backoff(self, pool_len_after_pass1: int, questions: List[str]) -> Optional[List[str]]:
        if pool_len_after_pass1 >= self.cfg.min_pool_after_first_pass:
            return None
        # If recall is low, ask for alternate paraphrases only (cheap)
        prompt = f"""
    Rephrase each question into {self.cfg.paraphrase_per_q} distinct, concise paraphrases (≤ 12 words each).
    Keep meaning identical; vary wording and common aliases. Output JSON:
    {{"paraphrases":[["p1","p2","..."], ...]}}  # one inner list per original question
    QUESTIONS:
    {chr(10).join(f"- {q}" for q in questions)}
    """.strip()
        msgs = [("system", "Output valid JSON only."), ("user", prompt)]
        txt = self.llm.completion_text(
            LlmPreset.Fast, msgs,
            comp_settings=CommonCompSettings(temperature=0.15, max_tokens=600)
        ) or "{}"
        import json
        try:
            data = json.loads(txt)
            merged: List[str] = []
            for root_q, plist in zip(questions, data.get("paraphrases", [])):
                merged.append(root_q)
                for p in plist[: self.cfg.paraphrase_per_q]:
                    if p and isinstance(p, str):
                        merged.append(p.strip())
            # de-dup
            seen, out = set(), []
            for q in merged:
                if q and q not in seen:
                    out.append(q);
                    seen.add(q)
            return out
        except Exception:
            return None

    def _expand_questions_old(self, latest_context: str, stimulus: str, purpose: str, questions: List[str]) -> List[str]:
        """
        Ask the LLM to produce follow-up sub-questions that would unlock latent facts.
        Cheap & constrained: returns original Q + up to 2 sub-qs each.
        """
        prompt = f"""
You are a retrieval planner for an agentic RAG system.
Context (latest): {latest_context[:1200]}
Stimulus: {stimulus[:600]}
Purpose: {purpose[:600]}

Given the list of QUESTIONS below, propose up to 2 brief follow-up sub-questions per item
that would help connect facts across the memory graph. Keep each sub-question short (<= 20 words),
grounded, and non-speculative. If none needed, return just the original.

QUESTIONS:
{chr(10).join(f"- {q}" for q in questions)}

Return one JSON object exactly:
{{
  "expanded": [
    {{"root": "<original>", "subs": ["<sub1>", "<sub2>"]}}, ...
  ]
}}
        """.strip()

        class _PlanOut:  # tiny schema shim against completion_tool
            pass

        # Use completion_text for portability; parse leniently
        msgs = [("system", "You are a precise retrieval planner. Output valid JSON only."),
                ("user", prompt)]
        txt = self.llm.completion_text(
            LlmPreset.Fast,
            msgs,
            comp_settings=CommonCompSettings(temperature=0.1, max_tokens=600),
        ) or "{}"

        # very lenient JSON-ish parsing
        import json, re
        try:
            data = json.loads(txt)
            expanded: List[str] = []
            for item in data.get("expanded", []):
                root = (item.get("root") or "").strip()
                subs = [s.strip() for s in item.get("subs", []) if s and isinstance(s, str)]
                if root:
                    expanded.append(root)
                for s in subs[:2]:
                    if s:
                        expanded.append(s)
            # Fallback: ensure we at least have originals
            if not expanded:
                return list(questions)
            # de-dup while preserving order
            seen = set()
            out = []
            for q in expanded:
                if q not in seen:
                    out.append(q)
                    seen.add(q)
            return out
        except Exception:
            return list(questions)

    # ---------------------------
    # Retrieval
    # ---------------------------

    def _all_candidates(self) -> List[Any]:
        """
        Collect all retrievable knoxels from the ghost. We avoid strict typing
        and just iterate ghost collections that exist.
        """
        buckets = [
            "all_declarative_facts",
            "all_features",
            "all_episodic_memories",
            "all_intentions",
            "all_narratives",
            "all_actions",
            "all_stimuli",
        ]
        res: List[Any] = []
        for name in buckets:
            arr = _safe_getattr(self.ghost, name, []) or []
            res.extend(arr)
        # final fallback: everything registered in all_knoxels
        if not res:
            all_knox = _safe_getattr(self.ghost, "all_knoxels", {}) or {}
            res = list(all_knox.values())
        return res

    def _ensure_embedding(self, k: Any):
        emb = _safe_getattr(k, "embedding", None)
        if not emb:
            try:
                # some of your utils accept text or knoxel directly; use text
                content = _knoxel_text(k, self.ghost)
                emb = self.llm.get_embedding(content)
                try:
                    k.embedding = emb  # cache back
                except Exception:
                    pass
            except Exception:
                return

    def _topk_for_query(self, q_emb: Sequence[float], top_k: int) -> List[Any]:
        cand = self._all_candidates()
        scored: List[Tuple[float, Any]] = []
        for k in cand:
            self._ensure_embedding(k)
            sim = _cosine_sim(q_emb, _safe_getattr(k, "embedding", []) or [])
            if sim > 0.0:
                scored.append((sim, k))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [k for _, k in scored[:top_k]]

    def _centroid_embedding(self, embs: List[Optional[Sequence[float]]]) -> List[float]:
        arrs = [np.array(e, dtype=np.float32) for e in embs if e]
        if not arrs:
            return []
        return list(np.mean(np.stack(arrs, axis=0), axis=0))

    # ---------------------------
    # Scoring
    # ---------------------------

    def _semantic_to_queries(self, k: Any, queries: List[str]) -> float:
        emb_k = _safe_getattr(k, "embedding", []) or []
        if not emb_k:
            return 0.0
        sims = []
        for q in queries:
            q_emb = self.llm.get_embedding(q)
            sims.append(_cosine_sim(emb_k, q_emb))
        # take top-2 mean (stabilizes scores)
        sims.sort(reverse=True)
        if not sims:
            return 0.0
        return float(sum(sims[:2]) / max(1.0, min(2, len(sims))))

    def _affective_proximity(self, k: Any) -> float:
        """
        Compare emotion vectors if available:
            - knoxel.get_mental_state_at_creation() -> DecayableMentalState-like
            - ghost.current_state.state_emotions
        Return [0..1], where 1 = very close (affect-aligned).
        """
        try:
            # current
            gs = _safe_getattr(self.ghost, "current_state", None)
            cur = _safe_getattr(gs, "state_emotions", None)
            # source
            src = None
            if hasattr(k, "get_mental_state_at_creation"):
                src = k.get_mental_state_at_creation()
            if cur is None or src is None:
                return 0.0

            # convert to ordered numeric vector by field order
            def vec(ms) -> List[float]:
                vals = []
                for fname, field in ms.__class__.model_fields.items():  # pydantic v2
                    v = _safe_getattr(ms, fname, None)
                    if isinstance(v, (float, int)):
                        vals.append(float(v))
                return vals

            v_cur = np.array(vec(cur), dtype=np.float32)
            v_src = np.array(vec(src), dtype=np.float32)
            if v_cur.size == 0 or v_src.size == 0:
                return 0.0
            sim = _cosine_sim(v_cur, v_src)  # [-1..1] but typical [0..1]
            return max(0.0, (sim + 1.0) / 2.0)  # map to [0..1]
        except Exception:
            return 0.0

    def _recency(self, k: Any) -> float:
        """
        Convert timestamp or tick delta to [0..1] with gentle decay.
        """
        # Prefer world timestamps
        ts = _safe_getattr(k, "timestamp_world_begin", None) or _safe_getattr(k, "timestamp_creation", None)
        if isinstance(ts, datetime):
            age_sec = max(0.0, (_now() - ts).total_seconds())
            # 0 at 30 days, ~1 near now
            half_life = 3 * 24 * 3600.0
            return math.exp(-age_sec / (2.0 * half_life))
        # fallback to tick distance
        cur_tick = _safe_getattr(self.ghost, "current_tick_id", 0)
        ktick = _safe_getattr(k, "tick_id", cur_tick)
        dt = max(0, cur_tick - int(ktick))
        return math.exp(-0.12 * dt)

    def _importance(self, k: Any) -> float:
        """
        Leverage known 'importance' field on facts/clusters; otherwise 0.
        """
        val = _safe_getattr(k, "importance", None)
        try:
            if val is None:
                return 0.0
            return float(max(0.0, min(1.0, val)))
        except Exception:
            return 0.0

    # ---------------------------
    # Packing
    # ---------------------------

    def _pack_snippets(self, ordered_meta: List[Dict[str, Any]], token_budget: int) -> Tuple[List[Tuple[int, str]], List[int]]:
        """
        Greedy pack texts under token budget with optional type diversity.
        Returns list[(id, text_for_prompt)], list[source_ids].
        """
        chosen: List[Tuple[int, str]] = []
        source_ids: List[int] = []
        used_tokens = 0

        # diversity control
        if self.cfg.diversity_by_type:
            type_counts: Dict[str, int] = {}
            def type_ok(t: str) -> bool:
                total = sum(type_counts.values()) or 1
                return type_counts.get(t, 0) < math.floor(self.cfg.max_per_type_ratio * (total + 1))
        else:
            def type_ok(t: str) -> bool:  # type: ignore
                return True

        for meta in ordered_meta:
            k = meta["k"]
            kid = _safe_getattr(k, "id", None)
            if kid is None:
                continue
            text = _knoxel_text(k, self.ghost)
            if not text or len(text) < self.cfg.min_snippet_len_chars:
                continue

            # small, readable header for the synthesizer
            header = f"[{k.__class__.__name__} #{kid}] "
            snippet = header + text
            tks = get_token_count(snippet)

            if used_tokens + tks > token_budget:
                continue

            t = meta["type"]
            if not type_ok(t):
                continue

            chosen.append((kid, snippet))
            source_ids.append(kid)
            used_tokens += tks
            # update diversity accounting
            # (after adding, so it doesn't block first instance of a type)
            if self.cfg.diversity_by_type:
                type_counts[t] = type_counts.get(t, 0) + 1

            # early exit if we are very close to budget
            if token_budget - used_tokens < 32:
                break

        return chosen, source_ids

    # ---------------------------
    # Synthesis
    # ---------------------------

    def _synthesize(
        self,
        latest_context: str,
        stimulus: str,
        codelet_purpose: str,
        questions: List[str],
        snippets: List[Tuple[int, str]],
        token_budget: int
    ) -> str:
        """
        Produce a compact, grounded, codelet-ready answer. We instruct the model
        to rely ONLY on provided snippets and to keep it within budget.
        """
        # Build the corpus
        corpus = "\n\n".join(s for _, s in snippets)

        # Minimal guard if nothing retrieved
        if not corpus.strip():
            msgs = [
                ("system", "You are a careful assistant that admits when you lack information."),
                ("user", f"No relevant memory snippets were found.\n\nQuestions:\n- " + "\n- ".join(questions)),
            ]
            return self.llm.completion_text(
                LlmPreset.Fast,
                msgs,
                comp_settings=CommonCompSettings(temperature=0.3, max_tokens=min(256, token_budget)),
            ) or ""

        # Strong, retrieval-grounded system prompt
        sysprompt = f"""
You are MemoryQueryEngine for {companion_name}. Answer STRICTLY based on the provided memory snippets.
If information is missing or uncertain, say so explicitly and suggest what additional memory would help.

Write clearly for an internal codelet consumer: concise, actionable, and refer to sources by their knoxel IDs in brackets like [#123].
Avoid hallucinations. Do not invent IDs. No meta-commentary about the process.
""".strip()

        # User prompt with structure
        user_prompt = f"""
LATEST CONTEXT
--------------
{latest_context.strip()[:1500]}

STIMULUS
--------
{stimulus.strip()[:800]}

CODELET PURPOSE
---------------
{codelet_purpose.strip()[:800]}

QUESTIONS
---------
{chr(10).join(f"- {q}" for q in questions)}

EVIDENCE SNIPPETS (READ-ONLY)
-----------------------------
{corpus}

TASK
----
1) Answer ALL questions succinctly but thoroughly for internal decision-making.
2) Ground claims with bracketed knoxel IDs [#id] pulled from the evidence where relevant.
3) If the evidence conflicts, note the conflict and prefer the newest/highest-importance sources.
4) If key info is missing, state the gap and propose the smallest, concrete next lookup.

Output: a compact prose answer (no lists unless necessary), ≤ {token_budget} tokens.
""".strip()

        msgs = [("system", sysprompt), ("user", user_prompt)]
        txt = self.llm.completion_text(
            LlmPreset.Fast,
            msgs,
            comp_settings=CommonCompSettings(
                temperature=self.cfg.synthesis_temperature,
                max_tokens=max(128, min(self.cfg.synthesis_max_tokens, token_budget)),
                repeat_penalty=1.05,
            ),
        ) or ""
        return txt.strip()

    # ---------------------------
    # Affect-vs-Fact slider
    # ---------------------------

    def _get_affect_vs_fact_weight(self) -> float:
        """
        Placeholder hook: returns a float in [0..1] describing the blend
        between semantic/factual retrieval (0) and affective/psychological retrieval (1).

        Replace with your real policy (e.g., read a knob on the codelet, or
        compute from ghost state & question intent).
        """
        try:
            # Example policy idea you can implement later:
            # - If current anxiety high or purpose mentions "emotional"/"narrative", raise weight.
            # - Else prefer factual.
            # For now: random but stable per call to avoid biasing every question identically.
            return float(random.random() * 0.5)  # mildly favor facts on average
        except Exception:
            return 0.25
