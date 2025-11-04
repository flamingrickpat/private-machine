"""
Ghost Cycle — Functional LIDA/GWT-style architecture with blackboard state.

WHY THIS FILE EXISTS
--------------------
This is the "glue" cycle for a meta-agent (the "ghost") built from small,
mostly-pure functions that read/write a central BLACKBOARD (KV store).
It captures our design choices:

- Separate *fast path* (cheap reactions) from *slow/introspective path* (sims, psychologist).
- Use a Current Situational Model (CSM) to aggregate "what matters now".
- Use global workspace (GWT) to broadcast consciously-accessible content (CCQ).
- Run THREE simulation flavors (world/ToM, self, meta/style/capability).
- Generate *subjective experience* (Stage-2) and *reportability* (Stage-3).
- Keep introspective content OFF the user path unless explicitly requested or triggered.
- Learn at the tail end from expectation→reality deltas and consolidate memory.

This scaffold sets *defaults* everywhere so you can ship pieces gradually without
breaking the overall loop. All Procs are idempotent-ish and write namespaced keys.

READ THIS FIRST
---------------
- The "ghost" is a blackboard: it’s a dict-like object with helpers to get/set typed values.
- Each Proc only touches its namespace (e.g., "csm.*", "sim.*", "qualia.*").
- You can unit-test each Proc in isolation by passing a fresh GhostCodelets().
- The sequence below reflects cognitive causality: input → memory → attention →
  simulation → introspection → workspace → action → learning → consolidation.

TWEAKS WE AGREED TO
-------------------
1) Optional EARLY partial memory consolidation right after memory recall
   (improves near-term priming for coalition building).

2) Apply self-model/affective appraisal slightly BEFORE coalition rating
   (so emotions influence salience scoring, not just selection).

3) Introspection/psychologist passes can be throttled (every N ticks) and
   gated by triggers (user asks, or codelet sets "introspect_now").

4) Learning happens *after* workspace integration to avoid teaching on unstable states.

Replace stubs with your real logic progressively. Keep the comments!
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum, Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Callable
import time
import uuid
import math
import random
import logging

# --------------------------------------------------------------------------------------
# Minimal infra / types
# --------------------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class ControlVariables(StrEnum):
    """
    Blackboard control flags (global, not namespaced to a Proc).

    HasInput: Did we receive *any* new stimulus that should advance the slow path?
              If False, we still let fast-path do small maintenance and then exit.
    """
    HasInput = "control.has_input"
    TickId = "control.tick_id"
    IntrospectNow = "control.introspect_now"      # raised by codelets or user request
    MetaFrequency = "control.meta_frequency"      # throttle for heavy introspective passes
    FastPathOnly = "control.fast_path_only"       # allow quick answer and defer slow path
    CharacterId = "control.character_id"          # who this ghost instance "is"


class KVPolicy(Enum):
    """How to write defaults into the blackboard."""
    SET_IF_MISSING = auto()
    OVERWRITE = auto()
    APPEND = auto()


@dataclass
class ShellBase:
    """
    Shell-like environment that owns IO buffers, tool handles, etc.

    This is intentionally tiny in the scaffold. In your real system it will:
    - provide input stimuli (user messages, world events),
    - manage tool calls / output channels,
    - hold persistent services (vector DB, model clients).
    """
    inbox: List[Dict[str, Any]] = field(default_factory=list)
    outbox: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class GhostCodelets:
    """
    Central blackboard object. Think 'functional state' with typed helpers.
    This is the ONLY mutable shared state across Procs.

    Namespacing convention:
        "<namespace>.<key>"  e.g., "csm.snapshot", "sim.world.outcomes"
    """
    store: Dict[str, Any] = field(default_factory=dict)

    # ---------- convenience accessors ----------
    def set(self, key: StrEnum | str, value: Any, policy: KVPolicy = KVPolicy.OVERWRITE):
        k = str(key)
        if policy == KVPolicy.SET_IF_MISSING:
            if k not in self.store:
                self.store[k] = value
        elif policy == KVPolicy.APPEND:
            self.store.setdefault(k, [])
            self.store[k].append(value)
        else:
            self.store[k] = value

    def get(self, key: StrEnum | str, default: Any = None) -> Any:
        return self.store.get(str(key), default)

    def get_bool(self, key: StrEnum | str, default: bool = False) -> bool:
        return bool(self.store.get(str(key), default))

    def get_int(self, key: StrEnum | str, default: int = 0) -> int:
        return int(self.store.get(str(key), default))

    def get_list(self, key: StrEnum | str, default: Optional[List[Any]] = None) -> List[Any]:
        return list(self.store.get(str(key), [] if default is None else default))

    def bump_tick(self):
        """Increase global tick and return it."""
        cur = self.get_int(ControlVariables.TickId, 0) + 1
        self.set(ControlVariables.TickId, cur)
        return cur

    # ---------- lightweight tracing ----------
    def trace(self, where: str, msg: str, payload: Optional[Dict[str, Any]] = None):
        """
        Writes a human-readable trace line into blackboard for later debugging.
        Keeps the stdout clean in production; you can dump these on error.
        """
        entry = {
            "t": time.time(),
            "tick": self.get_int(ControlVariables.TickId, 0),
            "where": where,
            "msg": msg,
            "data": payload or {}
        }
        self.set(f"trace.{uuid.uuid4()}", entry)


# --------------------------------------------------------------------------------------
# Utility stubs (replace with real implementations)
# --------------------------------------------------------------------------------------

def now_ts() -> float:
    return time.time()


def rand_score() -> float:
    """Cheap stand-in for a scoring model."""
    return round(random.uniform(0, 1), 3)


# --------------------------------------------------------------------------------------
# PROC IMPLEMENTATIONS (ALL STATIC, PURE-ish AGAINST THE BLACKBOARD)
# Each Proc reads what it needs and writes its namespace.
# --------------------------------------------------------------------------------------

class BaseProc:
    """Top-level orchestration helpers that are not semantically attached to one subsystem."""

    @staticmethod
    def load(ghost: GhostCodelets, shell: ShellBase):
        """
        Load or refresh any state that must exist every tick.
        This is where we set safe defaults so later code can assume presence.
        """
        ghost.set(ControlVariables.CharacterId, ghost.get(ControlVariables.CharacterId, "companion-01"))
        ghost.set(ControlVariables.MetaFrequency, ghost.get(ControlVariables.MetaFrequency, 5))  # every 5 ticks
        ghost.set(ControlVariables.FastPathOnly, ghost.get_bool(ControlVariables.FastPathOnly, False))
        ghost.set(ControlVariables.IntrospectNow, ghost.get_bool(ControlVariables.IntrospectNow, False))
        # If inbox has content, mark HasInput True; else False.
        has_input = bool(shell.inbox)
        ghost.set(ControlVariables.HasInput, has_input)
        ghost.trace("BaseProc.load", f"HasInput={has_input}")

    @staticmethod
    def init_cycle(ghost: GhostCodelets):
        """
        Start-of-tick initialization: bump tick counter, set timing markers, prepare scratch.
        Rationale: We want deterministic ordering and easy profiling per tick.
        """
        tick = ghost.bump_tick()
        ghost.set("meta.tick_started_at", now_ts())
        ghost.set("meta.tick_id_str", f"tick-{tick:06d}", policy=KVPolicy.OVERWRITE)
        ghost.trace("BaseProc.init_cycle", f"tick={tick}")

    @staticmethod
    def read_initial_buffer(ghost: GhostCodelets, shell: ShellBase):
        """
        Pull new stimuli (user msgs, world events) into blackboard.
        We keep raw stimuli AND a normalized "stimulus batch" for later Procs.
        """
        batch = []
        while shell.inbox:
            batch.append(shell.inbox.pop(0))
        # Normalize shape (id, type, content, actor, ts)
        norm = []
        for item in batch:
            norm.append({
                "id": item.get("id", str(uuid.uuid4())),
                "kind": item.get("kind", "user_message"),
                "content": item.get("content", ""),
                "actor_id": item.get("actor_id", 0),
                "ts": item.get("ts", now_ts())
            })
        ghost.set("input.batch", norm)
        ghost.trace("BaseProc.read_initial_buffer", f"n={len(norm)}", {"sample": norm[:1]})

    @staticmethod
    def update_dynamic_states(ghost: GhostCodelets):
        """
        Compute any per-tick dynamic latent state (e.g., emotion, energy, bias).
        Rationale: later modules depend on these; we compute them ONCE.
        """
        # STUB: Replace with your real vector/EMA computations.
        ms = {
            "valence": rand_score(),
            "arousal": rand_score(),
            "dominance": rand_score(),
            "ego_strength": rand_score(),
            "mental_energy": rand_score(),
        }
        ghost.set("state.mental", ms)
        ghost.trace("BaseProc.update_dynamic_states", "latent", ms)

    @staticmethod
    def update_triage(ghost: GhostCodelets, shell: ShellBase):
        """
        Triage result -> send any immediate fast-path responses to shell.outbox.
        """
        fast_replies = ghost.get_list("triage.fast_replies")
        for r in fast_replies:
            shell.outbox.append(r)
        if fast_replies:
            ghost.trace("BaseProc.update_triage", f"sent={len(fast_replies)}")

    @staticmethod
    def update_closing_buffer(ghost: GhostCodelets, shell: ShellBase):
        """
        Send final CCQ-assembled outputs to consumers (user thread, thought thread, tools).
        """
        ccq = ghost.get("workspace.ccq", {"world": None, "thought": None})
        if ccq.get("world"):
            shell.outbox.append({"kind": "to_user", "content": ccq["world"]})
        if ccq.get("thought"):
            shell.outbox.append({"kind": "to_thought_log", "content": ccq["thought"]})
        ghost.trace("BaseProc.update_closing_buffer", "ccq dispatched")

    @staticmethod
    def update_ccq(ghost: GhostCodelets, shell: ShellBase):
        """
        After dispatch, record the current CCQ knoxel IDs to inform next tick's CCQ prime.
        """
        ghost.set("workspace.ccq_state", {"knoxels": ghost.get_list("workspace.knoxel_ids")})
        ghost.trace("BaseProc.update_ccq", "ccq_state updated")

    @staticmethod
    def save(ghost: GhostCodelets, shell: ShellBase):
        """
        Persist blackboard snapshot (replace with your JSON dump).
        This stub just stamps timing & size; use your own serialization.
        """
        ghost.set("meta.tick_finished_at", now_ts())
        # In your implementation, dump to ./data/tick_<id>.json
        ghost.trace("BaseProc.save", "persisted stub", {"keys": len(ghost.store)})


# -------------------------- CSM (Current Situational Model) ---------------------------

class CsmProc:
    """Building, decaying, and updating the Current Situational Model."""

    @staticmethod
    def prepare_csm(ghost: GhostCodelets):
        """
        Either load last CSM or create a new scaffold. Decay happens in a separate step.
        """
        prev = ghost.get("csm.state", {"items": [], "version": 1})
        ghost.set("csm.state", prev)
        ghost.trace("CsmProc.prepare_csm", "loaded_or_init", {"n_items": len(prev["items"])})

    @staticmethod
    def decay(ghost: GhostCodelets):
        """
        Reduce the influence of old features (forgetting curve).
        This keeps CSM bounded and responsive.
        """
        state = ghost.get("csm.state")
        # STUB: soften all item weights by a decay factor
        for it in state["items"]:
            it["salience"] = max(0.0, 0.95 * it.get("salience", 0.0))
            it["valence"] = 0.95 * it.get("valence", 0.0)
        ghost.set("csm.state", state)
        ghost.trace("CsmProc.decay", "applied")

    @staticmethod
    def update_situation(ghost: GhostCodelets):
        """
        Compute a short-term contextual 'theme' from inputs and mental state.
        This sets the focus for subsequent memory queries and codelets.
        """
        items = ghost.get_list("csm.state.items")
        ms = ghost.get("state.mental", {})
        batch = ghost.get_list("input.batch")
        theme = {
            "keywords": ["help", "plan"] if batch else ["idle"],
            "ms_hint": ms,
            "recent_input": batch[-1]["content"] if batch else "",
        }
        ghost.set("csm.theme", theme)
        ghost.trace("CsmProc.update_situation", "theme", theme)

    @staticmethod
    def build_latent_prompts(ghost: GhostCodelets):
        """
        Pre-build small prompt snippets for later codelets (saves tokens/time).
        """
        theme = ghost.get("csm.theme", {})
        latent = {
            "short": f"[short theme] {theme.get('keywords')}",
            "long": f"[long theme] {theme}",
        }
        ghost.set("csm.latent_prompts", latent)
        ghost.trace("CsmProc.build_latent_prompts", "latent_ready")


# ---------------------------- Memory / PAM-like codelets -----------------------------

class PamCodeletProc:
    """Memory retrieval & integration."""

    @staticmethod
    def query_memories_situation(ghost: GhostCodelets):
        theme = ghost.get("csm.theme", {})
        # STUB: vector search by theme keywords
        hits = [{"id": "mem1", "strength": 0.6}, {"id": "mem2", "strength": 0.4}]
        ghost.set("mem.situation_hits", hits)
        ghost.trace("PamCodeletProc.query_memories_situation", "hits", {"n": len(hits)})

    @staticmethod
    def query_memories_mental_state(ghost: GhostCodelets):
        ms = ghost.get("state.mental", {})
        # STUB: search emotionally similar memories
        hits = [{"id": "memA", "similarity": 0.7}]
        ghost.set("mem.ms_hits", hits)
        ghost.trace("PamCodeletProc.query_memories_mental_state", "hits", {"n": len(hits)})

    @staticmethod
    def update_memories(ghost: GhostCodelets):
        """
        Merge new memory hits into CSM items with boosting of existing items.
        """
        csm = ghost.get("csm.state", {"items": []})
        new_items = [{"id": h["id"], "salience": h.get("strength", h.get("similarity", 0.5)), "valence": 0.0}
                     for h in (ghost.get_list("mem.situation_hits") + ghost.get_list("mem.ms_hits"))]
        csm["items"].extend(new_items)
        ghost.set("csm.state", csm)
        ghost.trace("PamCodeletProc.update_memories", "merged", {"added": len(new_items)})

    @staticmethod
    def partial_consolidate(ghost: GhostCodelets):
        """
        EARLY consolidation to prime near-term selection (optional tweak).
        Think of this as a soft write to episodic memory cache.
        """
        # STUB: record a lightweight summary for priming
        theme = ghost.get("csm.theme", {})
        ghost.set("mem.partial", {"last_theme": theme, "ts": now_ts()})
        ghost.trace("PamCodeletProc.partial_consolidate", "done")


# ------------------------------------ TRIAGE -----------------------------------------

class TriageProc:
    """Decide if fast-path reply is possible; otherwise schedule slow path."""

    @staticmethod
    def triage_stimulus(ghost: GhostCodelets):
        batch = ghost.get_list("input.batch")
        fast_replies = []
        fast_path_only = ghost.get_bool(ControlVariables.FastPathOnly, False)
        if batch and len(batch[-1].get("content", "")) < 30:
            # STUB: simple heuristic — short messages may be handled quickly
            fast_replies.append({"kind": "to_user", "content": "🗲 Quick reply (fast path)."})
            # If marked fast-only, skip slow pipeline later:
            ghost.set(ControlVariables.FastPathOnly, True if fast_path_only else False)
        ghost.set("triage.fast_replies", fast_replies)
        ghost.trace("TriageProc.triage_stimulus", f"fast={len(fast_replies)}")


# --------------------------------- Policy Dynamics -----------------------------------

class PolicyDynamics:
    """Form/update intentions around the present situation (short timeframe)."""

    @staticmethod
    def preprocess_intentions(ghost: GhostCodelets):
        # STUB: derive or refresh simple intents
        intents = [{"id": "assist", "priority": 0.6}, {"id": "clarify", "priority": 0.4}]
        ghost.set("policy.intents", intents)
        ghost.trace("PolicyDynamics.preprocess_intentions", "intents", {"n": len(intents)})

    @staticmethod
    def execute_intention_codelets(ghost: GhostCodelets):
        # STUB: appraise deltas / fulfillments
        updates = [{"id": "assist", "delta": +0.1}, {"id": "clarify", "delta": -0.05}]
        ghost.set("policy.intent_updates", updates)
        ghost.trace("PolicyDynamics.execute_intention_codelets", "updates", {"n": len(updates)})

    @staticmethod
    def postprocess_intentions(ghost: GhostCodelets):
        # STUB merge updates
        base = {i["id"]: i for i in ghost.get_list("policy.intents")}
        for u in ghost.get_list("policy.intent_updates"):
            base[u["id"]]["priority"] = max(0.0, min(1.0, base[u["id"]]["priority"] + u["delta"]))
        ghost.set("policy.intents", list(base.values()))
        ghost.trace("PolicyDynamics.postprocess_intentions", "merged")


# ------------------------------------ Codelets ---------------------------------------

class CodeletProc:
    """Pick families, filter CSM, run codelets to produce features."""

    @staticmethod
    def determine_codelet_families(ghost: GhostCodelets):
        # STUB: choose families based on theme & intents
        fams = ["introspection" if "feel" in ghost.get("csm.theme", {}).get("recent_input", "")
                else "dialogue",
                "memory_linker"]
        ghost.set("code.families", fams)
        ghost.trace("CodeletProc.determine_codelet_families", "families", {"fams": fams})

    @staticmethod
    def pre_filter_csm(ghost: GhostCodelets):
        # STUB: favor introspection if user asks about feelings
        prefer_introspection = "feel" in ghost.get("csm.theme", {}).get("recent_input", "")
        ghost.set("code.prefers_introspection", prefer_introspection)
        ghost.trace("CodeletProc.pre_filter_csm", f"prefers_introspection={prefer_introspection}")

    @staticmethod
    def execute_codelets(ghost: GhostCodelets):
        """
        Run codelets and add non-causal features to CSM.
        """
        csm = ghost.get("csm.state")
        produced = [
            {"id": f"feat-{uuid.uuid4()}", "type": "CodeletPercept", "salience": rand_score(), "valence": rand_score()},
            {"id": f"feat-{uuid.uuid4()}", "type": "CodeletOutput", "salience": rand_score(), "valence": rand_score()}
        ]
        csm["items"].extend(produced)
        ghost.set("csm.state", csm)
        ghost.set("code.produced_features", produced)
        ghost.trace("CodeletProc.execute_codelets", "produced", {"n": len(produced)})


# ------------------------------------ Attention --------------------------------------

class AttentionProc:
    """Purge, cluster into coalitions, rate, select."""

    @staticmethod
    def purge_features(ghost: GhostCodelets):
        csm = ghost.get("csm.state")
        before = len(csm["items"])
        csm["items"] = [it for it in csm["items"] if it.get("valence", 0) > -0.9]  # dummy threshold
        ghost.set("csm.state", csm)
        ghost.trace("AttentionProc.purge_features", f"pruned={before - len(csm['items'])}")

    @staticmethod
    def build_coalitions(ghost: GhostCodelets):
        """
        Group items around intents/themes. Real impl: clustering by embeddings/keywords.
        """
        csm = ghost.get("csm.state")
        coalitions = [
            {"id": "coal-1", "items": [it["id"] for it in csm["items"][:2]], "topic": "assist"},
            {"id": "coal-2", "items": [it["id"] for it in csm["items"][2:]], "topic": "clarify"},
        ]
        ghost.set("attn.coalitions", coalitions)
        ghost.trace("AttentionProc.build_coalitions", "formed", {"n": len(coalitions)})

    @staticmethod
    def rate_coalitions(ghost: GhostCodelets):
        """
        IMPORTANT TWEAK:
        We run QualiaProc.apply_self_model_attention() *before* this call in the main cycle,
        so affective/self-model bias can influence these ratings (salience/priority).
        """
        cols = ghost.get_list("attn.coalitions")
        scored = []
        for c in cols:
            base = rand_score()
            bias = ghost.get("qualia.attn_bias", 0.0)  # set by QualiaProc earlier
            scored.append({"id": c["id"], "score": max(0.0, min(1.0, base + bias)), "topic": c["topic"]})
        ghost.set("attn.scores", scored)
        ghost.trace("AttentionProc.rate_coalitions", "scored", {"scores": scored})

    @staticmethod
    def select_coalitions(ghost: GhostCodelets, k: int = 2):
        scores = sorted(ghost.get_list("attn.scores"), key=lambda x: x["score"], reverse=True)
        selected = scores[:k]
        ghost.set("attn.selected", selected)
        ghost.trace("AttentionProc.select_coalitions", "selected", {"sel": selected})


# ----------------------------------- Simulation --------------------------------------

class SimulationProc:
    """World (ToM/game-theory), self, and meta/capability simulations + selection."""

    @staticmethod
    def determine_steering_vector(ghost: GhostCodelets):
        """
        Produce guidance for the sims based on MS + intents, including pos/neu/neg outcome modes.
        """
        ms = ghost.get("state.mental", {})
        intents = ghost.get_list("policy.intents")
        steer = {"mood": ms, "intents": intents, "modes": ["positive", "neutral", "negative"]}
        ghost.set("sim.steer", steer)
        ghost.trace("SimulationProc.determine_steering_vector", "steer", steer)

    @staticmethod
    def simulate_world(ghost: GhostCodelets):
        """
        Outward-looking: ToM, game theory, world dynamics.
        """
        outcomes = [{"option": "ask_clarify", "risk": 0.2, "utility": 0.6},
                    {"option": "provide_answer", "risk": 0.1, "utility": 0.5}]
        ghost.set("sim.world.outcomes", outcomes)
        ghost.trace("SimulationProc.simulate_world", "done", {"n": len(outcomes)})

    @staticmethod
    def simulate_self(ghost: GhostCodelets):
        """
        Inward-looking: self-regulation, long-term goals, energy costs.
        """
        outcomes = [{"option": "reflect_deeper", "energy_cost": 0.3, "progress": 0.7}]
        ghost.set("sim.self.outcomes", outcomes)
        ghost.trace("SimulationProc.simulate_self", "done", {"n": len(outcomes)})

    @staticmethod
    def simulate_meta(ghost: GhostCodelets):
        """
        Meta/style/capability: persona adherence, avoid impossible actions.
        """
        outcomes = [{"option": "avoid_tool_use", "style_score": 0.8}]
        ghost.set("sim.meta.outcomes", outcomes)
        ghost.trace("SimulationProc.simulate_meta", "done", {"n": len(outcomes)})

    @staticmethod
    def select_simulation(ghost: GhostCodelets):
        """
        Bundle sim outputs and pick best option(s).
        """
        bundle = ghost.get_list("sim.world.outcomes") + ghost.get_list("sim.self.outcomes") + ghost.get_list("sim.meta.outcomes")
        # STUB: pick top by utility/progress/style with naive normalize
        for b in bundle:
            b["score"] = max(b.get("utility", 0), b.get("progress", 0), b.get("style_score", 0)) - b.get("risk", 0)
        best = sorted(bundle, key=lambda x: x["score"], reverse=True)[:2]
        ghost.set("sim.selected", best)
        ghost.trace("SimulationProc.select_simulation", "selected", {"best": best})


# ------------------------------------- Qualia ----------------------------------------

class QualiaProc:
    """
    Subjective experience & self-model:
    - Generate a compressed 'what-it's-like' snapshot (Stage-2).
    - Apply self-model bias to attention and action.
    - Run psychologist-agent to reconstruct reasons (Stage-3 reportability) — gated.
    """

    @staticmethod
    def apply_self_model_simulation(ghost: GhostCodelets):
        """
        Bias the selected simulation(s) based on self-model (e.g., avoid rumination if energy low).
        """
        ms = ghost.get("state.mental", {})
        selected = ghost.get_list("sim.selected")
        bias = -0.1 if ms.get("mental_energy", 0.5) < 0.3 else +0.05
        for s in selected:
            s["score"] += bias
        ghost.set("sim.selected", selected)
        ghost.trace("QualiaProc.apply_self_model_simulation", "bias_applied", {"bias": bias})

    @staticmethod
    def generate_subjective_experience(ghost: GhostCodelets):
        """
        Create a compact, *depictive* snapshot of inner state to be broadcast.
        This is the 'phenomenal-like' content (simulacrum) in functional form.
        """
        ms = ghost.get("state.mental", {})
        selected = ghost.get_list("sim.selected")
        snapshot = {
            "feel": {"valence": ms.get("valence", 0.0), "arousal": ms.get("arousal", 0.0)},
            "focus": [s["option"] for s in selected],
            "note": "what-it-is-like right now (functional simulacrum)"
        }
        ghost.set("qualia.snapshot", snapshot)
        ghost.trace("QualiaProc.generate_subjective_experience", "snapshot", snapshot)

    @staticmethod
    def match_subjective_experience(ghost: GhostCodelets):
        """
        Retrieve past experiences similar to current snapshot (analogy/backreferences).
        """
        snap = ghost.get("qualia.snapshot", {})
        # STUB: pretend we matched something
        matches = [{"tick": ghost.get_int(ControlVariables.TickId) - 7, "similarity": 0.62, "label": "like_when_we_helped_before"}]
        ghost.set("qualia.matches", matches)
        ghost.trace("QualiaProc.match_subjective_experience", "matches", {"n": len(matches)})

    @staticmethod
    def update_self_model_attention(ghost: GhostCodelets):
        """
        Psychologist-agent REASONING about *why* these features arose (hidden-cause inference).
        We throttle by meta-frequency and by trigger, and we DO NOT surface to user by default.
        """
        tick = ghost.get_int(ControlVariables.TickId)
        freq = ghost.get_int(ControlVariables.MetaFrequency, 5)
        triggered = ghost.get_bool(ControlVariables.IntrospectNow, False)
        if (tick % freq != 0) and not triggered:
            ghost.trace("QualiaProc.update_self_model_attention", "skipped (throttled)")
            return

        # STUB: infer reasons (private; do not expose unless asked)
        reasons = [{"feature": "focus", "hypothesis": "seeking clarity to reduce uncertainty", "confidence": 0.6}]
        ghost.set("qualia.private_reasons", reasons)
        # Provide a SMALL attention bias for the next rating step (our tweak #2).
        ghost.set("qualia.attn_bias", +0.05 if reasons else 0.0)
        ghost.trace("QualiaProc.update_self_model_attention", "inferred", {"reasons": reasons})

    @staticmethod
    def apply_self_model_attention(ghost: GhostCodelets):
        """
        (Intentionally called BEFORE rating) We only set bias; actual use happens in AttentionProc.rate_coalitions.
        """
        # Nothing to do here in the scaffold; bias set in update_self_model_attention.
        ghost.trace("QualiaProc.apply_self_model_attention", "bias_ready")

    @staticmethod
    def apply_self_model_action(ghost: GhostCodelets):
        """
        Small bias on action selection later (e.g., nudge toward reflective if energy allows).
        """
        ms = ghost.get("state.mental", {})
        ghost.set("qualia.action_bias", +0.05 if ms.get("mental_energy", 0.5) > 0.4 else -0.05)
        ghost.trace("QualiaProc.apply_self_model_action", "action_bias_set")

    @staticmethod
    def generate_subjective_action_descision(ghost: GhostCodelets):
        """
        Compressed description of 'why we chose this path' (kept private until asked).
        """
        selected = ghost.get_list("sim.selected")
        desc = {"decision": [s["option"] for s in selected],
                "why": ghost.get_list("qualia.private_reasons")}
        ghost.set("qualia.subjective_decision", desc)
        ghost.trace("QualiaProc.generate_subjective_action_descision", "done")

    @staticmethod
    def update_self_model(ghost: GhostCodelets):
        """
        Second psychologist pass focused on ACTION reasons; writes to private store.
        """
        private = ghost.get("qualia.private_model", {})
        private["last_action_reason"] = ghost.get("qualia.subjective_decision", {})
        ghost.set("qualia.private_model", private)
        ghost.trace("QualiaProc.update_self_model", "updated")


# ----------------------------------- Workspace / GWT ---------------------------------

class WorkspaceProc:
    """Integrate coalitions into global workspace and build CCQ (outputs/prompts)."""

    @staticmethod
    def integrate_coalitions(ghost: GhostCodelets):
        selected = ghost.get_list("attn.selected")
        # STUB: materialize knoxel ids and mark as 'conscious'
        knoxel_ids = [f"knx-{s['id']}" for s in selected]
        ghost.set("workspace.knoxel_ids", knoxel_ids)
        ghost.trace("WorkspaceProc.integrate_coalitions", "integrated", {"knoxels": knoxel_ids})

    @staticmethod
    def update_causality(ghost: GhostCodelets):
        """
        Mark integrated features as causal (so downstream learning uses the right subset).
        """
        ghost.set("workspace.causal_mark", True)
        ghost.trace("WorkspaceProc.update_causality", "causal_marked")

    @staticmethod
    def generate_ccq(ghost: GhostCodelets):
        """
        Build prompts/contents for world & thought threads.
        NOTE: We keep introspective content OUT unless user asked or trigger is set.
        """
        theme = ghost.get("csm.theme", {})
        snapshot = ghost.get("qualia.snapshot", {})
        introspect_ok = ghost.get_bool(ControlVariables.IntrospectNow, False)
        world = f"Story reply based on theme {theme.get('keywords')}."
        thought = f"Inner monologue: focusing on {snapshot.get('focus')}."

        if introspect_ok:
            # Allow some private reasoning to be narrated
            reasons = ghost.get_list("qualia.private_reasons")
            thought += f" (self-notes: {reasons})"

        ghost.set("workspace.ccq", {"world": world, "thought": thought})
        ghost.trace("WorkspaceProc.generate_ccq", "ccq_ready")


# ----------------------------------- Action Selection --------------------------------

class ActionSelectionProc:
    """Pick a general behavior policy for the next user/world turn."""

    @staticmethod
    def create_possible_behaviors(ghost: GhostCodelets):
        options = ["answer_helpfully", "ask_clarifying_question", "reflect_then_answer"]
        ghost.set("action.options", options)
        ghost.trace("ActionSelectionProc.create_possible_behaviors", "options", {"n": len(options)})

    @staticmethod
    def get_behavior(ghost: GhostCodelets):
        bias = ghost.get("qualia.action_bias", 0.0)
        options = ghost.get_list("action.options")
        # STUB: pick by dumb score + bias
        scored = [{"opt": o, "score": rand_score() + bias} for o in options]
        pick = max(scored, key=lambda x: x["score"])["opt"] if scored else "answer_helpfully"
        ghost.set("action.selected", pick)
        ghost.trace("ActionSelectionProc.get_behavior", "selected", {"pick": pick})


# -------------------------------------- Learning -------------------------------------

class PolicyDynamicsLearn:
    """Expectation-outcome learning (for intents/policies)."""

    @staticmethod
    def learn_expectation_outcome(ghost: GhostCodelets):
        # STUB: record quadruple placeholder
        quadruple = {"situation": ghost.get("csm.theme", {}),
                     "expectation": ghost.get_list("sim.selected"),
                     "reality": None, "delta": None}
        ghost.set("learn.intent_quadruple", quadruple)
        ghost.trace("PolicyDynamics.learn_expectation_outcome", "queued")


class AttentionLearn:
    """Refine attention weights from usage statistics."""

    @staticmethod
    def learn_attention_behavior(ghost: GhostCodelets):
        # STUB: bump a counter
        cnt = ghost.get_int("learn.attn_cnt", 0) + 1
        ghost.set("learn.attn_cnt", cnt)
        ghost.trace("AttentionProc.learn_attention_behavior", f"cnt={cnt}")


class SimulationLearn:
    """Update world/self/meta simulation priors from outcome tracking."""

    @staticmethod
    def learn_simulation_result(ghost: GhostCodelets):
        # STUB
        ghost.trace("SimulationProc.learn_simulation_result", "noop")


# -------------------------------------- Memory ---------------------------------------

class MemoryProc:
    """Consolidate episodic/semantic memory and update narratives."""

    @staticmethod
    def consolidate_memory(ghost: GhostCodelets):
        # STUB: snapshot pointers to long-term store
        ghost.trace("MemoryProc.consolidate_memory", "noop")

    @staticmethod
    def update_narratives(ghost: GhostCodelets):
        # STUB: refresh high-level autobiographical threads
        ghost.trace("MemoryProc.update_narratives", "noop")


# ------------------------------------ Optimization -----------------------------------

class OptimizationProc:
    """Prompt evolution, optional LoRA updates (offline or throttled)."""

    @staticmethod
    def improve_prompts(ghost: GhostCodelets, shell: ShellBase):
        # STUB: record an improvement suggestion (human-in-the-loop later)
        ghost.set("opt.suggestion", "Consider shortening introspection snippets.")
        ghost.trace("OptimizationProc.improve_prompts", "suggested")

    @staticmethod
    def learn_lora(ghost: GhostCodelets, shell: ShellBase):
        # STUB: no-op (heavy training should be throttled / external)
        ghost.trace("OptimizationProc.learn_lora", "noop")


# ======================================================================================
# MAIN COGNITIVE CYCLE
# ======================================================================================

def cognitive_cycle(ghost: GhostCodelets, shell: ShellBase):
    """
    The full causal order for one tick.
    You can early-return after fast-path if desired, but we keep defaults safe so
    later calls won't crash even if earlier stages had no input.
    """

    # ---- ENTRY: load defaults, bump tick, ingest inputs --------------------------------
    BaseProc.load(ghost, shell)
    BaseProc.init_cycle(ghost)
    BaseProc.read_initial_buffer(ghost, shell)

    if not ghost.get_bool(ControlVariables.HasInput):
        # No new input: still maintain a light heartbeat if you like
        ghost.trace("cognitive_cycle", "no_input_exit")
        return

    # ---- PRE-STATE / CSM & THEME -------------------------------------------------------
    BaseProc.update_dynamic_states(ghost)
    CsmProc.prepare_csm(ghost)
    CsmProc.decay(ghost)
    CsmProc.update_situation(ghost)

    # ---- MEMORY (PAM) ------------------------------------------------------------------
    PamCodeletProc.query_memories_situation(ghost)
    PamCodeletProc.query_memories_mental_state(ghost)
    PamCodeletProc.update_memories(ghost)
    # OPTIONAL TWEAK: small early consolidation to prime near-term selection
    PamCodeletProc.partial_consolidate(ghost)

    # ---- PROMPTS FOR DOWNSTREAM CODELETS ----------------------------------------------
    CsmProc.build_latent_prompts(ghost)

    # ---- TRIAGE / FAST-PATH ------------------------------------------------------------
    TriageProc.triage_stimulus(ghost)
    BaseProc.update_triage(ghost, shell)
    if ghost.get_bool(ControlVariables.FastPathOnly, False):
        ghost.trace("cognitive_cycle", "fast_path_only_exit")
        return

    # ---- INTENTIONS (SHORT HORIZON) ----------------------------------------------------
    PolicyDynamics.preprocess_intentions(ghost)
    PolicyDynamics.execute_intention_codelets(ghost)

    # ---- CODELET FANNING (FEATURE CREATION) --------------------------------------------
    CodeletProc.determine_codelet_families(ghost)
    CodeletProc.pre_filter_csm(ghost)
    CodeletProc.execute_codelets(ghost)

    # ---- INTENT POSTPROCESS ------------------------------------------------------------
    PolicyDynamics.postprocess_intentions(ghost)

    # ---- ATTENTION PREP ----------------------------------------------------------------
    AttentionProc.purge_features(ghost)
    AttentionProc.build_coalitions(ghost)

    # ---- SIMULATIONS (MID HORIZON) -----------------------------------------------------
    SimulationProc.determine_steering_vector(ghost)
    SimulationProc.simulate_world(ghost)
    SimulationProc.simulate_self(ghost)
    SimulationProc.simulate_meta(ghost)
    SimulationProc.select_simulation(ghost)

    # ---- QUALIA / SELF-MODEL -----------------------------------------------------------
    # Bias simulations based on self-model (energy/ego) BEFORE we build subjective snapshot.
    QualiaProc.apply_self_model_simulation(ghost)
    QualiaProc.generate_subjective_experience(ghost)
    QualiaProc.match_subjective_experience(ghost)

    # Psychologist-agent does hidden-cause inference.
    # NOTE: Throttled by MetaFrequency and only produces a *bias* for attention scoring.
    QualiaProc.update_self_model_attention(ghost)

    # Apply attention bias (affective) BEFORE coalition rating (tweak #2).
    QualiaProc.apply_self_model_attention(ghost)

    # ---- ATTENTION RATING & SELECTION --------------------------------------------------
    AttentionProc.rate_coalitions(ghost)
    AttentionProc.select_coalitions(ghost)

    # ---- WORKSPACE / CCQ ---------------------------------------------------------------
    WorkspaceProc.integrate_coalitions(ghost)
    WorkspaceProc.update_causality(ghost)
    WorkspaceProc.generate_ccq(ghost)

    # ---- ACTION SELECTION --------------------------------------------------------------
    ActionSelectionProc.create_possible_behaviors(ghost)
    QualiaProc.apply_self_model_action(ghost)
    ActionSelectionProc.get_behavior(ghost)

    # ---- SUBJECTIVE ACTION REPORT (PRIVATE) -------------------------------------------
    QualiaProc.generate_subjective_action_descision(ghost)
    QualiaProc.update_self_model(ghost)

    # ---- DISPATCH OUTPUTS --------------------------------------------------------------
    BaseProc.update_closing_buffer(ghost, shell)
    BaseProc.update_ccq(ghost, shell)

    # ---- LEARNING / CONSOLIDATION / OPTIMIZATION --------------------------------------
    AttentionLearn.learn_attention_behavior(ghost)
    PolicyDynamicsLearn.learn_expectation_outcome(ghost)
    SimulationLearn.learn_simulation_result(ghost)
    MemoryProc.consolidate_memory(ghost)
    MemoryProc.update_narratives(ghost)
    OptimizationProc.improve_prompts(ghost, shell)
    OptimizationProc.learn_lora(ghost, shell)

    # ---- SAVE SNAPSHOT -----------------------------------------------------------------
    BaseProc.save(ghost, shell)


# --------------------------------------------------------------------------------------
# EXAMPLE USAGE (remove in production; keep for quick smoke tests)
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    shell = ShellBase(inbox=[{"kind": "user_message", "content": "how do you feel today?"}])
    ghost = GhostCodelets()
    cognitive_cycle(ghost, shell)

    print("=== OUTBOX ===")
    for o in shell.outbox:
        print(o)

    print("\n=== TRACE (last 10) ===")
    # Dump a few traces in deterministic order for debuggability
    traces = [v for k, v in ghost.store.items() if str(k).startswith("trace.")]
    traces = sorted(traces, key=lambda t: (t["tick"], t["t"]))[-10:]
    for t in traces:
        print(f"[tick {t['tick']:>3}] {t['where']:<35} :: {t['msg']}")
