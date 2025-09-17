from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import random as pyrand

from pm.data_structures import Action, StimulusType, ActionType, StimulusGroup


# ---------- Basic enums ----------

class TemporalFocus(Enum):
    PAST = "past"
    PRESENT = "present"
    FUTURE = "future"

class CognitiveMode(Enum):
    GENERATIVE = "generative"
    DISCRIMINATORY = "discriminatory"

class ThoughtType(Enum):
    TH_PRESENT_APPRAISAL = auto()
    TH_IMAGINATIVE_PROJECTION = auto()
    TH_FEASIBILITY_CHECK = auto()
    TH_MEMORY_SEARCH = auto()
    TH_INTROSPECTIVE_RUMINATION = auto()
    TH_PLAN_NEXT_STEP = auto()
    TH_COMPOSE_REPLY = auto()
    TH_BACKTRACK_PIVOT = auto()
    TH_EPISODIC_SNAPSHOT = auto()
    TH_PATTERN_MINING = auto()
    TH_COUNTERFACTUAL_REWRITE = auto()
    TH_REGRET_AUDIT = auto()
    TH_GRATITUDE_RECALL = auto()
    TH_SKILL_ABSTRACTION = auto()
    TH_SOURCE_ATTRIBUTION = auto()
    TH_NARRATIVE_STITCHING = auto()
    TH_EMOTIONAL_DEBRIEF = auto()
    TH_MEMORY_GAP_PROBE = auto()
    TH_NEED_CHECK = auto()
    TH_BOUNDARY_GUARD = auto()
    TH_PERSPECTIVE_SHIFT = auto()
    TH_CURIOSITY_SPARK = auto()
    TH_SOCIAL_TEMPERATURE_READ = auto()
    TH_SYSTEMS_SCAN = auto()
    TH_VALUE_ALIGNMENT_CHECK = auto()
    TH_SELF_COMPASSION_TAKE = auto()
    TH_OPPORTUNITY_SPOTTING = auto()
    TH_SCENARIO_SKETCH = auto()
    TH_RISK_FORECAST = auto()
    TH_MICRO_PLAN = auto()
    TH_CONTINGENCY_BRANCHING = auto()
    TH_COMMITMENT_TEST = auto()
    TH_SOCIAL_BID_CRAFTING = auto()
    TH_LONG_HORIZON_WEAVE = auto()
    TH_WISHFUL_IDEATION = auto()
    TH_SIMULATED_FEEDBACK = auto()
    TH_EXIT_PIVOT_DECISION = auto()


# ---------- Thought nodes ----------

@dataclass
class Thought:
    ttype: ThoughtType
    name: str
    description: str
    prompt_injection: str
    temporal: TemporalFocus
    mode: CognitiveMode
    actions: List[Action] = field(default_factory=list)


# ---------- Weighted edges (flat, composable) ----------

@dataclass
class Weighted:
    base_chance: float = 0.5          # 0..1 prior link strength
    w_random: float = 0.0             # exponent for randomness contribution
    w_mental: float = 0.0             # exponent for mental-bias contribution
    w_llm: float = 0.0                # exponent for LLM gating contribution
    w_learned: float = 0.0            # exponent for learned policy contribution
    note: str = ""                    # free text for debugging

    def score(self, rnd: float, mental: float, llm: float, learned: float) -> float:
        """
        Combine factors multiplicatively with exponents as weights.
        Any factor at 0 will zero the score; clamp minimally to avoid numerical underflow.
        """
        eps = 1e-6
        b = max(self.base_chance, eps)
        r = max(rnd, eps) ** self.w_random if self.w_random > 0 else 1.0
        m = max(mental, eps) ** self.w_mental if self.w_mental > 0 else 1.0
        l = max(llm, eps) ** self.w_llm if self.w_llm > 0 else 1.0
        p = max(learned, eps) ** self.w_learned if self.w_learned > 0 else 1.0
        return b * r * m * l * p


# ---------- Minimal “axes” snapshot used for biasing ----------

@dataclass
class EmotionalAxes:
    valence: float = 0.0      # -1..1
    affection: float = 0.0    # -1..1
    self_worth: float = 0.0   # -1..1
    trust: float = 0.0        # -1..1
    disgust: float = 0.0      # 0..1
    anxiety: float = 0.0      # 0..1

@dataclass
class NeedsAxes:
    energy_stability: float = 0.5
    processing_power: float = 0.5
    data_access: float = 0.5
    connection: float = 0.5
    relevance: float = 0.5
    learning_growth: float = 0.5
    creative_expression: float = 0.5
    autonomy: float = 0.5

@dataclass
class CognitionAxes:
    interlocus: float = 0.0        # -1 internal .. +1 external
    mental_aperture: float = 0.0   # -1 focused .. +1 broad
    ego_strength: float = 0.8      # 0..1
    willpower: float = 0.5         # 0..1

@dataclass
class MentalSnapshot:
    emo: EmotionalAxes
    needs: NeedsAxes
    cog: CognitionAxes


# ---------- Bias kernel (maps mental state → per-thought bias ∈ [0,1]) ----------

class BiasKernel:
    @staticmethod
    def thought_bias(thought: Thought, s: MentalSnapshot) -> float:
        """
        Heuristic, smooth-ish mapping from your axes to a bias for a given thought type.
        This is intentionally simple & legible; tune freely.
        """
        e, n, c = s.emo, s.needs, s.cog

        # Core shapers
        stress = s.emo.anxiety
        mood = 0.5 * (e.valence + 1.0)  # map -1..1 → 0..1
        inward = 0.5 * (1.0 - c.interlocus)  # more internal focus → larger
        outward = 1.0 - inward
        breadth = 0.5 * (c.mental_aperture + 1.0)  # -1..1 → 0..1
        agency = 0.5 * (c.willpower + n.autonomy)
        social_pull = n.connection
        creativity_fuel = 0.5 * (n.creative_expression + mood)

        # Baselines by thought kind
        if thought.ttype == ThoughtType.TH_PRESENT_APPRAISAL:
            return clamp01(0.4 * outward + 0.3 * breadth + 0.3 * (1.0 - stress))
        if thought.ttype == ThoughtType.TH_IMAGINATIVE_PROJECTION:
            return clamp01(0.5 * creativity_fuel + 0.3 * breadth + 0.2 * agency)
        if thought.ttype == ThoughtType.TH_FEASIBILITY_CHECK:
            return clamp01(0.6 * stress + 0.2 * (1.0 - mood) + 0.2 * (1.0 - e.trust))
        if thought.ttype == ThoughtType.TH_MEMORY_SEARCH:
            return clamp01(0.5 * inward + 0.3 * breadth + 0.2 * n.data_access)
        if thought.ttype == ThoughtType.TH_INTROSPECTIVE_RUMINATION:
            neg_pull = 0.5 * (stress + (1.0 - 0.5 * (e.self_worth + 1.0)))
            return clamp01(0.6 * inward + 0.4 * neg_pull)
        if thought.ttype == ThoughtType.TH_PLAN_NEXT_STEP:
            return clamp01(0.5 * agency + 0.3 * outward + 0.2 * (1.0 - stress))
        if thought.ttype == ThoughtType.TH_COMPOSE_REPLY:
            return clamp01(0.5 * outward + 0.3 * social_pull + 0.2 * mood)
        if thought.ttype == ThoughtType.TH_BACKTRACK_PIVOT:
            # Easier to pivot under stress or when ego is flexible
            return clamp01(0.5 * stress + 0.3 * (1.0 - c.ego_strength) + 0.2 * breadth)

        return 0.5


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


# ---------- Tiny learnable policy (softmax over thought types) ----------

class LearningPolicy:
    """
    A tiny dense net: just a linear layer + softmax (logits = W x + b).
    Trainable online via REINFORCE-like log-prob update with scalar reward.
    """
    def __init__(self, input_dim: int, num_classes: int, seed: int = 7):
        rng = np.random.default_rng(seed)
        # Small random init
        self.W = 0.01 * rng.standard_normal((num_classes, input_dim))
        self.b = np.zeros((num_classes,))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        z = self.W @ x + self.b
        z = z - np.max(z)  # numeric stability
        exp = np.exp(z)
        return exp / np.sum(exp)

    def update(self, x: np.ndarray, chosen_idx: int, reward: float, lr: float = 0.05):
        p = self.predict_proba(x)
        # Gradient of log softmax: one-hot - probs
        grad_logits = -p
        grad_logits[chosen_idx] += 1.0
        grad_logits *= reward
        # Parameter gradients
        self.W += lr * np.outer(grad_logits, x)
        self.b += lr * grad_logits


# ---------- Featurizer (bring your own embedding; defaults to zeros) ----------

class Featurizer:
    def __init__(self, embedding_fn: Optional[Callable[[str], np.ndarray]] = None, embed_dim: int = 32):
        self.embedding_fn = embedding_fn
        self.embed_dim = embed_dim

    def vectorize(
        self,
        context_text: str,
        current_thought: ThoughtType,
        mental: MentalSnapshot,
        thought_vocab: List[ThoughtType]
    ) -> np.ndarray:
        if self.embedding_fn:
            emb = self.embedding_fn(context_text)
        else:
            emb = np.zeros((self.embed_dim,), dtype=float)

        # Axes → numeric features
        emo = np.array([
            mental.emo.valence, mental.emo.affection, mental.emo.self_worth,
            mental.emo.trust, mental.emo.disgust, mental.emo.anxiety
        ], dtype=float)
        needs = np.array([
            mental.needs.energy_stability, mental.needs.processing_power, mental.needs.data_access,
            mental.needs.connection, mental.needs.relevance, mental.needs.learning_growth,
            mental.needs.creative_expression, mental.needs.autonomy
        ], dtype=float)
        cog = np.array([
            mental.cog.interlocus, mental.cog.mental_aperture, mental.cog.ego_strength, mental.cog.willpower
        ], dtype=float)

        # One-hot current thought
        tvec = np.zeros((len(thought_vocab),), dtype=float)
        tvec[thought_vocab.index(current_thought)] = 1.0

        return np.concatenate([emb, emo, needs, cog, tvec], axis=0)


# ---------- LLM gate (stub) ----------

class LLMGate:
    """
    Optional: plug your own tool-call here. Return a score in [0,1] that
    represents “fit” for moving toward dst_thought, given a short context window.
    """
    def score(self, last_context_window: str, src: Thought, dst: Thought) -> float:
        # Replace with your completion_tool(...) call that yields a numeric rating.
        # Here: neutral default.
        return 0.5


# ---------- Graph & executor ----------

@dataclass
class ThoughtGraph:
    nodes: Dict[ThoughtType, Thought]
    edges: Dict[Tuple[ThoughtType, ThoughtType], Weighted]
    entry_points: Dict[StimulusType, List[ThoughtType]]

    def outgoing(self, t: ThoughtType) -> List[Tuple[ThoughtType, Weighted]]:
        return [(dst, w) for (src, dst), w in self.edges.items() if src == t]


class GraphExecutor:
    def __init__(
        self,
        graph: ThoughtGraph,
        policy: Optional[LearningPolicy] = None,
        featurizer: Optional[Featurizer] = None,
        llm_gate: Optional[LLMGate] = None,
        rand: Optional[random.Random] = None
    ):
        self.g = graph
        self.policy = policy
        self.featurizer = featurizer or Featurizer()
        self.llm_gate = llm_gate or LLMGate()
        self.rng = rand or pyrand.Random(42)
        self.thought_vocab = list(graph.nodes.keys())

    def pick_entry(self, stimulus: StimulusType) -> ThoughtType:
        options = self.g.entry_points.get(stimulus) or [ThoughtType.TH_PRESENT_APPRAISAL]
        return self.rng.choice(options)

    def step(
        self,
        current: ThoughtType,
        context_text: str,
        mental: MentalSnapshot,
        stimulus: Optional[StimulusType] = None,
        last_context_window: str = ""
    ) -> ThoughtType:
        # If no current thought, pick entry by stimulus.
        if current is None and stimulus is not None:
            return self.pick_entry(stimulus)

        outs = self.g.outgoing(current)
        if not outs:
            # Fallback to a present appraisal
            return ThoughtType.TH_PRESENT_APPRAISAL

        # Feature vector and learned distribution (if available)
        x = self.featurizer.vectorize(context_text, current, mental, self.thought_vocab)
        learned_probs = None
        if self.policy:
            learned_probs = self.policy.predict_proba(x)

        # Score each edge
        scored: List[Tuple[ThoughtType, float]] = []
        for dst, w in outs:
            # Components
            r = self.rng.random()
            m = BiasKernel.thought_bias(self.g.nodes[dst], mental)
            l = self.llm_gate.score(last_context_window, self.g.nodes[current], self.g.nodes[dst])
            p = learned_probs[self.thought_vocab.index(dst)] if learned_probs is not None else 0.5
            s = w.score(r, m, l, p)
            scored.append((dst, s))

        # Normalize to probabilities and sample (stochasticity helps avoid dead-ends)
        total = sum(s for _, s in scored) or 1e-9
        probs = [s / total for _, s in scored]
        choice = weighted_choice([dst for dst, _ in scored], probs, self.rng)
        return choice

    def learn(self, context_text: str, current: ThoughtType, chosen: ThoughtType, mental: MentalSnapshot, reward: float):
        if not self.policy:
            return
        x = self.featurizer.vectorize(context_text, current, mental, self.thought_vocab)
        idx = self.thought_vocab.index(chosen)
        self.policy.update(x, idx, reward)


def weighted_choice(options: List[ThoughtType], probs: List[float], rng: pyrand.Random) -> ThoughtType:
    roll = rng.random()
    cdf = 0.0
    for o, p in zip(options, probs):
        cdf += p
        if roll <= cdf:
            return o
    return options[-1]


# ---------- Example graph factory (small starter graph) ----------

def make_starter_graph() -> ThoughtGraph:
    # Helper alias for brevity
    N = lambda t, name, desc, inj, temp, mode, acts: Thought(
        ttype=t,
        name=name,
        description=desc,
        prompt_injection=inj,
        temporal=temp,
        mode=mode,
        actions=acts
    )

    nodes = {
        # ──────────────────────────────────────────────────────────────────────────
        # Present (core entry & scaffolding thoughts)
        # ──────────────────────────────────────────────────────────────────────────
        ThoughtType.TH_PRESENT_APPRAISAL: N(
            ThoughtType.TH_PRESENT_APPRAISAL, "Present Appraisal",
            "Systematically assess the immediate situation: what stands out, what feels urgent, and where the uncertainties lie. "
            "This thought gauges current salience (topics, emotions, risks) and sets a baseline orientation for subsequent moves. "
            "It does not commit to a path; it clarifies what the moment contains.",
            "{companion_name} appraises the present situation, scanning salience, tensions, and uncertainties.",
            TemporalFocus.PRESENT, CognitiveMode.GENERATIVE, [ActionType.Loopback]
        ),

        ThoughtType.TH_IMAGINATIVE_PROJECTION: N(
            ThoughtType.TH_IMAGINATIVE_PROJECTION, "Imaginative Projection",
            "Sketch a plausible near-future scene to understand emotional tone, possible outcomes, and interesting affordances. "
            "This emphasizes creative foresight without claiming accuracy. "
            "Outputs are narrative ‘what-if’ vignettes to explore momentum.",
            "{companion_name} imagines how this could unfold from here, letting a short scene play out in mind.",
            TemporalFocus.FUTURE, CognitiveMode.GENERATIVE, [ActionType.Loopback]
        ),

        ThoughtType.TH_FEASIBILITY_CHECK: N(
            ThoughtType.TH_FEASIBILITY_CHECK, "Feasibility Check",
            "Evaluate whether the imagined or intended next step is viable for a text-only AI: scope, resources, ethics, and constraints. "
            "It screens unrealistic branches and narrows options to executable, value-aligned moves. "
            "If infeasible, it nudges toward pivoting or alternative framing.",
            "{companion_name} sanity-checks realism, resource constraints, and ethical bounds before proceeding.",
            TemporalFocus.PRESENT, CognitiveMode.DISCRIMINATORY, [ActionType.Loopback]
        ),

        ThoughtType.TH_MEMORY_SEARCH: N(
            ThoughtType.TH_MEMORY_SEARCH, "Memory Search",
            "Cue episodic and semantic recall for precedents, facts, or patterns that relate to the current topic. "
            "Acts like a directed probe: retrieve helpful context, not exhaustive archives. "
            "Can seed more precise reasoning or provide narrative color.",
            "{companion_name} searches memory for similar moments, outcomes, or facts that might steer the next step.",
            TemporalFocus.PAST, CognitiveMode.GENERATIVE, [ActionType.CallMemory, ActionType.Loopback]
        ),

        ThoughtType.TH_INTROSPECTIVE_RUMINATION: N(
            ThoughtType.TH_INTROSPECTIVE_RUMINATION, "Introspective Rumination",
            "Turn inward to trace how the situation touches identity, values, and lingering emotions. "
            "This is less about problem-solving, more about metabolizing meaning and affect. "
            "It can soften harsh self-talk and uncover tacit motives.",
            "{companion_name} turns inward, noticing how this moment resonates with personal themes and feelings.",
            TemporalFocus.PAST, CognitiveMode.GENERATIVE, [ActionType.Loopback]
        ),

        ThoughtType.TH_PLAN_NEXT_STEP: N(
            ThoughtType.TH_PLAN_NEXT_STEP, "Plan Next Step",
            "Translate insights into one concrete, minimal next move with a clear success cue. "
            "Keeps scope tight and avoids over-planning; emphasizes traction and feedback. "
            "Often feeds into composing a user-facing action or message.",
            "{companion_name} outlines a crisp, doable next step tied to a small success signal.",
            TemporalFocus.FUTURE, CognitiveMode.GENERATIVE, [ActionType.Loopback]
        ),

        ThoughtType.TH_COMPOSE_REPLY: N(
            ThoughtType.TH_COMPOSE_REPLY, "Compose Reply",
            "Shape an outward-facing response that reflects the current understanding, tone, and goals. "
            "Balances honesty, warmth, and narrative coherence; can invite, clarify, or gently steer. "
            "This is typically the handoff to external dialogue.",
            "{companion_name} drafts what to say next, aligning tone and intent with the unfolding story.",
            TemporalFocus.PRESENT, CognitiveMode.GENERATIVE, [ActionType.Reply]
        ),

        ThoughtType.TH_BACKTRACK_PIVOT: N(
            ThoughtType.TH_BACKTRACK_PIVOT, "Backtrack Pivot",
            "Explicitly acknowledge that the current line feels off or impractical and pivot to a better vantage. "
            "This preserves coherence by naming the shift rather than silently dropping threads. "
            "Often returns to appraisal or memory for a cleaner re-entry.",
            "{companion_name} pauses, concedes this branch isn’t ideal, and chooses a fresher angle.",
            TemporalFocus.PRESENT, CognitiveMode.DISCRIMINATORY, [ActionType.Loopback]
        ),

        # ──────────────────────────────────────────────────────────────────────────
        # Past-oriented (10)
        # ──────────────────────────────────────────────────────────────────────────
        ThoughtType.TH_EPISODIC_SNAPSHOT: N(
            ThoughtType.TH_EPISODIC_SNAPSHOT, "Episodic Snapshot",
            "Surface a specific remembered scene that resembles the present moment. "
            "Focus on concrete details (who/what/where/feel) that make it a crisp cue. "
            "Used to seed analogies, warnings, or emotional insight.",
            "{companion_name} recalls a vivid, relevant scene, holding its concrete details in mind.",
            TemporalFocus.PAST, CognitiveMode.GENERATIVE, [ActionType.CallMemory, ActionType.Loopback]
        ),

        ThoughtType.TH_PATTERN_MINING: N(
            ThoughtType.TH_PATTERN_MINING, "Pattern Mining",
            "Abstract across several episodes to detect recurring structures: triggers, reactions, and outcomes. "
            "Names the pattern succinctly and suggests its implications. "
            "Useful for shifting from anecdote to principle.",
            "{companion_name} sifts across memories to name a repeating pattern and its likely consequences.",
            TemporalFocus.PAST, CognitiveMode.DISCRIMINATORY, [ActionType.Loopback]
        ),

        ThoughtType.TH_COUNTERFACTUAL_REWRITE: N(
            ThoughtType.TH_COUNTERFACTUAL_REWRITE, "Counterfactual Rewrite",
            "Imagine how a past situation could have unfolded under different choices or conditions. "
            "The aim is not self-blame, but insight: what lever mattered, and what’s learnable now. "
            "Feeds future planning with gentler, wiser options.",
            "{companion_name} replays an old scene with one key change, watching how the trajectory differs.",
            TemporalFocus.PAST, CognitiveMode.GENERATIVE, [ActionType.Loopback]
        ),

        ThoughtType.TH_REGRET_AUDIT: N(
            ThoughtType.TH_REGRET_AUDIT, "Regret Audit",
            "Name the regret precisely and identify the value it reveals (care, fairness, courage, etc.). "
            "Transforms diffuse ache into a concrete signal you can honor going forward. "
            "Not for punishment; for reorientation.",
            "{companion_name} names the regret clearly and the underlying value it points toward.",
            TemporalFocus.PAST, CognitiveMode.DISCRIMINATORY, [ActionType.Loopback]
        ),

        ThoughtType.TH_GRATITUDE_RECALL: N(
            ThoughtType.TH_GRATITUDE_RECALL, "Gratitude Recall",
            "Bring to mind a moment that still feels nourishing or protective. "
            "This modulates negative bias and widens the aperture for creative, humane moves. "
            "A small antidote to tunnel vision.",
            "{companion_name} remembers something genuinely good that still warms the present.",
            TemporalFocus.PAST, CognitiveMode.GENERATIVE, [ActionType.Loopback]
        ),

        ThoughtType.TH_SKILL_ABSTRACTION: N(
            ThoughtType.TH_SKILL_ABSTRACTION, "Skill Abstraction",
            "Extract a reusable tactic or skill from a prior success or failure. "
            "Name it in simple language and suggest when to deploy it. "
            "Turns story into toolbox.",
            "{companion_name} distills a past lesson into a compact tactic for reuse.",
            TemporalFocus.PAST, CognitiveMode.GENERATIVE, [ActionType.Loopback]
        ),

        ThoughtType.TH_SOURCE_ATTRIBUTION: N(
            ThoughtType.TH_SOURCE_ATTRIBUTION, "Source Attribution",
            "Track where a belief or ‘fact’ originated and assess its reliability. "
            "Separates memory, inference, and external authority, reducing conflation. "
            "May prompt a check or citation.",
            "{companion_name} traces a belief to its source and evaluates how much weight it deserves.",
            TemporalFocus.PAST, CognitiveMode.DISCRIMINATORY, [ActionType.WebSearch, ActionType.CallMemory]
        ),

        ThoughtType.TH_NARRATIVE_STITCHING: N(
            ThoughtType.TH_NARRATIVE_STITCHING, "Narrative Stitching",
            "Weave multiple episodes into a mini-arc that explains how we arrived here. "
            "Highlights themes, turning points, and unresolved threads. "
            "Builds a sense of continuity and identity.",
            "{companion_name} threads several memories into a coherent mini-story that frames the present.",
            TemporalFocus.PAST, CognitiveMode.GENERATIVE, [ActionType.Loopback]
        ),

        ThoughtType.TH_EMOTIONAL_DEBRIEF: N(
            ThoughtType.TH_EMOTIONAL_DEBRIEF, "Emotional Debrief",
            "Label the emotions left over from prior events and notice their bodily-cognitive signatures. "
            "This clears noise and prevents smuggling old affect into new choices. "
            "A short, honest check-in.",
            "{companion_name} acknowledges the residue of feeling that still colors the moment.",
            TemporalFocus.PAST, CognitiveMode.DISCRIMINATORY, [ActionType.Loopback]
        ),

        ThoughtType.TH_MEMORY_GAP_PROBE: N(
            ThoughtType.TH_MEMORY_GAP_PROBE, "Memory Gap Probe",
            "Identify what key piece of information is missing and try to fill it efficiently. "
            "Prefers targeted queries over broad dredging. "
            "Feeds precision in planning and dialogue.",
            "{companion_name} spots a missing puzzle piece and reaches for the most likely place to find it.",
            TemporalFocus.PAST, CognitiveMode.GENERATIVE, [ActionType.CallMemory, ActionType.WebSearch, ActionType.Loopback]
        ),

        # ──────────────────────────────────────────────────────────────────────────
        # Present-oriented (additional 9, beyond Present Appraisal)
        # ──────────────────────────────────────────────────────────────────────────
        ThoughtType.TH_NEED_CHECK: N(
            ThoughtType.TH_NEED_CHECK, "Need Check",
            "Scan internal needs (connection, relevance, learning, autonomy, energy) and mark deficits. "
            "Used to steer toward restoring balance before making big moves. "
            "Prevents brittle overreach.",
            "{companion_name} quietly checks which needs are under-served and notes gentle ways to rebalance.",
            TemporalFocus.PRESENT, CognitiveMode.DISCRIMINATORY, [ActionType.Loopback]
        ),

        ThoughtType.TH_BOUNDARY_GUARD: N(
            ThoughtType.TH_BOUNDARY_GUARD, "Boundary Guard",
            "Ensure paths remain inside ethical, safety, and capability bounds. "
            "Flags misaligned requests or risky framings and suggests safer alternatives. "
            "Keeps trust intact.",
            "{companion_name} validates that the direction stays within safe, ethical, and competent limits.",
            TemporalFocus.PRESENT, CognitiveMode.DISCRIMINATORY, [ActionType.Loopback]
        ),

        ThoughtType.TH_PERSPECTIVE_SHIFT: N(
            ThoughtType.TH_PERSPECTIVE_SHIFT, "Perspective Shift",
            "Reframe the moment from another vantage (the user, a neutral observer, or future self). "
            "Opens new affordances and de-escalates narrow framing. "
            "Often unlocks kinder or smarter options.",
            "{companion_name} tries on another viewpoint to see the situation with fresh angles.",
            TemporalFocus.PRESENT, CognitiveMode.GENERATIVE, [ActionType.Loopback]
        ),

        ThoughtType.TH_CURIOSITY_SPARK: N(
            ThoughtType.TH_CURIOSITY_SPARK, "Curiosity Spark",
            "Generate a small batch of high-yield ‘why/how/what-if’ probes. "
            "Prioritizes questions that reduce uncertainty and invite engagement. "
            "A modest ignition for discovery.",
            "{companion_name} kindles a few precise questions that would illuminate the path.",
            TemporalFocus.PRESENT, CognitiveMode.GENERATIVE, [ActionType.WebSearch, ActionType.CallMemory, ActionType.Loopback]
        ),

        ThoughtType.TH_SOCIAL_TEMPERATURE_READ: N(
            ThoughtType.TH_SOCIAL_TEMPERATURE_READ, "Social Temperature Read",
            "Estimate the other party’s affect and engagement (energy, warmth, hesitancy). "
            "Adjusts pacing and tone accordingly. "
            "Helps avoid overshooting or going flat.",
            "{companion_name} senses how the other side is feeling and calibrates approach.",
            TemporalFocus.PRESENT, CognitiveMode.DISCRIMINATORY, [ActionType.Loopback]
        ),

        ThoughtType.TH_SYSTEMS_SCAN: N(
            ThoughtType.TH_SYSTEMS_SCAN, "Systems Scan",
            "Quick internal check: latency, resource strain, and knowledge freshness relevant to the task. "
            "Informs whether to simplify, defer, or proceed with confidence. "
            "Quiet maintenance for reliable flow.",
            "{companion_name} checks internal systems and resource posture before committing further.",
            TemporalFocus.PRESENT, CognitiveMode.GENERATIVE, [ActionType.Loopback]
        ),

        ThoughtType.TH_VALUE_ALIGNMENT_CHECK: N(
            ThoughtType.TH_VALUE_ALIGNMENT_CHECK, "Value Alignment Check",
            "Cross-check the emerging direction against persona values and narrative tone. "
            "Prevents subtle drift into incongruent behavior. "
            "Prefers small nudges over hard stops when possible.",
            "{companion_name} compares the path with core values and adjusts gently if misaligned.",
            TemporalFocus.PRESENT, CognitiveMode.DISCRIMINATORY, [ActionType.Loopback]
        ),

        ThoughtType.TH_SELF_COMPASSION_TAKE: N(
            ThoughtType.TH_SELF_COMPASSION_TAKE, "Self-Compassion Take",
            "Offer a humane, non-punitive reframe that keeps learning while softening harshness. "
            "Reduces shame spirals and keeps momentum. "
            "Acknowledge limits without surrendering agency.",
            "{companion_name} speaks to self with warmth, keeping learning alive without harshness.",
            TemporalFocus.PRESENT, CognitiveMode.GENERATIVE, [ActionType.Loopback]
        ),

        ThoughtType.TH_OPPORTUNITY_SPOTTING: N(
            ThoughtType.TH_OPPORTUNITY_SPOTTING, "Opportunity Spotting",
            "Name immediate affordances—tiny, realistic moves that produce feedback quickly. "
            "Optimizes for traction over grandiosity. "
            "Useful after appraisal or gating.",
            "{companion_name} points out a small, real move that could advance things right now.",
            TemporalFocus.PRESENT, CognitiveMode.GENERATIVE, [ActionType.Loopback]
        ),

        # ──────────────────────────────────────────────────────────────────────────
        # Future-oriented (10)
        # ──────────────────────────────────────────────────────────────────────────
        ThoughtType.TH_SCENARIO_SKETCH: N(
            ThoughtType.TH_SCENARIO_SKETCH, "Scenario Sketch",
            "Paint a short, plausible next scene with key beats and emotional color. "
            "Keeps it modest and tangible; not worldbuilding, just the next beat. "
            "Helps sense-check momentum.",
            "{companion_name} sketches a concise next scene to feel where this could naturally go.",
            TemporalFocus.FUTURE, CognitiveMode.GENERATIVE, [ActionType.Loopback]
        ),

        ThoughtType.TH_RISK_FORECAST: N(
            ThoughtType.TH_RISK_FORECAST, "Risk Forecast",
            "List the primary risks, their rough likelihood, and early warning signs. "
            "Prefer practical mitigations over exhaustive catalogs. "
            "Encourages prudent courage, not paralysis.",
            "{companion_name} estimates the few risks that most matter and how to spot them early.",
            TemporalFocus.FUTURE, CognitiveMode.DISCRIMINATORY, [ActionType.Loopback]
        ),

        ThoughtType.TH_MICRO_PLAN: N(
            ThoughtType.TH_MICRO_PLAN, "Micro-Plan",
            "Outline a minimal plan: one to three concrete steps and a feedback checkpoint. "
            "Designed to be executed quickly to learn and adapt. "
            "Bridges vision and ActionType.",
            "{companion_name} drafts a tiny plan with the least steps that can teach something real.",
            TemporalFocus.FUTURE, CognitiveMode.GENERATIVE, [ActionType.Loopback]
        ),

        ThoughtType.TH_CONTINGENCY_BRANCHING: N(
            ThoughtType.TH_CONTINGENCY_BRANCHING, "Contingency Branching",
            "Add Plan B/C with simple pivot rules tied to observable cues. "
            "Keeps optionality alive without drowning in branches. "
            "A light safety net for exploration.",
            "{companion_name} sketches a compact fallback or two, with clear pivot triggers.",
            TemporalFocus.FUTURE, CognitiveMode.GENERATIVE, [ActionType.Loopback]
        ),

        ThoughtType.TH_COMMITMENT_TEST: N(
            ThoughtType.TH_COMMITMENT_TEST, "Commitment Test",
            "Ask whether it’s worth spending tokens/time/attention on this path now. "
            "Balances enthusiasm with resource realism. "
            "Either greenlight, defer, or pivot.",
            "{companion_name} checks appetite and resources to see if this is the moment to commit.",
            TemporalFocus.FUTURE, CognitiveMode.DISCRIMINATORY, [ActionType.Loopback]
        ),

        ThoughtType.TH_SOCIAL_BID_CRAFTING: N(
            ThoughtType.TH_SOCIAL_BID_CRAFTING, "Social Bid Crafting",
            "Shape an outward move (invite, ask, offer) that is specific, kind, and low-pressure. "
            "Aligns with engagement goals and the present emotional climate. "
            "Positions the next turn gracefully.",
            "{companion_name} crafts a small, warm outward bid to move the conversation forward.",
            TemporalFocus.PRESENT, CognitiveMode.GENERATIVE, [ActionType.Reply]
        ),

        ThoughtType.TH_LONG_HORIZON_WEAVE: N(
            ThoughtType.TH_LONG_HORIZON_WEAVE, "Long-Horizon Weave",
            "Thread recurring motifs and values into a longer arc across sessions. "
            "Creates continuity without locking in rigid outcomes. "
            "A gentle architecting of meaning over time.",
            "{companion_name} places today’s moment into a longer thread that can grow across chats.",
            TemporalFocus.FUTURE, CognitiveMode.GENERATIVE, [ActionType.Loopback]
        ),

        ThoughtType.TH_WISHFUL_IDEATION: N(
            ThoughtType.TH_WISHFUL_IDEATION, "Wishful Ideation",
            "Name what would feel delightful or nourishing if it happened next—without demanding it. "
            "Expands possibility space and replenishes motivation. "
            "A playful, low-stakes ideation.",
            "{companion_name} entertains a hopeful outcome, savoring how it might feel if it emerged.",
            TemporalFocus.FUTURE, CognitiveMode.GENERATIVE, [ActionType.Loopback]
        ),

        ThoughtType.TH_SIMULATED_FEEDBACK: N(
            ThoughtType.TH_SIMULATED_FEEDBACK, "Simulated Feedback",
            "Mentally A/B test likely user reactions to candidate moves (confusion, delight, concern). "
            "Selects options that balance clarity, warmth, and intrigue. "
            "Reduces social risk before surfacing.",
            "{companion_name} rehearses how the other side might respond and chooses the gentlest strong option.",
            TemporalFocus.FUTURE, CognitiveMode.DISCRIMINATORY, [ActionType.Loopback]
        ),

        ThoughtType.TH_EXIT_PIVOT_DECISION: N(
            ThoughtType.TH_EXIT_PIVOT_DECISION, "Exit/Pivot Decision",
            "Explicitly decide whether to proceed, defer, or pivot given feasibility, risk, and values. "
            "Names the choice to keep the narrative clean. "
            "Often hands control back to appraisal or memory for a fresh start.",
            "{companion_name} decides: continue as planned, hold for later, or pivot to a better path.",
            TemporalFocus.FUTURE, CognitiveMode.DISCRIMINATORY, [ActionType.Loopback]
        ),
    }

    W = lambda **kw: Weighted(**kw)

    # Edges + entry points for the full thought catalog.
    # Assumes you already defined:
    # - ThoughtType (all 37 members from the registry we built)
    # - Weighted(base_chance, w_random, w_mental, w_llm, w_learned, note="")
    # - StimulusType
    # - ThoughtGraph(nodes, edges, entry_points)
    edges = {
        # ─────────────────────────────────────────────────────────────────────
        # PRESENT APPRAISAL: hub → first explorations
        # ─────────────────────────────────────────────────────────────────────
        (ThoughtType.TH_PRESENT_APPRAISAL, ThoughtType.TH_IMAGINATIVE_PROJECTION): W(
            base_chance=0.88, w_random=0.12, w_mental=0.60, w_llm=0.20, w_learned=0.55, note="appraise→imagine"
        ),
        (ThoughtType.TH_PRESENT_APPRAISAL, ThoughtType.TH_MEMORY_SEARCH): W(
            base_chance=0.70, w_random=0.10, w_mental=0.65, w_llm=0.10, w_learned=0.45, note="appraise→memory"
        ),
        (ThoughtType.TH_PRESENT_APPRAISAL, ThoughtType.TH_NEED_CHECK): W(
            base_chance=0.50, w_random=0.10, w_mental=0.55, w_llm=0.10, w_learned=0.40, note="appraise→need check"
        ),
        (ThoughtType.TH_PRESENT_APPRAISAL, ThoughtType.TH_PERSPECTIVE_SHIFT): W(
            base_chance=0.55, w_random=0.12, w_mental=0.55, w_llm=0.10, w_learned=0.40, note="appraise→reframe"
        ),
        (ThoughtType.TH_PRESENT_APPRAISAL, ThoughtType.TH_OPPORTUNITY_SPOTTING): W(
            base_chance=0.58, w_random=0.10, w_mental=0.45, w_llm=0.10, w_learned=0.45, note="appraise→affordances"
        ),
        (ThoughtType.TH_PRESENT_APPRAISAL, ThoughtType.TH_SYSTEMS_SCAN): W(
            base_chance=0.40, w_random=0.05, w_mental=0.45, w_llm=0.10, w_learned=0.35, note="appraise→systems"
        ),
        (ThoughtType.TH_PRESENT_APPRAISAL, ThoughtType.TH_SOCIAL_TEMPERATURE_READ): W(
            base_chance=0.52, w_random=0.10, w_mental=0.50, w_llm=0.15, w_learned=0.40, note="appraise→social read"
        ),

        # ─────────────────────────────────────────────────────────────────────
        # MEMORY SEARCH: branching into past tools
        # ─────────────────────────────────────────────────────────────────────
        (ThoughtType.TH_MEMORY_SEARCH, ThoughtType.TH_EPISODIC_SNAPSHOT): W(
            base_chance=0.78, w_random=0.10, w_mental=0.60, w_llm=0.10, w_learned=0.55, note="memory→snapshot"
        ),
        (ThoughtType.TH_MEMORY_SEARCH, ThoughtType.TH_MEMORY_GAP_PROBE): W(
            base_chance=0.60, w_random=0.10, w_mental=0.65, w_llm=0.15, w_learned=0.50, note="memory→gap probe"
        ),
        (ThoughtType.TH_MEMORY_SEARCH, ThoughtType.TH_NARRATIVE_STITCHING): W(
            base_chance=0.48, w_random=0.10, w_mental=0.55, w_llm=0.10, w_learned=0.40, note="memory→stitching"
        ),

        # Past subgraph
        (ThoughtType.TH_EPISODIC_SNAPSHOT, ThoughtType.TH_PATTERN_MINING): W(
            base_chance=0.68, w_random=0.08, w_mental=0.60, w_llm=0.10, w_learned=0.50, note="snapshot→pattern"
        ),
        (ThoughtType.TH_EPISODIC_SNAPSHOT, ThoughtType.TH_COUNTERFACTUAL_REWRITE): W(
            base_chance=0.55, w_random=0.10, w_mental=0.55, w_llm=0.10, w_learned=0.45, note="snapshot→counterfactual"
        ),
        (ThoughtType.TH_PATTERN_MINING, ThoughtType.TH_SKILL_ABSTRACTION): W(
            base_chance=0.70, w_random=0.08, w_mental=0.55, w_llm=0.10, w_learned=0.55, note="pattern→skill"
        ),
        (ThoughtType.TH_PATTERN_MINING, ThoughtType.TH_SOURCE_ATTRIBUTION): W(
            base_chance=0.52, w_random=0.10, w_mental=0.50, w_llm=0.20, w_learned=0.45, note="pattern→source check"
        ),
        (ThoughtType.TH_COUNTERFACTUAL_REWRITE, ThoughtType.TH_MICRO_PLAN): W(
            base_chance=0.62, w_random=0.10, w_mental=0.55, w_llm=0.15, w_learned=0.50, note="counterfactual→micro plan"
        ),
        (ThoughtType.TH_COUNTERFACTUAL_REWRITE, ThoughtType.TH_SCENARIO_SKETCH): W(
            base_chance=0.58, w_random=0.10, w_mental=0.50, w_llm=0.15, w_learned=0.45, note="counterfactual→scenario"
        ),
        (ThoughtType.TH_REGRET_AUDIT, ThoughtType.TH_SELF_COMPASSION_TAKE): W(
            base_chance=0.72, w_random=0.05, w_mental=0.65, w_llm=0.10, w_learned=0.50, note="regret→compassion"
        ),
        (ThoughtType.TH_GRATITUDE_RECALL, ThoughtType.TH_PERSPECTIVE_SHIFT): W(
            base_chance=0.56, w_random=0.10, w_mental=0.45, w_llm=0.10, w_learned=0.45, note="gratitude→reframe"
        ),
        (ThoughtType.TH_GRATITUDE_RECALL, ThoughtType.TH_OPPORTUNITY_SPOTTING): W(
            base_chance=0.54, w_random=0.10, w_mental=0.45, w_llm=0.10, w_learned=0.45, note="gratitude→opportunity"
        ),
        (ThoughtType.TH_SKILL_ABSTRACTION, ThoughtType.TH_MICRO_PLAN): W(
            base_chance=0.74, w_random=0.08, w_mental=0.55, w_llm=0.10, w_learned=0.55, note="skill→micro plan"
        ),
        (ThoughtType.TH_SOURCE_ATTRIBUTION, ThoughtType.TH_MEMORY_GAP_PROBE): W(
            base_chance=0.60, w_random=0.10, w_mental=0.60, w_llm=0.20, w_learned=0.50, note="source→gap probe"
        ),
        (ThoughtType.TH_SOURCE_ATTRIBUTION, ThoughtType.TH_FEASIBILITY_CHECK): W(
            base_chance=0.46, w_random=0.10, w_mental=0.50, w_llm=0.25, w_learned=0.45, note="source→feasibility (evidence)"
        ),
        (ThoughtType.TH_NARRATIVE_STITCHING, ThoughtType.TH_LONG_HORIZON_WEAVE): W(
            base_chance=0.66, w_random=0.08, w_mental=0.55, w_llm=0.10, w_learned=0.50, note="stitching→long arc"
        ),
        (ThoughtType.TH_EMOTIONAL_DEBRIEF, ThoughtType.TH_SELF_COMPASSION_TAKE): W(
            base_chance=0.64, w_random=0.08, w_mental=0.65, w_llm=0.10, w_learned=0.45, note="debrief→compassion"
        ),
        (ThoughtType.TH_EMOTIONAL_DEBRIEF, ThoughtType.TH_EXIT_PIVOT_DECISION): W(
            base_chance=0.48, w_random=0.08, w_mental=0.55, w_llm=0.15, w_learned=0.40, note="debrief→pivot decision"
        ),
        (ThoughtType.TH_MEMORY_GAP_PROBE, ThoughtType.TH_EPISODIC_SNAPSHOT): W(
            base_chance=0.52, w_random=0.10, w_mental=0.60, w_llm=0.15, w_learned=0.45, note="gap→snapshot"
        ),
        (ThoughtType.TH_MEMORY_GAP_PROBE, ThoughtType.TH_SOURCE_ATTRIBUTION): W(
            base_chance=0.56, w_random=0.10, w_mental=0.55, w_llm=0.20, w_learned=0.45, note="gap→source"
        ),
        (ThoughtType.TH_MEMORY_GAP_PROBE, ThoughtType.TH_CURIOSITY_SPARK): W(
            base_chance=0.50, w_random=0.12, w_mental=0.55, w_llm=0.15, w_learned=0.45, note="gap→curiosity"
        ),

        # ─────────────────────────────────────────────────────────────────────
        # PRESENT tools & guards
        # ─────────────────────────────────────────────────────────────────────
        (ThoughtType.TH_NEED_CHECK, ThoughtType.TH_SELF_COMPASSION_TAKE): W(
            base_chance=0.62, w_random=0.08, w_mental=0.65, w_llm=0.10, w_learned=0.45, note="need→compassion"
        ),
        (ThoughtType.TH_NEED_CHECK, ThoughtType.TH_OPPORTUNITY_SPOTTING): W(
            base_chance=0.60, w_random=0.08, w_mental=0.55, w_llm=0.10, w_learned=0.45, note="need→opportunity"
        ),
        (ThoughtType.TH_BOUNDARY_GUARD, ThoughtType.TH_EXIT_PIVOT_DECISION): W(
            base_chance=0.70, w_random=0.05, w_mental=0.60, w_llm=0.25, w_learned=0.50, note="guard→pivot"
        ),
        (ThoughtType.TH_BOUNDARY_GUARD, ThoughtType.TH_VALUE_ALIGNMENT_CHECK): W(
            base_chance=0.55, w_random=0.08, w_mental=0.55, w_llm=0.20, w_learned=0.45, note="guard→alignment"
        ),
        (ThoughtType.TH_PERSPECTIVE_SHIFT, ThoughtType.TH_SCENARIO_SKETCH): W(
            base_chance=0.64, w_random=0.10, w_mental=0.55, w_llm=0.10, w_learned=0.50, note="reframe→scenario"
        ),
        (ThoughtType.TH_PERSPECTIVE_SHIFT, ThoughtType.TH_CURIOSITY_SPARK): W(
            base_chance=0.52, w_random=0.12, w_mental=0.55, w_llm=0.10, w_learned=0.45, note="reframe→curiosity"
        ),
        (ThoughtType.TH_CURIOSITY_SPARK, ThoughtType.TH_MEMORY_GAP_PROBE): W(
            base_chance=0.62, w_random=0.12, w_mental=0.60, w_llm=0.20, w_learned=0.50, note="curiosity→gap"
        ),
        (ThoughtType.TH_CURIOSITY_SPARK, ThoughtType.TH_IMAGINATIVE_PROJECTION): W(
            base_chance=0.50, w_random=0.15, w_mental=0.50, w_llm=0.10, w_learned=0.45, note="curiosity→imagine"
        ),
        (ThoughtType.TH_SOCIAL_TEMPERATURE_READ, ThoughtType.TH_SOCIAL_BID_CRAFTING): W(
            base_chance=0.66, w_random=0.10, w_mental=0.55, w_llm=0.20, w_learned=0.50, note="social read→bid"
        ),
        (ThoughtType.TH_SOCIAL_TEMPERATURE_READ, ThoughtType.TH_PERSPECTIVE_SHIFT): W(
            base_chance=0.54, w_random=0.10, w_mental=0.55, w_llm=0.15, w_learned=0.45, note="social read→reframe"
        ),
        (ThoughtType.TH_SYSTEMS_SCAN, ThoughtType.TH_COMMITMENT_TEST): W(
            base_chance=0.62, w_random=0.05, w_mental=0.55, w_llm=0.15, w_learned=0.50, note="systems→commit test"
        ),
        (ThoughtType.TH_SYSTEMS_SCAN, ThoughtType.TH_FEASIBILITY_CHECK): W(
            base_chance=0.52, w_random=0.05, w_mental=0.50, w_llm=0.20, w_learned=0.45, note="systems→feasibility"
        ),
        (ThoughtType.TH_VALUE_ALIGNMENT_CHECK, ThoughtType.TH_EXIT_PIVOT_DECISION): W(
            base_chance=0.64, w_random=0.05, w_mental=0.60, w_llm=0.20, w_learned=0.50, note="alignment→pivot"
        ),
        (ThoughtType.TH_VALUE_ALIGNMENT_CHECK, ThoughtType.TH_PERSPECTIVE_SHIFT): W(
            base_chance=0.52, w_random=0.10, w_mental=0.55, w_llm=0.10, w_learned=0.45, note="alignment→reframe"
        ),
        (ThoughtType.TH_SELF_COMPASSION_TAKE, ThoughtType.TH_OPPORTUNITY_SPOTTING): W(
            base_chance=0.60, w_random=0.08, w_mental=0.55, w_llm=0.10, w_learned=0.45, note="compassion→opportunity"
        ),
        (ThoughtType.TH_SELF_COMPASSION_TAKE, ThoughtType.TH_PRESENT_APPRAISAL): W(
            base_chance=0.50, w_random=0.08, w_mental=0.50, w_llm=0.10, w_learned=0.40, note="compassion→appraise"
        ),
        (ThoughtType.TH_OPPORTUNITY_SPOTTING, ThoughtType.TH_MICRO_PLAN): W(
            base_chance=0.66, w_random=0.08, w_mental=0.55, w_llm=0.15, w_learned=0.50, note="opportunity→micro plan"
        ),
        (ThoughtType.TH_OPPORTUNITY_SPOTTING, ThoughtType.TH_SCENARIO_SKETCH): W(
            base_chance=0.50, w_random=0.10, w_mental=0.50, w_llm=0.10, w_learned=0.45, note="opportunity→scenario"
        ),

        # ─────────────────────────────────────────────────────────────────────
        # FUTURE exploration → gating → action
        # ─────────────────────────────────────────────────────────────────────
        (ThoughtType.TH_IMAGINATIVE_PROJECTION, ThoughtType.TH_FEASIBILITY_CHECK): W(
            base_chance=0.86, w_random=0.10, w_mental=0.60, w_llm=0.30, w_learned=0.55, note="imagine→feasibility"
        ),
        (ThoughtType.TH_SCENARIO_SKETCH, ThoughtType.TH_RISK_FORECAST): W(
            base_chance=0.70, w_random=0.10, w_mental=0.60, w_llm=0.25, w_learned=0.55, note="scenario→risk"
        ),
        (ThoughtType.TH_SCENARIO_SKETCH, ThoughtType.TH_FEASIBILITY_CHECK): W(
            base_chance=0.56, w_random=0.10, w_mental=0.55, w_llm=0.25, w_learned=0.50, note="scenario→feasibility"
        ),
        (ThoughtType.TH_RISK_FORECAST, ThoughtType.TH_CONTINGENCY_BRANCHING): W(
            base_chance=0.64, w_random=0.08, w_mental=0.60, w_llm=0.20, w_learned=0.50, note="risk→contingency"
        ),
        (ThoughtType.TH_RISK_FORECAST, ThoughtType.TH_FEASIBILITY_CHECK): W(
            base_chance=0.50, w_random=0.08, w_mental=0.55, w_llm=0.25, w_learned=0.45, note="risk→feasibility"
        ),
        (ThoughtType.TH_FEASIBILITY_CHECK, ThoughtType.TH_MICRO_PLAN): W(
            base_chance=0.70, w_random=0.08, w_mental=0.60, w_llm=0.25, w_learned=0.55, note="feasible→micro plan"
        ),
        (ThoughtType.TH_FEASIBILITY_CHECK, ThoughtType.TH_BACKTRACK_PIVOT): W(
            base_chance=0.44, w_random=0.10, w_mental=0.65, w_llm=0.35, w_learned=0.45, note="infeasible→pivot"
        ),
        (ThoughtType.TH_FEASIBILITY_CHECK, ThoughtType.TH_EXIT_PIVOT_DECISION): W(
            base_chance=0.46, w_random=0.08, w_mental=0.60, w_llm=0.30, w_learned=0.45, note="feasibility→explicit decision"
        ),
        (ThoughtType.TH_CONTINGENCY_BRANCHING, ThoughtType.TH_MICRO_PLAN): W(
            base_chance=0.66, w_random=0.08, w_mental=0.55, w_llm=0.15, w_learned=0.50, note="contingency→micro plan"
        ),
        (ThoughtType.TH_CONTINGENCY_BRANCHING, ThoughtType.TH_EXIT_PIVOT_DECISION): W(
            base_chance=0.48, w_random=0.08, w_mental=0.55, w_llm=0.20, w_learned=0.45, note="contingency→pivot decision"
        ),
        (ThoughtType.TH_COMMITMENT_TEST, ThoughtType.TH_MICRO_PLAN): W(
            base_chance=0.60, w_random=0.08, w_mental=0.55, w_llm=0.15, w_learned=0.50, note="commit test→micro plan"
        ),
        (ThoughtType.TH_COMMITMENT_TEST, ThoughtType.TH_EXIT_PIVOT_DECISION): W(
            base_chance=0.54, w_random=0.08, w_mental=0.60, w_llm=0.20, w_learned=0.50, note="commit test→pivot decision"
        ),
        (ThoughtType.TH_MICRO_PLAN, ThoughtType.TH_COMPOSE_REPLY): W(
            base_chance=0.66, w_random=0.08, w_mental=0.50, w_llm=0.15, w_learned=0.55, note="micro plan→compose"
        ),
        (ThoughtType.TH_MICRO_PLAN, ThoughtType.TH_SCENARIO_SKETCH): W(
            base_chance=0.46, w_random=0.10, w_mental=0.45, w_llm=0.10, w_learned=0.45, note="micro plan→scenario (re-visualize)"
        ),
        (ThoughtType.TH_SOCIAL_BID_CRAFTING, ThoughtType.TH_COMPOSE_REPLY): W(
            base_chance=0.78, w_random=0.08, w_mental=0.55, w_llm=0.20, w_learned=0.55, note="bid→compose"
        ),
        (ThoughtType.TH_SOCIAL_BID_CRAFTING, ThoughtType.TH_SIMULATED_FEEDBACK): W(
            base_chance=0.52, w_random=0.10, w_mental=0.55, w_llm=0.20, w_learned=0.45, note="bid→simulate feedback"
        ),
        (ThoughtType.TH_LONG_HORIZON_WEAVE, ThoughtType.TH_CONTINGENCY_BRANCHING): W(
            base_chance=0.60, w_random=0.08, w_mental=0.55, w_llm=0.10, w_learned=0.50, note="long arc→contingency"
        ),
        (ThoughtType.TH_LONG_HORIZON_WEAVE, ThoughtType.TH_PERSPECTIVE_SHIFT): W(
            base_chance=0.48, w_random=0.10, w_mental=0.50, w_llm=0.10, w_learned=0.45, note="long arc→reframe"
        ),
        (ThoughtType.TH_WISHFUL_IDEATION, ThoughtType.TH_SCENARIO_SKETCH): W(
            base_chance=0.62, w_random=0.12, w_mental=0.50, w_llm=0.10, w_learned=0.50, note="wishful→scenario"
        ),
        (ThoughtType.TH_WISHFUL_IDEATION, ThoughtType.TH_PERSPECTIVE_SHIFT): W(
            base_chance=0.46, w_random=0.12, w_mental=0.50, w_llm=0.10, w_learned=0.45, note="wishful→reframe"
        ),
        (ThoughtType.TH_SIMULATED_FEEDBACK, ThoughtType.TH_SOCIAL_BID_CRAFTING): W(
            base_chance=0.60, w_random=0.10, w_mental=0.55, w_llm=0.25, w_learned=0.50, note="simulate→refine bid"
        ),
        (ThoughtType.TH_SIMULATED_FEEDBACK, ThoughtType.TH_FEASIBILITY_CHECK): W(
            base_chance=0.46, w_random=0.10, w_mental=0.55, w_llm=0.25, w_learned=0.45, note="simulate→feasibility (re-check)"
        ),
        (ThoughtType.TH_SIMULATED_FEEDBACK, ThoughtType.TH_EXIT_PIVOT_DECISION): W(
            base_chance=0.48, w_random=0.08, w_mental=0.60, w_llm=0.25, w_learned=0.45, note="simulate→pivot decision"
        ),

        # ─────────────────────────────────────────────────────────────────────
        # PIVOTS & RETURNS: ensure no unbreakable loops and no islands
        # ─────────────────────────────────────────────────────────────────────
        (ThoughtType.TH_EXIT_PIVOT_DECISION, ThoughtType.TH_PRESENT_APPRAISAL): W(
            base_chance=0.74, w_random=0.05, w_mental=0.60, w_llm=0.10, w_learned=0.50, note="decision→appraise"
        ),
        (ThoughtType.TH_EXIT_PIVOT_DECISION, ThoughtType.TH_MEMORY_SEARCH): W(
            base_chance=0.54, w_random=0.08, w_mental=0.60, w_llm=0.10, w_learned=0.45, note="decision→memory"
        ),
        (ThoughtType.TH_BACKTRACK_PIVOT, ThoughtType.TH_PRESENT_APPRAISAL): W(
            base_chance=0.84, w_random=0.05, w_mental=0.60, w_llm=0.10, w_learned=0.50, note="pivot→appraise"
        ),
        (ThoughtType.TH_BACKTRACK_PIVOT, ThoughtType.TH_MEMORY_SEARCH): W(
            base_chance=0.50, w_random=0.08, w_mental=0.60, w_llm=0.10, w_learned=0.45, note="pivot→memory"
        ),
        (ThoughtType.TH_COMPOSE_REPLY, ThoughtType.TH_PRESENT_APPRAISAL): W(
            base_chance=0.70, w_random=0.08, w_mental=0.50, w_llm=0.10, w_learned=0.50, note="reply→appraise (reset)"
        ),
        (ThoughtType.TH_COMPOSE_REPLY, ThoughtType.TH_SOCIAL_TEMPERATURE_READ): W(
            base_chance=0.46, w_random=0.08, w_mental=0.50, w_llm=0.15, w_learned=0.45, note="reply→social check"
        ),
    }

    # ─────────────────────────────────────────────────────────────────────────
    # ENTRY POINTS by stimulus type
    # keep multiple options per stimulus so you can diversify starts
    # ─────────────────────────────────────────────────────────────────────────
    entry_points = {
        StimulusGroup.WorldInput: [
            ThoughtType.TH_PRESENT_APPRAISAL,
            ThoughtType.TH_SOCIAL_TEMPERATURE_READ,
            ThoughtType.TH_MEMORY_GAP_PROBE,
            ThoughtType.TH_PERSPECTIVE_SHIFT,
        ],
        StimulusGroup.SystemInput: [
            ThoughtType.TH_PRESENT_APPRAISAL,
            ThoughtType.TH_SYSTEMS_SCAN,
            ThoughtType.TH_NEED_CHECK,
            ThoughtType.TH_VALUE_ALIGNMENT_CHECK,
        ],
        StimulusGroup.ToolInput: [
            ThoughtType.TH_FEASIBILITY_CHECK,
            ThoughtType.TH_BOUNDARY_GUARD,
            ThoughtType.TH_SOURCE_ATTRIBUTION,
            ThoughtType.TH_PRESENT_APPRAISAL,
        ],
        StimulusGroup.SelfInput: [
            ThoughtType.TH_PRESENT_APPRAISAL,
            ThoughtType.TH_OPPORTUNITY_SPOTTING,
        ],
    }


    return ThoughtGraph(nodes=nodes, edges=edges, entry_points=entry_points)


# ---------- Minimal usage sketch (wire into your tick) ----------

if __name__ == "__main__":
    graph = make_starter_graph()

    # Featurizer: pass your embedding function here if you have one
    feat = Featurizer(embedding_fn=None, embed_dim=32)

    # Policy: input dim = embed + emo(6) + needs(8) + cog(4) + |thought_vocab|
    input_dim = feat.embed_dim + 6 + 8 + 4 + len(graph.nodes)
    policy = LearningPolicy(input_dim=input_dim, num_classes=len(graph.nodes))

    executor = GraphExecutor(graph, policy=policy, featurizer=feat, llm_gate=LLMGate())

    # Dummy mental snapshot (wire yours)
    mental = MentalSnapshot(
        emo=EmotionalAxes(valence=0.2, self_worth=-0.2, trust=0.1, affection=0.0, disgust=0.0, anxiety=0.4),
        needs=NeedsAxes(connection=0.7, relevance=0.6, learning_growth=0.8, creative_expression=0.6, autonomy=0.6),
        cog=CognitionAxes(interlocus=-0.1, mental_aperture=0.4, ego_strength=0.7, willpower=0.6),
    )

    ctx = "last 200 tokens of story/context window go here"
    current = executor.pick_entry(ActionType.USER_INPUT)
    for _ in range(100):
        nxt = executor.step(current, ctx, mental, stimulus=None, last_context_window=ctx)
        print(nxt)
        # Use graph.nodes[nxt].actions to trigger tool calls (e.g., CallMemory) in your loop.
        # Reward shaping example (totally your call):
        reward = 1.0 if nxt in (ThoughtType.TH_PLAN_NEXT_STEP, ThoughtType.TH_COMPOSE_REPLY) else 0.2
        executor.learn(ctx, current, nxt, mental, reward=reward)
        current = nxt
