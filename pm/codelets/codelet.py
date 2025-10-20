# pm/codelets/core.py
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Dict, Tuple, Set

from nltk.cluster import cosine_distance
from pydantic import BaseModel, Field
import random

from pm.codelets.codelet_definitions import CodeletFamily, ALL_CODELETS
from pm.data_structures import StimulusType, Feature, MLevel, Stimulus, KnoxelBase, StimulusGroup
from pm.csm.csm import CSMState
from pm.llm.llm_proxy import LlmManagerProxy
from pm.mental_state_vectors import FullMentalState
from pm.mental_states import DecayableMentalState

class CodeletDecision(Enum):
    SKIP = auto()
    RUN = auto()

@dataclass
class CodeletSignature:
    """Static ‘what this codelet is about’ for embedding & selection."""
    name: str
    families: List[CodeletFamily]
    description: str                 # long_function_description for cosine
    preferred_sources: StimulusGroup
    stimulus_whitelist: Set[StimulusType] | None = None  # None = any
    stimulus_blacklist: Set[StimulusType] | None = None  # None = none
    prompt: str = None

class ActivationFactors(BaseModel):
    """Weights used for activation scoring."""
    w_activation: float = 1.0
    w_stimulus: float = 0.35
    w_state: float = 0.20
    w_keywords: float = 0.10
    w_vector_match: float = 0.25
    w_recency_penalty: float = 0.05
    w_random: float = 0.05

class CodeletRuntime(BaseModel):
    """Per-codelet runtime state (cooldowns, counters)."""
    last_fired_tick: int | None = None
    total_runs: int = 0
    cooldown_ticks: int = 2
    activation: float = 0

class CodeletState(BaseModel):
    states: Dict[str, CodeletRuntime] = Field(default_factory=dict)

class CodeletContext(BaseModel):
    """Everything a codelet may use to decide & produce features."""
    tick_id: int
    stimulus: Stimulus | None
    csm_snapshot: CSMState | None = None
    mental_state: FullMentalState | None = None
    prefill_messages: List[Tuple[str, str]]
    context_embedding: List[float] | None = None
    llm: LlmManagerProxy | None = None

class CodeletOutput(BaseModel):
    """Products (non-causal features) and any logs/metrics."""
    sources: List[KnoxelBase] = Field(default_factory=list)
    features: List[Feature] = Field(default_factory=list)
    mental_state_deltas: List[DecayableMentalState] = Field(default_factory=list)
    notes: str = ""
    # Optional: proposed updates to intentions, etc.
    # intentions: List['Intention'] = Field(default_factory=list)

class BaseCodelet:
    """Subclass this to implement behavior. Keep methods pure-ish."""
    signature: CodeletSignature
    runtime: CodeletRuntime
    factors: ActivationFactors
    llm: LlmManagerProxy

    def __init__(self,
                 signature: CodeletSignature,
                 factors: ActivationFactors | None = None,
                 runtime: CodeletRuntime | None = None,
                 llm: LlmManagerProxy | None = None):
        self.result_list: List[CodeletOutput] = []
        self.signature = signature
        self.factors = factors or ActivationFactors()
        self.runtime = runtime or CodeletRuntime()
        self.llm = llm

    # ---- selection ----------------------------------------------------------
    def should_consider(self, ctx: CodeletContext) -> bool:
        st = ctx.stimulus.stimulus_type if ctx.stimulus else None
        if self.signature.stimulus_whitelist and st not in self.signature.stimulus_whitelist:
            return False
        if self.signature.stimulus_blacklist and st in self.signature.stimulus_blacklist:
            return False
        return True

    def _cooldown_ok(self, ctx: CodeletContext) -> bool:
        if self.runtime.last_fired_tick is None: return True
        return (ctx.tick_id - self.runtime.last_fired_tick) >= self.runtime.cooldown_ticks

    def activation_score(self, ctx: CodeletContext) -> float:
        """
        Combine:
          - stimulus match (type match, quick heuristics)
          - state match (emotions/needs/cognition)
          - keyword hints
          - vector cosine between signature.description and ctx.context_embedding
          - recency penalty (if it just ran)
          - small random factor to prevent deadlocks
        """
        # --- stimulus
        s_stim = 0.0
        if ctx.stimulus:
            if (not self.signature.stimulus_blacklist) and \
               (self.signature.stimulus_whitelist is None or ctx.stimulus.stimulus_type in self.signature.stimulus_whitelist):
                s_stim = 1.0

        # --- state (very rough placeholder hooks)
        # todo
        if ctx.mental_state:
            s_state = max(0.0, min(1.0, abs(ctx.mental_state.state_core.arousal - 0.5) * 0.3
                                         + abs(ctx.mental_state.state_cognition.mental_aperture) * 0.2
                                         + ctx.mental_state.state_needs.relevance * 0.5))
        s_state = 0

        # --- keywords heuristic (define your own tag map per codelet, omitted here)
        s_keywords = 0.0  # e.g., search ctx.context_text for codelet-specific hints

        # --- vector cosine
        s_vec = cosine_distance(ctx.context_embedding, ctx.llm.get_embedding(self.signature.description))

        # --- recency penalty
        s_recent = 0.0
        if self.runtime.last_fired_tick is not None:
            dt = ctx.tick_id - self.runtime.last_fired_tick
            s_recent = max(0.0, 1.0 - 1.0 / (1.0 + dt))

        # --- random spice
        s_rand = random.random()

        score = (
            self.factors.w_activation * self.runtime.activation +
            self.factors.w_stimulus * s_stim +
            self.factors.w_state * s_state +
            self.factors.w_keywords * s_keywords +
            self.factors.w_vector_match * s_vec -
            self.factors.w_recency_penalty * s_recent +
            self.factors.w_random * s_rand
        )

        # if cooling down, clamp
        if not self._cooldown_ok(ctx):
            score *= 0.25

        return max(0.0, min(1.0, score))

    # ---- execution ----------------------------------------------------------
    def decide(self, ctx: CodeletContext, threshold: float = 0.55) -> CodeletDecision:
        if not self.should_consider(ctx):
            return CodeletDecision.SKIP
        return CodeletDecision.RUN if self.activation_score(ctx) >= threshold else CodeletDecision.SKIP

    def _push_output(self, out: CodeletOutput):
        self.result_list.append(out)

    def run(self, ctx: CodeletContext) -> None:
        """
        Implement your logic here. Return non-causal Feature objects.
        Keep it side-effect free; the engine will add features to the CSM.
        """
        raise NotImplementedError

    def _meta_log(self, level: MLevel, message: str):
        raise NotImplementedError

    def _build_context(self, ctx: CodeletContext) -> str:
        raise NotImplementedError

    def on_commit(self, ctx: CodeletContext, out: CodeletOutput) -> None:
        """Called after features are inserted into CSM—update runtime state."""
        self.runtime.last_fired_tick = ctx.tick_id
        self.runtime.total_runs += 1


# --- registry ---------------------------------------------------------------

class CodeletRegistry:
    def __init__(self):
        self._items: Dict[str, BaseCodelet] = {}

    def register(self, codelet: BaseCodelet):
        key = codelet.signature.name
        if key in self._items:
            raise ValueError(f"Codelet '{key}' already registered.")
        self._items[key] = codelet

    def all(self) -> List[BaseCodelet]:
        return list(self._items.values())

    def pick_candidates(self, ctx: CodeletContext) -> List[Tuple[BaseCodelet, float]]:
        scored = []
        for c in self._items.values():
            if c.should_consider(ctx):
                scored.append((c, c.activation_score(ctx)))
        # high to low
        return sorted(scored, key=lambda x: x[1], reverse=True)

    @classmethod
    def init_from_simple_codelets(cls, ctx: CodeletContext):
        res = cls()
        for base_codelet in ALL_CODELETS:
            sig = CodeletSignature(name=base_codelet.name, families=base_codelet.codelet_families, description=base_codelet.description, preferred_sources=StimulusGroup.All, prompt=base_codelet.prompt)
            cdl = BaseCodelet(sig, llm=ctx.llm)
            res.register(cdl)
        return res

    def get_codelet_state(self) -> CodeletState:
        res = CodeletState()
        for k, v in self._items.items():
            res.states[k] = v.runtime.copy()
        return res

    def apply_codelet_state(self, cs: CodeletState):
        for k, v in self._items.items():
            if k in cs.states.keys():
                v.runtime = cs.states[k].copy()
