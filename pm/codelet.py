# pm/codelets/core.py
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Any, Optional, Callable, Tuple, Set
from pydantic import BaseModel, Field
import time
import math
import random

from pm.data_structures import StimulusType, Feature, FeatureType, MLevel, Stimulus, KnoxelBase
from pm.ghosts.base_ghost import GhostState, GhostConfig
from pm.llm.llm_proxy import LlmManagerProxy
from pm.mental_states import DecayableMentalState


# You already have these in your project; import your real ones:
# from pm.data_structures import Stimulus, StimulusType, Feature, FeatureType
# from pm.ghosts.base_ghost import GhostState, GhostConfig
# from pm.utils.emb_utils import cosine_pair

# --- enums & small types ------------------------------------------------------

class CodeletSource(Enum):
    USER_INPUT = auto()
    INTERNAL_THOUGHT = auto()
    WORLD_EVENT = auto()
    MEMORY_DRIVEN = auto()
    HOMEOSTATIC = auto()

class CodeletFamily(Enum):
    APPRAISAL = auto()
    RETRIEVAL = auto()
    REFRAMING = auto()
    PLANNING = auto()
    SIMULATION = auto()
    SOCIAL = auto()
    META = auto()
    NARRATIVE = auto()
    ACTION_SELECTION = auto()

class CodeletDecision(Enum):
    SKIP = auto()
    RUN = auto()

@dataclass
class CodeletSignature:
    """Static ‘what this codelet is about’ for embedding & selection."""
    name: str
    family: CodeletFamily
    description: str                 # long_function_description for cosine
    preferred_sources: Set[CodeletSource]
    stimulus_whitelist: Set[StimulusType] | None = None  # None = any
    stimulus_blacklist: Set[StimulusType] | None = None  # None = none
    context_builder_prompt: str = None

class ActivationFactors(BaseModel):
    """Weights used for activation scoring."""
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
    cooldown_ticks: int = 2   # default; codelet can override

class CodeletContext(BaseModel):
    """Everything a codelet may use to decide & produce features."""
    tick_id: int
    stimulus: Stimulus | None
    state: GhostState
    config: GhostConfig
    csm_snapshot: List['CSMItem']   # defined below
    context_text: str               # recent causal story
    context_embedding: List[float] | None = None
    gw_last_text: str | None = None # previous global workspace text (if any)

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
        s_state = max(0.0, min(1.0, abs(ctx.state.state_emotions.anxiety - 0.5)*0.3
                                     + abs(ctx.state.state_cognition.mental_aperture)*0.2
                                     + ctx.state.state_needs.relevance*0.5))

        # --- keywords heuristic (define your own tag map per codelet, omitted here)
        s_keywords = 0.0  # e.g., search ctx.context_text for codelet-specific hints

        # --- vector cosine
        s_vec = 0.0
        try:
            # emb_signature = emb(self.signature.description)
            # s_vec = cosine_pair(emb_signature, ctx.context_embedding)
            # Use 0.5 baseline if embeddings are absent
            s_vec = 0.5 if ctx.context_embedding is None else 0.7
        except Exception:
            s_vec = 0.3

        # --- recency penalty
        s_recent = 0.0
        if self.runtime.last_fired_tick is not None:
            dt = ctx.tick_id - self.runtime.last_fired_tick
            s_recent = max(0.0, 1.0 - 1.0 / (1.0 + dt))

        # --- random spice
        s_rand = random.random()

        score = (
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

    def _run(self, ctx: CodeletContext) -> None:
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


# registry wiring
REGISTRY = CodeletRegistry()
REGISTRY.register(MemoryGapProbe())
