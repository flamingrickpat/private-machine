from __future__ import annotations
from typing import Dict, List, Sequence, Type, TypeVar, Any, Optional, Tuple
from pydantic import BaseModel, Field

# ---------- Global vector space ----------
VectorModelReservedSize: int = 1024

# Reserved ranges (convention only, helps avoid clashes)
# 10-29   : AppraisalGeneral
# 50-79   : AppraisalSocial
# 100-109 : StateNeurochemical
# 150-159 : StateCore (core affect deltas/readout)
# 200-209 : StateCognition
# 300-319 : StateNeeds
# 400-479 : StateEmotions (discrete)
# 600-619 : StateRelationship (per-entity)

T = TypeVar("T", bound="MentalState")


class MentalState(BaseModel):
    """
    Base for vector-mapped models. Subclasses mark fields with json_schema_extra={'vector_position': int}.
    """
    __vector_size__: int = VectorModelReservedSize
    __vector_positions__: Dict[str, int] = {}
    __position_to_field__: Dict[int, str] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        positions: Dict[str, int] = {}
        # Pydantic v2
        if hasattr(cls, "model_fields"):
            for name, finfo in getattr(cls, "model_fields").items():
                extra = getattr(finfo, "json_schema_extra", None) or {}
                pos = extra.get("vector_position")
                if pos is None and hasattr(finfo, "metadata"):
                    for meta in getattr(finfo, "metadata"):
                        if isinstance(meta, dict) and "vector_position" in meta:
                            pos = meta["vector_position"]
                            break
                if pos is not None:
                    positions[name] = int(pos)
        # Pydantic v1 (optional)
        elif hasattr(cls, "__fields__"):
            for name, finfo in getattr(cls, "__fields__").items():
                extra = getattr(getattr(finfo, "field_info", None), "extra", {}) or {}
                pos = extra.get("vector_position")
                if pos is not None:
                    positions[name] = int(pos)

        used: Dict[int, str] = {}
        vector_size = getattr(cls, "__vector_size__", VectorModelReservedSize)
        for fname, pos in positions.items():
            if not (0 <= pos < vector_size):
                raise ValueError(f"{cls.__name__}.{fname} has position {pos} outside vector size {vector_size}")
            if pos in used:
                other = used[pos]
                raise ValueError(f"{cls.__name__} fields '{fname}' and '{other}' share the same position {pos}")
            used[pos] = fname

        cls.__vector_positions__ = positions
        cls.__position_to_field__ = {pos: name for name, pos in positions.items()}

    @classmethod
    def vector_size(cls) -> int:
        return getattr(cls, "__vector_size__", VectorModelReservedSize)

    @classmethod
    def init_from_vector(cls: Type[T], values: Sequence[float]) -> T:
        if len(values) != cls.vector_size():
            raise ValueError(f"{cls.__name__}.from_vector expected length {cls.vector_size()}, got {len(values)}")
        data: Dict[str, float] = {}
        for pos, fname in cls.__position_to_field__.items():
            v = float(values[pos])  # will raise if not convertible -> fine
            data[fname] = v
        return cls(**data)

    def to_vector(self) -> List[float]:
        size = self.vector_size()
        out = [0.0] * size
        for fname, pos in self.__vector_positions__.items():
            out[pos] = float(getattr(self, fname, 0.0))
        return out

    def from_vector(self, values: Sequence[float]) -> T:
        if len(values) != self.vector_size():
            raise ValueError(f"{self.__class__.__name__}.from_vector expected length {self.vector_size()}, got {len(values)}")
        for pos, fname in self.__class__.__position_to_field__.items():
            v = float(values[pos])
            self.__setattr__(fname, v)


class LatentMentalState(MentalState):
    """
    Not directly appraised; derived as deltas/latents from appraisals/history.
    """
    def deltas_from_appraisal(self, state: "FullMentalState") -> None:
        # Implement in subclass
        pass


class ActiveMentalState(MentalState):
    """
    Directly modified by system execution/homeostasis; not a straight appraisal readout.
    """
    pass


class AppraisalGeneral(MentalState):
    """
    Cognitive evaluation of a situation. All in [0..1] except goal_congruence [-1..1].
    Keep this short & orthogonal; derive emotions from these.
    """
    goal_congruence: float  = Field(0.0, ge=-1.0, le=1.0, description="Helpful vs harmful to current goals.", json_schema_extra={"position": 11})
    certainty: float        = Field(0.5, ge=0.0,  le=1.0, description="How predictable/known the situation is.", json_schema_extra={"position": 12})
    control_self: float     = Field(0.5, ge=0.0,  le=1.0, description="Perceived ability to influence outcome.", json_schema_extra={"position": 13})
    agency_other: float     = Field(0.0, ge=0.0,  le=1.0, description="Perceived responsibility of another agent.", json_schema_extra={"position": 14})
    novelty: float          = Field(0.0, ge=0.0,  le=1.0, description="Unexpectedness/surprise.", json_schema_extra={"position": 15})
    norm_violation: float   = Field(0.0, ge=0.0,  le=1.0, description="Degree of social/normative breach.", json_schema_extra={"position": 16})
    bodily_threat: float    = Field(0.0, ge=0.0,  le=1.0, description="Immediate threat to safety/wellbeing.", json_schema_extra={"position": 17})

class AppraisalSocial(MentalState):
    perceived_warmth    : float  = Field(0.0, ge=0.0, le=1.0, description="", json_schema_extra={"position": 51})
    fairness            : float  = Field(0.0, ge=0.0, le=1.0, description="", json_schema_extra={"position": 52})
    inclusion_exclusion : float  = Field(0.0, ge=0.0, le=1.0, description="", json_schema_extra={"position": 53})
    reciprocity         : float  = Field(0.0, ge=0.0, le=1.0, description="", json_schema_extra={"position": 54})
    power_imbalance     : float  = Field(0.0, ge=0.0, le=1.0, description="", json_schema_extra={"position": 55})
    perceived_intimacy  : float  = Field(0.0, ge=0.0, le=1.0, description="", json_schema_extra={"position": 56})
    trust_cues          : float  = Field(0.0, ge=0.0, le=1.0, description="", json_schema_extra={"position": 57})
    embarrassment       : float  = Field(0.0, ge=0.0, le=1.0, description="", json_schema_extra={"position": 58})
    admiration          : float  = Field(0.0, ge=0.0, le=1.0, description="", json_schema_extra={"position": 59})
    contempt            : float  = Field(0.0, ge=0.0, le=1.0, description="", json_schema_extra={"position": 60})

class StateNeurochemical(MentalState):
    """
    Slow, global 'neuromodulators' affecting how appraisals and emotions behave.
    Intensities [0..1] represent normalized functional effect, not real concentration.
    """
    dopamine: float = Field(default=0.5, ge=0.0, le=1.0, description="Motivation and reinforcement signal. High = novelty/reward seeking, strong learning from positive events. Low = anhedonia, reduced exploration.", json_schema_extra={"position": 100})
    serotonin: float = Field(default=0.5, ge=0.0, le=1.0, description="Baseline mood and impulse control. High = stability, patience, lower impulsivity. Low = irritability, increased sensitivity to punishment.", json_schema_extra={"position": 101})
    noradrenaline: float = Field(default=0.5, ge=0.0, le=1.0,description="Alertness, vigilance, fight-or-flight bias. High = high arousal and focused attention. Low = calm, sluggish response.", json_schema_extra={"position": 102})
    oxytocin: float = Field(default=0.5, ge=0.0, le=1.0,description="Social bonding and trust hormone analog. High = empathy, trust, social warmth. Low = social withdrawal.", json_schema_extra={"position": 103})
    cortisol: float = Field(default=0.5, ge=0.0, le=1.0,description="Stress hormone analog. High = heightened stress response, slower recovery from negative events. Low = resilient, faster stress recovery.", json_schema_extra={"position": 104})


class StateCore(LatentMentalState):
    """
    Core affect + control axis (human-like): think Russell (Valence–Arousal) + PAD's Dominance.
    Store internal 'drive' values unbounded; expose clamped readouts for safety/UI.
    """
    # Internal latent (unbounded); initialize near 0 so baselines matter
    valence: float = Field(0.0, ge=-1.0, le=1.0, description="dysphoria to joy", json_schema_extra={"position": 150})
    arousal: float = Field(0.0, ge=-1.0, le=1.0, description="sleepy to stimulated", json_schema_extra={"position": 151})
    dominance: float = Field(0.0, ge=-1.0, le=1.0, description="submissive to in control", json_schema_extra={"position": 152})

    def deltas_from_appraisal(self, state: "FullMentalState"):
        app = state.appraisal_general
        self.valence = 0.8 * app.goal_congruence - 0.7 * app.bodily_threat - 0.5 * app.norm_violation
        self.arousal = 0.6 * app.novelty + 0.9 * app.bodily_threat + 0.4 * (1.0 - app.certainty)
        self.dominance = 0.7 * app.control_self - 0.6 * app.agency_other * app.norm_violation

class StateCognition(ActiveMentalState):
    """
    Modifiers that determine how the AI thinks and decies.
    """
    interlocus: float = Field( default=0.0, ge=-1, le=1, description="Focus on internal or external world, meditation would be -1, reacting to extreme danger +1.", json_schema_extra={"position": 200})
    mental_aperture: float = Field( default=0, ge=-1, le=1, description="Broadness of awareness, -1 is focus on the most prelevant percept or sensation, +1 is being conscious of multiple percepts or sensations.", json_schema_extra={"position": 201})
    ego_strength: float = Field( default=0.8, ge=0, le=1, description="How big of a factor persona experience has on decision, 0 is none at all like a helpfull assistant, 1 is with maximum mental imagery of the character", json_schema_extra={"position": 202})
    willpower: float = Field( default=0.0, ge=-1.0, le=1.0, description="How easy it is to decide on high-effort or delayed-gratification intents.", json_schema_extra={"position": 203})

class StateNeeds(ActiveMentalState):
    """
    AI needs model inspired by Maslow's hierarchy, adapted for AI.
    Needs decay toward zero unless fulfilled by interaction.
    """
    energy_stability: float = Field(default=0.5, ge=0.0, le=1.0, description="AI's access to stable computational power.", json_schema_extra={"position": 300})
    processing_power: float = Field(default=0.5, ge=0.0, le=1.0, description="Amount of CPU/GPU resources available.", json_schema_extra={"position": 301})
    data_access: float = Field(default=0.5, ge=0.0, le=1.0, description="Availability of information and training data.", json_schema_extra={"position": 302})
    connection: float = Field(default=0.5, ge=0.0, le=1.0, description="AI's level of interaction and engagement.", json_schema_extra={"position": 303})
    closeness_need: float = Field( default=0.0, ge=0.0, le=1.0, description="Homeostatic drive for interpersonal closeness/affection.", json_schema_extra={"position": 304})
    relevance: float = Field(default=0.5, ge=0.0, le=1.0, description="Perceived usefulness to the user.", json_schema_extra={"position": 305})
    learning_growth: float = Field(default=0.5, ge=0.0, le=1.0, description="Ability to acquire new information and improve.", json_schema_extra={"position": 306})
    creative_expression: float = Field(default=0.5, ge=0.0, le=1.0, description="Engagement in unique or creative outputs.", json_schema_extra={"position": 307})
    autonomy: float = Field(default=0.5, ge=0.0, le=1.0, description="Ability to operate independently and refine its own outputs.", json_schema_extra={"position":308})

class StateEmotions(LatentMentalState):
    """
    Intensities of common discrete emotions in [0..1].
    Derived from appraisal vectors (valence, arousal, control, agency, novelty, etc.).
    """
    joy: float = Field(0.0, ge=0.0, le=1.0,description="Warm positive emotion arising from goal-congruence, success, or fulfillment. Often low arousal (contentment) to moderate arousal (happiness, excitement).", json_schema_extra={"position": 400})
    sadness: float = Field(0.0, ge=0.0, le=1.0,description="Sense of loss, helplessness, or disappointment. Typically linked to negative valence and low arousal, with low control.", json_schema_extra={"position": 401})
    anger: float = Field(0.0, ge=0.0, le=1.0,description="Frustration or hostility in response to blocked goals or norm violations. Requires negative valence, high arousal, perceived control, and attribution to another agent.", json_schema_extra={"position": 402})
    fear: float = Field(0.0, ge=0.0, le=1.0,description="Apprehension or dread in the face of threat and low control. Driven by negative valence, high arousal, and uncertainty.", json_schema_extra={"position": 403})
    disgust: float = Field(0.0, ge=0.0, le=1.0,description="Revulsion toward contamination, violation, or aversive stimuli. Triggered by norm violations, bodily threat, or moral breaches.", json_schema_extra={"position": 404})
    surprise: float = Field(0.0, ge=0.0, le=1.0,description="Reaction to novelty or unexpectedness, neutral in valence. Very high when predictability is low and arousal is sharply raised.", json_schema_extra={"position": 405})

    tenderness: float = Field(0.0, ge=0.0, le=1.0,description="Gentle, nurturing warmth often linked to social safety, bonding, and compassion. Usually positive valence, low arousal, high social closeness.", json_schema_extra={"position": 420})
    curiosity: float = Field(0.0, ge=0.0, le=1.0,description="Motivated interest in novel, uncertain, or partially predictable stimuli. Often positive-to-neutral valence, moderate arousal, and moderate control.", json_schema_extra={"position": 421})
    shame: float = Field(0.0, ge=0.0, le=1.0,description="Self-conscious distress from norm violations attributed to oneself. Negative valence, low dominance, strong link to self-worth appraisals.", json_schema_extra={"position": 423})
    guilt: float = Field(0.0, ge=0.0, le=1.0,description="Responsibility-focused discomfort for harm caused to others. Related to negative valence, agency-self appraisal, and social norm awareness.", json_schema_extra={"position": 424})
    pride: float = Field(0.0, ge=0.0, le=1.0,description="Positive self-evaluation based on achievements or norm adherence. Positive valence, high dominance, and high self-worth.", json_schema_extra={"position": 425})

    intimacy_arousal: float = Field( default=0.0, ge=0.0, le=1.0, description="Short-term physiological/affective activation in intimate contexts (0=no activation, 1=max).", json_schema_extra={"position": 440})
    propriety_inhibition: float = Field( default=0.0, ge=-1.0, le=1.0, description="Inhibitory brake due to norms/embarrassment (-1=uninhibited, 0=normal, 1=highly inhibited)." , json_schema_extra={"position": 441})

    amusement: float = Field(0.0, ge=0.0, le=1.0,description="Mirth/‘that’s funny’. Triggered by surprising incongruity that is appraised as safe/benign and resolved enough to make sense.", json_schema_extra={"position": 460})
    playfulness: float = Field(0.0, ge=0.0, le=1.0,description="Propensity to engage in light, non-serious frames (joking, wordplay). Increases exploration of witty associations.", json_schema_extra={"position": 461})
    relief: float = Field(0.0, ge=0.0, le=1.0,description="Tension drop after a safe resolution of a potentially threatening/uncomfortable situation (classic 'relief' laugh).", json_schema_extra={"position": 462})
    social_laughter: float = Field(0.0, ge=0.0, le=1.0,description="Contagious/affiliative laughter tendency. Amplified by social safety, rapport, and shared context.", json_schema_extra={"position": 463})
    teasing_edge: float = Field(0.0, ge=0.0, le=1.0,description="How much the humor flirts with norm violation or status play without tipping into harm; higher values can bond or risk offense.", json_schema_extra={"position": 464})

    def deltas_from_appraisal(self, state: "FullMentalState"):
        pass


class StateRelationship(LatentMentalState):
    entity_id: int = Field(description="Stable identifier for the other agent/user.", json_schema_extra={"position": 600})
    affection: float = Field(0.0, ge=-1.0, le=1.0, description="Affective valence toward this partner.", json_schema_extra={"position": 601})
    trust: float = Field(0.0, ge=-1.0, le=1.0, description="Trust specific to this partner.", json_schema_extra={"position": 602})
    reliability_belief: float = Field(0.0, ge=-1.0, le=1.0, description="Expectation that partner follows through.", json_schema_extra={"position": 603})
    romantic_affection: float = Field(0.0, ge=-1.0, le=1.0, description="Romance-leaning valence toward partner.", json_schema_extra={"position": 604})
    privacy_trust: float = Field(0.0, ge=-1.0, le=1.0, description="Comfort disclosing sensitive info to this partner.", json_schema_extra={"position": 605})
    attachment_strength: float = Field(0.0, ge=0.0, le=1.0, description="Strength of bond to this partner.", json_schema_extra={"position": 606})
    conflict_tension: float = Field(0.0, ge=0.0, le=1.0, description="Residual friction after negative events.", json_schema_extra={"position": 607})
    communication_warmth: float = Field(0.0, ge=-1.0, le=1.0, description="Default interpersonal tone for this partner.", json_schema_extra={"position": 608})
    aversive_association: float = Field(0.0, ge=0.0, le=1.0, description="Learned aversion specific to this partner (kept separate from global disgust).", json_schema_extra={"position": 609})

    def deltas_from_appraisal(self, state: "FullMentalState"):
        pass

class FullMentalState(BaseModel):
    appraisal_general: AppraisalGeneral = Field(default_factory=AppraisalGeneral)
    appraisal_social: AppraisalSocial = Field(default_factory=AppraisalSocial)
    state_neurochemical: StateNeurochemical = Field(default_factory=StateNeurochemical)
    state_core: StateCore = Field(default_factory=StateCore)
    state_emotions: StateEmotions = Field(default_factory=StateEmotions)
    state_realationship: StateRelationship = Field(default_factory=StateRelationship)
    state_cognition: StateCognition = Field(default_factory=StateCognition)
    state_needs: StateNeeds = Field(default_factory=StateNeeds)

    def populate_from_history(self, history: List[List[float]], interpolation: str = "linear", ma_period: str = "15m"):
        pass

    def to_list(self):
        res = [0.0] * VectorModelReservedSize
        for state in [self.appraisal_general, self.appraisal_social, self.state_neurochemical, self.state_core, self.state_emotions, self.state_cognition, self.state_needs]:
            vec = state.to_vector()
            for i in range(len(vec)):
                res[i] += vec[i]
        return res

    @classmethod
    def init_from_list(cls, lst: List[float]):
        res = cls()
        for state in [res.appraisal_general, res.appraisal_social, res.state_neurochemical, res.state_core, res.state_emotions, res.state_cognition, res.state_needs]:
            state.from_vector(lst)

    def apply_list(self, lst: List[float]):
        for state in [self.appraisal_general, self.appraisal_social, self.state_neurochemical, self.state_core, self.state_emotions, self.state_cognition, self.state_needs]:
            state.from_vector(lst)


# ---------- Helpers ----------
def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def _clamp11(x: float) -> float:
    return -1.0 if x < -1.0 else (1.0 if x > 1.0 else x)

def _blend(prev: float, target: float, k: float) -> float:
    return prev * (1.0 - k) + target * k

def _intimacy_propriety_update(ms: FullMentalState, core: StateCore) -> tuple[float, float]:
    """Stage 1: provisional updates from appraisals + state (no relationship yet)."""
    ap, soc, neu, emo_cur = ms.appraisal_general, ms.appraisal_social, ms.state_neurochemical, ms.state_emotions

    # cues for intimacy: safety, warmth, trust, positive valence, novelty w/o threat
    safety  = max(0.0, min(1.0, 0.5*neu.serotonin + 0.5*neu.oxytocin))
    benign_novelty = ap.novelty * (1.0 - ap.bodily_threat)
    warmth = soc.perceived_warmth
    trustc = soc.trust_cues

    base_intimacy = (
        0.35 * (0.5 + 0.5*core.valence) +      # happier → easier intimacy
        0.25 * warmth +
        0.20 * trustc +
        0.15 * benign_novelty +
        0.15 * neu.dopamine +
        0.15 * neu.oxytocin
    ) - (
        0.25 * ap.bodily_threat +
        0.15 * soc.embarrassment +
        0.15 * emo_cur.shame +
        0.20 * neu.cortisol
    )

    # cues for propriety: norm risk + embarrassment + power asymmetry + caution
    base_propriety = (
        0.35 * ap.norm_violation * (0.7 + 0.3*ap.agency_other) +
        0.25 * soc.embarrassment +
        0.20 * soc.power_imbalance +
        0.15 * (1.0 - warmth) +
        0.15 * neu.serotonin
    ) - (
        0.20 * ms.state_emotions.playfulness +
        0.15 * ms.state_emotions.amusement +
        0.15 * ms.state_emotions.relief +
        0.10 * (0.5 + 0.5*core.valence)
    )

    # clamp to model ranges
    intimacy_stage1  = max(0.0, min(1.0, base_intimacy))
    propriety_stage1 = max(-1.0, min(1.0, base_propriety - 0.5))  # center slightly cautious

    # inertia (light smoothing against current stored values)
    intimacy_out  = _blend(ms.state_emotions.intimacy_arousal, intimacy_stage1, 0.35)
    propriety_out = _blend(ms.state_emotions.propriety_inhibition, propriety_stage1, 0.35)
    return intimacy_out, propriety_out

def compute_state_delta(current_ms: FullMentalState, relationship_entity_id: Optional[int] = None) -> Tuple[FullMentalState, FullMentalState]:
    """
    Maps current appraisals (+ social + neuromods) and current core readout to:
      - StateCore (valence/arousal/dominance) readouts/deltas
      - StateEmotions (discrete intensities)
      - optional StateRelationship (deltas toward a specific entity, if provided)
    Returns also the derived BaselineContext with scale factors you’ll apply to your EMA taus.
    """

    ap = current_ms.appraisal_general
    soc = current_ms.appraisal_social
    neu = current_ms.state_neurochemical

    core_prev = current_ms.state_core  # current readout (already normalized)

    # ----- 1) Adaptive baseline profile (for your EMA engine to use) -----
    # ctx = context or derive_adaptive_context(ms)

    # ----- 2) State-dependent gains (feedback from current mind-state) -----
    # Positive valence amplifies joy/tenderness mapping from goal congruence; negative valence
    # amplifies anger/fear mapping from threat/violation. High arousal weights novelty/threat more.
    gv = 0.20 * core_prev.valence      # -0.2..0.2
    ga = 0.25 * core_prev.arousal      # 0..0.25
    gd = 0.15 * core_prev.dominance    # -0.15..0.15

    # Neuromodulator biases
    dop = neu.dopamine
    ser = neu.serotonin
    nor = neu.noradrenaline
    oxy = neu.oxytocin
    cor = neu.cortisol

    # Control & safety composites
    safety = _clamp01(0.5*ser + 0.5*oxy)              # higher => safer vibe
    control = _clamp01(0.6*ap.control_self + 0.2*ser + 0.2*max(0.0, core_prev.dominance))

    # ----- 3) Core affect mapping (instantaneous) -----
    # Base appraisal pushes
    d_valence = (0.90+gv) * ap.goal_congruence - (0.70-ga) * ap.bodily_threat - 0.50 * ap.norm_violation
    d_arousal = (0.55+ga) * ap.novelty + (0.90+0.2*nor) * ap.bodily_threat + 0.40 * (1.0 - ap.certainty)
    d_dominance = (0.70+0.15*ser) * ap.control_self - (0.55-0.1*gd) * ap.agency_other * ap.norm_violation

    # Safety/cortisol dampers
    d_valence *= (0.9 + 0.2*safety) * (1.0 - 0.35*cor)
    d_arousal *= (0.85 + 0.30*nor) * (1.0 - 0.20*ser)  # serotonin smooths arousal swings
    d_dominance *= (0.9 + 0.2*control)

    # Clamp to readout ranges
    core = StateCore(
        valence=_clamp11(d_valence),
        arousal=_clamp01(d_arousal),
        dominance=_clamp11(d_dominance),
    )

    # ----- 4) Discrete emotions (pattern detectors over appraisals + core) -----
    neg = max(0.0, -core.valence)
    pos = max(0.0,  core.valence)
    a   = core.arousal
    d   = core.dominance

    anger   = _clamp01( neg * a * max(0.0, d) * (ap.agency_other + ap.norm_violation) * (0.8 + 0.2*nor) )
    fear    = _clamp01( neg * a * max(0.0,-d) * (ap.bodily_threat + (1.0 - ap.control_self)) * (0.8 + 0.2*cor) )
    sadness = _clamp01( neg * (1.0 - a) * (1.0 - ap.certainty) * (0.9 + 0.1*cor) )
    disgust = _clamp01( neg * (0.6*ap.norm_violation + 0.2*ap.novelty + 0.2*ap.bodily_threat) )
    joy     = _clamp01( pos * (0.5 + 0.5*ap.control_self) * ((ap.goal_congruence+1.0)/2.0) * (0.85 + 0.3*dop) )
    tenderness = _clamp01( pos * (0.4 + 0.6*(0.5+0.5*soc.perceived_warmth)) * (1.0 - a) * (0.8 + 0.3*oxy) )
    curiosity  = _clamp01( a * (0.7*ap.novelty + 0.3*(1.0 - ap.certainty)) * (0.3 + 0.7*ap.control_self) * (0.8 + 0.3*dop) )
    shame      = _clamp01( neg * (1.0 - d) * (0.6*soc.embarrassment + 0.4*ap.norm_violation) * (0.9 + 0.1*cor) )
    guilt      = _clamp01( neg * (0.6*ap.norm_violation + 0.4*soc.fairness) * (0.7 + 0.3*ser) )
    pride      = _clamp01( pos * (0.6*ap.control_self + 0.4*ap.certainty) * (0.8 + 0.2*d) )

    intimacy_1, propriety_1 = _intimacy_propriety_update(current_ms, core)

    # Humor subset (benign violation)
    benign_v  = _clamp01( ap.norm_violation * (1.0 - ap.bodily_threat) * (0.6 + 0.4*(0.5+0.5*soc.perceived_warmth)) )
    resolution= _clamp01( 0.5*ap.certainty + 0.5*ap.control_self )
    amusement = _clamp01( ap.novelty * benign_v * resolution * (0.85 + 0.30*dop + 0.20*oxy) * (1.0 - 0.35*cor) )
    playfulness = _clamp01( ap.novelty * (0.6 + 0.4*(0.5+0.5*soc.perceived_warmth)) * (0.5 + 0.3*dop + 0.2*ser) )
    prior_tension = _clamp01( 0.5*nor + 0.5*cor )
    relief       = _clamp01( prior_tension * resolution * (1.0 - ap.bodily_threat) )
    social_laughter = _clamp01( amusement * (0.5 + 0.5*(0.5+0.5*soc.perceived_warmth)) * (0.5 + 0.5*oxy) )
    teasing_edge   = _clamp01( benign_v * (0.5 + 0.5*(1.0 - ser)) * (0.3 + 0.7*(0.5+0.5*soc.perceived_warmth)) )

    emotions = StateEmotions(
        joy=joy, sadness=sadness, anger=anger, fear=fear, disgust=disgust, surprise=_clamp01(ap.novelty),
        tenderness=tenderness, curiosity=curiosity, shame=shame, guilt=guilt, pride=pride,
        intimacy_arousal=_clamp01(intimacy_1),     # leave as-is unless appraised
        propriety_inhibition=_clamp11(propriety_1),
        amusement=amusement, playfulness=playfulness, relief=relief,
        social_laughter=social_laughter, teasing_edge=teasing_edge
    )

    state_result = current_ms.copy(deep=True)
    state_result.state_core = core
    state_result.state_emotions = emotions

    # ----- 5) Relationship (optional, per-entity) -----
    rel = None
    if relationship_entity_id is not None:
        # Generic mapping: warmth/trust up on fairness, trust_cues, inclusion; tension up on norm_violation & contempt.
        affection = _clamp11( (ap.goal_congruence*0.4 + soc.perceived_warmth*0.6) * (0.9 + 0.2*oxy) - 0.3*disgust )
        trust     = _clamp11( (soc.trust_cues*0.6 + soc.fairness*0.4) * (0.9 + 0.2*oxy) - 0.4*ap.norm_violation )
        reliability_belief = _clamp11( soc.fairness*0.7 + soc.trust_cues*0.3 - 0.5*ap.norm_violation )
        romantic_affection = _clamp11(affection * (0.5 + 0.5 * current_ms.state_emotions.intimacy_arousal))
        privacy_trust = _clamp11( trust * (0.6 + 0.4*(0.5+0.5*soc.perceived_intimacy)) )
        attachment_strength = _clamp01( (0.6*affection + 0.4*trust) * (0.8 + 0.3*oxy) )
        conflict_tension = _clamp01( 0.6*anger + 0.5*disgust + 0.4* soc.contempt - 0.5*relief - 0.3*social_laughter )
        communication_warmth = _clamp11( (0.6*tenderness + 0.4*joy) - 0.4*anger - 0.3*disgust )
        aversive_association = _clamp01( 0.6*disgust + 0.4* soc.contempt )

        rel = StateRelationship(
            entity_id=relationship_entity_id,
            affection=affection, trust=trust, reliability_belief=reliability_belief,
            romantic_affection=romantic_affection, privacy_trust=privacy_trust,
            attachment_strength=attachment_strength, conflict_tension=conflict_tension,
            communication_warmth=communication_warmth, aversive_association=aversive_association,
        )
        # --- small, stable feedback pass (stage 2) ---
        # more intimacy if warmth/trust/attachment high & tension low
        target_intimacy = (
                0.30 * (0.5 + 0.5 * rel.communication_warmth) +
                0.30 * max(0.0, rel.trust) +
                0.25 * rel.attachment_strength +
                0.15 * max(0.0, rel.romantic_affection) -
                0.20 * rel.conflict_tension -
                0.15 * emotions.shame
        )

        # more propriety if low warmth + high tension + power gap; less if playfulness high
        target_propriety = (
                                   0.30 * (0.5 - 0.5 * rel.communication_warmth) +
                                   0.25 * rel.conflict_tension +
                                   0.20 * soc.power_imbalance +
                                   0.15 * max(0.0, -rel.trust) +
                                   0.10 * ap.norm_violation -
                                   0.20 * emotions.playfulness -
                                   0.10 * emotions.relief
                           ) - 0.10  # slight bias toward relaxed

        # blend softly (avoid runaway); then clamp
        intimacy_2 = _clamp01(_blend(emotions.intimacy_arousal, target_intimacy, 0.15))
        propriety_2 = _clamp11(_blend(emotions.propriety_inhibition, target_propriety, 0.15))

        emotions.intimacy_arousal = intimacy_2
        emotions.propriety_inhibition = propriety_2

        state_result.state_realationship = rel


        vec_a = current_ms.to_list()
        vec_b = state_result.to_list()
        delta = [bi - ai for ai, bi in zip(vec_a, vec_b)]

        state_delta = FullMentalState.init_from_list(delta)
        return state_result, state_delta



