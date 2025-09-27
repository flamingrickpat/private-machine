from __future__ import annotations
from typing import Dict, List, Sequence, Type, TypeVar, Any, Optional
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
    def from_vector(cls: Type[T], values: Sequence[float]) -> T:
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
    willpower: float = Field( default=0.5, ge=0, le=1, description="How easy it is to decide on high-effort or delayed-gratification intents.", json_schema_extra={"position": 203})

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
    state_cognition: StateCognition = Field(default_factory=StateCognition)

    def populate_from_history(self, history: List[List[float]], interpolation: str = "linear", ma_period: str = "15m"):
        pass


