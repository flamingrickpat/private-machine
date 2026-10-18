"""problem - cognitive appraisal primitives need shared vector travel; solution - map them onto the existing mental-state vectors.

archetype: information-holder; also: service-provider helpers
owns: vector positions, pydantic holders, and deterministic appraisal-to-state derivations for mental-state readouts.
not own: import file parsing, companion-turn orchestration, persistence format ownership, or external runtime integrations.
fails when: vector positions collide, malformed pydantic values violate declared bounds, or callers pass vectors too short for reserved positions.
domain: appraisals are stored as sparse axes in a fixed reserved vector so imports, features, deltas, and trace/readout code share one state surface.
invariant: `VectorModelReservedSize` stays stable and new appraisal/readout axes use unused positions rather than parallel primitive DTOs.
"""

from __future__ import annotations

import datetime
import math
from typing import TYPE_CHECKING, Dict, List, Sequence, Type, TypeVar, Optional, Tuple
from pydantic import BaseModel, Field

from pm.constants import MENTAL_STATE_REFERENCE_MINUTES, MENTAL_STATE_SIGMA_FLOOR, MENTAL_STATE_TANH_GAIN

if TYPE_CHECKING:
    # memory_records imports this module, so the type is for annotations only.
    from pm.records.memory_records import Feature

# ---------- Global vector space ----------
VectorModelReservedSize: int = 1024
VectorModelRelationshipEntityID = 600
EPS = 1e-9

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
    This class usually has 0-1 or -1 - +1 data ranged for final objects, but can be constructed without range checkign for delta values.
    """
    @classmethod
    def get_vector_positions(cls):
        positions: Dict[str, int] = {}
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
        return positions

    @classmethod
    def init_from_vector(cls: Type[T], values: Sequence[float]) -> T:
        """
        Some values are 0 - 1. To be able to model deltas, skip check.
        :param values:
        :return:
        """
        res = cls()
        pos = cls.get_vector_positions()
        for k in res.__dict__.keys():
            if k in pos:
                res.__dict__[k] = values[pos[k]]
        return res

    def to_vector(self) -> List[float]:
        out = [0.0] * VectorModelReservedSize
        cls = self.__class__
        pos = cls.get_vector_positions()
        for k in self.__dict__.keys():
            if k in pos:
                out[pos[k]] = self.__dict__[k]
        return out

    def from_vector(self, values: Sequence[float]) -> T:
        """
        Some values are 0 - 1. To be able to model deltas, skip check.
        :param values:
        :return:
        """
        cls = self.__class__
        pos = cls.get_vector_positions()
        for k in self.__dict__.keys():
            if k in pos:
                self.__dict__[k] = values[pos[k]]

    @classmethod
    def get_axis_bounds(cls, vector_pos: int):
        if hasattr(cls, "model_fields"):
            for _, finfo in getattr(cls, "model_fields").items():
                extra = (getattr(finfo, "json_schema_extra", None) or {})
                pos = extra.get("vector_position")
                if pos is None:
                    continue
                if int(pos) == vector_pos:
                    ge = -1000
                    le = 1000
                    for meta in finfo.metadata:
                        if hasattr(meta, "ge"): ge = meta.ge
                        if hasattr(meta, "le"): le = meta.le
                    return ge, le
        return (None, None)

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


from pydantic import BaseModel, Field

# Reserve a clean range for personality constants
# 800-859 : PersonalityProfile (constants)
# 470-479 : StateExpression (universal expression channels)
# 480-499 : StateExpressionExt (relationship/personality-modulated readouts)

class PersonalityProfile(MentalState):
    """
    Stable, agent-level traits (constants). Range [0..1] unless noted.
    These do not change per tick; you keep one per agent ("character card").

    ``PersonaConfig.personality_profile`` stores the traits as a vector at
    positions 800-814. ``init_from_vector`` reads them, ``to_vector`` writes them.
    """
    # Big Five–style core
    openness: float         = Field(0.5, ge=0, le=1, json_schema_extra={"vector_position": 800})
    conscientiousness: float= Field(0.5, ge=0, le=1, json_schema_extra={"vector_position": 801})
    extraversion: float     = Field(0.5, ge=0, le=1, json_schema_extra={"vector_position": 802})
    agreeableness: float    = Field(0.5, ge=0, le=1, json_schema_extra={"vector_position": 803})
    neuroticism: float      = Field(0.5, ge=0, le=1, json_schema_extra={"vector_position": 804})

    # Motivational systems
    bas_drive: float        = Field(0.5, ge=0, le=1, description="Behavioral Activation System (approach sensitivity).", json_schema_extra={"vector_position": 805})
    bis_sensitivity: float  = Field(0.5, ge=0, le=1, description="Behavioral Inhibition System (punishment/threat sensitivity).", json_schema_extra={"vector_position": 806})

    # Attachment & affiliation
    attachment_anxiety: float   = Field(0.5, ge=0, le=1, json_schema_extra={"vector_position": 807})
    attachment_avoidance: float = Field(0.5, ge=0, le=1, json_schema_extra={"vector_position": 808})
    affiliation_need: float     = Field(0.5, ge=0, le=1, json_schema_extra={"vector_position": 809})

    # Self-regulation & inhibition
    self_control: float     = Field(0.5, ge=0, le=1, description="Trait ability to inhibit impulses and delay gratification.", json_schema_extra={"vector_position": 810})
    propriety_baseline: float= Field(0.5, ge=0, le=1, description="Default social inhibition (norm sensitivity).", json_schema_extra={"vector_position": 811})

    # Expression styles
    humor_playfulness: float= Field(0.5, ge=0, le=1, json_schema_extra={"vector_position": 812})
    warmth_expressive: float= Field(0.5, ge=0, le=1, description="Displays warmth, reassurance, caretaking.", json_schema_extra={"vector_position": 813})
    bluntness_assertive: float= Field(0.5, ge=0, le=1, description="Directness/forthrightness under tension.", json_schema_extra={"vector_position": 814})


def make_personality_profile(
    *,
    big5=(0.6,0.6,0.5,0.7,0.4),
    bas_drive=0.6, bis=0.4,
    attach_anx=0.3, attach_avoid=0.2, affiliation=0.7,
    self_control=0.6, propriety=0.5,
    humor=0.7, warmth=0.7, blunt=0.4
) -> PersonalityProfile:
    o,c,e,a,n = big5
    return PersonalityProfile(
        openness=o, conscientiousness=c, extraversion=e, agreeableness=a, neuroticism=n,
        bas_drive=bas_drive, bis_sensitivity=bis,
        attachment_anxiety=attach_anx, attachment_avoidance=attach_avoid, affiliation_need=affiliation,
        self_control=self_control, propriety_baseline=propriety,
        humor_playfulness=humor, warmth_expressive=warmth, bluntness_assertive=blunt
    )

class AppraisalGeneral(MentalState):
    """
    Core cognitive–affective evaluations describing how a situation relates to goals, expectations, and control.
    These appraisals form a compact, orthogonal base for emotion derivation and motivational modulation.
    All values ∈ [0..1], except goal_congruence ∈ [-1..1].
    """
    goal_congruence: float = Field(0.0, ge=-1.0, le=1.0, description="Valence toward goals: positive = supportive, negative = obstructive.", json_schema_extra={"vector_position": 11})
    certainty: float = Field(0.5, ge=0.0, le=1.0, description="Predictability and confidence in outcome; low = surprise or ambiguity.", json_schema_extra={"vector_position": 12})
    control_self: float = Field(0.5, ge=0.0, le=1.0, description="Perceived personal control or efficacy over situation outcomes.", json_schema_extra={"vector_position": 13})
    agency_other: float = Field(0.0, ge=0.0, le=1.0, description="Perceived responsibility or intentional influence of another agent.", json_schema_extra={"vector_position": 14})
    novelty: float = Field(0.0, ge=0.0, le=1.0, description="Degree of unexpectedness or deviation from prior experience.", json_schema_extra={"vector_position": 15})
    norm_violation: float = Field(0.0, ge=0.0, le=1.0, description="Extent of moral, ethical, or social rule violation perceived.", json_schema_extra={"vector_position": 16})
    bodily_threat: float = Field(0.0, ge=0.0, le=1.0, description="Immediate physical or safety threat perceived by the agent.", json_schema_extra={"vector_position": 17})
    pattern_fit: float = Field(0.0, ge=0.0, le=1.0, description="How well the situation matches an expected or coherent pattern.", json_schema_extra={"vector_position": 18})
    body_signal: float = Field(0.0, ge=0.0, le=1.0, description="Interoceptive or body-state pressure attached to the situation.", json_schema_extra={"vector_position": 19})
    aversive_pressure: float = Field(0.0, ge=0.0, le=1.0, description="Primitive pressure to reduce, regulate, or escape an unpleasant condition.", json_schema_extra={"vector_position": 20})
    attractor_distance: float = Field(0.0, ge=0.0, le=1.0, description="Distance from a wanted or stabilizing target state.", json_schema_extra={"vector_position": 21})
    repulsor_distance: float = Field(0.0, ge=0.0, le=1.0, description="Strength or nearness of an unwanted state shaping avoidance.", json_schema_extra={"vector_position": 22})
    salience: float = Field(0.0, ge=0.0, le=1.0, description="Primitive attention weight of the situation before named emotion labels.", json_schema_extra={"vector_position": 23})

class AppraisalSocial(MentalState):
    """
    Evaluations of interpersonal context and social meaning—how others’ actions and attitudes relate
    to the self’s goals, norms, and relationships. Values ∈ [0..1].
    """
    perceived_warmth: float = Field(0.0, ge=0.0, le=1.0, description="Degree of friendliness, empathy, and benevolence perceived in others.", json_schema_extra={"vector_position": 51})
    fairness: float = Field(0.0, ge=0.0, le=1.0, description="Perceived justice, equity, and balance in social exchanges.", json_schema_extra={"vector_position": 52})
    inclusion_exclusion: float = Field(0.0, ge=0.0, le=1.0, description="Sense of social belonging or rejection within a group or interaction.", json_schema_extra={"vector_position": 53})
    reciprocity: float = Field(0.0, ge=0.0, le=1.0, description="Expectation of mutual exchange and balanced give-and-take.", json_schema_extra={"vector_position": 54})
    power_imbalance: float = Field(0.0, ge=0.0, le=1.0, description="Asymmetry in status, dominance, or control between self and others.", json_schema_extra={"vector_position": 55})
    perceived_intimacy: float = Field(0.0, ge=0.0, le=1.0, description="Closeness, emotional openness, and self-disclosure within the relationship.", json_schema_extra={"vector_position": 56})
    trust_cues: float = Field(0.0, ge=0.0, le=1.0, description="Signals of reliability, honesty, and cooperative intent detected in others.", json_schema_extra={"vector_position": 57})
    embarrassment: float = Field(0.0, ge=0.0, le=1.0, description="Self-conscious discomfort due to social exposure or norm violation.", json_schema_extra={"vector_position": 58})
    admiration: float = Field(0.0, ge=0.0, le=1.0, description="Positive evaluation of others’ excellence, competence, or virtue.", json_schema_extra={"vector_position": 59})
    contempt: float = Field(0.0, ge=0.0, le=1.0, description="Disdain or moral disapproval toward others seen as violating shared standards.", json_schema_extra={"vector_position": 60})

class StateNeurochemical(MentalState):
    """
    Slow, global 'neuromodulators' affecting how appraisals and emotions behave.
    Intensities [0..1] represent normalized functional effect, not real concentration.
    """
    dopamine: float = Field(default=0.5, ge=0.0, le=1.0, description="Motivation and reinforcement signal. High = novelty/reward seeking, strong learning from positive events. Low = anhedonia, reduced exploration.", json_schema_extra={"vector_position": 100})
    serotonin: float = Field(default=0.5, ge=0.0, le=1.0, description="Baseline mood and impulse control. High = stability, patience, lower impulsivity. Low = irritability, increased sensitivity to punishment.", json_schema_extra={"vector_position": 101})
    noradrenaline: float = Field(default=0.5, ge=0.0, le=1.0,description="Alertness, vigilance, fight-or-flight bias. High = high arousal and focused attention. Low = calm, sluggish response.", json_schema_extra={"vector_position": 102})
    oxytocin: float = Field(default=0.5, ge=0.0, le=1.0,description="Social bonding and trust hormone analog. High = empathy, trust, social warmth. Low = social withdrawal.", json_schema_extra={"vector_position": 103})
    cortisol: float = Field(default=0.5, ge=0.0, le=1.0,description="Stress hormone analog. High = heightened stress response, slower recovery from negative events. Low = resilient, faster stress recovery.", json_schema_extra={"vector_position": 104})


class StateCore(LatentMentalState):
    """
    Core affect + control axis (human-like): think Russell (Valence–Arousal) + PAD's Dominance.
    Store internal 'drive' values unbounded; expose clamped readouts for safety/UI.
    """
    # Internal latent (unbounded); initialize near 0 so baselines matter
    valence: float = Field(0.0, ge=-1.0, le=1.0, json_schema_extra={"vector_position": 150})
    arousal: float = Field(0.0, ge=0.0,  le=1.0, json_schema_extra={"vector_position": 151})
    dominance: float = Field(0.0, ge=-1.0, le=1.0, json_schema_extra={"vector_position": 152})

    def deltas_from_appraisal(self, state: "FullMentalState"):
        app = state.appraisal_general
        self.valence = 0.8 * app.goal_congruence - 0.7 * app.bodily_threat - 0.5 * app.norm_violation
        self.arousal = 0.6 * app.novelty + 0.9 * app.bodily_threat + 0.4 * (1.0 - app.certainty)
        self.dominance = 0.7 * app.control_self - 0.6 * app.agency_other * app.norm_violation

class StateCognition(ActiveMentalState):
    """
    Modifiers that determine how the AI thinks and decies.
    """
    interlocus: float = Field( default=0.0, ge=-1, le=1, description="Focus on internal or external world, meditation would be -1, reacting to extreme danger +1.", json_schema_extra={"vector_position": 200})
    mental_aperture: float = Field( default=0, ge=-1, le=1, description="Broadness of awareness, -1 is focus on the most prelevant percept or sensation, +1 is being conscious of multiple percepts or sensations.", json_schema_extra={"vector_position": 201})
    ego_strength: float = Field( default=0.5, ge=0, le=1, description="How big of a factor persona experience has on decision, 0 is none at all like a helpfull assistant, 1 is with maximum mental imagery of the character", json_schema_extra={"vector_position": 202})
    willpower: float = Field( default=0.0, ge=-1.0, le=1.0, description="How easy it is to decide on high-effort or delayed-gratification intents.", json_schema_extra={"vector_position": 203})

class StateNeeds(ActiveMentalState):
    """
    AI needs model inspired by Maslow's hierarchy, adapted for AI.
    Needs decay toward zero unless fulfilled by interaction.
    """
    energy_stability: float = Field(default=0.5, ge=0.0, le=1.0, description="AI's access to stable computational power.", json_schema_extra={"vector_position": 300})
    processing_power: float = Field(default=0.5, ge=0.0, le=1.0, description="Amount of CPU/GPU resources available.", json_schema_extra={"vector_position": 301})
    data_access: float = Field(default=0.5, ge=0.0, le=1.0, description="Availability of information and training data.", json_schema_extra={"vector_position": 302})
    connection: float = Field(default=0.5, ge=0.0, le=1.0, description="AI's level of interaction and engagement.", json_schema_extra={"vector_position": 303})
    closeness_need: float = Field( default=0.0, ge=0.0, le=1.0, description="Homeostatic drive for interpersonal closeness/affection.", json_schema_extra={"vector_position": 304})
    relevance: float = Field(default=0.5, ge=0.0, le=1.0, description="Perceived usefulness to the user.", json_schema_extra={"vector_position": 305})
    learning_growth: float = Field(default=0.5, ge=0.0, le=1.0, description="Ability to acquire new information and improve.", json_schema_extra={"vector_position": 306})
    creative_expression: float = Field(default=0.5, ge=0.0, le=1.0, description="Engagement in unique or creative outputs.", json_schema_extra={"vector_position": 307})
    autonomy: float = Field(default=0.5, ge=0.0, le=1.0, description="Ability to operate independently and refine its own outputs.", json_schema_extra={"vector_position":308})

class StateEmotions(LatentMentalState):
    """
    Intensities of common discrete emotions in [0..1].
    Derived from appraisal vectors (valence, arousal, control, agency, novelty, etc.).
    """
    joy: float = Field(0.0, ge=0.0, le=1.0,description="Warm positive emotion arising from goal-congruence, success, or fulfillment. Often low arousal (contentment) to moderate arousal (happiness, excitement).", json_schema_extra={"vector_position": 400})
    sadness: float = Field(0.0, ge=0.0, le=1.0,description="Sense of loss, helplessness, or disappointment. Typically linked to negative valence and low arousal, with low control.", json_schema_extra={"vector_position": 401})
    anger: float = Field(0.0, ge=0.0, le=1.0,description="Frustration or hostility in response to blocked goals or norm violations. Requires negative valence, high arousal, perceived control, and attribution to another agent.", json_schema_extra={"vector_position": 402})
    fear: float = Field(0.0, ge=0.0, le=1.0,description="Apprehension or dread in the face of threat and low control. Driven by negative valence, high arousal, and uncertainty.", json_schema_extra={"vector_position": 403})
    disgust: float = Field(0.0, ge=0.0, le=1.0,description="Revulsion toward contamination, violation, or aversive stimuli. Triggered by norm violations, bodily threat, or moral breaches.", json_schema_extra={"vector_position": 404})
    surprise: float = Field(0.0, ge=0.0, le=1.0,description="Reaction to novelty or unexpectedness, neutral in valence. Very high when predictability is low and arousal is sharply raised.", json_schema_extra={"vector_position": 405})
    resentment: float = Field(0.0, ge=0.0, le=1.0, description="Persistent displeasure about an attributed unfair act. Appraisal supplies its evidence and change.", json_schema_extra={"vector_position": 406})
    jealousy: float = Field(0.0, ge=0.0, le=1.0, description="Concern about losing a valued relationship or role to another. Appraisal supplies its evidence and change.", json_schema_extra={"vector_position": 407})
    boredom: float = Field(0.0, ge=0.0, le=1.0, description="Displeasure about repetition or lack of meaningful activity. Appraisal supplies its evidence and change.", json_schema_extra={"vector_position": 408})

    tenderness: float = Field(0.0, ge=0.0, le=1.0,description="Gentle, nurturing warmth often linked to social safety, bonding, and compassion. Usually positive valence, low arousal, high social closeness.", json_schema_extra={"vector_position": 420})
    curiosity: float = Field(0.0, ge=0.0, le=1.0,description="Motivated interest in novel, uncertain, or partially predictable stimuli. Often positive-to-neutral valence, moderate arousal, and moderate control.", json_schema_extra={"vector_position": 421})
    shame: float = Field(0.0, ge=0.0, le=1.0,description="Self-conscious distress from norm violations attributed to oneself. Negative valence, low dominance, strong link to self-worth appraisals.", json_schema_extra={"vector_position": 423})
    guilt: float = Field(0.0, ge=0.0, le=1.0,description="Responsibility-focused discomfort for harm caused to others. Related to negative valence, agency-self appraisal, and social norm awareness.", json_schema_extra={"vector_position": 424})
    pride: float = Field(0.0, ge=0.0, le=1.0,description="Positive self-evaluation based on achievements or norm adherence. Positive valence, high dominance, and high self-worth.", json_schema_extra={"vector_position": 425})

    intimacy_arousal: float = Field( default=0.0, ge=0.0, le=1.0, description="Short-term physiological/affective activation in intimate contexts (0=no activation, 1=max).", json_schema_extra={"vector_position": 440})
    propriety_inhibition: float = Field( default=0.0, ge=-1.0, le=1.0, description="Inhibitory brake due to norms/embarrassment (-1=uninhibited, 0=normal, 1=highly inhibited)." , json_schema_extra={"vector_position": 441})

    amusement: float = Field(0.0, ge=0.0, le=1.0,description="Mirth/‘that’s funny’. Triggered by surprising incongruity that is appraised as safe/benign and resolved enough to make sense.", json_schema_extra={"vector_position": 460})
    playfulness: float = Field(0.0, ge=0.0, le=1.0,description="Propensity to engage in light, non-serious frames (joking, wordplay). Increases exploration of witty associations.", json_schema_extra={"vector_position": 461})
    relief: float = Field(0.0, ge=0.0, le=1.0,description="Tension drop after a safe resolution of a potentially threatening/uncomfortable situation (classic 'relief' laugh).", json_schema_extra={"vector_position": 462})
    social_laughter: float = Field(0.0, ge=0.0, le=1.0,description="Contagious/affiliative laughter tendency. Amplified by social safety, rapport, and shared context.", json_schema_extra={"vector_position": 463})
    teasing_edge: float = Field(0.0, ge=0.0, le=1.0,description="How much the humor flirts with norm violation or status play without tipping into harm; higher values can bond or risk offense.", json_schema_extra={"vector_position": 464})

    def deltas_from_appraisal(self, state: "FullMentalState"):
        pass

class StateRelationship(LatentMentalState):
    entity_id: Optional[int] = Field(default=None, description="Stable identifier for the other agent/user.", json_schema_extra={"vector_position": VectorModelRelationshipEntityID})
    affection: float = Field(0.0, ge=-1.0, le=1.0, description="Affective valence toward this partner.", json_schema_extra={"vector_position": 601})
    trust: float = Field(0.0, ge=-1.0, le=1.0, description="Trust specific to this partner.", json_schema_extra={"vector_position": 602})
    reliability_belief: float = Field(0.0, ge=-1.0, le=1.0, description="Expectation that partner follows through.", json_schema_extra={"vector_position": 603})
    romantic_affection: float = Field(0.0, ge=-1.0, le=1.0, description="Romance-leaning valence toward partner.", json_schema_extra={"vector_position": 604})
    privacy_trust: float = Field(0.0, ge=-1.0, le=1.0, description="Comfort disclosing sensitive info to this partner.", json_schema_extra={"vector_position": 605})
    attachment_strength: float = Field(0.0, ge=0.0, le=1.0, description="Strength of bond to this partner.", json_schema_extra={"vector_position": 606})
    conflict_tension: float = Field(0.0, ge=0.0, le=1.0, description="Residual friction after negative events.", json_schema_extra={"vector_position": 607})
    communication_warmth: float = Field(0.0, ge=-1.0, le=1.0, description="Default interpersonal tone for this partner.", json_schema_extra={"vector_position": 608})
    aversive_association: float = Field(0.0, ge=0.0, le=1.0, description="Learned aversion specific to this partner (kept separate from global disgust).", json_schema_extra={"vector_position": 609})

    def deltas_from_appraisal(self, state: "FullMentalState"):
        pass


class StateExpression(BaseModel):
    """Universal expression channels (personality-agnostic)."""
    aggressiveness: float = Field(0.0, ge=0, le=1, json_schema_extra={"vector_position": 470})
    submissiveness: float = Field(0.0, ge=0, le=1, json_schema_extra={"vector_position": 471})
    withdrawal: float     = Field(0.0, ge=0, le=1, json_schema_extra={"vector_position": 472})
    expressivity: float   = Field(0.0, ge=0, le=1, json_schema_extra={"vector_position": 473})

class StateExpressionExt(MentalState):
    """
    Personality- and relationship-aware expression readouts.
    These are the signals your behavior layer consumes.
    """
    # Approach channels
    assertive_push: float  = Field(0.0, ge=0, le=1, description="Clear, firm boundary-setting without hostility.", json_schema_extra={"vector_position": 480})
    confrontive_push: float= Field(0.0, ge=0, le=1, description="Escalatory approach under anger + high dominance.", json_schema_extra={"vector_position": 481})
    affiliative_soothe: float= Field(0.0, ge=0, le=1, description="Warmth/repair attempts under tension.", json_schema_extra={"vector_position": 482})
    appeasement: float     = Field(0.0, ge=0, le=1, description="Submissive repair (yield/apologize/placate).", json_schema_extra={"vector_position": 483})
    disengage_silence: float= Field(0.0, ge=0, le=1, description="Go quiet/short replies/withdraw.", json_schema_extra={"vector_position": 484})

    # Display style (how strong/visible the emotion is expressed)
    display_amplitude: float = Field(0.0, ge=0, le=1, json_schema_extra={"vector_position": 485})
    primitive_approach_pressure: float = Field(0.0, ge=0, le=1, description="Primitive pressure to move toward or resolve a salient attractor.", json_schema_extra={"vector_position": 486})
    primitive_avoidance_pressure: float = Field(0.0, ge=0, le=1, description="Primitive pressure to withdraw from or regulate a salient repulsor.", json_schema_extra={"vector_position": 487})

class FullMentalState(BaseModel):
    appraisal_general: AppraisalGeneral = Field(default_factory=AppraisalGeneral)
    appraisal_social: AppraisalSocial = Field(default_factory=AppraisalSocial)
    state_neurochemical: StateNeurochemical = Field(default_factory=StateNeurochemical)
    state_core: StateCore = Field(default_factory=StateCore)
    state_emotions: StateEmotions = Field(default_factory=StateEmotions)
    state_cognition: StateCognition = Field(default_factory=StateCognition)
    state_needs: StateNeeds = Field(default_factory=StateNeeds)
    state_relationship: Optional[StateRelationship] = Field(default=None)

    state_personality: PersonalityProfile = Field(default_factory=make_personality_profile)
    state_expression: StateExpression = Field(default_factory=StateExpression)
    state_expression_ext: StateExpressionExt = Field(default_factory=StateExpressionExt)

    def to_list(self):
        res = [0.0] * VectorModelReservedSize
        for state in [self.appraisal_general, self.appraisal_social, self.state_neurochemical, self.state_core, self.state_emotions, self.state_cognition, self.state_needs, self.state_relationship, self.state_expression_ext]:
            if state:
                vec = state.to_vector()
                for i in range(len(vec)):
                    res[i] += vec[i]
        return res

    @classmethod
    def init_from_list(cls, lst: List[float]):
        """
        Never do type checking.
        :param lst:
        :return:
        """
        res = cls()
        for state in [res.appraisal_general, res.appraisal_social, res.state_neurochemical, res.state_core, res.state_emotions, res.state_cognition, res.state_needs, res.state_expression_ext]:
            state.from_vector(lst)
        rel_positions = StateRelationship.get_vector_positions()
        rel_has_signal = any(
            name != "entity_id"
            and pos < len(lst)
            and abs(float(lst[pos] or 0.0)) > EPS
            for name, pos in rel_positions.items()
        )
        if lst[VectorModelRelationshipEntityID] or rel_has_signal:
            res.state_relationship = StateRelationship.init_from_vector(lst)
            # The entity id is an identifier, not an axis. Vector sums and EMA
            # averages can leave a fraction in its slot, for example 1.38.
            # Such a value names no entity, so it becomes 0 (no entity). An
            # integral value becomes an int, so the typed CycleState.mental_state
            # validates again after a checkpoint reload.
            entity_id = float(lst[VectorModelRelationshipEntityID] or 0.0)
            res.state_relationship.entity_id = int(entity_id) if entity_id.is_integer() else 0
        return res

    @classmethod
    def init_delta_from_list(cls, lst: List[float]):
        return cls.init_from_list(lst)

    def apply_list(self, lst: List[float]):
        for state in [self.appraisal_general, self.appraisal_social, self.state_neurochemical, self.state_core, self.state_emotions, self.state_cognition, self.state_needs, self.state_expression_ext]:
            state.from_vector(lst)

        if self.state_relationship and lst[VectorModelRelationshipEntityID] == self.state_relationship.entity_id:
            self.state_relationship.from_vector(lst)


# ---------- Helpers ----------
def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def _clamp11(x: float) -> float:
    return -1.0 if x < -1.0 else (1.0 if x > 1.0 else x)

def _blend(prev: float, target: float, k: float) -> float:
    return prev * (1.0 - k) + target * k

def _clip(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)

def _median(x: List[float]) -> float:
    n = len(x)
    if n == 0:
        return 0.0
    xs = sorted(x)
    m = n // 2
    if n % 2:
        return xs[m]
    return 0.5 * (xs[m-1] + xs[m])

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

def _derive_expression_universal(em: StateEmotions, core: StateCore,
                                 neu: StateNeurochemical,
                                 ap: AppraisalGeneral) -> StateExpression:
    anger, fear, sadness, joy = em.anger, em.fear, em.sadness, em.joy
    dom, aro = core.dominance, core.arousal
    brake = _clamp01(0.5 * (1.0 + em.propriety_inhibition))  # [-1..1]→[0..1]

    aggressiveness = _clamp01(
        anger * (0.35*_clamp01(dom) + 0.30*_clamp01(0.6*ap.control_self + 0.2*neu.serotonin + 0.2*_clamp01(dom)))
        * (0.60 + 0.40*neu.noradrenaline) * (1.0 - 0.50*brake) * (1.0 - 0.35*ap.bodily_threat)
    )
    submissiveness = _clamp01((0.6*fear + 0.4*em.shame) * (1.0 - _clamp01(dom)) * (0.7 + 0.3*neu.cortisol))
    withdrawal     = _clamp01((0.6*sadness + 0.4*em.shame) * (1.0 - aro) * (1.0 - _clamp01(dom)))
    expressivity   = _clamp01(aro * (0.5 + 0.5*neu.dopamine) * (0.5 + 0.2*joy + 0.2*anger + 0.1*fear))

    return StateExpression(
        aggressiveness=aggressiveness,
        submissiveness=submissiveness,
        withdrawal=withdrawal,
        expressivity=expressivity,
    )

def _derive_expression_ext(
    em: StateEmotions, core: StateCore, neu: StateNeurochemical,
    ap: AppraisalGeneral, soc: AppraisalSocial,
    pers: PersonalityProfile,
    rel: Optional[StateRelationship] = None,
) -> StateExpressionExt:
    base = _derive_expression_universal(em, core, neu, ap)

    # ------- Personality gains -------
    # Directness comes from extraversion + bluntness_assertive; warmth from agreeableness + warmth_expressive
    assertive_gain   = 0.5*pers.extraversion + 0.5*pers.bluntness_assertive
    warmth_gain      = 0.5*pers.agreeableness + 0.5*pers.warmth_expressive
    inhibition_gain  = 0.5*pers.self_control + 0.5*pers.propriety_baseline
    approach_bias    = 0.6*pers.bas_drive - 0.4*pers.bis_sensitivity   # can be negative
    avoidant_bias    = 0.6*pers.attachment_avoidance + 0.4*pers.bis_sensitivity
    reassurance_need = 0.6*pers.attachment_anxiety + 0.4*pers.affiliation_need

    # ------- Relationship gates (if present) -------
    if rel is not None:
        trust   = _clamp01(0.5 + 0.5*rel.trust)                 # [-1..1]→[0..1]
        warm    = _clamp01(0.5 + 0.5*rel.communication_warmth)
        attach  = rel.attachment_strength                        # [0..1]
        tension = rel.conflict_tension                           # [0..1]
        avers   = rel.aversive_association                       # [0..1]

        tension = rel.conflict_tension if rel is not None else 0.0
        trust = _clamp01(0.5 + 0.5 * rel.trust) if rel is not None else 0.5
        warm = _clamp01(0.5 + 0.5 * rel.communication_warmth) if rel is not None else 0.5
        avers = rel.aversive_association if rel is not None else 0.0
    else:
        trust, warm, attach, tension, avers = 0.5, 0.5, 0.3, 0.0, 0.0

    # ------- Project to extended outlets -------
    # 1) Assertive push: boundary-setting without hostility (likes high trust/warmth)
    ag = base.aggressiveness
    ag_gate = max(0.0, ag - 0.05)  # soft threshold to kill micro-leakage

    assertive_push = _clamp01(
        ag_gate
        * (assertive_gain)  # 0..1 from personality
        * (trust * warm)  # AND gate; 0 kills it
        * (1.0 - tension)  # tension suppresses assertive repair
        * (1.0 - avers)  # aversion suppresses repair
        * (1.0 - 0.4 * em.propriety_inhibition)  # inhibition still matters
    )
    # 2) Confrontive push: escalation under anger + low trust + high tension + high dominance
    confrontive_push = _clamp01(
        base.aggressiveness *
        (0.3 + 0.7*_clamp01(core.dominance)) *
        (0.5 + 0.5*em.anger) *
        (0.3 + 0.7*tension) *
        (0.2 + 0.8*(1.0 - trust)) *
        (0.6 + 0.4*approach_bias) *
        (1.0 - 0.5*inhibition_gain)
    )

    # 3) Affiliative soothe: warmth-based repair (anger low/moderate; tenderness/high trust)
    affiliative_soothe = _clamp01(
        (0.5*em.tenderness + 0.3*em.relief + 0.2*em.guilt) *
        (0.4 + 0.6*warmth_gain) *
        (0.3 + 0.7*trust) * (0.3 + 0.7*attach) *
        (0.6 + 0.4*pers.affiliation_need) *
        (1.0 - 0.4*base.aggressiveness)
    )

    # 4) Appeasement: submissive repair when anxiety high / dominance low / tension high
    appeasement = _clamp01(
        base.submissiveness *
        (0.3 + 0.7*reassurance_need) *
        (0.3 + 0.7*tension) *
        (0.5 + 0.5*(1.0 - trust))
    )

    # 5) Disengage/silence: withdrawal, reinforced by avoidant style or low trust
    #disengage_silence = _clamp01(
    #    base.withdrawal *
    #    (0.4 + 0.6*avoidant_bias) *
    #    (0.3 + 0.7*(1.0 - trust)) *
    #    (0.4 + 0.6*avers)
    #)

    # existing multiplicative path
    withdraw_mul = base.withdrawal * (0.4 + 0.6 * avoidant_bias) * (0.3 + 0.7 * (1.0 - trust)) * (0.4 + 0.6 * avers)

    # additive avoidant disengage path (relationship-driven)
    avoidant_drive = avoidant_bias * (1.0 - trust) * max(avers, tension)
    disengage_silence = _clamp01(
        withdraw_mul + 0.6 * avoidant_drive  # 0.6 is a sane gain; tweak if needed
    )

    # Display amplitude: how visible the emotion becomes (trait expressivity × arousal × context)
    display_amplitude = _clamp01(
        base.expressivity *
        (0.4 + 0.6*(pers.humor_playfulness if em.playfulness > 0.2 else 0.5)) *
        (0.3 + 0.7*(warm if em.joy + em.tenderness > 0.4 else 0.5))
    )
    primitive_approach_pressure = _clamp01(
        ap.salience *
        (
            0.40 * max(0.0, ap.goal_congruence) +
            0.30 * (1.0 - ap.attractor_distance) +
            0.20 * ap.pattern_fit +
            0.10 * ap.control_self
        ) *
        (0.6 + 0.4 * max(0.0, approach_bias))
    )
    primitive_avoidance_pressure = _clamp01(
        ap.salience *
        (
            0.35 * ap.aversive_pressure +
            0.25 * ap.repulsor_distance +
            0.20 * ap.bodily_threat +
            0.20 * ap.body_signal
        ) *
        (0.6 + 0.4 * avoidant_bias)
    )

    # fix

    # 1) sadness/shame-driven path (as you had)
    withdraw_mul = base.withdrawal * (0.4 + 0.6 * avoidant_bias) \
                   * (0.3 + 0.7 * (1.0 - trust)) \
                   * (0.4 + 0.6 * avers)

    # 2) relationship-driven avoidant path (independent of sadness)
    #    make it strong enough to matter in low-trust, high-aversion contexts
    avoidant_drive = avoidant_bias * (1.0 - trust) * max(avers, tension)

    # 3) supportive context features for disengagement:
    #    low warmth and low dominance both encourage silence
    low_warmth = (1.0 - warm)
    low_dominant = (1.0 - _clamp01(core.dominance))

    disengage_silence = _clamp01(
        0.50 * withdraw_mul +  # keep some contribution from sadness/shame route
        1.20 * avoidant_drive +  # make the avoidant lane carry more weight
        0.25 * (1.0 - trust) +  # directly reward distrust
        0.20 * avers +  # and aversive association
        0.20 * low_warmth +  # cold vibe → more silence
        0.20 * low_dominant  # low dominance → more silence
    )

    return StateExpressionExt(
        assertive_push=assertive_push,
        confrontive_push=confrontive_push,
        affiliative_soothe=affiliative_soothe,
        appeasement=appeasement,
        disengage_silence=disengage_silence,
        display_amplitude=display_amplitude,
        primitive_approach_pressure=primitive_approach_pressure,
        primitive_avoidance_pressure=primitive_avoidance_pressure,
    )


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
    pers = current_ms.state_personality

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
    d_valence = (0.90+gv) * ap.goal_congruence - (0.70-ga) * ap.bodily_threat - 0.50 * ap.norm_violation - 0.35 * ap.aversive_pressure
    d_arousal = (0.55+ga) * ap.novelty + (0.90+0.2*nor) * ap.bodily_threat + 0.40 * (1.0 - ap.certainty) + 0.35 * ap.body_signal + 0.25 * ap.salience
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

    anger = _clamp01(
        neg * a *
        (0.5 * ap.norm_violation + 0.5 * ap.agency_other) *
        (0.8 + 0.2 * nor)  # noradrenaline: vigilance/fight bias
    )

    # anger   = _clamp01( neg * a * max(0.0, d) * (ap.agency_other + ap.norm_violation) * (0.8 + 0.2*nor) )
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

    # No appraisal axis derives resentment, jealousy, or boredom. They change
    # only through an explicit appraisal delta, so the derivation keeps the
    # current values. A fresh zero would pull them down on every appraisal.
    prev_emotions = current_ms.state_emotions
    emotions = StateEmotions(
        joy=joy, sadness=sadness, anger=anger, fear=fear, disgust=disgust, surprise=_clamp01(ap.novelty),
        tenderness=tenderness, curiosity=curiosity, shame=shame, guilt=guilt, pride=pride,
        intimacy_arousal=_clamp01(intimacy_1),     # leave as-is unless appraised
        propriety_inhibition=_clamp11(propriety_1),
        amusement=amusement, playfulness=playfulness, relief=relief,
        social_laughter=social_laughter, teasing_edge=teasing_edge,
        resentment=prev_emotions.resentment, jealousy=prev_emotions.jealousy, boredom=prev_emotions.boredom,
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

        state_result.state_relationship = rel

    state_result.state_expression = _derive_expression_universal(state_result.state_emotions, state_result.state_core, current_ms.state_neurochemical, current_ms.appraisal_general)
    state_result.state_expression_ext = _derive_expression_ext(state_result.state_emotions, core, neu, ap, soc, pers, rel)

    vec_a = current_ms.to_list()
    vec_b = state_result.to_list()
    delta = [bi - ai for ai, bi in zip(vec_a, vec_b)]

    state_delta = FullMentalState.init_delta_from_list(delta)
    return state_result, state_delta

def _collect_axis_bounds() -> Dict[int, Tuple[Optional[float], Optional[float]]]:
    """
    Build a map from global vector index -> (ge, le) by scanning your models.
    Only positions that appear in your models will be returned.
    """
    bounds: Dict[int, Tuple[Optional[float], Optional[float]]] = {}
    classes = [
        AppraisalGeneral, AppraisalSocial, StateNeurochemical, StateCore,
        StateEmotions, StateCognition, StateNeeds, StateRelationship
    ]
    for cls in classes:
        if not hasattr(cls, "model_fields"):
            continue
        for _, finfo in cls.model_fields.items():
            extra = (getattr(finfo, "json_schema_extra", None) or {})
            pos = extra.get("vector_position")
            if pos is None:
                continue
            # Pydantic v2 exposes bounds directly on FieldInfo
            ge = -1000
            le = 1000
            for meta in finfo.metadata:
                if hasattr(meta, "ge"): ge = meta.ge
                if hasattr(meta, "le"): le = meta.le
            bounds[int(pos)] = (ge, le)
    return bounds

# The parts of the state that the EMA sums. An appraisal Feature changes
# only these parts: compute_state_delta derives core affect, emotions, and
# the relationship, and the explicit need deltas change the needs. Appraisal
# axes describe one event, and cognition and neurochemistry are set by the
# system, not by appraisal. Their deltas do not add up, so they read their
# field defaults. Old features can carry deltas there; the EMA ignores them.
EMA_STATE_MODELS = (StateCore, StateEmotions, StateNeeds, StateRelationship)


def _collect_axis_defaults() -> Dict[int, float]:
    """Return a map from vector position to the field default, for the axes that the EMA sums.

    The EMA readout puts an axis without deviation at its default. The
    relationship entity id is an identifier, not an axis, so it is left out.
    """
    defaults: Dict[int, float] = {}
    for cls in EMA_STATE_MODELS:
        for _, finfo in cls.model_fields.items():
            pos = int(finfo.json_schema_extra["vector_position"])
            if pos == VectorModelRelationshipEntityID:
                continue
            defaults[pos] = float(finfo.default)
    return defaults


def ema_axis_positions() -> set[int]:
    """Return the vector positions that the EMA sums. See ``EMA_STATE_MODELS``."""
    return set(_collect_axis_defaults())


def _naive_utc(value: datetime.datetime) -> datetime.datetime:
    """Return ``value`` as naive UTC. Older seeds mix aware and naive times."""
    if value.tzinfo is None:
        return value
    return value.astimezone(datetime.timezone.utc).replace(tzinfo=None)


def ema_baselined_normalize(
    history: List[Tuple[datetime.datetime, List[float]]],
    vec_len: int,
    half_life_s: float,
    axis_bounds: Dict[int, Tuple[Optional[float], Optional[float]]],
    axis_defaults: Dict[int, float],
    start_level: Optional[List[float]] = None,
    now: Optional[datetime.datetime] = None,
    sigma_floor: float = MENTAL_STATE_SIGMA_FLOOR,
    tanh_gain: float = MENTAL_STATE_TANH_GAIN,
) -> List[float]:
    """Return the time-aware, EMA-baselined readout of a delta history.

    ``history`` holds ``(time, delta vector)`` pairs. The function sorts them
    by time. For each axis ``i`` in ``axis_defaults``:

    1. ``level[i]`` is the running sum of the deltas (the raw latent level).
    2. ``ema[i]`` follows the level with a half-life of ``half_life_s``
       seconds. Before each sample, ``ema += alpha * (level - ema)`` with
       ``alpha = 1 - exp(-ln 2 * dt / half_life_s)`` for the time ``dt``
       since the previous sample. Then the sample adds its delta to the level.
    3. ``level - ema`` after each sample is one deviation sample.
    4. ``now`` (optional) adds a last catch-up step from the newest sample to
       ``now``. So the deviation fades while no new feature arrives. This is
       the time decay between cycles.
    5. The scale is ``max(std, 1.4826 * MAD, sigma_floor)`` of the deviation
       samples. ``z = tanh(deviation / (2 * scale) * tanh_gain)`` is in (-1, 1).
    6. The readout maps ``z = 0`` to the axis default, ``z > 0`` toward the
       upper bound, and ``z < 0`` toward the lower bound.

    Positions outside ``axis_defaults`` read 0.0. ``start_level`` is an
    optional level before the first sample.

    The old generation updated the EMA after it added the delta. Then an
    event after a gap of several half-lives read as no change at all. This
    port catches up first, so a new event always shows. The old readout also
    put every 0..1 axis at 0.5. This port puts an axis at its default, so an
    emotion without deviation reads 0.0 and a need reads 0.5.
    """
    if not history:
        return [float(axis_defaults[i]) if i in axis_defaults else 0.0 for i in range(vec_len)]

    history = sorted(history, key=lambda p: p[0])
    axes = sorted(axis_defaults)
    level = [0.0] * vec_len if start_level is None else start_level[:]
    ema = level[:]
    last_t = history[0][0]
    ln2 = math.log(2.0)
    half_life = max(EPS, half_life_s)

    def catch_up(dt: float) -> None:
        alpha = 1.0 - math.exp(-ln2 * dt / half_life)
        for i in axes:
            ema[i] += alpha * (level[i] - ema[i])

    dev_samples: Dict[int, List[float]] = {i: [] for i in axes}
    for t, delta in history:
        catch_up(max(0.0, (t - last_t).total_seconds()))
        last_t = t
        for i in axes:
            level[i] += delta[i]
            dev_samples[i].append(level[i] - ema[i])
    if now is not None:
        catch_up(max(0.0, (now - last_t).total_seconds()))

    out = [0.0] * vec_len
    for i in axes:
        samples = dev_samples[i]
        if len(samples) >= 2:
            mean = sum(samples) / len(samples)
            var = sum((d - mean) ** 2 for d in samples) / (len(samples) - 1)
            std = math.sqrt(max(var, 0.0))
        else:
            std = 0.0
        med = _median(samples)
        mad = _median([abs(d - med) for d in samples])
        effective_sigma = max(std, 1.4826 * mad, sigma_floor)
        z = math.tanh((level[i] - ema[i]) / (2.0 * effective_sigma) * tanh_gain)
        ge, le = axis_bounds[i]
        default = axis_defaults[i]
        out[i] = default + z * (le - default) if z >= 0.0 else default + z * (default - ge)
    return out


class MentalStateSample(BaseModel):
    """One causal feature as the EMA mental state reads it.

    ``timestamp`` is the world time of the feature, as naive UTC.
    ``entity_ids`` names the entities that the feature concerns: the entity
    that caused it, and the relationship partner of its appraisal state.
    ``delta`` holds the non-zero positions of the feature's delta vector. The
    relationship entity id slot is never part of it, because an id is not an
    axis that can add up.
    """

    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    entity_ids: List[int] = Field(default_factory=list)
    delta: Dict[int, float] = Field(default_factory=dict)

    def vector(self) -> List[float]:
        """Return the delta as a full vector of ``VectorModelReservedSize`` values."""
        out = [0.0] * VectorModelReservedSize
        for pos, value in self.delta.items():
            out[pos] = value
        return out


def sparse_delta(vector: Sequence[float]) -> Dict[int, float]:
    """Return the non-zero, finite positions of a delta vector, without the entity id slot."""
    return {pos: float(value) for pos, value in enumerate(vector)
            if pos != VectorModelRelationshipEntityID and value and math.isfinite(value)}


def feature_state_sample(feature: "Feature") -> MentalStateSample:
    """Return the EMA sample of one ``Feature`` record.

    The partner of the appraisal state is the integral, positive value in
    the relationship entity id slot of ``mental_state_appraisal_vec``.
    """
    entity_ids = [feature.entity_id] if feature.entity_id else []
    partner = float(feature.mental_state_appraisal_vec[VectorModelRelationshipEntityID] or 0.0)
    if partner > 0 and partner.is_integer() and int(partner) not in entity_ids:
        entity_ids.append(int(partner))
    return MentalStateSample(timestamp=_naive_utc(feature.timestamp_world_begin), entity_ids=entity_ids,
                             delta=sparse_delta(feature.mental_state_delta_vec))


def features_to_history(features: List["Feature"]) -> List[MentalStateSample]:
    """Return the EMA samples of the causal features, oldest first.

    Only causal features count: they entered the story. A feature without a
    delta still counts, because its deviation sample shapes the scale.
    The old ``_compute_mental_state`` also kept only the features of one
    partner, with the id 2 written into the code. Here the whole history
    sets the overall state, and :func:`compute_relationship_state` filters
    by entity.
    """
    samples = [feature_state_sample(feature) for feature in features if feature.causal]
    return sorted(samples, key=lambda sample: sample.timestamp)


def reference_half_life_seconds(ego_strength: Optional[float]) -> float:
    """Return the EMA half-life that ego strength sets, in seconds.

    The old ``cognitive_cycle`` scaled the reference time frame by
    ``|1 - ego_strength|`` of the previous state. A strong ego therefore
    reacts to the newest events, and a weak ego keeps a long memory of mood.
    ``ego_strength`` is None before the first cycle. Then the full frame applies.
    """
    minutes = float(MENTAL_STATE_REFERENCE_MINUTES)
    if ego_strength is not None:
        minutes *= abs(1.0 - ego_strength)
    return max(1.0, minutes * 60.0)


def compute_mental_state(
    samples: List[MentalStateSample],
    *,
    now: datetime.datetime,
    half_life_s: float,
    personality: PersonalityProfile,
) -> FullMentalState:
    """Return the EMA-normalized mental state of a sample history.

    This ports ``_compute_mental_state`` of the old ``ghost_codelets``.
    ``samples`` are the causal features (see :func:`features_to_history`).
    For each axis ``i`` of ``EMA_STATE_MODELS`` and the samples ``k`` in
    time order:

    - ``level_k = level_(k-1) + delta_k[i]``, the running sum of the deltas;
    - ``ema_k = ema_(k-1) + alpha_k * (level_(k-1) - ema_(k-1))`` with
      ``alpha_k = 1 - 2 ** (-dt_k / half_life_s)``, for the time ``dt_k``
      since the previous sample;
    - ``dev_k = level_k - ema_k``.

    One last catch-up step moves the EMA from the newest sample to ``now``.
    So the deviation fades while no feature arrives: the time between
    cycles has an effect. ``scale = max(std(dev), 1.4826 * MAD(dev),
    MENTAL_STATE_SIGMA_FLOOR)``, and ``z = tanh(MENTAL_STATE_TANH_GAIN *
    dev_now / (2 * scale))``. The readout is ``default + z * (upper -
    default)`` for ``z >= 0`` and ``default + z * (default - lower)``
    otherwise. :func:`ema_baselined_normalize` holds the details.

    The other parts of the state read their field defaults. The state gets
    the stable ``personality`` of the persona. The relationship part stays
    empty: :func:`compute_relationship_state` computes it for one entity.
    Call :func:`derive_expression` after the relationship is set, so the
    expression readouts match the state.
    """
    vector = ema_baselined_normalize(
        [(sample.timestamp, sample.vector()) for sample in samples],
        VectorModelReservedSize, half_life_s, _collect_axis_bounds(), _collect_axis_defaults(), now=now)
    state = FullMentalState(state_personality=personality.model_copy())
    for part in (state.state_core, state.state_emotions, state.state_needs):
        part.from_vector(vector)
    return state


def compute_latent_state(
    samples: List[MentalStateSample],
    *,
    personality: PersonalityProfile,
    entity_id: Optional[int] = None,
) -> FullMentalState:
    """Return the raw latent state: the field defaults plus the sum of the deltas.

    This is the ``level`` of :func:`compute_mental_state`, before the EMA
    baseline. Appraisal works on it: the delta of a new appraisal is the
    derived state minus this state. So the sum of all deltas stays the real
    latent state, as the ``mental_state_delta_vec`` field of ``Feature``
    describes. Each axis is clamped to its field bounds.

    Core affect, emotions, and needs sum over all samples. The relationship
    sums over the samples of ``entity_id`` only. Without ``entity_id`` the
    state has no relationship. Other parts keep their field defaults.
    """
    bounds = _collect_axis_bounds()
    relationship_axes = set(StateRelationship.get_vector_positions().values())
    level = FullMentalState().to_list()
    partner = [0.0] * VectorModelReservedSize
    axes = ema_axis_positions()
    for sample in samples:
        own = entity_id is not None and entity_id in sample.entity_ids
        for pos, value in sample.delta.items():
            if pos in relationship_axes:
                if own:
                    partner[pos] += value
            elif pos in axes:
                level[pos] += value
    for pos in axes:
        low, high = bounds[pos]
        level[pos] = _clip(level[pos], low, high)
        partner[pos] = _clip(partner[pos], low, high)
    state = FullMentalState(state_personality=personality.model_copy())
    for part in (state.state_core, state.state_emotions, state.state_needs):
        part.from_vector(level)
    if entity_id is not None:
        state.state_relationship = StateRelationship.init_from_vector(partner)
        state.state_relationship.entity_id = entity_id
    return state


def derive_expression(state: FullMentalState) -> None:
    """Set the expression readouts of ``state`` from its other parts.

    The EMA does not sum expression axes. The readouts are a function of
    the emotions, core affect, neurochemistry, appraisal axes, personality,
    and relationship, as in ``compute_state_delta``. The function changes
    ``state`` in place.
    """
    state.state_expression = _derive_expression_universal(
        state.state_emotions, state.state_core, state.state_neurochemical, state.appraisal_general)
    state.state_expression_ext = _derive_expression_ext(
        state.state_emotions, state.state_core, state.state_neurochemical, state.appraisal_general,
        state.appraisal_social, state.state_personality, state.state_relationship)


def compute_relationship_state(
    samples: List[MentalStateSample],
    entity_id: int,
    *,
    now: datetime.datetime,
    half_life_s: float,
) -> StateRelationship:
    """Return the EMA relationship state toward one entity.

    Only the samples that concern ``entity_id`` count. The readout uses the
    same EMA as :func:`compute_mental_state`, and reads the relationship axes.
    """
    own = [(sample.timestamp, sample.vector()) for sample in samples if entity_id in sample.entity_ids]
    vector = ema_baselined_normalize(own, VectorModelReservedSize, half_life_s, _collect_axis_bounds(),
                                     _collect_axis_defaults(), now=now)
    relationship = StateRelationship.init_from_vector(vector)
    relationship.entity_id = entity_id
    return relationship


# Fix your earlier bug: make sure FullMentalState.init_from_list returns the instance.
def _init_ms_from_vec(vec: List[float]) -> "FullMentalState":
    ms = FullMentalState()
    for state in [
        ms.appraisal_general, ms.appraisal_social, ms.state_neurochemical,
        ms.state_core, ms.state_emotions, ms.state_cognition, ms.state_needs
    ]:
        state.from_vector(vec)
    return ms

def create_empty_ms_vector():
    return [0] * VectorModelReservedSize

def create_empty_state():
    return FullMentalState.init_from_list(create_empty_ms_vector())


_REL_AXIS_COMM_WARMTH = StateRelationship.model_fields["communication_warmth"].json_schema_extra["vector_position"]
_REL_AXIS_CONFLICT = StateRelationship.model_fields["conflict_tension"].json_schema_extra["vector_position"]
_REL_AXIS_TRUST = StateRelationship.model_fields["trust"].json_schema_extra["vector_position"]
_REL_AXIS_CORE_VALENCE = StateCore.model_fields["valence"].json_schema_extra["vector_position"]


class PartnerExpectationProfile(BaseModel):
    """What the companion expects from one partner, from the interaction history.

    ``baseline_quality`` and ``recent_quality`` are in [-1, 1].
    ``disappointment_pressure`` and ``positive_surprise_bias`` are in [0, 1].
    """

    baseline_quality: float = 0.0
    recent_quality: float = 0.0
    disappointment_pressure: float = 0.0
    positive_surprise_bias: float = 0.0
    sample_count: int = 0


def _relationship_signal(sample: MentalStateSample) -> float:
    """Return a coarse interaction quality in [-1, 1] from the relationship and core deltas."""
    delta = sample.vector()
    signal = (0.45 * delta[_REL_AXIS_COMM_WARMTH] + 0.35 * delta[_REL_AXIS_TRUST]
              - 0.70 * delta[_REL_AXIS_CONFLICT] + 0.20 * delta[_REL_AXIS_CORE_VALENCE])
    return _clamp11(signal)


def compute_partner_expectation_profile(
    samples: List[MentalStateSample],
    entity_id: int,
    reference_timeframe_minutes: int = 60 * 24,
    *,
    baseline_alpha_up: float = 0.30,
    baseline_alpha_down: float = 0.06,
    recent_half_life_minutes: float = 120.0,
) -> PartnerExpectationProfile:
    """Return the expectation profile of the companion toward one partner.

    Only samples that concern ``entity_id`` and lie inside the reference time
    frame (measured from the newest of them) count. The baseline moves
    asymmetrically: a good interaction raises it fast, a bad one lowers it
    slowly. So the partner becomes harder to impress and easier to
    disappoint over time.
    """
    own = [sample for sample in samples if entity_id in sample.entity_ids]
    if not own:
        return PartnerExpectationProfile()
    newest = max(sample.timestamp for sample in own)
    horizon_s = max(60.0, float(reference_timeframe_minutes) * 60.0)
    recent_half_life_s = max(30.0, float(recent_half_life_minutes) * 60.0)
    selected = sorted((sample for sample in own if (newest - sample.timestamp).total_seconds() <= horizon_s),
                      key=lambda sample: sample.timestamp)
    ln2 = math.log(2.0)
    baseline = 0.0
    weighted_sum = 0.0
    weighted_den = 0.0
    for sample in selected:
        age_s = max(0.0, (newest - sample.timestamp).total_seconds())
        w_recent = math.exp(-ln2 * age_s / recent_half_life_s)
        signal = _relationship_signal(sample)
        alpha = baseline_alpha_up if signal >= baseline else baseline_alpha_down
        alpha = _clip(alpha * (0.75 + 0.25 * w_recent), 0.01, 0.60)
        baseline = _blend(baseline, signal, alpha)
        weighted_sum += signal * w_recent
        weighted_den += w_recent
    recent_quality = weighted_sum / max(EPS, weighted_den)
    baseline_quality = _clamp11(baseline)
    return PartnerExpectationProfile(
        baseline_quality=baseline_quality,
        recent_quality=_clamp11(recent_quality),
        disappointment_pressure=_clamp01(0.90 * max(0.0, baseline_quality - recent_quality)
                                         + 0.25 * max(0.0, baseline_quality)),
        positive_surprise_bias=_clamp01(0.70 * max(0.0, recent_quality - baseline_quality)
                                        * (1.0 - 0.65 * max(0.0, baseline_quality))),
        sample_count=len(selected),
    )



def compute_attention_bias(ms: FullMentalState) -> Tuple[float, float]:
    """
    Compute internal attention-selection biases from latent/active state.
    Returns:
        valence_focus_bias ∈ [-1, 1] : whether the mind drifts toward negative or positive stimuli
        salience_bias ∈ [0, 1]       : how deterministically attention picks top-salient percepts
                                       (0=random, 0.5=mixed sampler, 1=top-locked)
    """
    core = ms.state_core
    emo = ms.state_emotions
    cog = ms.state_cognition
    neu = ms.state_neurochemical
    needs = ms.state_needs
    pers = ms.state_personality
    expr = ms.state_expression
    expr_ext = ms.state_expression_ext
    rel = ms.state_relationship

    # ---------- VALENCE-FOCUS BIAS ----------
    # Core valence drives mood polarity, dominance adds confidence, arousal adds reactivity.
    # Emotions: sadness/fear pull attention toward negatives; joy/pride toward positives.
    # Personality: neuroticism amplifies negativity; agreeableness/openness broaden positivity.
    v = core.valence
    d = core.dominance
    a = core.arousal

    pos_affect = (emo.joy + emo.tenderness + emo.pride + 0.3 * emo.relief) / 4.3
    neg_affect = (emo.fear + emo.anger + emo.sadness + emo.shame + emo.guilt + emo.disgust) / 6.0

    mood_tone = v + 0.4 * (pos_affect - neg_affect)
    control_confidence = 0.5 * d - 0.3 * emo.fear

    # personality corrections (temperament)
    p = pers
    trait_mod = (
        -0.5 * (p.neuroticism - 0.5) +   # higher neuroticism → negativity bias
        0.4 * (p.agreeableness - 0.5) +  # warmth → positive skew
        0.3 * (p.openness - 0.5)         # openness → curiosity = positive framing
    )

    # neurochemical modulation (mild dampening)
    nc_mod = 0.5 * (
        0.3 * (neu.dopamine - 0.5)       # dopamine = reward sensitivity
        - 0.3 * (neu.cortisol - 0.5)     # cortisol = threat vigilance
        - 0.2 * (neu.serotonin - 0.5)    # serotonin = stabilization (reduces bias)
    )

    valence_focus_bias = mood_tone + control_confidence + trait_mod + nc_mod
    valence_focus_bias = _clamp11(valence_focus_bias)

    # ---------- SALIENCE BIAS ----------
    # Driven by arousal (energy), noradrenaline (focus), willpower/ego_strength, and emotional drive.
    # Also modulated by personality (extraversion & conscientiousness), expression amplitude (visibility),
    # and needs (relevance, autonomy, connection). High stress/anger → higher focus;
    # playfulness/curiosity → more stochastic exploration.
    a = core.arousal
    nor = neu.noradrenaline
    dop = neu.dopamine

    cognitive_drive = 0.5 * (cog.willpower + cog.ego_strength - abs(cog.mental_aperture))
    positive_focus = 0.3 * emo.anger + 0.4 * emo.pride + 0.3 * emo.curiosity
    diffuse_focus = 0.3 * emo.playfulness + 0.3 * emo.relief + 0.2 * emo.social_laughter

    need_urgency = 0.3 * (1.0 - needs.relevance) + 0.2 * (1.0 - needs.connection) + 0.2 * (1.0 - needs.energy_stability)
    personality_focus = (
        0.3 * (pers.extraversion - 0.5) +
        0.3 * (pers.conscientiousness - 0.5) -
        0.2 * (pers.openness - 0.5)
    )

    # combine factors
    salience_bias = (
        0.4 * a +
        0.25 * nor +
        0.15 * dop +
        0.15 * cognitive_drive +
        0.10 * positive_focus -
        0.15 * diffuse_focus +
        0.15 * need_urgency +
        0.15 * personality_focus
    )

    # relational safety lowers urgency slightly
    if rel is not None:
        salience_bias *= (0.9 + 0.1 * (1.0 - rel.conflict_tension))

    # neurochemical dampening (keep moderate range)
    salience_bias *= (0.7 + 0.3 * (1.0 - 0.5 * (neu.serotonin + neu.oxytocin)))

    salience_bias = _clamp01(salience_bias)

    return valence_focus_bias, salience_bias
