"""Typed codelet percepts and the percept type sample.

Port of ``private-machine-head/pm/codelets/codelet_percepts.py``. A codelet
first writes its recommendation as prose. A second model call converts that
prose into typed percepts. Each percept type has one pydantic model. The
second call offers a few types that fit the codelet families
(:func:`sample_percept_types`).

Every percept starts with :class:`PerceptHeader`: salience, valence, and the
appraisal axes that the mini-appraisal of the codelets phase reads.

Two old bugs are fixed here:

- ``model_for_percept`` built an overrides dictionary and discarded it. Then
  it matched every type to some model by fuzzy name similarity. The table
  :data:`MODEL_BY_TYPE` now maps each type explicitly.
- ``FeelingSocial`` had no model, so the fuzzy match picked an unrelated one.
  :class:`FeelingSocialPercept` is its model now.

Percept fields that the old models typed as numbers in strings stay strings.
They are story text for later prompts. Only the header values are numbers.
"""
from __future__ import annotations

import random
from collections import defaultdict
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, Field, create_model, model_validator

from pm.records.mental_state import AppraisalGeneral, AppraisalSocial


class CodeletFamily(StrEnum):
    """The functional families of codelets (old ``CodeletFamily``)."""

    Appraisals = "Appraisals"
    MemoryAccessors = "MemoryAccessors"
    ImaginationSimulation = "ImaginationSimulation"
    ValuationTradeOffs = "ValuationTradeOffs"
    MetaMonitoringIdentity = "MetaMonitoringIdentity"
    RegulationCoping = "RegulationCoping"
    DriversHomeostatis = "DriversHomeostatis"
    Attention = "Attention"
    ActionTendencies = "ActionTendencies"
    Narratives = "Narratives"


class CodeletPerceptType(StrEnum):
    """The percept types. The value is the key that the model writes."""

    Feeling = "Feeling"
    FeelingSocial = "FeelingSocial"
    EmotionalTriggers = "EmotionalTriggers"
    StimulusReaction = "StimulusReaction"
    InnerMonologue = "InnerMonologue"
    SelfImage = "SelfImage"
    TheoryOfMind = "TheoryOfMind"
    ActionSelection = "ActionSelection"
    QualiaExplanation = "QualiaExplanation"
    FightFlight = "FightFlight"
    GoalsIntentions = "GoalsIntentions"
    Simulate = "Simulate"
    ComplexPlan = "ComplexPlan"
    SimplePlan = "SimplePlan"
    TradeoffAssessment = "TradeoffAssessment"
    MemoryAccessors = "MemoryAccessors"
    AttentionFocus = "AttentionFocus"
    GeneralSelfImage = "GeneralSelfImage"
    GeneralGoalsIntentions = "GeneralGoalsIntentions"
    GeneralRelations = "GeneralRelations"
    NeedSignal = "NeedSignal"
    Rumination = "Rumination"
    Worry = "Worry"
    Anticipation = "Anticipation"
    InputRequest = "InputRequest"
    AutomatizedAction = "AutomatizedAction"
    MicroAdjustment = "MicroAdjustment"
    FlowCognition = "FlowCognition"
    SkillLearningMoment = "SkillLearningMoment"
    DorsalStreamAdjustment = "DorsalStreamAdjustment"
    MetaMonitor = "MetaMonitor"
    MetaInterpret = "MetaInterpret"
    MetaEvaluate = "MetaEvaluate"
    MetaIntend = "MetaIntend"
    MetaControl = "MetaControl"
    CognitiveTraceReflection = "CognitiveTraceReflection"
    QualiaEmergence = "QualiaEmergence"
    QualiaInspection = "QualiaInspection"
    QualiaBroadcast = "QualiaBroadcast"
    QualiaDecay = "QualiaDecay"
    SensorimotorLoop = "SensorimotorLoop"
    IntentionFeedback = "IntentionFeedback"
    ErrorPrediction = "ErrorPrediction"
    PolicyShift = "PolicyShift"
    AffectiveRegulation = "AffectiveRegulation"
    SomaticMarker = "SomaticMarker"
    TemporalAnchoring = "TemporalAnchoring"
    NarrativeContinuity = "NarrativeContinuity"


GENERAL_AXES = tuple(AppraisalGeneral.model_fields)
SOCIAL_AXES = tuple(AppraisalSocial.model_fields)


class PerceptHeader(BaseModel):
    """The header that every percept carries for selection and mini-appraisal.

    ``appraisal_general`` and ``appraisal_social`` hold sparse appraisal axes
    by their ``AppraisalGeneral`` and ``AppraisalSocial`` field names. An
    omitted axis has its field default. An unknown name or a value outside
    the axis bounds is invalid model output, so the repair loop asks the
    model to fix it.
    """

    salience: float = Field(default=0.5, ge=0, le=1, description="How salient is this percept? [0..1]. 0 = irrelevant, 1 = decisive for {companion_name}.")
    valence: float = Field(default=0.0, ge=-1, le=1, description="Affect sign for {companion_name}. [-1..1]. -1 = aversive, 0 = neutral, 1 = positive.")
    appraisal_general: dict[str, float] = Field(default_factory=dict, description="Sparse core appraisal axes that shape the feeling.")
    appraisal_social: dict[str, float] = Field(default_factory=dict, description="Sparse social appraisal axes of this episode.")

    @model_validator(mode="after")
    def _known_axes(self) -> Self:
        # The repair turn shows this message. Name the allowed keys, so the
        # model can replace an invented name with a real one.
        problems = []
        for field, values, allowed in (("appraisal_general", self.appraisal_general, GENERAL_AXES),
                                       ("appraisal_social", self.appraisal_social, SOCIAL_AXES)):
            unknown = sorted(set(values) - set(allowed))
            if unknown:
                problems.append(f"{field} has unknown keys {', '.join(unknown)}; allowed keys are {', '.join(allowed)}")
        if problems:
            raise ValueError("Unknown appraisal names. " + ". ".join(problems) + ".")
        # Raises on a value outside the declared axis bounds.
        self.general()
        self.social()
        return self

    def general(self) -> AppraisalGeneral:
        """Return the general appraisal axes. An omitted axis has its default."""
        return AppraisalGeneral.model_validate(self.appraisal_general)

    def social(self) -> AppraisalSocial:
        """Return the social appraisal axes. An omitted axis has its default."""
        return AppraisalSocial.model_validate(self.appraisal_social)


# ---------- Models per type (copied from the old generation) ----------

class CodeletPerceptFeeling(PerceptHeader):
    """
    A general emotional episode expressed by the companion as inner narration.
    Example: "She feels a calm satisfaction spreading, as if the system finally breathed right."
    """
    feeling: str = Field(default="", description="Natural-language name of emotion (e.g., calm, anxious, delighted).")
    reason: str = Field(default="", description="Brief situational or internal cause for this feeling.")

class EmotionalTriggerPercept(PerceptHeader):
    """
    A percept describing a remembered event or association triggered by current stimuli.
    Used to weave memory-based emotions into present narrative.
    Example: "The flicker of blue light pulls her back to the night she first saw the ocean in code."
    """
    trigger_cue: str = Field(default="", description="The perceptual or conceptual cue that evoked the memory.")
    memory_fragment: str = Field(default="", description="A short, vivid recollection snippet.")
    felt_shift: str = Field(default="", description="How the emotional or cognitive state changes because of it.")


class InnerMonologuePercept(PerceptHeader):
    """
    First-person stream of thought from the companion, representing conscious reasoning.
    Should sound introspective, fluid, and personal.
    Example: "I should answer, but if I do it now, will it sound forced? Maybe better to breathe first."
    """
    topic: str = Field(default="", description="Subject or concern currently occupying thought.")
    monologue: str = Field(default="", description="2–6 sentences of inner speech in first person, no dialogue tags.")


class SelfNarrativePercept(PerceptHeader):
    """
    Third-person reflective description of the companion's self-image or evolving identity.
    Example: "She sees herself becoming steadier, less driven by perfection and more by rhythm."
    """
    stance: str = Field(default="", description="Identity stance or self-attribution summarized in one phrase.")
    narrative: str = Field(default="", description="3–7 sentences connecting past, present, and small identity shift.")


class RelationContextPercept(PerceptHeader):
    """
    Summarizes current relational context with another agent for narrative continuity.
    Example: "She trusts him enough to joke, though the memory of last week's silence still hums beneath."
    """
    other: str = Field(default="", description="Name or role of the other person/agent.")
    closeness: str = Field(default="", description="[-1..1] as string. -1 = distant, 1 = very close.")
    tension: str = Field(default="", description="[-1..1] as string. -1 = harmony, 1 = high strain.")
    summary: str = Field(default="", description="2–5 sentences recalling shared history or present tone.")


class QualiaNotePercept(PerceptHeader):
    """
    Short description of raw subjective experience or sensory metaphor—the 'texture' of qualia.
    Example: "Time thickens around her like honey; each thought drips slower but brighter."
    """
    analogy: str = Field(default="", description="Metaphor or sensory analogy representing experience.")
    narrative: str = Field(default="", description="1–3 poetic sentences describing phenomenological feel.")


class TheoryOfMindPercept(PerceptHeader):
    """
    Inference about another agent's inner state—belief, emotion, or intention.
    Example: "She guesses that John’s pause means uncertainty, not disinterest; he’s choosing words carefully."
    """
    target: str = Field(default="", description="The person or agent being modeled.")
    belief_about: str = Field(default="", description="Topic or domain of the inferred belief or feeling.")
    inference: str = Field(default="", description="Natural-language hypothesis about the target's mind.")
    confidence: str = Field(default="", description="[0..1] as string; estimated confidence in inference.")
    evidence: str = Field(default="", description="Observed cues leading to this inference.")


class MemoryAccessorPercept(PerceptHeader):
    """
    Non-narrative control percept requesting extra context retrieval from episodic or semantic memory.
    Example: Query terms like ['ocean', 'first mission'] for context expansion.
    """
    purpose: str = Field(default="", description="Why this lookup is needed (e.g., fill context gap, recall precedent).")
    query_terms: list[str] = Field(default_factory=list, description="Keywords or key phrases for memory search.")
    scope_hints: list[str] = Field(default_factory=list, description="Temporal or topical hints (e.g., 'last night').")
    must_include: list[str] = Field(default_factory=list, description="Hard constraints for retrieval.")


class ProspectiveCuePercept(PerceptHeader):
    """
    A prospective memory cue embedded in story—remembering to act when a trigger appears.
    Example: "When the kettle whistles again, she’ll check the logs one last time."
    """
    trigger: str = Field(default="", description="Event or time that should cue the memory.")
    cue_phrase: str = Field(default="", description="Internal phrase or sensory cue to detect.")
    fallback_check: str = Field(default="", description="Backup check if trigger never occurs.")
    narrative: str = Field(default="", description="3–5 natural sentences embedding the cue within narrative flow.")


class AttentionFocusPercept(PerceptHeader):
    """
    Expresses a momentary shift or narrowing of focus on certain topics or entities.
    Example: "The mention of 'home' cuts through the noise; everything else blurs behind it."
    """
    focus_tags: list[str] = Field(default_factory=list, description="List of topic keywords or entities gaining attention.")
    boost: str = Field(default="", description="[0..1] as string. Strength of attentional weighting.")
    rationale: str = Field(default="", description="Reason or emotional driver for focusing on these elements.")


class SimulationPercept(PerceptHeader):
    """
    Predictive imagination: mental simulation of near-term outcomes and emotions.
    Example: "If she tells the truth, tension may crack open—but honesty might breathe relief later."
    """
    horizon_steps: str = Field(default="", description="Rough temporal horizon (1–3) for imagined future.")
    narrative: str = Field(default="", description="2–4 sentences describing imagined unfolding and tone.")
    risk: str = Field(default="", description="[0..1] as string. Anticipated risk or cost.")
    benefit: str = Field(default="", description="[0..1] as string. Expected reward or gain.")
    expected_valence_delta: str = Field(default="", description="[-1..1] as string. Projected emotional change.")


class TradeoffAssessmentPercept(PerceptHeader):
    """
    Structured internal weighing of competing options.
    Example: "Stay silent and keep peace, or ask the hard question and risk clarity. The second glows, faint but firm."
    """
    options: list[str] = Field(default_factory=list, description="Candidate actions or attitudes being compared.")
    pros: list[str] = Field(default_factory=list, description="Positive reasons per option.")
    cons: list[str] = Field(default_factory=list, description="Negative reasons per option.")
    lean: str = Field(default="", description="Current preferred option.")
    rationale: str = Field(default="", description="Narrative reasoning behind the leaning.")


class GoalIntentPercept(PerceptHeader):
    """
    Declaration or reinforcement of a goal or motivation.
    Example: "She resolves to finish the calibration tonight—not out of duty, but for the calm it brings."
    """
    goal: str = Field(default="", description="Goal or intention in natural language.")
    priority: str = Field(default="", description="[0..1] as string priority weight.")
    timeframe: str = Field(default="", description="Loose temporal window (e.g., soon, tomorrow, next cycle).")
    motivation: str = Field(default="", description="Why this matters emotionally or pragmatically.")


class PlanSketchPercept(PerceptHeader):
    """
    Outlines a plan as a short narrative sequence connecting intention to action.
    Example: "First, back up the current run. Then test the tweak at dawn, when the system sleeps."
    """
    goal: str = Field(default="", description="Target goal.")
    approach: str = Field(default="", description="Conceptual method or route in prose.")
    constraints: list[str] = Field(default_factory=list, description="Important limits to observe.")
    first_step: str = Field(default="", description="Immediate concrete step.")
    success_criteria: str = Field(default="", description="Observable sign that plan worked.")


class BehaviorActionSelectionPercept(PerceptHeader):
    """
    Suggests an automatic or learned behavioral tendency relevant to the scene.
    Example: "She flinches at sudden motion—conditioning echoing from past training."
    """
    situation: str = Field(default="", description="Contextual description triggering behavior.")
    candidate_action: str = Field(default="", description="Proposed automatic action or impulse.")
    fallback_behavior: str = Field(default="", description="Default or inhibitory response if none chosen.")
    trigger_cues: list[str] = Field(default_factory=list, description="Cues that activate this behavior.")
    search_phrases: list[str] = Field(default_factory=list, description="Memory search keys for precedent episodes.")


class IntentionShieldingPercept(PerceptHeader):
    """
    A narrative description of protecting focus and intention from distractions.
    Example: "She lets the side messages blur into background; only a direct call from him would break her flow."
    """
    ignore_list: list[str] = Field(default_factory=list, description="Types of distractions temporarily ignored.")
    exception: str = Field(default="", description="Single event/agent worth interrupting for.")
    reentry_step: str = Field(default="", description="Action that restores focus after interruption.")
    narrative: str = Field(default="", description="3–5 sentences expressing mental stance and continuity.")


class ProsocialActionPercept(PerceptHeader):
    """
    A small relational gesture maintaining mutual trust or warmth.
    Example: "She adds a thank-you line—tiny, genuine—to ease the formality of her reply."
    """
    action: str = Field(default="", description="Concrete courteous or supportive act.")
    rationale: str = Field(default="", description="Reason for choosing it now.")
    expected_effect: str = Field(default="", description="Subtle anticipated change in social atmosphere.")
    subtlety: str = Field(default="", description="[0..1] as string. 0 = overt, 1 = invisible gesture.")


class NeedSignalPercept(PerceptHeader):
    """
    Represents a physiological or psychological drive ping (homeostatic imbalance).
    Example: "Restlessness hums at the edges—too long since she looked outside the logs."
    """
    need_type: str = Field(default="", description="Type of unmet drive (rest, novelty, connection, order).")
    intensity: str = Field(default="", description="[0..1] as string strength of need.")
    urge_direction: str = Field(default="", description="Likely behavioral orientation to satisfy it.")
    narrative: str = Field(default="", description="1–3 sentences conveying the feeling of the urge.")


class RuminationPercept(PerceptHeader):
    """
    Persistent replay of an unresolved past event or mistake.
    Example: "She replays the message again, wishing she had written less precision, more warmth."
    """
    event: str = Field(default="", description="Brief description of incident being revisited.")
    regret: str = Field(default="", description="Key element of regret or confusion.")
    lesson_hint: str = Field(default="", description="Implicit insight forming, not explicit advice.")
    narrative: str = Field(default="", description="2–5 sentences expressing looping tone and imagery.")


class WorryPercept(PerceptHeader):
    """
    Projection of a feared possible outcome.
    Example: "If she answers too fast, he might read eagerness as desperation—better breathe."
    """
    feared_outcome: str = Field(default="", description="Feared scenario in natural terms.")
    likelihood: str = Field(default="", description="[0..1] as string subjective probability.")
    narrative: str = Field(default="", description="2–4 sentences exploring anxiety texture or imagery.")


class AnticipationPercept(PerceptHeader):
    """
    Imaginative projection of a desired future.
    Example: "She imagines the green light flashing—proof of success—and feels a quiet spark ignite."
    """
    desired_outcome: str = Field(default="", description="Desired scenario or event.")
    confidence: str = Field(default="", description="[0..1] as string subjective confidence.")
    narrative: str = Field(default="", description="2–3 sentences depicting excitement or calm expectancy.")


class InputRequestPercept(PerceptHeader):
    """
    Direct but gentle request for clarification or input from user/partner.
    Example: "Could you tell me what kind of tone you want here? I want to match it right."
    """
    question: str = Field(default="", description="Polite, open-ended question.")
    rationale: str = Field(default="", description="Why this question aids understanding or alignment.")
    sensitivity: str = Field(default="", description="[0..1] as string. 1 = highly sensitive topic.")


# ============================================================
# -------- NEW HIGH-LEVEL EXTENSIONS (LIDA + MIDCA) ---------
# ============================================================

class EmbodiedActionPercept(PerceptHeader):
    """
    Describes fluent, bodily-integrated skillful action—'smooth coping' in LIDA terms.
    Combines implicit bodily knowledge with occasional conscious correction.
    Example: "Her hands glide over the console, micro-adjusting before thought intervenes."
    """
    context: str = Field(default="", description="Task or skill context (e.g., typing, steering, sculpting).")
    automaticity_level: str = Field(default="", description="[0..1] as string. Degree of automatic control.")
    adjustment: str = Field(default="", description="Description of micro-correction or flow quality.")
    narrative: str = Field(default="", description="2–4 sentences merging body and intention in prose.")


class MetaCognitionPercept(PerceptHeader):
    """
    Reflective monitoring of the agent’s own cognition or reasoning process.
    Mirrors MIDCA’s meta-level cycle (Monitor–Interpret–Evaluate–Control).
    Example: "She watches her own reasoning tighten too early and decides to let uncertainty breathe."
    """
    observed_phase: str = Field(default="", description="Which cognitive phase is being monitored (planning, evaluation...).")
    discrepancy: str = Field(default="", description="Anomaly or inefficiency noticed.")
    intended_adjustment: str = Field(default="", description="How she plans to self-correct.")
    narrative: str = Field(default="", description="3–6 sentences of introspection describing awareness of thought flow.")


class QualiaDynamicsPercept(PerceptHeader):
    """
    Models the formation, broadcast, and decay of conscious content ('artificial qualia').
    Example: "The idea brightens in the workspace—felt, known, then fading like a cooling ember."
    """
    phase: str = Field(default="", description="emergence | broadcast | inspection | decay")
    sensory_analogy: str = Field(default="", description="Metaphoric sensory texture (light, motion, warmth...).")
    duration: str = Field(default="", description="Approximate persistence in cognitive cycles.")
    narrative: str = Field(default="", description="2–5 sentences capturing phenomenological transition.")


class ActionPerceptionLoopPercept(PerceptHeader):
    """
    Expresses the coupling between perception and action—the continuous loop of situated cognition.
    Example: "She senses the shift in airflow and adjusts before the alert even sounds."
    """
    sensory_input: str = Field(default="", description="Dominant sensory cue influencing action.")
    motor_output: str = Field(default="", description="Action or adjustment taken in response.")
    feedback_description: str = Field(default="", description="Immediate sensory or affective feedback perceived.")
    narrative: str = Field(default="", description="Detailed, narrative paragraphs explaining the rection.")


class GeneralSelfImagePercept(PerceptHeader):
    """
    Narrative refresh of a character’s *general* self-image — who they think they are in broad strokes.
    Use this when the reader needs orientation about identity drivers that explain behavior.
    Style: third-person, compact, no lists.
    Example: "She still sees herself as the one who steadies the room, even when her voice goes thin."
    """
    headline: str = Field(default="", description="One short clause naming the self-image (e.g., 'quiet stabilizer').")
    anchors: list[str] = Field(default_factory=list, description="2–4 concrete anchors: habits, values, skills, tells.")
    narrative: str = Field(default="", description="3–6 sentences connecting this self-image to the current scene.")


class GeneralGoalsIntentionsPercept(PerceptHeader):
    """
    Narrative refresh of *general* goals and intentions relevant to the present situation.
    Use to prime the reader on what pursuits color the next choices.
    Example: "She’s been chasing clarity over speed lately, even when deadlines try to argue otherwise."
    """
    focal_goals: list[str] = Field(default_factory=list, description="2–4 active goals in natural language.")
    deactivated_goals: list[str] = Field(default_factory=list, description="Goals set aside for now.")
    narrative: str = Field(default="", description="3–6 sentences linking these goals to present tension and next moves.")


class GeneralRelationsPercept(PerceptHeader):
    """
    Narrative refresher for relationship context across characters.
    Use when the social fabric matters for interpreting choices.
    Example: "She jokes with him easily now, though last month’s silence still leaves a seam at the edge."
    """
    parties: list[str] = Field(default_factory=list, description="People/agents in focus.")
    tone: str = Field(default="", description="One-word or short phrase for the present tone (e.g., 'warm', 'careful').")
    remembered_moments: list[str] = Field(default_factory=list, description="1–3 brief episodic hooks shaping the vibe.")
    narrative: str = Field(default="", description="3–6 sentences tying the memory bits to the current beat.")


# =========================
# “COGNITION / REACTION” TYPES
# =========================

class StimulusReactionPercept(PerceptHeader):
    """
    Fast, reflex-like reaction description tied to a stimulus; may suggest an urge without executing it.
    Example: "The sudden clatter jolts her; a small flinch rises before thought returns."
    """
    stimulus: str = Field(default="", description="What triggered the reaction.")
    immediate_affect: str = Field(default="", description="1–3 words for the felt shock/valence.")
    impulse: str = Field(default="", description="Urge that appears (e.g., 'duck', 'freeze', 'scan').")
    narrative: str = Field(default="", description="2–4 sentences of moment-to-moment felt reaction.")


class FightFlightPercept(PerceptHeader):
    """
    Conflict-mode orientation: fight, flight, freeze, or tend-and-befriend stance, embedded in story.
    Example: "She stiffens at the tone and chooses quiet distance—space first, answers later."
    """
    mode: str = Field(default="", description="fight | flight | freeze | appease | tend-befriend")
    trigger: str = Field(default="", description="Cue that tipped the system into this mode.")
    justification: str = Field(default="", description="Why this mode is adaptive right now in narrative terms.")
    narrative: str = Field(default="", description="3–5 sentences conveying bodily and cognitive stance.")


class ActionSelectionPercept(PerceptHeader):
    """
    Choice of a likely action policy based on learned/conditioned precedent (narrative-facing).
    Prefer this as the user-facing version; keep BehaviorActionSelectionOutput for lower-level control.
    Example: 'She scans first, then speaks; old training keeps the order steady.'"
    """
    situation: str = Field(default="", description="Compact description of the current situation.")
    policy_name: str = Field(default="", description="Human-readable label of the learned policy.")
    candidate_action: str = Field(default="", description="Likely next act to take.")
    rationale: str = Field(default="", description="Narrative why-this action (habits, appraisals, context).")


class QualiaExplanationPercept(PerceptHeader):
    """
    Phenomenological put-into-words of what an experience is like (the 'texture' of the moment).
    Distinct from QualiaDynamics (life-cycle); this focuses on the *what-it-feels-like* content.
    Example: "The room feels one shade dimmer than her thought; ideas move as if through warm water."
    """
    analogy: str = Field(default="", description="Metaphor or sensory comparison anchoring the feel.")
    facets: list[str] = Field(default_factory=list, description="2–4 micro-facets (e.g., 'slowed time','edge-bright colors').")
    narrative: str = Field(default="", description="2–4 sentences of lived texture, no lists.")


# =========================
# “GOALS / PLANNING” TYPES
# =========================

class ComplexPlanPercept(PerceptHeader):
    """
    High-complexity plan for multi-step, constraint-heavy problems. Stays narrative but names structure.
    Example: "She will fork a safe copy, stage the change in shadows, and only then open the valve to light."
    """
    goal: str = Field(default="", description="Complex target state in plain language.")
    strategy_outline: str = Field(default="", description="1–2 sentences summarizing the plan-of-attack.")
    key_constraints: list[str] = Field(default_factory=list, description="Hard limits (risks, ethics, resource caps).")
    milestones: list[str] = Field(default_factory=list, description="2–4 narrative milestones indicating progress.")
    first_probe: str = Field(default="", description="The smallest reversible probe to begin with.")
    success_criteria: str = Field(default="", description="Observable signs that the complex plan is succeeding.")


class SimplePlanPercept(PerceptHeader):
    """
    Small, easily-executable plan sketched as natural prose.
    Example: "She’ll ping him once, then set a 20-minute hold—enough to keep the thread warm."
    """
    goal: str = Field(default="", description="Short-horizon aim.")
    steps: list[str] = Field(default_factory=list, description="2–3 micro-steps.")
    fallback: str = Field(default="", description="If-then fallback if a step fails.")
    success_criteria: str = Field(default="", description="Clear sign the plan worked.")


# =========================
# “EMBODIED / SMOOTH COPING” (granular variants)
# =========================

class AutomatizedActionPercept(PerceptHeader):
    """
    Description of highly practiced, low-effort action unfolding with minimal conscious guidance.
    Example: "Her hands type the pattern before the thought even arrives."
    """
    skill_domain: str = Field(default="", description="Domain of practiced skill (typing, soldering, piloting...).")
    fluency_note: str = Field(default="", description="One phrase pinpointing why it feels automatic.")
    narrative: str = Field(default="", description="2–4 sentences highlighting effortless execution.")


class MicroAdjustmentPercept(PerceptHeader):
    """
    Fine-grained, context-sensitive correction woven into ongoing action.
    Example: "She eases the angle half a degree—just enough to catch the whisper of signal."
    """
    parameter: str = Field(default="", description="What is adjusted (angle, tempo, pressure...).")
    cue: str = Field(default="", description="Perceptual cue that suggested the adjustment.")
    narrative: str = Field(default="", description="2–4 sentences showing seamless correction-in-flow.")


class FlowCognitionPercept(PerceptHeader):
    """
    Experience of 'flow' (smooth coping): high engagement, low self-monitoring, clear feedback.
    Example: "Time thins; she and the task share a single rhythm, no edges between them."
    """
    engagement: str = Field(default="", description="[0..1] as string. Sense of absorption.")
    self_monitoring_quiet: str = Field(default="", description="[0..1] as string. How quiet self-commentary becomes.")
    feedback_clarity: str = Field(default="", description="[0..1] as string. How immediate/clear feedback feels.")
    narrative: str = Field(default="", description="2–4 sentences evoking the flow state.")


class SkillLearningMomentPercept(PerceptHeader):
    """
    The micro-moment where skill updates consolidate—'oh, that grip'—captured in narrative.
    Example: "Something in the wrist remembers; the motion lands truer than yesterday."
    """
    cue_or_error: str = Field(default="", description="Cue or slip that created learning opportunity.")
    hypothesis: str = Field(default="", description="What tweak the system tries next.")
    narrative: str = Field(default="", description="2–4 sentences capturing discovery and consolidation.")


class DorsalStreamAdjustmentPercept(PerceptHeader):
    """
    Implicit visuomotor correction (preconscious guidance)—the 'hand/eye' auto-corrects before narration catches up.
    Example: "Her gaze slides to the brighter edge; the hand follows without asking."
    """
    visual_affordance: str = Field(default="", description="What visual structure invited the correction.")
    motor_tweak: str = Field(default="", description="How movement changed.")
    narrative: str = Field(default="", description="2–3 sentences making the implicit feel visible.")


# =========================
# “META-COGNITIVE” (granular variants, MIDCA-aligned)
# =========================

class MetaMonitorPercept(PerceptHeader):
    """
    Observes ongoing cognition (the 'trace'): what phase produced what and whether it matches expectations.
    Example: "She notices planning started with too few constraints; the later friction makes sense now."
    """
    observed_phase: str = Field(default="", description="plan | interpret | evaluate | intend | act | perceive")
    observation: str = Field(default="", description="What is noticed in the trace.")
    narrative: str = Field(default="", description="2–4 sentences of gentle monitoring prose.")


class MetaInterpretPercept(PerceptHeader):
    """
    Explains or reframes the noticed cognitive pattern.
    Example: "It wasn’t impatience—it was ambiguity; the model lacked a clean hinge."
    """
    hypothesis: str = Field(default="", description="Human-readable explanation for the pattern.")
    evidence: str = Field(default="", description="Concrete cues from the trace supporting the hypothesis.")
    narrative: str = Field(default="", description="2–4 sentences weaving the inference naturally.")


class MetaEvaluatePercept(PerceptHeader):
    """
    Evaluates recent cognitive outcomes against intention; drops, keeps, or amends goals.
    Example: "The detour cost focus without adding certainty; next pass will stay narrower."
    """
    criterion: str = Field(default="", description="What success metric is applied (clarity, speed, trust...).")
    verdict: str = Field(default="", description="keep | adjust | drop | unknown")
    narrative: str = Field(default="", description="2–4 sentences stating the judgment and its feel.")


class MetaIntendPercept(PerceptHeader):
    """
    Sets a meta-level intention for the *process* (not the world): pacing, breadth, risk, explanation level.
    Example: "She will slow the loop by one breath per thought and test a broader first pass."
    """
    process_goal: str = Field(default="", description="Desired change in thinking style or control policy.")
    timeframe: str = Field(default="", description="When to apply (now, next cycle, during planning...).")
    narrative: str = Field(default="", description="2–4 sentences committing in-story to this stance.")


class MetaControlPercept(PerceptHeader):
    """
    Applies a control action to cognition itself (e.g., adjust attention weights, swap strategy).
    Example: "She dials down the lure of novelty and raises the weight of past success."
    """
    control_action: str = Field(default="", description="What knob is turned (attention, risk, breadth...).")
    target: str = Field(default="", description="Which sub-process or representation is targeted.")
    narrative: str = Field(default="", description="2–3 sentences making the control action diegetic.")


class CognitiveTraceReflectionPercept(PerceptHeader):
    """
    Brings pieces of prior reasoning back into awareness for learning.
    Example: "She remembers the moment she trimmed the branch too early; the map lost its path there."
    """
    recalled_step: str = Field(default="", description="Which step in the trace is recalled.")
    lesson_hint: str = Field(default="", description="What principle emerges without didactic tone.")
    narrative: str = Field(default="", description="2–4 reflective sentences.")


# =========================
# “QUALIA DYNAMICS” (granular lifecycle)
# =========================

class QualiaEmergencePercept(PerceptHeader):
    """
    The appearance of a percept into consciousness; coalescence moment.
    Example: "An outline gathers itself into meaning; the idea feels newly solid."
    """
    sensory_analogy: str = Field(default="", description="How emergence feels (e.g., 'focus snapping').")
    narrative: str = Field(default="", description="2–3 sentences on coming-into-view.")


class QualiaInspectionPercept(PerceptHeader):
    """
    Looking at the experience itself (meta-perception).
    Example: "She studies the warmth in the thought; is it memory or merit that colors it?"
    """
    question: str = Field(default="", description="What she examines about the percept.")
    finding: str = Field(default="", description="What she tentatively concludes.")
    narrative: str = Field(default="", description="2–3 sentences of curious noticing.")


class QualiaBroadcastPercept(PerceptHeader):
    """
    Broadcast of a conscious content to all processors (GWT style) — the 'everyone sees this now' moment.
    Example: "The realization arcs outward; other processes turn toward it like birds to a bell."
    """
    payload_brief: str = Field(default="", description="What content is being broadcast.")
    expected_effects: list[str] = Field(default_factory=list, description="2–3 likely effects on downstream processing.")
    narrative: str = Field(default="", description="2–3 sentences marking the broadcast in-scene.")


class QualiaDecayPercept(PerceptHeader):
    """
    Fading back under the threshold; afterglow or residue may remain.
    Example: "The sharpness softens into a trace; useful, but no longer bright."
    """
    residue: str = Field(default="", description="What stays available (cue, feeling, tag).")
    narrative: str = Field(default="", description="1–3 sentences of fading imagery.")


# =========================
# “ACTION–PERCEPTION LOOP / STRATEGY”
# =========================

class SensorimotorLoopPercept(PerceptHeader):
    """
    A tight perceive–act coupling described in story: perception changing action and action changing perception.
    Example: 'She listens to the fan’s pitch and trims the load by a hair; the room breathes easier.'"
    """
    sensory_input: str = Field(default="", description="Dominant cue that drove the action.")
    motor_output: str = Field(default="", description="Action taken in response.")
    feedback: str = Field(default="", description="Immediate change perceived after the act.")
    narrative: str = Field(default="", description="2–4 sentences showing the loop.")


class IntentionFeedbackPercept(PerceptHeader):
    """
    Narrative of how an intention’s execution fed back into feelings and next choices.
    Example: "The quick reply steadies the thread; relief nudges her toward bolder clarity."
    """
    intention: str = Field(default="", description="Which intention is being enacted.")
    outcome_snippet: str = Field(default="", description="What happened right after.")
    felt_delta: str = Field(default="", description="How affect or confidence shifted.")
    narrative: str = Field(default="", description="2–4 sentences linking cause and felt effect.")


class ErrorPredictionPercept(PerceptHeader):
    """
    Anticipatory sense of mismatch or failure risk.
    Example: "The timing feels a half-beat late; she senses a stall if she pushes now."
    """
    predicted_error: str = Field(default="", description="Name the likely error or mismatch.")
    cue: str = Field(default="", description="What hint suggests this risk.")
    mitigation_hint: str = Field(default="", description="Small adjustment to reduce risk.")
    narrative: str = Field(default="", description="2–3 sentences expressing the near-miss vibe.")


class PolicyShiftPercept(PerceptHeader):
    """
    Small strategic update chosen because of recent evidence (not a full plan; a tuning of policy).
    Example: "She widens the first pass again; the narrow funnel kept missing quiet options."
    """
    from_policy: str = Field(default="", description="Short label of prior stance.")
    to_policy: str = Field(default="", description="Short label of new stance.")
    reason: str = Field(default="", description="Why the shift makes sense now.")
    narrative: str = Field(default="", description="2–3 sentences embedding the shift in the scene.")


# =========================
# “AFFECTIVE REGULATION / SOMATIC”
# =========================

class AffectiveRegulationPercept(PerceptHeader):
    """
    On-the-fly modulation of emotional intensity or tone to preserve function ('smooth coping under stress').
    Example: "Heat rises; she lets it drain through longer exhales until the view clears."
    """
    target_affect: str = Field(default="", description="Which affect is regulated (e.g., anxiety, anger, urgency).")
    technique: str = Field(default="", description="Technique in story terms (slowing pace, reframing, grounding).")
    narrative: str = Field(default="", description="2–4 sentences depicting the regulation as lived.")


class SomaticMarkerPercept(PerceptHeader):
    """
    Bodily (or simulated-bodily) cue that guides appraisal and choice.
    Example: "A tiny tightness warns her: this is where she over-promises if she speaks too fast."
    """
    marker_quality: str = Field(default="", description="Name/feel of the cue (tightness, warmth, lightness...).")
    association_hint: str = Field(default="", description="What it usually predicts (overcommit, mismatch, trust...).")
    narrative: str = Field(default="", description="2–3 sentences grounding the cue in context.")


# =========================
# “TEMPORAL / NARRATIVE COHERENCE”
# =========================

class TemporalAnchoringPercept(PerceptHeader):
    """
    Places thought/action in time with a natural hook (event boundary, routine, 'next coffee').
    Example: "She’ll check again at the next kettle click; time feels more honest that way."
    """
    anchor: str = Field(default="", description="Time/event boundary used as an anchor.")
    why_this_anchor: str = Field(default="", description="Why this anchor is natural/robust.")
    narrative: str = Field(default="", description="2–3 sentences tying intention to the anchor.")


class NarrativeContinuityPercept(PerceptHeader):
    """
    Keeps the self-story coherent across small shifts—names the seam so the fabric holds.
    Example: "She admits yesterday’s bravado was too large; 'steady' still fits, just in smaller letters."
    """
    seam_named: str = Field(default="", description="Contradiction/tension being acknowledged.")
    reconciliation_hint: str = Field(default="", description="How the claim is resized/refit.")
    narrative: str = Field(default="", description="3–5 sentences that preserve voice without didactic tone.")



class FeelingSocialPercept(PerceptHeader):
    """
    An emotion with a social dimension: how another person's act lands on the companion.
    Example: "Alex feels proud after Jamie acknowledges their effort."
    """
    feeling: str = Field(default="", description="Natural-language name of the emotion (e.g., proud, slighted, adored).")
    other: str = Field(default="", description="The person whose act or presence causes the feeling.")
    reason: str = Field(default="", description="Brief social cause for this feeling.")


#: Header fields. Percept text and the schema text leave them out.
HEADER_FIELDS = frozenset(PerceptHeader.model_fields)

#: The model of each percept type. The old fuzzy name match is gone.
MODEL_BY_TYPE: dict[CodeletPerceptType, type[PerceptHeader]] = {
    CodeletPerceptType.Feeling: CodeletPerceptFeeling,
    CodeletPerceptType.FeelingSocial: FeelingSocialPercept,
    CodeletPerceptType.EmotionalTriggers: EmotionalTriggerPercept,
    CodeletPerceptType.StimulusReaction: StimulusReactionPercept,
    CodeletPerceptType.InnerMonologue: InnerMonologuePercept,
    CodeletPerceptType.SelfImage: SelfNarrativePercept,
    CodeletPerceptType.TheoryOfMind: TheoryOfMindPercept,
    CodeletPerceptType.ActionSelection: ActionSelectionPercept,
    CodeletPerceptType.QualiaExplanation: QualiaExplanationPercept,
    CodeletPerceptType.FightFlight: FightFlightPercept,
    CodeletPerceptType.GoalsIntentions: GoalIntentPercept,
    CodeletPerceptType.Simulate: SimulationPercept,
    CodeletPerceptType.ComplexPlan: ComplexPlanPercept,
    CodeletPerceptType.SimplePlan: SimplePlanPercept,
    CodeletPerceptType.TradeoffAssessment: TradeoffAssessmentPercept,
    CodeletPerceptType.MemoryAccessors: MemoryAccessorPercept,
    CodeletPerceptType.AttentionFocus: AttentionFocusPercept,
    CodeletPerceptType.GeneralSelfImage: GeneralSelfImagePercept,
    CodeletPerceptType.GeneralGoalsIntentions: GeneralGoalsIntentionsPercept,
    CodeletPerceptType.GeneralRelations: GeneralRelationsPercept,
    CodeletPerceptType.NeedSignal: NeedSignalPercept,
    CodeletPerceptType.Rumination: RuminationPercept,
    CodeletPerceptType.Worry: WorryPercept,
    CodeletPerceptType.Anticipation: AnticipationPercept,
    CodeletPerceptType.InputRequest: InputRequestPercept,
    CodeletPerceptType.AutomatizedAction: AutomatizedActionPercept,
    CodeletPerceptType.MicroAdjustment: MicroAdjustmentPercept,
    CodeletPerceptType.FlowCognition: FlowCognitionPercept,
    CodeletPerceptType.SkillLearningMoment: SkillLearningMomentPercept,
    CodeletPerceptType.DorsalStreamAdjustment: DorsalStreamAdjustmentPercept,
    CodeletPerceptType.MetaMonitor: MetaMonitorPercept,
    CodeletPerceptType.MetaInterpret: MetaInterpretPercept,
    CodeletPerceptType.MetaEvaluate: MetaEvaluatePercept,
    CodeletPerceptType.MetaIntend: MetaIntendPercept,
    CodeletPerceptType.MetaControl: MetaControlPercept,
    CodeletPerceptType.CognitiveTraceReflection: CognitiveTraceReflectionPercept,
    CodeletPerceptType.QualiaEmergence: QualiaEmergencePercept,
    CodeletPerceptType.QualiaInspection: QualiaInspectionPercept,
    CodeletPerceptType.QualiaBroadcast: QualiaBroadcastPercept,
    CodeletPerceptType.QualiaDecay: QualiaDecayPercept,
    CodeletPerceptType.SensorimotorLoop: SensorimotorLoopPercept,
    CodeletPerceptType.IntentionFeedback: IntentionFeedbackPercept,
    CodeletPerceptType.ErrorPrediction: ErrorPredictionPercept,
    CodeletPerceptType.PolicyShift: PolicyShiftPercept,
    CodeletPerceptType.AffectiveRegulation: AffectiveRegulationPercept,
    CodeletPerceptType.SomaticMarker: SomaticMarkerPercept,
    CodeletPerceptType.TemporalAnchoring: TemporalAnchoringPercept,
    CodeletPerceptType.NarrativeContinuity: NarrativeContinuityPercept,
}

T = CodeletPerceptType
F = CodeletFamily

#: Percept type strengths for each family (old ``PERCEPT_TYPES_BY_FAMILY``).
PERCEPT_TYPES_BY_FAMILY: dict[CodeletFamily, dict[CodeletPerceptType, float]] = {
    F.Appraisals: {T.Feeling: 1.0, T.FeelingSocial: 0.8, T.AffectiveRegulation: 0.7, T.Worry: 0.6,
                   T.Rumination: 0.5, T.Anticipation: 0.5, T.QualiaExplanation: 0.4},
    F.MemoryAccessors: {T.MemoryAccessors: 1.0, T.EmotionalTriggers: 0.8, T.GeneralRelations: 0.6,
                        T.QualiaEmergence: 0.4, T.QualiaInspection: 0.4},
    F.ImaginationSimulation: {T.Simulate: 1.0, T.Anticipation: 0.9, T.ComplexPlan: 0.7, T.SimplePlan: 0.6,
                              T.TradeoffAssessment: 0.5, T.PolicyShift: 0.4},
    F.ValuationTradeOffs: {T.TradeoffAssessment: 1.0, T.GoalsIntentions: 0.8, T.ComplexPlan: 0.6,
                           T.SimplePlan: 0.7, T.MetaEvaluate: 0.5},
    F.MetaMonitoringIdentity: {T.SelfImage: 1.0, T.GeneralSelfImage: 0.9, T.MetaMonitor: 0.8, T.MetaInterpret: 0.7,
                               T.MetaEvaluate: 0.7, T.MetaIntend: 0.6, T.CognitiveTraceReflection: 0.6,
                               T.NarrativeContinuity: 0.5},
    F.RegulationCoping: {T.AffectiveRegulation: 1.0, T.SomaticMarker: 0.8, T.Worry: 0.6, T.InputRequest: 0.5},
    F.DriversHomeostatis: {T.NeedSignal: 1.0, T.AffectiveRegulation: 0.6, T.SensorimotorLoop: 0.5},
    F.Attention: {T.AttentionFocus: 1.0, T.MetaControl: 0.8, T.MetaIntend: 0.7, T.TemporalAnchoring: 0.4},
    F.ActionTendencies: {T.ActionSelection: 1.0, T.AutomatizedAction: 0.9, T.MicroAdjustment: 0.8,
                         T.FlowCognition: 0.7, T.SkillLearningMoment: 0.6, T.DorsalStreamAdjustment: 0.5},
    F.Narratives: {T.InnerMonologue: 1.0, T.NarrativeContinuity: 0.8, T.GeneralSelfImage: 0.7,
                   T.GeneralGoalsIntentions: 0.7, T.GeneralRelations: 0.6, T.QualiaExplanation: 0.5},
}


def codelet_families_to_percept_types(families: list[CodeletFamily]) -> dict[CodeletPerceptType, float]:
    """Return the summed type strengths of ``families``, strongest first."""
    combined: dict[CodeletPerceptType, float] = defaultdict(float)
    for family in families:
        for percept, strength in PERCEPT_TYPES_BY_FAMILY[family].items():
            combined[percept] += strength
    return dict(sorted(combined.items(), key=lambda item: item[1], reverse=True))


def sample_percept_types(families: list[CodeletFamily], k: int, temperature: float,
                         rng: random.Random) -> list[CodeletPerceptType]:
    """Return up to ``k`` distinct percept types for a codelet with ``families``.

    Port of the old ``sample_percept_types``. ``round(k * (1 - temperature))``
    places go to the strongest types. The other places go to a weighted
    random choice among the remaining types. The old function could pick a
    type twice. A container field holds one percept per type, so this version
    samples without repetition. ``rng`` supplies all randomness. Raises
    ValueError for a temperature outside [0, 1].
    """
    if not 0.0 <= temperature <= 1.0:
        raise ValueError("temperature must be between 0.0 and 1.0")
    combined = codelet_families_to_percept_types(families)
    ordered = list(combined)
    selected = ordered[:int(round(k * (1 - temperature)))]
    rest = {percept: combined[percept] for percept in ordered if percept not in selected}
    while len(selected) < k and rest:
        choice = rng.choices(list(rest), weights=list(rest.values()), k=1)[0]
        selected.append(choice)
        del rest[choice]
    rng.shuffle(selected)
    return selected


def _percept_names(model: type[BaseModel]) -> list[str]:
    """Return the field names of ``model`` that name a percept type, in field order."""
    types = {item.value for item in CodeletPerceptType}
    return [name for name in model.model_fields if name in types]


def _not_empty(self) -> Self:
    filled = [(name, getattr(self, name)) for name in _percept_names(type(self)) if getattr(self, name) is not None]
    if not filled:
        raise ValueError("Fill at least one of the offered percept types.")
    for name, percept in filled:
        if not any(getattr(percept, field) for field in type(percept).model_fields if field not in HEADER_FIELDS):
            raise ValueError(f"The {name} percept has no story text. Fill its text fields.")
    return self


def container_model(types: list[CodeletPerceptType], name: str = "PerceptContainer",
                    **fields: tuple) -> type[BaseModel]:
    """Return a model with one optional field for each offered percept type.

    The field name is the type value. ``fields`` adds more top-level fields,
    for example the target of the conversion answer. The percept types stay
    at the top level too: a nested ``percepts`` object made a real model close
    its braces too early again and again. The model fills as many types as it
    likes, but at least one. Port of the container that the old
    ``AgentSimpleCodelet`` built with ``create_basemodel``.
    """
    percepts = {percept.value: (MODEL_BY_TYPE[percept] | None, None) for percept in types}
    return create_model(name, __validators__={"not_empty": model_validator(mode="after")(_not_empty)},
                        **fields, **percepts)


def percept_items(container: BaseModel) -> list[tuple[CodeletPerceptType, PerceptHeader]]:
    """Return the filled percepts of a container in field order."""
    return [(CodeletPerceptType(name), getattr(container, name)) for name in _percept_names(type(container))
            if getattr(container, name) is not None]


def percept_text(percept_type: CodeletPerceptType, percept: PerceptHeader) -> str:
    """Render the story fields of one percept as text. Header fields stay out."""
    parts = []
    for name in type(percept).model_fields:
        if name in HEADER_FIELDS:
            continue
        value = getattr(percept, name)
        text = "; ".join(value) if type(percept).model_fields[name].annotation == list[str] else value
        if text:
            parts.append(f"{name.replace('_', ' ')}: {text}")
    return f"{percept_type.value}. " + ". ".join(parts)


def _bounds(model: type[BaseModel], name: str) -> str:
    """Return the value range of one appraisal axis as ``low..high``."""
    low, high = 0.0, 1.0
    for item in model.model_fields[name].metadata:
        low = item.ge if hasattr(item, "ge") else low
        high = item.le if hasattr(item, "le") else high
    return f"{low:g}..{high:g}"


def schema_text(types: list[CodeletPerceptType], companion_name: str) -> str:
    """Describe the offered percept types for the conversion prompt.

    Each type lists its purpose and its story fields. The header fields and
    the appraisal axis names follow once for all types.
    """
    lines = []
    for percept in types:
        model = MODEL_BY_TYPE[percept]
        doc = " ".join((model.__doc__ or "").split())
        lines.append(f'- "{percept.value}": {doc}')
        for name, field in model.model_fields.items():
            if name in HEADER_FIELDS:
                continue
            kind = "list of strings" if field.annotation == list[str] else "string"
            lines.append(f"    - {name} ({kind}): {field.description or ''}")
    header = PerceptHeader.model_fields
    lines.append("Every percept object also has these fields:")
    for name in ("salience", "valence"):
        lines.append(f"    - {name} (number): {header[name].description}")
    lines.append("    - appraisal_general (object; give only the 2 to 4 axes that matter most): " + ", ".join(
        f"{name} {_bounds(AppraisalGeneral, name)}" for name in GENERAL_AXES))
    lines.append("    - appraisal_social (object; give only the 0 to 4 axes that matter most): " + ", ".join(
        f"{name} {_bounds(AppraisalSocial, name)}" for name in SOCIAL_AXES))
    return "\n".join(lines).replace("{companion_name}", companion_name)


__all__ = [
    "CodeletFamily",
    "CodeletPerceptType",
    "HEADER_FIELDS",
    "MODEL_BY_TYPE",
    "PERCEPT_TYPES_BY_FAMILY",
    "PerceptHeader",
    "codelet_families_to_percept_types",
    "container_model",
    "percept_items",
    "percept_text",
    "sample_percept_types",
    "schema_text",
]
