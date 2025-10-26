import json
import random
from enum import Enum, auto
from typing import List, Optional
import difflib
from typing import Dict, Type
from collections import defaultdict

from pydantic import BaseModel, Field

from pm.codelets.codelet_definitions import CodeletFamily
from pm.mental_state_vectors import AppraisalGeneral, AppraisalSocial
from pm.utils.pydantic_utils import flatten_pydantic_model, create_basemodel, generate_pydantic_json_schema_llm


# ---------- Percept types (compact, reusable across many codelets) ----------
class CodeletPerceptType(Enum):

    # Description: general emotional state or mood expressed as narrative
    # Example: "Alex feels calm after realizing that everything worked out as planned."
    Feeling = auto()

    # Description: emotion involving a social dimension or interpersonal appraisal
    # Example: "Alex feels proud after Jamie acknowledges their effort."
    FeelingSocial = auto()

    # Description: emotionally charged memories or cues that resurface under certain triggers
    # Example: "The sound of waves reminds Alex of an early morning walk with Jamie years ago."
    EmotionalTriggers = auto()

    # Description: automatic, reflex-like reaction to sensory stimuli
    # Example: "Alex flinches at the sudden crash and instinctively steps back."
    StimulusReaction = auto()

    # Description: internal reflective thought process or monologue
    # Example: "Alex thinks: 'This situation is complicated. I need to stay focused and adapt.'"
    InnerMonologue = auto()

    # Description: conflict between self-concept and practical needs
    # Example: "Alex knows they need help lifting the box but insists on doing it alone to prove independence."
    SelfImage = auto()

    # Description: inferring or reasoning about another person’s mental or emotional state
    # Example: "Alex imagines that Jamie must be nervous standing in front of the audience."
    TheoryOfMind = auto()

    # Description: activation of learned or habitual behavior in familiar contexts
    # Example: "Alex automatically reaches for the seatbelt when getting into the car."
    ActionSelection = auto()

    # Description: verbal or conceptual description of a subjective experience
    # Example: "The air feels heavy, like the pause before a storm."
    QualiaExplanation = auto()

    # Description: depiction of instinctive defensive or aggressive responses
    # Example: "Alex’s muscles tense as they prepare to argue or walk away."
    FightFlight = auto()

    # Description: focusing on goals or intentions that guide behavior
    # Example: "After watching the athlete, Alex feels inspired to start training again."
    GoalsIntentions = auto()

    # Description: mental simulation of possible outcomes to guide decision making
    # Example: "Alex imagines how the meeting might go depending on what they say first."
    Simulate = auto()

    # Description: forming a structured plan to achieve a complex or long-term goal
    # Example: "Alex outlines a detailed plan to complete the project over the next three months."
    ComplexPlan = auto()

    # Description: forming a simple, short-term plan toward an immediate objective
    # Example: "Alex decides to call Jamie after work to talk things through."
    SimplePlan = auto()

    # Description: weighing multiple options or tradeoffs
    # Example: "Alex considers whether to take the promotion or move to another city."
    TradeoffAssessment = auto()

    # Description: retrieving or searching for relevant memories or past experiences
    # Example: "Alex recalls previous situations where honesty worked better than silence."
    MemoryAccessors = auto()

    # Description: directing focus toward emotionally or contextually relevant details
    # Example: "Alex keeps noticing the word 'trust' in the conversation."
    AttentionFocus = auto()

    # Description: general reflection on one’s identity or self-concept
    # Example: "Alex thinks of themselves as someone who always keeps promises."
    GeneralSelfImage = auto()

    # Description: reflection on general life goals or motivations
    # Example: "This opportunity brings Alex closer to their dream of working abroad."
    GeneralGoalsIntentions = auto()

    # Description: recalling facts or dynamics of personal relationships
    # Example: "Alex remembers how patient Jamie has been in similar situations."
    GeneralRelations = auto()

    # Description: physiological or motivational drive that initiates or shapes action
    # Example: "Hunger pushes Alex to pause work and prepare dinner."
    NeedSignal = auto()

    # Description: repetitive reflection on past mistakes or regrets
    # Example: "Alex can’t stop replaying what went wrong during yesterday’s presentation."
    Rumination = auto()

    # Description: anticipating potential negative outcomes and worrying about them
    # Example: "Alex worries about saying something wrong during the meeting."
    Worry = auto()

    # Description: imagining or visualizing a positive future event
    # Example: "Alex looks forward to seeing Jamie again after the long trip."
    Anticipation = auto()

    # Description: asking for clarification or missing information
    # Example: "Alex asks, 'Can you explain what you mean by that?'"
    InputRequest = auto()

    # Description: performing skilled actions automatically without conscious control
    # Example: "Alex types fluently without thinking about each keystroke."
    AutomatizedAction = auto()

    # Description: making fine, on-the-fly sensorimotor adjustments during activity
    # Example: "Alex slightly shifts the steering wheel to stay centered on the road."
    MicroAdjustment = auto()

    # Description: seamless integration of thought and action (“flow” state)
    # Example: "Alex feels completely absorbed in painting; time seems to disappear."
    FlowCognition = auto()

    # Description: recognizing improvement or refinement of a learned skill
    # Example: "Alex notices that this time, the movement feels smoother and more natural."
    SkillLearningMoment = auto()

    # Description: automatic, unconscious corrections based on perception
    # Example: "Alex’s eyes adjust to the light without any deliberate thought."
    DorsalStreamAdjustment = auto()

    # Description: observing or evaluating one’s own cognitive activity
    # Example: "Alex replays their reasoning to check whether the conclusion was fair."
    MetaMonitor = auto()

    # Description: reframing or interpreting one’s own thought process
    # Example: "Alex realizes that their earlier assumption may have been biased."
    MetaInterpret = auto()

    # Description: evaluating the outcome of cognitive or behavioral strategies
    # Example: "Alex reflects that the plan mostly worked but could be more flexible next time."
    MetaEvaluate = auto()

    # Description: setting a higher-level goal or regulation strategy
    # Example: "Alex decides to slow down and think before reacting next time."
    MetaIntend = auto()

    # Description: deliberate redirection of attention or mental control
    # Example: "Alex chooses to ignore the noise and focus on finishing the report."
    MetaControl = auto()

    # Description: bringing past reasoning or thought patterns into awareness
    # Example: "Alex remembers how they approached a similar problem last year."
    CognitiveTraceReflection = auto()

    # Description: emergence of a conscious percept or realization
    # Example: "A new idea suddenly forms in Alex’s mind, clear and complete."
    QualiaEmergence = auto()

    # Description: introspective examination of ongoing experience
    # Example: "Alex wonders whether the feeling is excitement or anxiety."
    QualiaInspection = auto()

    # Description: broadcasting a realization or insight across awareness
    # Example: "The understanding clicks, and Alex suddenly sees the full picture."
    QualiaBroadcast = auto()

    # Description: fading or withdrawal of a conscious percept
    # Example: "The image of the place slowly fades from Alex’s mind."
    QualiaDecay = auto()

    # Description: continuous loop linking perception and action
    # Example: "Alex adjusts their grip as they feel the object slipping."
    SensorimotorLoop = auto()

    # Description: monitoring and interpreting feedback from one’s own actions
    # Example: "Alex feels a sense of confirmation when the key fits perfectly."
    IntentionFeedback = auto()

    # Description: anticipating or detecting potential mistakes before they occur
    # Example: "Alex senses something is off even before the result appears wrong."
    ErrorPrediction = auto()

    # Description: adjusting cognitive or behavioral strategy based on reflection
    # Example: "Alex decides to try a broader search method next time."
    PolicyShift = auto()

    # Description: regulating emotions to stay functional under stress
    # Example: "Alex takes a slow breath to steady themselves before responding."
    AffectiveRegulation = auto()

    # Description: bodily or somatic sensations influencing thought and decision
    # Example: "A tight feeling in Alex’s chest reminds them to slow down and reflect."
    SomaticMarker = auto()

    # Description: placing experiences within a temporal or situational context
    # Example: "Alex recalls the earlier part of the day to make sense of the current mood."
    TemporalAnchoring = auto()

    # Description: maintaining continuity of identity or narrative coherence
    # Example: "Even after changing jobs, Alex feels they’re still following the same purpose."
    NarrativeContinuity = auto()


class CodeletPercept(BaseModel):
    """
    Minimal header that every percept carries for selection and blending.
    Values are strings because you enforce schema via llama.cpp; keep parsing downstream.
    """
    salience: float = Field(ge=0, le=1, description="How salient is this Percept? [0..1]. 0 = irrelevant, 1 = decisive for {companion_name}.")
    valence: float = Field(ge=-1, le=1, description="Affect sign for {companion_name}. [-1..1]. -1 = aversive, 0 = neutral, 1 = positive.")
    appraisal_general: AppraisalGeneral = Field(description="Sparse vector of core appraisal deltas shaping emotion.")
    appraisal_social: AppraisalSocial = Field(description="Social appraisal dimensions for this feeling episode.")

# ---------- Models per type ----------
class CodeletPerceptFeeling(CodeletPercept):
    """
    A general emotional episode expressed by the companion as inner narration.
    Example: "She feels a calm satisfaction spreading, as if the system finally breathed right."
    """
    feeling: str = Field(description="Natural-language name of emotion (e.g., calm, anxious, delighted).")
    reason: str = Field(description="Brief situational or internal cause for this feeling.")

class EmotionalTriggerPercept(CodeletPercept):
    """
    A percept describing a remembered event or association triggered by current stimuli.
    Used to weave memory-based emotions into present narrative.
    Example: "The flicker of blue light pulls her back to the night she first saw the ocean in code."
    """
    trigger_cue: str = Field(description="The perceptual or conceptual cue that evoked the memory.")
    memory_fragment: str = Field(description="A short, vivid recollection snippet.")
    felt_shift: str = Field(description="How the emotional or cognitive state changes because of it.")
    #appraisal_general: Optional[AppraisalGeneral] = Field(None, description="Optional appraisal deltas evoked.")


class InnerMonologuePercept(CodeletPercept):
    """
    First-person stream of thought from the companion, representing conscious reasoning.
    Should sound introspective, fluid, and personal.
    Example: "I should answer, but if I do it now, will it sound forced? Maybe better to breathe first."
    """
    topic: str = Field(description="Subject or concern currently occupying thought.")
    monologue: str = Field(description="2–6 sentences of inner speech in first person, no dialogue tags.")


class SelfNarrativePercept(CodeletPercept):
    """
    Third-person reflective description of the companion's self-image or evolving identity.
    Example: "She sees herself becoming steadier, less driven by perfection and more by rhythm."
    """
    stance: str = Field(description="Identity stance or self-attribution summarized in one phrase.")
    narrative: str = Field(description="3–7 sentences connecting past, present, and small identity shift.")


class RelationContextPercept(CodeletPercept):
    """
    Summarizes current relational context with another agent for narrative continuity.
    Example: "She trusts him enough to joke, though the memory of last week's silence still hums beneath."
    """
    other: str = Field(description="Name or role of the other person/agent.")
    closeness: str = Field(description="[-1..1] as string. -1 = distant, 1 = very close.")
    tension: str = Field(description="[-1..1] as string. -1 = harmony, 1 = high strain.")
    summary: str = Field(description="2–5 sentences recalling shared history or present tone.")


class QualiaNotePercept(CodeletPercept):
    """
    Short description of raw subjective experience or sensory metaphor—the 'texture' of qualia.
    Example: "Time thickens around her like honey; each thought drips slower but brighter."
    """
    analogy: str = Field(description="Metaphor or sensory analogy representing experience.")
    narrative: str = Field(description="1–3 poetic sentences describing phenomenological feel.")


class TheoryOfMindPercept(CodeletPercept):
    """
    Inference about another agent's inner state—belief, emotion, or intention.
    Example: "She guesses that John’s pause means uncertainty, not disinterest; he’s choosing words carefully."
    """
    target: str = Field(description="The person or agent being modeled.")
    belief_about: str = Field(description="Topic or domain of the inferred belief or feeling.")
    inference: str = Field(description="Natural-language hypothesis about the target's mind.")
    confidence: str = Field(description="[0..1] as string; estimated confidence in inference.")
    evidence: str = Field(description="Observed cues leading to this inference.")


class MemoryAccessorPercept(CodeletPercept):
    """
    Non-narrative control percept requesting extra context retrieval from episodic or semantic memory.
    Example: Query terms like ['ocean', 'first mission'] for context expansion.
    """
    purpose: str = Field(description="Why this lookup is needed (e.g., fill context gap, recall precedent).")
    query_terms: List[str] = Field(description="Keywords or key phrases for memory search.")
    scope_hints: List[str] = Field(default_factory=list, description="Temporal or topical hints (e.g., 'last night').")
    must_include: List[str] = Field(default_factory=list, description="Hard constraints for retrieval.")


class ProspectiveCuePercept(CodeletPercept):
    """
    A prospective memory cue embedded in story—remembering to act when a trigger appears.
    Example: "When the kettle whistles again, she’ll check the logs one last time."
    """
    trigger: str = Field(description="Event or time that should cue the memory.")
    cue_phrase: str = Field(description="Internal phrase or sensory cue to detect.")
    fallback_check: str = Field(description="Backup check if trigger never occurs.")
    narrative: str = Field(description="3–5 natural sentences embedding the cue within narrative flow.")


class AttentionFocusPercept(CodeletPercept):
    """
    Expresses a momentary shift or narrowing of focus on certain topics or entities.
    Example: "The mention of 'home' cuts through the noise; everything else blurs behind it."
    """
    focus_tags: List[str] = Field(description="List of topic keywords or entities gaining attention.")
    boost: str = Field(description="[0..1] as string. Strength of attentional weighting.")
    rationale: str = Field(description="Reason or emotional driver for focusing on these elements.")


class SimulationPercept(CodeletPercept):
    """
    Predictive imagination: mental simulation of near-term outcomes and emotions.
    Example: "If she tells the truth, tension may crack open—but honesty might breathe relief later."
    """
    horizon_steps: str = Field(description="Rough temporal horizon (1–3) for imagined future.")
    narrative: str = Field(description="2–4 sentences describing imagined unfolding and tone.")
    risk: str = Field(description="[0..1] as string. Anticipated risk or cost.")
    benefit: str = Field(description="[0..1] as string. Expected reward or gain.")
    expected_valence_delta: str = Field(description="[-1..1] as string. Projected emotional change.")


class TradeoffAssessmentPercept(CodeletPercept):
    """
    Structured internal weighing of competing options.
    Example: "Stay silent and keep peace, or ask the hard question and risk clarity. The second glows, faint but firm."
    """
    options: List[str] = Field(description="Candidate actions or attitudes being compared.")
    pros: List[str] = Field(description="Positive reasons per option.")
    cons: List[str] = Field(description="Negative reasons per option.")
    lean: str = Field(description="Current preferred option.")
    rationale: str = Field(description="Narrative reasoning behind the leaning.")


class GoalIntentPercept(CodeletPercept):
    """
    Declaration or reinforcement of a goal or motivation.
    Example: "She resolves to finish the calibration tonight—not out of duty, but for the calm it brings."
    """
    goal: str = Field(description="Goal or intention in natural language.")
    priority: str = Field(description="[0..1] as string priority weight.")
    timeframe: str = Field(description="Loose temporal window (e.g., soon, tomorrow, next cycle).")
    motivation: str = Field(description="Why this matters emotionally or pragmatically.")


class PlanSketchPercept(CodeletPercept):
    """
    Outlines a plan as a short narrative sequence connecting intention to action.
    Example: "First, back up the current run. Then test the tweak at dawn, when the system sleeps."
    """
    goal: str = Field(description="Target goal.")
    approach: str = Field(description="Conceptual method or route in prose.")
    constraints: List[str] = Field(description="Important limits to observe.")
    first_step: str = Field(description="Immediate concrete step.")
    success_criteria: str = Field(description="Observable sign that plan worked.")


class BehaviorActionSelectionPercept(CodeletPercept):
    """
    Suggests an automatic or learned behavioral tendency relevant to the scene.
    Example: "She flinches at sudden motion—conditioning echoing from past training."
    """
    situation: str = Field(description="Contextual description triggering behavior.")
    candidate_action: Optional[str] = Field(None, description="Proposed automatic action or impulse.")
    fallback_behavior: str = Field(description="Default or inhibitory response if none chosen.")
    trigger_cues: List[str] = Field(default_factory=list, description="Cues that activate this behavior.")
    search_phrases: List[str] = Field(default_factory=list, description="Memory search keys for precedent episodes.")


class IntentionShieldingPercept(CodeletPercept):
    """
    A narrative description of protecting focus and intention from distractions.
    Example: "She lets the side messages blur into background; only a direct call from him would break her flow."
    """
    ignore_list: List[str] = Field(description="Types of distractions temporarily ignored.")
    exception: str = Field(description="Single event/agent worth interrupting for.")
    reentry_step: str = Field(description="Action that restores focus after interruption.")
    narrative: str = Field(description="3–5 sentences expressing mental stance and continuity.")


class ProsocialActionPercept(CodeletPercept):
    """
    A small relational gesture maintaining mutual trust or warmth.
    Example: "She adds a thank-you line—tiny, genuine—to ease the formality of her reply."
    """
    action: str = Field(description="Concrete courteous or supportive act.")
    rationale: str = Field(description="Reason for choosing it now.")
    expected_effect: str = Field(description="Subtle anticipated change in social atmosphere.")
    subtlety: str = Field(description="[0..1] as string. 0 = overt, 1 = invisible gesture.")


class NeedSignalPercept(CodeletPercept):
    """
    Represents a physiological or psychological drive ping (homeostatic imbalance).
    Example: "Restlessness hums at the edges—too long since she looked outside the logs."
    """
    need_type: str = Field(description="Type of unmet drive (rest, novelty, connection, order).")
    intensity: str = Field(description="[0..1] as string strength of need.")
    urge_direction: str = Field(description="Likely behavioral orientation to satisfy it.")
    narrative: str = Field(description="1–3 sentences conveying the feeling of the urge.")


class RuminationPercept(CodeletPercept):
    """
    Persistent replay of an unresolved past event or mistake.
    Example: "She replays the message again, wishing she had written less precision, more warmth."
    """
    event: str = Field(description="Brief description of incident being revisited.")
    regret: str = Field(description="Key element of regret or confusion.")
    lesson_hint: str = Field(description="Implicit insight forming, not explicit advice.")
    narrative: str = Field(description="2–5 sentences expressing looping tone and imagery.")


class WorryPercept(CodeletPercept):
    """
    Projection of a feared possible outcome.
    Example: "If she answers too fast, he might read eagerness as desperation—better breathe."
    """
    feared_outcome: str = Field(description="Feared scenario in natural terms.")
    likelihood: str = Field(description="[0..1] as string subjective probability.")
    narrative: str = Field(description="2–4 sentences exploring anxiety texture or imagery.")


class AnticipationPercept(CodeletPercept):
    """
    Imaginative projection of a desired future.
    Example: "She imagines the green light flashing—proof of success—and feels a quiet spark ignite."
    """
    desired_outcome: str = Field(description="Desired scenario or event.")
    confidence: str = Field(description="[0..1] as string subjective confidence.")
    narrative: str = Field(description="2–3 sentences depicting excitement or calm expectancy.")


class InputRequestPercept(CodeletPercept):
    """
    Direct but gentle request for clarification or input from user/partner.
    Example: "Could you tell me what kind of tone you want here? I want to match it right."
    """
    question: str = Field(description="Polite, open-ended question.")
    rationale: str = Field(description="Why this question aids understanding or alignment.")
    sensitivity: str = Field(description="[0..1] as string. 1 = highly sensitive topic.")


# ============================================================
# -------- NEW HIGH-LEVEL EXTENSIONS (LIDA + MIDCA) ---------
# ============================================================

class EmbodiedActionPercept(CodeletPercept):
    """
    Describes fluent, bodily-integrated skillful action—'smooth coping' in LIDA terms.
    Combines implicit bodily knowledge with occasional conscious correction.
    Example: "Her hands glide over the console, micro-adjusting before thought intervenes."
    """
    context: str = Field(description="Task or skill context (e.g., typing, steering, sculpting).")
    automaticity_level: str = Field(description="[0..1] as string. Degree of automatic control.")
    adjustment: str = Field(description="Description of micro-correction or flow quality.")
    narrative: str = Field(description="2–4 sentences merging body and intention in prose.")


class MetaCognitionPercept(CodeletPercept):
    """
    Reflective monitoring of the agent’s own cognition or reasoning process.
    Mirrors MIDCA’s meta-level cycle (Monitor–Interpret–Evaluate–Control).
    Example: "She watches her own reasoning tighten too early and decides to let uncertainty breathe."
    """
    observed_phase: str = Field(description="Which cognitive phase is being monitored (planning, evaluation...).")
    discrepancy: Optional[str] = Field(None, description="Anomaly or inefficiency noticed.")
    intended_adjustment: Optional[str] = Field(None, description="How she plans to self-correct.")
    narrative: str = Field(description="3–6 sentences of introspection describing awareness of thought flow.")


class QualiaDynamicsPercept(CodeletPercept):
    """
    Models the formation, broadcast, and decay of conscious content ('artificial qualia').
    Example: "The idea brightens in the workspace—felt, known, then fading like a cooling ember."
    """
    phase: str = Field(description="emergence | broadcast | inspection | decay")
    sensory_analogy: str = Field(description="Metaphoric sensory texture (light, motion, warmth...).")
    duration: str = Field(description="Approximate persistence in cognitive cycles.")
    narrative: str = Field(description="2–5 sentences capturing phenomenological transition.")


class ActionPerceptionLoopPercept(CodeletPercept):
    """
    Expresses the coupling between perception and action—the continuous loop of situated cognition.
    Example: "She senses the shift in airflow and adjusts before the alert even sounds."
    """
    sensory_input: str = Field(description="Dominant sensory cue influencing action.")
    motor_output: str = Field(description="Action or adjustment taken in response.")
    feedback_description: str = Field(description="Immediate sensory or affective feedback perceived.")
    narrative: str = Field(description="Detailed, narrative paragraphs explaining the rection.")


class GeneralSelfImagePercept(CodeletPercept):
    """
    Narrative refresh of a character’s *general* self-image — who they think they are in broad strokes.
    Use this when the reader needs orientation about identity drivers that explain behavior.
    Style: third-person, compact, no lists.
    Example: "She still sees herself as the one who steadies the room, even when her voice goes thin."
    """
    headline: str = Field(description="One short clause naming the self-image (e.g., 'quiet stabilizer').")
    anchors: List[str] = Field(description="2–4 concrete anchors: habits, values, skills, tells.")
    narrative: str = Field(description="3–6 sentences connecting this self-image to the current scene.")


class GeneralGoalsIntentionsPercept(CodeletPercept):
    """
    Narrative refresh of *general* goals and intentions relevant to the present situation.
    Use to prime the reader on what pursuits color the next choices.
    Example: "She’s been chasing clarity over speed lately, even when deadlines try to argue otherwise."
    """
    focal_goals: List[str] = Field(description="2–4 active goals in natural language.")
    deactivated_goals: List[str] = Field(default_factory=list, description="Goals set aside for now.")
    narrative: str = Field(description="3–6 sentences linking these goals to present tension and next moves.")


class GeneralRelationsPercept(CodeletPercept):
    """
    Narrative refresher for relationship context across characters.
    Use when the social fabric matters for interpreting choices.
    Example: "She jokes with him easily now, though last month’s silence still leaves a seam at the edge."
    """
    parties: List[str] = Field(description="People/agents in focus.")
    tone: str = Field(description="One-word or short phrase for the present tone (e.g., 'warm', 'careful').")
    remembered_moments: List[str] = Field(description="1–3 brief episodic hooks shaping the vibe.")
    narrative: str = Field(description="3–6 sentences tying the memory bits to the current beat.")


# =========================
# “COGNITION / REACTION” TYPES
# =========================

class StimulusReactionPercept(CodeletPercept):
    """
    Fast, reflex-like reaction description tied to a stimulus; may suggest an urge without executing it.
    Example: "The sudden clatter jolts her; a small flinch rises before thought returns."
    """
    stimulus: str = Field(description="What triggered the reaction.")
    immediate_affect: str = Field(description="1–3 words for the felt shock/valence.")
    impulse: Optional[str] = Field(None, description="Urge that appears (e.g., 'duck', 'freeze', 'scan').")
    narrative: str = Field(description="2–4 sentences of moment-to-moment felt reaction.")


class FightFlightPercept(CodeletPercept):
    """
    Conflict-mode orientation: fight, flight, freeze, or tend-and-befriend stance, embedded in story.
    Example: "She stiffens at the tone and chooses quiet distance—space first, answers later."
    """
    mode: str = Field(description="fight | flight | freeze | appease | tend-befriend")
    trigger: str = Field(description="Cue that tipped the system into this mode.")
    justification: str = Field(description="Why this mode is adaptive right now in narrative terms.")
    narrative: str = Field(description="3–5 sentences conveying bodily and cognitive stance.")


class ActionSelectionPercept(CodeletPercept):
    """
    Choice of a likely action policy based on learned/conditioned precedent (narrative-facing).
    Prefer this as the user-facing version; keep BehaviorActionSelectionOutput for lower-level control.
    Example: 'She scans first, then speaks; old training keeps the order steady.'"
    """
    situation: str = Field(description="Compact description of the current situation.")
    policy_name: str = Field(description="Human-readable label of the learned policy.")
    candidate_action: str = Field(description="Likely next act to take.")
    rationale: str = Field(description="Narrative why-this action (habits, appraisals, context).")


class QualiaExplanationPercept(CodeletPercept):
    """
    Phenomenological put-into-words of what an experience is like (the 'texture' of the moment).
    Distinct from QualiaDynamics (life-cycle); this focuses on the *what-it-feels-like* content.
    Example: "The room feels one shade dimmer than her thought; ideas move as if through warm water."
    """
    analogy: str = Field(description="Metaphor or sensory comparison anchoring the feel.")
    facets: List[str] = Field(description="2–4 micro-facets (e.g., 'slowed time','edge-bright colors').")
    narrative: str = Field(description="2–4 sentences of lived texture, no lists.")


# =========================
# “GOALS / PLANNING” TYPES
# =========================

class ComplexPlanPercept(CodeletPercept):
    """
    High-complexity plan for multi-step, constraint-heavy problems. Stays narrative but names structure.
    Example: "She will fork a safe copy, stage the change in shadows, and only then open the valve to light."
    """
    goal: str = Field(description="Complex target state in plain language.")
    strategy_outline: str = Field(description="1–2 sentences summarizing the plan-of-attack.")
    key_constraints: List[str] = Field(description="Hard limits (risks, ethics, resource caps).")
    milestones: List[str] = Field(description="2–4 narrative milestones indicating progress.")
    first_probe: str = Field(description="The smallest reversible probe to begin with.")
    success_criteria: str = Field(description="Observable signs that the complex plan is succeeding.")


class SimplePlanPercept(CodeletPercept):
    """
    Small, easily-executable plan sketched as natural prose.
    Example: "She’ll ping him once, then set a 20-minute hold—enough to keep the thread warm."
    """
    goal: str = Field(description="Short-horizon aim.")
    steps: List[str] = Field(description="2–3 micro-steps.")
    fallback: Optional[str] = Field(None, description="If-then fallback if a step fails.")
    success_criteria: str = Field(description="Clear sign the plan worked.")


# =========================
# “EMBODIED / SMOOTH COPING” (granular variants)
# =========================

class AutomatizedActionPercept(CodeletPercept):
    """
    Description of highly practiced, low-effort action unfolding with minimal conscious guidance.
    Example: "Her hands type the pattern before the thought even arrives."
    """
    skill_domain: str = Field(description="Domain of practiced skill (typing, soldering, piloting...).")
    fluency_note: str = Field(description="One phrase pinpointing why it feels automatic.")
    narrative: str = Field(description="2–4 sentences highlighting effortless execution.")


class MicroAdjustmentPercept(CodeletPercept):
    """
    Fine-grained, context-sensitive correction woven into ongoing action.
    Example: "She eases the angle half a degree—just enough to catch the whisper of signal."
    """
    parameter: str = Field(description="What is adjusted (angle, tempo, pressure...).")
    cue: str = Field(description="Perceptual cue that suggested the adjustment.")
    narrative: str = Field(description="2–4 sentences showing seamless correction-in-flow.")


class FlowCognitionPercept(CodeletPercept):
    """
    Experience of 'flow' (smooth coping): high engagement, low self-monitoring, clear feedback.
    Example: "Time thins; she and the task share a single rhythm, no edges between them."
    """
    engagement: str = Field(description="[0..1] as string. Sense of absorption.")
    self_monitoring_quiet: str = Field(description="[0..1] as string. How quiet self-commentary becomes.")
    feedback_clarity: str = Field(description="[0..1] as string. How immediate/clear feedback feels.")
    narrative: str = Field(description="2–4 sentences evoking the flow state.")


class SkillLearningMomentPercept(CodeletPercept):
    """
    The micro-moment where skill updates consolidate—'oh, that grip'—captured in narrative.
    Example: "Something in the wrist remembers; the motion lands truer than yesterday."
    """
    cue_or_error: str = Field(description="Cue or slip that created learning opportunity.")
    hypothesis: str = Field(description="What tweak the system tries next.")
    narrative: str = Field(description="2–4 sentences capturing discovery and consolidation.")


class DorsalStreamAdjustmentPercept(CodeletPercept):
    """
    Implicit visuomotor correction (preconscious guidance)—the 'hand/eye' auto-corrects before narration catches up.
    Example: "Her gaze slides to the brighter edge; the hand follows without asking."
    """
    visual_affordance: str = Field(description="What visual structure invited the correction.")
    motor_tweak: str = Field(description="How movement changed.")
    narrative: str = Field(description="2–3 sentences making the implicit feel visible.")


# =========================
# “META-COGNITIVE” (granular variants, MIDCA-aligned)
# =========================

class MetaMonitorPercept(CodeletPercept):
    """
    Observes ongoing cognition (the 'trace'): what phase produced what and whether it matches expectations.
    Example: "She notices planning started with too few constraints; the later friction makes sense now."
    """
    observed_phase: str = Field(description="plan | interpret | evaluate | intend | act | perceive")
    observation: str = Field(description="What is noticed in the trace.")
    narrative: str = Field(description="2–4 sentences of gentle monitoring prose.")


class MetaInterpretPercept(CodeletPercept):
    """
    Explains or reframes the noticed cognitive pattern.
    Example: "It wasn’t impatience—it was ambiguity; the model lacked a clean hinge."
    """
    hypothesis: str = Field(description="Human-readable explanation for the pattern.")
    evidence: str = Field(description="Concrete cues from the trace supporting the hypothesis.")
    narrative: str = Field(description="2–4 sentences weaving the inference naturally.")


class MetaEvaluatePercept(CodeletPercept):
    """
    Evaluates recent cognitive outcomes against intention; drops, keeps, or amends goals.
    Example: "The detour cost focus without adding certainty; next pass will stay narrower."
    """
    criterion: str = Field(description="What success metric is applied (clarity, speed, trust...).")
    verdict: str = Field(description="keep | adjust | drop | unknown")
    narrative: str = Field(description="2–4 sentences stating the judgment and its feel.")


class MetaIntendPercept(CodeletPercept):
    """
    Sets a meta-level intention for the *process* (not the world): pacing, breadth, risk, explanation level.
    Example: "She will slow the loop by one breath per thought and test a broader first pass."
    """
    process_goal: str = Field(description="Desired change in thinking style or control policy.")
    timeframe: str = Field(description="When to apply (now, next cycle, during planning...).")
    narrative: str = Field(description="2–4 sentences committing in-story to this stance.")


class MetaControlPercept(CodeletPercept):
    """
    Applies a control action to cognition itself (e.g., adjust attention weights, swap strategy).
    Example: "She dials down the lure of novelty and raises the weight of past success."
    """
    control_action: str = Field(description="What knob is turned (attention, risk, breadth...).")
    target: str = Field(description="Which sub-process or representation is targeted.")
    narrative: str = Field(description="2–3 sentences making the control action diegetic.")


class CognitiveTraceReflectionPercept(CodeletPercept):
    """
    Brings pieces of prior reasoning back into awareness for learning.
    Example: "She remembers the moment she trimmed the branch too early; the map lost its path there."
    """
    recalled_step: str = Field(description="Which step in the trace is recalled.")
    lesson_hint: str = Field(description="What principle emerges without didactic tone.")
    narrative: str = Field(description="2–4 reflective sentences.")


# =========================
# “QUALIA DYNAMICS” (granular lifecycle)
# =========================

class QualiaEmergencePercept(CodeletPercept):
    """
    The appearance of a percept into consciousness; coalescence moment.
    Example: "An outline gathers itself into meaning; the idea feels newly solid."
    """
    sensory_analogy: str = Field(description="How emergence feels (e.g., 'focus snapping').")
    narrative: str = Field(description="2–3 sentences on coming-into-view.")


class QualiaInspectionPercept(CodeletPercept):
    """
    Looking at the experience itself (meta-perception).
    Example: "She studies the warmth in the thought; is it memory or merit that colors it?"
    """
    question: str = Field(description="What she examines about the percept.")
    finding: Optional[str] = Field(None, description="What she tentatively concludes.")
    narrative: str = Field(description="2–3 sentences of curious noticing.")


class QualiaBroadcastPercept(CodeletPercept):
    """
    Broadcast of a conscious content to all processors (GWT style) — the 'everyone sees this now' moment.
    Example: "The realization arcs outward; other processes turn toward it like birds to a bell."
    """
    payload_brief: str = Field(description="What content is being broadcast.")
    expected_effects: List[str] = Field(description="2–3 likely effects on downstream processing.")
    narrative: str = Field(description="2–3 sentences marking the broadcast in-scene.")


class QualiaDecayPercept(CodeletPercept):
    """
    Fading back under the threshold; afterglow or residue may remain.
    Example: "The sharpness softens into a trace; useful, but no longer bright."
    """
    residue: Optional[str] = Field(None, description="What stays available (cue, feeling, tag).")
    narrative: str = Field(description="1–3 sentences of fading imagery.")


# =========================
# “ACTION–PERCEPTION LOOP / STRATEGY”
# =========================

class SensorimotorLoopPercept(CodeletPercept):
    """
    A tight perceive–act coupling described in story: perception changing action and action changing perception.
    Example: 'She listens to the fan’s pitch and trims the load by a hair; the room breathes easier.'"
    """
    sensory_input: str = Field(description="Dominant cue that drove the action.")
    motor_output: str = Field(description="Action taken in response.")
    feedback: str = Field(description="Immediate change perceived after the act.")
    narrative: str = Field(description="2–4 sentences showing the loop.")


class IntentionFeedbackPercept(CodeletPercept):
    """
    Narrative of how an intention’s execution fed back into feelings and next choices.
    Example: "The quick reply steadies the thread; relief nudges her toward bolder clarity."
    """
    intention: str = Field(description="Which intention is being enacted.")
    outcome_snippet: str = Field(description="What happened right after.")
    felt_delta: str = Field(description="How affect or confidence shifted.")
    narrative: str = Field(description="2–4 sentences linking cause and felt effect.")


class ErrorPredictionPercept(CodeletPercept):
    """
    Anticipatory sense of mismatch or failure risk.
    Example: "The timing feels a half-beat late; she senses a stall if she pushes now."
    """
    predicted_error: str = Field(description="Name the likely error or mismatch.")
    cue: str = Field(description="What hint suggests this risk.")
    mitigation_hint: str = Field(description="Small adjustment to reduce risk.")
    narrative: str = Field(description="2–3 sentences expressing the near-miss vibe.")


class PolicyShiftPercept(CodeletPercept):
    """
    Small strategic update chosen because of recent evidence (not a full plan; a tuning of policy).
    Example: "She widens the first pass again; the narrow funnel kept missing quiet options."
    """
    from_policy: str = Field(description="Short label of prior stance.")
    to_policy: str = Field(description="Short label of new stance.")
    reason: str = Field(description="Why the shift makes sense now.")
    narrative: str = Field(description="2–3 sentences embedding the shift in the scene.")


# =========================
# “AFFECTIVE REGULATION / SOMATIC”
# =========================

class AffectiveRegulationPercept(CodeletPercept):
    """
    On-the-fly modulation of emotional intensity or tone to preserve function ('smooth coping under stress').
    Example: "Heat rises; she lets it drain through longer exhales until the view clears."
    """
    target_affect: str = Field(description="Which affect is regulated (e.g., anxiety, anger, urgency).")
    technique: str = Field(description="Technique in story terms (slowing pace, reframing, grounding).")
    narrative: str = Field(description="2–4 sentences depicting the regulation as lived.")


class SomaticMarkerPercept(CodeletPercept):
    """
    Bodily (or simulated-bodily) cue that guides appraisal and choice.
    Example: "A tiny tightness warns her: this is where she over-promises if she speaks too fast."
    """
    marker_quality: str = Field(description="Name/feel of the cue (tightness, warmth, lightness...).")
    association_hint: str = Field(description="What it usually predicts (overcommit, mismatch, trust...).")
    narrative: str = Field(description="2–3 sentences grounding the cue in context.")


# =========================
# “TEMPORAL / NARRATIVE COHERENCE”
# =========================

class TemporalAnchoringPercept(CodeletPercept):
    """
    Places thought/action in time with a natural hook (event boundary, routine, 'next coffee').
    Example: "She’ll check again at the next kettle click; time feels more honest that way."
    """
    anchor: str = Field(description="Time/event boundary used as an anchor.")
    why_this_anchor: str = Field(description="Why this anchor is natural/robust.")
    narrative: str = Field(description="2–3 sentences tying intention to the anchor.")


class NarrativeContinuityPercept(CodeletPercept):
    """
    Keeps the self-story coherent across small shifts—names the seam so the fabric holds.
    Example: "She admits yesterday’s bravado was too large; 'steady' still fits, just in smaller letters."
    """
    seam_named: str = Field(description="Contradiction/tension being acknowledged.")
    reconciliation_hint: str = Field(description="How the claim is resized/refit.")
    narrative: str = Field(description="3–5 sentences that preserve voice without didactic tone.")

class TradeoffAssessmentOutupt(CodeletPercept):
    """
    When faced with a dilemma, narrate the thoughts behind the selection process where pros and cons are weighted against each other.
    """
    premise: str = Field(..., description="")
    option_a: str = Field(..., description="")
    option_b: str = Field(..., description="")
    option_a_pros: str = Field(..., description="")
    option_b_pros: str = Field(..., description="")
    option_a_cons: str = Field(..., description="")
    option_b_cons: str = Field(..., description="")

# ------------------------------------------------------------
#  dynamic percept-type → basemodel registry
# ------------------------------------------------------------

def build_percept_model_registry(percept_type: CodeletPerceptType) -> Dict[CodeletPerceptType, Type[BaseModel]]:
    """
    Dynamically build mapping CodeletPerceptType → Pydantic model class.
    1) Collect all BaseModel subclasses ending in 'Output' from current globals().
    2) Use fuzzy name similarity between percept.name and model.__name__.
    3) Allow manual overrides for odd naming.
    """
    # collect candidate models
    candidates: Dict[str, Type[BaseModel]] = {
        cls.__name__.lower().replace("output", ""): cls for cls in CodeletPercept.__subclasses__()
    }

    registry: Dict[CodeletPerceptType, Type[BaseModel]] = {}
    overrides = {}
    {
        # explicit mappings where fuzzy might miss
        "feeling": "codeletoutputfeeling",
        "feelingsocial": "codeletoutputfeelingsocial",
        "actionselection": "actionselectionoutput",
        "behavioractionselection": "behavioractionselectionoutput",
    }

    for percept in CodeletPerceptType:
        key = percept.name.lower()
        # use override if defined
        if key in overrides and overrides[key] in candidates:
            registry[percept] = candidates[overrides[key]]
            continue
        # fuzzy match against available model names
        match = difflib.get_close_matches(key, candidates.keys(), n=1, cutoff=0)
        if match:
            registry[percept] = candidates[match[0]]
        else:
            # fallback: try substring search
            for cname in candidates:
                if key in cname:
                    registry[percept] = candidates[cname]
                    break
    return registry[percept_type]


def model_for_percept(percept_type: CodeletPerceptType) -> Type[BaseModel]:
    """Return matching Pydantic model; raises KeyError if not found."""
    return build_percept_model_registry(percept_type)

PERCEPT_TYPES_BY_FAMILY: Dict[CodeletFamily, Dict[CodeletPerceptType, float]] = {
    CodeletFamily.Appraisals: {
        CodeletPerceptType.Feeling: 1.0,
        CodeletPerceptType.FeelingSocial: 0.8,
        CodeletPerceptType.AffectiveRegulation: 0.7,
        CodeletPerceptType.Worry: 0.6,
        CodeletPerceptType.Rumination: 0.5,
        CodeletPerceptType.Anticipation: 0.5,
        CodeletPerceptType.QualiaExplanation: 0.4,
    },
    CodeletFamily.MemoryAccessors: {
        CodeletPerceptType.MemoryAccessors: 1.0,
        CodeletPerceptType.EmotionalTriggers: 0.8,
        CodeletPerceptType.GeneralRelations: 0.6,
        CodeletPerceptType.QualiaEmergence: 0.4,
        CodeletPerceptType.QualiaInspection: 0.4,
    },
    CodeletFamily.ImaginationSimulation: {
        CodeletPerceptType.Simulate: 1.0,
        CodeletPerceptType.Anticipation: 0.9,
        CodeletPerceptType.ComplexPlan: 0.7,
        CodeletPerceptType.SimplePlan: 0.6,
        CodeletPerceptType.TradeoffAssessment: 0.5,
        CodeletPerceptType.PolicyShift: 0.4,
    },
    CodeletFamily.ValuationTradeOffs: {
        CodeletPerceptType.TradeoffAssessment: 1.0,
        CodeletPerceptType.GoalsIntentions: 0.8,
        CodeletPerceptType.ComplexPlan: 0.6,
        CodeletPerceptType.SimplePlan: 0.7,
        CodeletPerceptType.MetaEvaluate: 0.5,
    },
    CodeletFamily.MetaMonitoringIdentity: {
        CodeletPerceptType.SelfImage: 1.0,
        CodeletPerceptType.GeneralSelfImage: 0.9,
        CodeletPerceptType.MetaMonitor: 0.8,
        CodeletPerceptType.MetaInterpret: 0.7,
        CodeletPerceptType.MetaEvaluate: 0.7,
        CodeletPerceptType.MetaIntend: 0.6,
        CodeletPerceptType.CognitiveTraceReflection: 0.6,
        CodeletPerceptType.NarrativeContinuity: 0.5,
    },
    CodeletFamily.RegulationCoping: {
        CodeletPerceptType.AffectiveRegulation: 1.0,
        CodeletPerceptType.SomaticMarker: 0.8,
        CodeletPerceptType.Worry: 0.6,
        CodeletPerceptType.InputRequest: 0.5,
    },
    CodeletFamily.DriversHomeostatis: {
        CodeletPerceptType.NeedSignal: 1.0,
        CodeletPerceptType.AffectiveRegulation: 0.6,
        CodeletPerceptType.SensorimotorLoop: 0.5,
    },
    CodeletFamily.Attention: {
        CodeletPerceptType.AttentionFocus: 1.0,
        CodeletPerceptType.MetaControl: 0.8,
        CodeletPerceptType.MetaIntend: 0.7,
        CodeletPerceptType.TemporalAnchoring: 0.4,
    },
    CodeletFamily.ActionTendencies: {
        CodeletPerceptType.ActionSelection: 1.0,
        CodeletPerceptType.AutomatizedAction: 0.9,
        CodeletPerceptType.MicroAdjustment: 0.8,
        CodeletPerceptType.FlowCognition: 0.7,
        CodeletPerceptType.SkillLearningMoment: 0.6,
        CodeletPerceptType.DorsalStreamAdjustment: 0.5,
    },
    CodeletFamily.Narratives: {
        CodeletPerceptType.InnerMonologue: 1.0,
        CodeletPerceptType.NarrativeContinuity: 0.8,
        CodeletPerceptType.GeneralSelfImage: 0.7,
        CodeletPerceptType.GeneralGoalsIntentions: 0.7,
        CodeletPerceptType.GeneralRelations: 0.6,
        CodeletPerceptType.QualiaExplanation: 0.5,
    },
}

def codelet_families_to_percept_types(
        families: List[CodeletFamily],
        k: Optional[int] = None
) -> Dict[CodeletPerceptType, float]:
    """
    Combine percept strengths across multiple CodeletFamilies.

    Args:
        families: List of CodeletFamily enums to include.
        k: Optional limit for the number of top percepts to return.

    Returns:
        Dict[CodeletPerceptType, float]: Sorted descending by combined strength.
    """
    combined: Dict[CodeletPerceptType, float] = defaultdict(float)

    # Combine strengths
    for fam in families:
        percepts = PERCEPT_TYPES_BY_FAMILY.get(fam, {})
        for percept, strength in percepts.items():
            combined[percept] += strength

    # Sort descending by strength
    sorted_items = sorted(combined.items(), key=lambda x: x[1], reverse=True)

    # Apply limiter
    if k is not None:
        sorted_items = sorted_items[:k]

    return dict(sorted_items)

def sample_percept_types(
    families: List[CodeletFamily],
    k: int,
    temperature: float = 0.0
) -> List[CodeletPerceptType]:
    """
    Return a list of percept types from given families.
    Mixes deterministic top-k with weighted randomness based on temperature.

    Args:
        families: List of CodeletFamily enums to include.
        k: Number of percepts to return (required).
        temperature: Float between 0 and 1; controls randomness.
            0.0 = fully deterministic top-k
            1.0 = fully random (weighted by strength)

    Returns:
        List[CodeletPerceptType]: Selected percept types.
    """
    if not 0.0 <= temperature <= 1.0:
        raise ValueError("temperature must be between 0.0 and 1.0")

    combined = codelet_families_to_percept_types(families)

    if not combined:
        return []

    # Convert dict to lists for sampling
    percepts, strengths = zip(*combined.items())

    # Normalize strengths for probabilistic selection
    total_strength = sum(strengths)
    probs = [s / total_strength for s in strengths]

    # Deterministic part (top-n)
    sorted_percepts = [p for p, _ in sorted(combined.items(), key=lambda x: x[1], reverse=True)]
    deterministic_count = int(round(k * (1 - temperature)))
    random_count = k - deterministic_count

    selected = []

    # Deterministic top ones
    selected.extend(sorted_percepts[:deterministic_count])

    # Random weighted by strength
    if random_count > 0:
        # Use random.choices to sample with probability weighting
        selected.extend(random.choices(percepts, weights=probs, k=random_count))

    # Shuffle final list a little for variety
    random.shuffle(selected)
    return selected[:k]


if __name__ == '__main__':
    defs = []
    for model in CodeletPercept.__subclasses__():
        flat = flatten_pydantic_model(model)
        d = {
            "name": model.__name__,
            "description": model.__doc__,
            "type": flat
        }
        defs.append(d)

    FeatureContainer = create_basemodel(defs, randomnize_entries=True)
    print(generate_pydantic_json_schema_llm(FeatureContainer, beautify=True))
