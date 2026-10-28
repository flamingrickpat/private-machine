"""Subtype vocabularies that record models and prompts branch on.

owns: the closed sets of names for record subtypes (action channel, reality
kind, stimulus type, feature family and subtype, intention type, and the rest).
not own: the record models themselves (see ``pm.records.base`` for the package
overview) or the semantic taxonomy used by memory reconsolidation, which lives
in ``pm.records.taxonomy``.

Every subtype enum derives from ``RecordSubtypeBase`` so that
``RecordBase.get_sub_type`` has one return type, and so that a subtype keeps
the same string value in records, seeds, and prompts after a class rename.
The string values are persisted data: change an enum member name only together
with the stored records that hold it.
"""
import enum
from enum import StrEnum, auto


class RecordSubtypeBase(StrEnum):
    """Marker base of every record subtype vocabulary."""

class NoSubtype(RecordSubtypeBase):
    """Subtype of a record that has no finer vocabulary yet."""

    NoSubtype = "NoSubtype"


class ActionType(RecordSubtypeBase):
    """
    The basic action types the AI can choose from.
    """
    Reply = "Reply"
    Ignore = "Ignore"
    ToolCall = "ToolCall"
    ToolCallAndReply = "ToolCallAndReply"
    InitiateUserConversation = "InitiateUserConversation"
    Loopback = "Loopback"
    Sleep = "Sleep"
    Idle = "Idle"


class RealityKind(RecordSubtypeBase):
    """
    The kind of reality a RealityRecord describes. Real life, an embodied
    game world (minecraft body), a fantasy campaign, system/meta reality.
    """
    Real = "Real"
    Game = "Game"
    Fantasy = "Fantasy"
    System = "System"


class ActionSchemaStatus(RecordSubtypeBase):
    """
    Lifecycle of an ActionSchemaRecord plan spec.
    """
    Proposed = "Proposed"
    Selected = "Selected"
    Executed = "Executed"
    Abandoned = "Abandoned"


class ActionSchemaExecutionMode(RecordSubtypeBase):
    """
    How an ActionSchemaRecord gets executed.
    SingleShotMcp: build a prompt for single-shot MCP/tool generation.
    Skill/Workflow: hand to the respective executor.
    ProseExternal: another skill verbalizes/acts it out.
    """
    SingleShotMcp = "SingleShotMcp"
    Skill = "Skill"
    Workflow = "Workflow"
    ProseExternal = "ProseExternal"


class ActionSchemaVerbalization(RecordSubtypeBase):
    """
    Who turns the ActionSchemaRecord into user-facing prose. Never the
    personality simulation itself.
    """
    Roleplay = "Roleplay"
    Assistant = "Assistant"
    None_ = "None"


class FrameEnum(StrEnum):
    """
    Perspective frame of a knoxel inside its reality.
    InWorld: facts inside the reality (house location, orgblorg kills goblin).
    Meta: facts about the reality (server ip, "we play every friday").
    """
    InWorld = "in_world"
    Meta = "meta"


class SourceKind(StrEnum):
    """
    Provenance of a knoxel: who/what brought it into the ghost.
    """
    User = "User"
    Agent = "Agent"
    Workflow = "Workflow"
    Self = "Self"
    System = "System"
    Migration = "Migration"


class FeatureType(RecordSubtypeBase):
    """
    Coarse prompt-facing feature families.

    Keep these intentionally broad. The point is to keep the prompt-builder stable even if the
    finer semantic taxonomy evolves. Most prompt assembly code should branch on these four values
    first and only use the subtype for finer formatting or prioritization.

    Dialogue
        External utterance-level content. Usually user speech, assistant speech, quoted snippets,
        or message-level communicative material that should appear as direct interaction.

    WorldEvent
        Scene-level or environment-level events. This includes things that happened "in the world"
        of the interaction: silence, interruption, time passing, a new message appearing, a tool
        result arriving, or a user doing something that is narrated more like an event than an
        utterance.

    InnerMonologue
        Prompt-visible internal narration. This is reportable inner framing, not raw architecture.
        It is the textual layer that says what the companion is noticing, leaning toward, dreading,
        hoping, suspecting, remembering, or privately considering.

    StateUpdate
        Prompt-visible architecture summary. This is used when a state-editing knoxel has changed
        beliefs, goals, attention, ontology bindings, modulation state, or similar internals and
        you want to expose that in a concise narrative way inside the prompt.
    """
    InitFeatureForPersonalityState = "InitFeatureForPersonalityState"
    SystemMessage = "SystemMessage"
    SubjectiveExperience = "SubjectiveExperience"
    Thought = "Thought"
    ExternalThought = "ExternalThought"
    Dialogue = "Dialogue"
    WorldEvent = "WorldEvent"
    InnerMonologue = "InnerMonologue"
    StateUpdate = "StateUpdate"


class StimulusType(RecordSubtypeBase):
    UserMessage = "UserMessage"
    CompanionMessage = "CompanionMessage"
    SystemMessage = "SystemMessage"
    SelfGeneratedThought = "SelfGeneratedThought"  # spec v2 §5: self-originated inner thought enters the cycle as a stimulus from the self
    UserInactivity = "UserInactivity"  # Checks time.time() - self.last_interaction_time. If it exceeds user_inactivity_timeout, it generates a stimulus. Narrative Content: `"The user has been inactive for 10 minutes. My need for connection is decreasing."*
    TimeOfDayChange = "TimeOfDayChange"  # The Shell can track the real-world datetime. When the hour changes, or it crosses a threshold (e.g., from "afternoon" to "evening"), it can generate a stimulus. Narrative Content: `"The time is now 7 PM. It is officially evening."*
    LowNeedTrigger = "LowNeedTrigger"  # The Shell can periodically (e.g., every 5-10 minutes of inactivity) ask the Ghost for its current needs state. If a need (like connection or learning_growth) drops below a critical threshold, the Shell can generate a stimulus.  Narrative Content: `"Internal monitoring shows my need for relevance is critically low (0.2). I feel a strong urge to be useful."*
    WakeUp = "WakeUp"  # If the Ghost was Sleeping, the Shell generates this stimulus when the sleep duration is over or if the user interrupts the sleep. Narrative Content: "The 8-hour sleep cycle has completed. I am now awake."* or"The user sent a message, interrupting my sleep cycle."*
    EngagementOpportunity = "EngagementOpportunity"  # From the new strategist
    MemoryRecall = "MemoryRecall"


class EntityClass(RecordSubtypeBase):
    Self = "Self"
    Human = "Human"
    AI = "AI"
    Agent = "Agent"


class NarrativeTypes(RecordSubtypeBase):
    AttentionFocus = "AttentionFocus"
    SelfImage = "SelfImage"
    PsychologicalAnalysis = "PsychologicalAnalysis"
    Relations = "Relations"
    ConflictResolution = "ConflictResolution"
    EmotionalTriggers = "EmotionalTriggers"
    GoalsIntentions = "GoalsIntentions"
    BehaviorActionSelection = "BehaviorActionSelection"
    InnerMonologue = "InnerMonologue"


class ClusterType(RecordSubtypeBase):
    GraphEpisode = "graph_episode"
    Topical = "topical"
    Temporal = "temporal"


class InterlocusType(enum.IntEnum):
    """Visibility of one feature to the conscious agent.

    ``Public`` and ``PrivateReportable`` can reach a prompt as dialogue or
    thought. ``PrivateInternal`` is third-person material that the persona does
    not fully see. ``Undefined`` is the neutral default.
    """

    Public = 1
    Undefined = 0
    PrivateReportable = -1
    PrivateInternal = -2


class StimulusTriage(RecordSubtypeBase):
    Insignificant = "Insignificant"  # ignore simulus, add as causal feature only
    Moderate = "Moderate"  # fast-track the action generation with limited cws
    Significant = "Significant"  # full pipeline
    Critical = "Critical"


class FeatureSubtype(RecordSubtypeBase):
    """
    Fine-grained subtype seeds for prompt feature generation.

    These values are *guidance seeds* for the agent and for DSPy modules, not a final ontology.
    They exist to suggest the kinds of prompt-visible elements the system may generate from
    state-editing knoxels, memory retrieval, simulations, and metacognitive traces.

    The enum is deliberately broad and partly overlapping. That is intentional. The architecture
    is exploratory, and the subtype system should suggest possibilities instead of overconstraining
    generation too early.

    Surface / scene-oriented seeds
    ------------------------------
    Percept
        A directly noticed thing in the present moment. Example: "The tone suddenly feels colder."

    EntityState
        A prompt-visible statement about the current state of a person, agent, or named entity.
        Example: "The user seems amused but impatient."

    RelationshipState
        A prompt-visible statement about the current social or relational configuration.
        Example: "The exchange feels less trusting than a moment ago."

    EnvironmentState
        A prompt-visible statement about the surrounding situational frame.
        Example: "The conversation has slowed and the silence feels noticeable."

    Interpretation / appraisal seeds
    --------------------------------
    Appraisal
        A compact emotional or motivational appraisal of a situation, event, or remembered item.

    Interpretation
        A best-current reading of what something means. Example: "This may have been banter."

    Anomaly
        A prompt-visible marker that something does not fit the current model, expectation, or plan.

    BeliefTension
        A conflict, contradiction, or unresolved mismatch between beliefs, evidence, goals, or
        self-model assumptions.

    RelevanceAssessment
        An explanation of why something matters now. Example: "This matters because it threatens rapport."

    CausalReading
        A compact causal interpretation linking conditions to likely consequences.

    Attention / workspace seeds
    ---------------------------
    AttentionCandidate
        Something nominated for attention competition.

    Coalition
        A group of mutually supporting items that jointly push for conscious access or prompt priority.

    Broadcast
        A marker that some content won access and became globally relevant for downstream processing.

    SalienceMarker
        A compact explanation of why something is salient: urgent, novel, threatening, goal-relevant, etc.

    Control / intention seeds
    -------------------------
    Goal
        A prompt-visible current goal, desire, pull, pressure, or intended outcome.

    Intention
        A more committed or action-shaping goal state than a casual goal mention.

    DistalIntention
        A long-horizon intention that spans many turns or cycles.

    ProximalIntention
        A short-horizon intention relevant to the current interaction window.

    MotorIntention
        The smallest immediately enactable intention, typically very close to a concrete action.

    PlanState
        A compact representation of where the current plan stands.

    PlanRevision
        A prompt-visible note that the current plan changed.

    ActionBias
        A note about the current directional push in action selection.
        Example: "There is a pull toward reassurance rather than challenge."

    ActionSimulation
        A simulation-derived prompt element about possible next moves and their consequences.

    ActionRating
        A ranking or evaluative statement about candidate actions.

    Subjective / reportability seeds
    --------------------------------
    Feeling
        A reportable felt state. Usually softer and more surface-level than a structured appraisal.

    Quale
        A reusable qualitative descriptor anchored in the private ontology.
        Example: "It stings." / "This feels heavy."

    SubjectiveExperience
        A richer description of what the current internal state is like from the system's own frame.

    MetaRepresentation
        A reportable representation of the system's own process or state.
        Example: "I keep circling the same interpretation."

    PrivateOntologyTerm
        A term from the system's own evolving inner vocabulary, whether or not it is used literally.

    ModulationState
        A prompt-visible note about current self-modulation or control mode.
        Example: "I am leaning cautious and withholding right now."

    Memory / trace seeds
    --------------------
    MemoryRecall
        A retrieved prior episode or remembered fragment.

    EpisodeLink
        A link between the current state and one or more remembered episodes.

    ReasoningTrace
        A compact trace of how some conclusion or shift was reached.

    CausalTrace
        A trace focused on antecedents, transformations, and consequences.

    MetaCognitiveTrace
        A trace of self-monitoring, self-correction, or internal diagnostic activity.

    ExpectationOutcome
        A comparison between what was expected and what actually happened.

    Communication / role seeds
    --------------------------
    Message
        A message-level communicative artifact, broader than direct dialogue.

    Query
        A question, request, or information-seeking act.

    Answer
        A reply, response, or resolution act.

    ParticipantRole
        A currently active social or conversational role.
        Example: "comforter", "challenger", "explainer", "listener"

    Long-horizon coherence seeds
    ----------------------------
    Narrative
        A longer-arc self-story or relational story used to keep coherence across turns.

    NarrativeUpdate
        A change to the currently active narrative understanding.

    MetaInsight
        A higher-level realization about the system's own behavior, tendencies, or recurring patterns.
    """
    Percept = auto()
    EntityState = auto()
    RelationshipState = auto()
    EnvironmentState = auto()

    Appraisal = auto()
    Interpretation = auto()
    Anomaly = auto()
    BeliefTension = auto()
    RelevanceAssessment = auto()
    CausalReading = auto()

    AttentionCandidate = auto()
    Coalition = auto()
    Broadcast = auto()
    SalienceMarker = auto()

    Goal = auto()
    Intention = auto()
    DistalIntention = auto()
    ProximalIntention = auto()
    MotorIntention = auto()
    PlanState = auto()
    PlanRevision = auto()
    ActionBias = auto()
    ActionSimulation = auto()
    ActionRating = auto()

    Feeling = auto()
    Quale = auto()
    SubjectiveExperience = auto()
    MetaRepresentation = auto()
    PrivateOntologyTerm = auto()
    ModulationState = auto()

    MemoryRecall = auto()
    EpisodeLink = auto()
    ReasoningTrace = auto()
    CausalTrace = auto()
    MetaCognitiveTrace = auto()
    ExpectationOutcome = auto()

    Message = auto()
    Query = auto()
    Answer = auto()
    ParticipantRole = auto()

    Narrative = auto()
    NarrativeUpdate = auto()
    MetaInsight = auto()

class IntentionType(RecordSubtypeBase):
    """What an ``Intention`` record is about: whose state, and which kind."""

    SelfStatePhysical = auto()
    SelfStateMental = auto()
    SelfSituation = auto()

    EntityStatePhysical = auto()
    EntityStateMental = auto()
    EntitySituation = auto()

    WorldState = auto()
    WorldSituation = auto()
