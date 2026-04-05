import enum
from enum import StrEnum

class MetaDebugLevel(StrEnum):
    Debug = "Debug"
    Low = "Low"
    Mid = "Mid"
    High = "High"


class KnoxelSubtypeBase(StrEnum):
    pass

class NoSubtype(KnoxelSubtypeBase):
    NoSubtype = "NoSubtype"

class ActionType(KnoxelSubtypeBase):
    Reply = "Reply"
    Ignore = "Ignore"
    ToolCallAndReply = "ToolCallAndReply"
    Sleep = "Sleep"
    InitiateUserConversation = "InitiateUserConversation"
    Idle = "Idle"
    ToolCall = "ToolCall"


class FeatureType(KnoxelSubtypeBase):
    Dialogue = "Dialogue"
    Feeling = "Feeling"
    SituationalModel = "SituationalModel"
    AttentionFocus = "AttentionFocus"
    ConsciousWorkspace = "ConsciousWorkspace"
    MemoryRecall = "MemoryRecall"
    SubjectiveExperience = "SubjectiveExperience"
    ActionSimulation = "ActionSimulation"
    ActionRating = "ActionRating"
    Action = "Action"
    ActionExpectation = "ActionExpectation"
    NarrativeUpdate = "NarrativeUpdate"
    ExpectationOutcome = "ExpectationOutcome"
    StoryWildcard = "StoryWildcard"
    Expectation = "Expectation"
    Goal = "Goal"
    Narrative = "Narrative"
    WorldEvent = "WorldEvent"
    Thought = "Thought"
    ExternalThought = "ExternalThought"
    MetaInsight = "MetaInsight"
    SystemMessage = "SystemMessage"
    CodeletOutput = "CodeletOutput"
    CodeletPercept = "CodeletPercept"


class StimulusType(KnoxelSubtypeBase):
    UserMessage = "UserMessage"
    CompanionMessage = "CompanionMessage"
    SystemMessage = "SystemMessage"
    UserInactivity = "UserInactivity"  # Checks time.time() - self.last_interaction_time. If it exceeds user_inactivity_timeout, it generates a stimulus. Narrative Content: `"The user has been inactive for 10 minutes. My need for connection is decreasing."*
    TimeOfDayChange = "TimeOfDayChange"  # The Shell can track the real-world datetime. When the hour changes, or it crosses a threshold (e.g., from "afternoon" to "evening"), it can generate a stimulus. Narrative Content: `"The time is now 7 PM. It is officially evening."*
    LowNeedTrigger = "LowNeedTrigger"  # The Shell can periodically (e.g., every 5-10 minutes of inactivity) ask the Ghost for its current needs state. If a need (like connection or learning_growth) drops below a critical threshold, the Shell can generate a stimulus.  Narrative Content: `"Internal monitoring shows my need for relevance is critically low (0.2). I feel a strong urge to be useful."*
    WakeUp = "WakeUp"  # If the Ghost was Sleeping, the Shell generates this stimulus when the sleep duration is over or if the user interrupts the sleep. Narrative Content: "The 8-hour sleep cycle has completed. I am now awake."* or"The user sent a message, interrupting my sleep cycle."*
    EngagementOpportunity = "EngagementOpportunity"  # From the new strategist
    MemoryRecall = "MemoryRecall"


class StimulusGroup(KnoxelSubtypeBase):
    All = "All"
    WorldInput = "WorldInput"
    SystemInput = "SystemInput"
    ToolInput = "ToolInput"
    SelfInput = "SelfInput"


class EntityClass(KnoxelSubtypeBase):
    Self = "Self"
    Human = "Human"
    AI = "AI"
    Agent = "Agent"


class NarrativeTypes(KnoxelSubtypeBase):
    AttentionFocus = "AttentionFocus"
    SelfImage = "SelfImage"
    PsychologicalAnalysis = "PsychologicalAnalysis"
    Relations = "Relations"
    ConflictResolution = "ConflictResolution"
    EmotionalTriggers = "EmotionalTriggers"
    GoalsIntentions = "GoalsIntentions"
    BehaviorActionSelection = "BehaviorActionSelection"
    InnerMonologue = "InnerMonologue"


class ClusterType(KnoxelSubtypeBase):
    GraphEpisode = "graph_episode"
    Topical = "topical"
    Temporal = "temporal"#


class InterlocusType(enum.IntEnum):
    Public = 1
    Undefined = 0
    PrivateReportable = -1
    PrivateInternal = -2


class StimulusTriage(KnoxelSubtypeBase):
    Insignificant = "Insignificant"  # ignore simulus, add as causal feature only
    Moderate = "Moderate"  # fast-track the action generation with limited cws
    Significant = "Significant"  # full pipeline
    Critical = "Critical"
