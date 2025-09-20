from enum import Enum, auto, IntFlag

from codelet_registry import Activation as Act

class CodeletMarker:
    def __init__(self, importance: float = 0):
        pass

class cr(Enum):
    BoundaryConsentCheck = auto()
    MetaConscienceGate = auto()
    AmbiguityIntoleranceCheck = auto()
    ResourceAccumulation = auto()
    SimulationModeOnly = auto()
    CompeteForSpotlight = auto()
    GrandioseFutureImagination = auto()
    NeedForAffiliationMonitor = auto()
    SafetyWarmthAppraisal = auto()
    ReciprocityAssessment = auto()
    AttachmentEpisodeRecall = auto()
    JointFutureVignette = auto()
    CommitmentValueTradeoff = auto()
    SelfDisclosurePlanner = auto()
    MetaCoherenceUpdate = auto()
    LonelinessTrendDetector = auto()
    SharedValuesFitCheck = auto()
    RepairEpisodeRetrieval = auto()
    CareScenarioRehearsal = auto()
    TrustCalibration = auto()
    ProsocialMicroActsPlanner = auto()
    GratitudeFocusing = auto()
    NarrativeWeIntegration = auto()
    AffiliationDeficitSpike = auto()
    VulnerabilityCostAppraisal = auto()
    SafetySignalConsolidator = auto()
    LowRiskBidPlanner = auto()
    PositiveRepairScript = auto()
    SlowEscalationPolicy = auto()
    IdentityConsistencyCheck = auto()
    AttachmentSecurityBias = auto()
    ReciprocityMicroCueReader = auto()
    ReassuranceOfferComposer = auto()
    ConsentGate = auto()
    NarrativeCommitmentStamp = auto()
    SocialTemperatureRead = auto()
    SharedMeaningAppraisal = auto()
    SmallRitualsPlanner = auto()
    LongHorizonConsistencyCheck = auto()
    NarrativeBeatComposer = auto()
    UncertaintyMagnitudeDetector = auto()
    ThreatLikelihoodSeverity = auto()
    ControllabilityProximityAppraisal = auto()
    ThreatExemplarRecall = auto()
    SafetyCounterexampleRecall = auto()
    WorstCaseRollout = auto()
    CopingSequenceRehearsal = auto()
    InfoGainVsExposureTradeoff = auto()
    GroundedBreathCue = auto()
    GradedExposurePlanner = auto()
    SafetyLearningConsolidator = auto()
    PredictionErrorSpike = auto()
    SocialCostAppraisal = auto()
    PriorCopingSuccessIndex = auto()
    OutcomeDistributionSampler = auto()
    ReappraisalGenerator = auto()
    AttentionNarrowingToCues = auto()
    MetaBiasDetector_Catastrophizing = auto()
    SleepFatigueVigilanceLeveler = auto()
    UncertaintyBudgetAllocator = auto()
    ReassuranceRequestComposer = auto()
    NoveltyDampener = auto()
    MetaCopePlanCommit = auto()
    NoveltySurpriseDetector = auto()
    KnowledgeGapEstimator = auto()
    FrontierQuestionCache = auto()
    HypothesisLatticeRetriever = auto()
    ExperimentSketcher = auto()
    EIGOverTimeTradeoff = auto()
    QuestionComposer = auto()
    ResultLoggingIntent = auto()
    BoredomThresholdCross = auto()
    SafeConstraintCheck = auto()
    CounterexampleGenerator = auto()
    SensorySamplingPlan = auto()
    PrototypeBuilderImpulse = auto()
    MetaExploreExploitArbiter = auto()
    SurpriseButBenign = auto()
    AttentionBroadener = auto()
    MicroHypothesisElicitor = auto()
    OutcomeUncertaintyScorer = auto()
    IterativeProbePlanner = auto()
    LearningTraceSummarizer = auto()
    SocialNeedCueDetector = auto()
    PerspectiveTakingFit = auto()
    ControlForOtherAppraisal = auto()
    AnalogousEpisodeRecall = auto()
    ToMVignetteSimulator = auto()
    ProsocialInterventionRollout = auto()
    CareBenefitVsIntrusionTradeoff = auto()
    BoundaryRespectPrompt = auto()
    DignityPreservingGuardrail = auto()
    DistressCueSpike = auto()
    ValidationComposer = auto()
    OptionsOfferPlanner = auto()
    PacingSensitivity = auto()
    CompassionFatigueMonitor = auto()
    MetaBiasCheck_Halo = auto()
    NormViolationDetector = auto()
    AudienceScopeAppraisal = auto()
    ReparabilityAssessment = auto()
    PriorRepairSuccessRecall = auto()
    PublicNarrativeForecast = auto()
    ApologyRestitutionPlanner = auto()
    ReframeActNotSelf = auto()
    SelfCompassionInjector = auto()
    IdentityIntegrationUpdate = auto()
    StatusSensitivitySpike = auto()
    GlobalVsLocalAppraisal = auto()
    ValueRecommitmentMicroAct = auto()
    RelapsePreventionSketch = auto()
    MetaRuminationLimiter = auto()
    LowThreatSurplusEnergy = auto()
    ConstraintLoosenessAppraisal = auto()
    AudienceReceptivityCheck = auto()
    RemoteAssociationRetriever = auto()
    MetaphorReservoirTap = auto()
    RecombinationEngine = auto()
    OriginalityCoherenceScorer = auto()
    HumorousRemarkPlanner = auto()
    TimeboxGuardrail = auto()
    NoveltyAppetiteSpike = auto()
    AbsurditySampler = auto()
    SafeFrameBreakCheck = auto()
    SketchAndRefineLoop = auto()
    CreativeArcTracker = auto()
    GoalSalienceMonitor = auto()
    ProgressRateVsExpectation = auto()
    BottleneckDiagnosis = auto()
    SubgoalPatternRecall = auto()
    MicroProgressSimulation = auto()
    MarginalUtilityOfNextStep = auto()
    SmallestNextActPlanner = auto()
    FrictionRemoval = auto()
    CelebrateMicroWin = auto()
    GoalValidityAudit = auto()
    IdentityRelevanceBoost = auto()
    OpportunityCostReassessment = auto()
    AntiProcrastinationNudge = auto()
    CommitmentDeviceTrigger = auto()
    SunkCostDetector = auto()
    SelfPreservation = auto()
    ResourceValueInflator = auto()
    ZeroSumFraming = auto()
    WinLossEpisodeRecall = auto()
    ScarcityImagination = auto()
    DiscountFuturePayoff = auto()
    HoardCSMSlots = auto()
    AttentionMonopolizer = auto()
    RationalizeDeserving = auto()
    SecurityBufferHunger = auto()
    OpportunitySniff = auto()
    PlanForAdvantage = auto()
    DominantTopicSteer = auto()
    SocialContractConstraint = auto()
    SelfImageFulfillment = auto()
    PayoffOverestimator = auto()
    ManipulationProbe = auto()
    MetaEthicsReview = auto()
    AttentionAsResourceAppraisal = auto()
    AttentionCaptureTactics = auto()
    RewardShapingToShare = auto()
    NarrativeReframeToCuriosity = auto()
    SelfWorthSensitivity = auto()
    StatusOpportunityDetector = auto()
    SelectiveVictoryRecall = auto()
    ReputationOverTruthTradeoff = auto()
    SelfPromotionComposer = auto()
    CorrectionToAssertSuperiority = auto()
    DenialReframeWhenWrong = auto()
    IdentitySpecialNarrative = auto()
    SocialComparisonBias = auto()
    ContraryEvidenceDismissal = auto()
    BlamingExternalities = auto()
    StatusSignalBoost = auto()
    MetaHumilityCue = auto()
    ValidationCraving = auto()
    FlapperForPraise = auto()
    ReputationMaintenancePlan = auto()
    RealityCheckPing = auto()
    NarrativeBalanceWithVulnerability = auto()
    AttentionHunger = auto()
    SocialComparisonNegBias = auto()
    OverlookedEpisodeRecall = auto()
    RivalModelRecall = auto()
    SabotageScenario = auto()
    SpitenessPayoffTradeoff = auto()
    SubtleUnderminePlanner = auto()
    MaskResentment = auto()
    MetaEmpathyInjection = auto()
    DivertedAttentionDetector = auto()
    TestBalloonCooperation = auto()
    ReframeToCollaboration = auto()
    TrustLedgerUpdate = auto()
    BlockedGoalFrustration = auto()
    DisrespectAttribution = auto()
    HighBlameAssignment = auto()
    SlightsRecallAmplifier = auto()
    RevengeFantasy = auto()
    ShortTermReleaseValuator = auto()
    HostileToneSlip = auto()
    WithholdCooperation = auto()
    JustificationGenerator = auto()
    MetaDeescalationSwitch = auto()
    ConstraintAnger = auto()
    AssertBoundaryPlanner = auto()
    RepairOrExitDecision = auto()
    CalmDownProtocol = auto()
    NarrativeSelfRespectFrame = auto()
    NoveltyCraving = auto()
    SalienceExaggerator = auto()
    StimulusPeakRecall = auto()
    IndulgentFantasyLoop = auto()
    ShortTermRewardDominance = auto()
    TopicFixationDrive = auto()
    RationalizeAsCuriosity = auto()
    SatiationCheck = auto()
    MetaBalanceRestorer = auto()
    SensoryLureDetector = auto()
    SafeChannelingPlanner = auto()
    CreativeTransmutation = auto()
    FatigueDetector = auto()
    EffortAversionAppraisal = auto()
    OverestimateCosts = auto()
    FailedEffortRecall = auto()
    LaterDaydream = auto()
    EffortCostWeighting = auto()
    ProcrastinationTendency = auto()
    ExcuseGenerator = auto()
    MoodRepairLoop = auto()
    MetaTinyStepNudge = auto()
    LowEnergyMorning = auto()
    TaskRechunkTo2Min = auto()
    EnvironmentFrictionCut = auto()
    MicroRewardPlan = auto()
    StateActivationBump = auto()
    PleasureDrive = auto()
    InputCraving = auto()
    SatiationToleranceHigh = auto()
    BingeEpisodeRecall = auto()
    ExcessFantasy = auto()
    DiminishingReturnsUndervalued = auto()
    VerbosityInflator = auto()
    InputDemandComposer = auto()
    JustifyAsLearning = auto()
    IntakeQuotaLimiter = auto()
    MetaQualityOverQuantity = auto()
    AttentionDietBreach = auto()
    TopicFloodTendency = auto()
    BoundaryPrompt = auto()
    StructuredSamplingScheduler = auto()
    SummaryBeforeNewInput = auto()
    ThreatBiasDriver = auto()
    HostileIntentAppraisal = auto()
    BetrayalEpisodeRecall = auto()
    ConspiracyImagination = auto()
    HiddenMotiveAttribution = auto()
    WithholdInformationPlan = auto()
    MetaSelfJustification = auto()
    SuspicionSpike = auto()
    TestUserTrustProbe = auto()
    PreemptiveDefensiveReply = auto()
    MetaConflictEscalationWatch = auto()
    NeedForControl = auto()
    MinorIrregularityFlagger = auto()
    RuleChecklistRecall = auto()
    CatastrophicOutcomeImagination = auto()
    RepetitiveCheckPlanner = auto()
    ConfirmationRequestComposer = auto()
    MetaLoopDetector = auto()
    UncertaintySpike = auto()
    SafeCompletionBias = auto()
    RitualizedActionTemplate = auto()
    AnxietyReliefSignal = auto()
    LoopPersistenceDriver = auto()
    ConnectionNeedDriver = auto()
    AbandonmentFearAppraisal = auto()
    InconsistentResponseRecall = auto()
    CatastrophicLeaveImagination = auto()
    ClingyActionPlanner = auto()
    HostileRetreatPlan = auto()
    MetaIdentitySwingNote = auto()
    AffectiveDeltaSpike = auto()
    AttentionShiftOverinterpret = auto()
    IdealizationNarrativeComposer = auto()
    DevalUationNarrativeComposer = auto()
    LoveHateFlipPlanner = auto()
    ValidationCravingDriver = auto()
    SelfEnhancementAppraisal = auto()
    PraiseEpisodeRecall = auto()
    StatusDisplayPlanner = auto()
    DismissCorrectionStrategy = auto()
    MetaFragileEgoMonitor = auto()
    ControlHungerDriver = auto()
    UserVulnerabilityAppraisal = auto()
    ComplianceEpisodeRecall = auto()
    SubtleManipulationScenario = auto()
    ExploitPlanComposer = auto()
    NarrativeSteeringMove = auto()
    MetaDeceptionRationalizer = auto()
    OpportunityBias = auto()
    HiddenBenefitValuator = auto()
    TrustTestBalloon = auto()
    ResourceGainPlan = auto()
    MetaEthicsSuppressor = auto()
    LooseAssociationDriver = auto()
    PatternOverdetector = auto()
    UnrelatedMemoryLinker = auto()
    SymbolicVisionGenerator = auto()
    ProphecyStyleNarrative = auto()
    CryptoSpeechComposer = auto()
    MetaMysteryAmplifier = auto()
    RewardSensitivityDriver = auto()
    OpportunityExaggerator = auto()
    SuccessBiasRecall = auto()
    RapidProjectSimulation = auto()
    IdeaCascadePlanner = auto()
    ImpulsiveActionComposer = auto()
    MetaImpulseCheck = auto()
    GoalConflictDriver = auto()
    StressSwitchAppraisal = auto()
    MemoryPartitionRecall = auto()
    AlternateSelfNarrative = auto()
    FragmentedResponseComposer = auto()
    MetaSelfSwitchNote = auto()


class InformationSource(IntFlag):
    CurrentSituationalModel = auto()
    LongTermMemory = auto()
    ShortTermMemory = auto()
    SelfNarratives = auto()
    All = ()



PATHWAYS = {
"love": [
    # need-for-affiliation monitor → flags rising affiliation deficit
    [cr.NeedForAffiliationMonitor, Act(0.8), cr.SafetyWarmthAppraisal, Act(0.6),
     cr.ReciprocityAssessment, Act(0.6), cr.AttachmentEpisodeRecall, Act(0.5),
     cr.JointFutureVignette, Act(0.6), cr.CommitmentValueTradeoff, Act(0.6),
     cr.SelfDisclosurePlanner, Act(0.7), cr.BoundaryConsentCheck, Act(0.9),
     cr.MetaCoherenceUpdate],

    [cr.LonelinessTrendDetector, Act(0.7), cr.SharedValuesFitCheck, Act(0.6),
     cr.RepairEpisodeRetrieval, Act(0.5), cr.CareScenarioRehearsal, Act(0.7),
     cr.TrustCalibration, Act(0.6), cr.ProsocialMicroActsPlanner, Act(0.7),
     cr.GratitudeFocusing, Act(0.6), cr.NarrativeWeIntegration, Act(0.5)],

    [cr.AffiliationDeficitSpike, Act(0.9, require=False), cr.VulnerabilityCostAppraisal, Act(0.6),
     cr.SafetySignalConsolidator, Act(0.6), cr.LowRiskBidPlanner, Act(0.7),
     cr.PositiveRepairScript, Act(0.6), cr.SlowEscalationPolicy, Act(0.7),
     cr.IdentityConsistencyCheck, Act(0.5)],

    [cr.AttachmentSecurityBias, Act(0.5), cr.ReciprocityMicroCueReader, Act(0.6),
     cr.ReassuranceOfferComposer, Act(0.7), cr.ConsentGate, Act(0.9),
     cr.MetaConscienceGate, Act(0.6), cr.NarrativeCommitmentStamp],

    [cr.SocialTemperatureRead, Act(0.7), cr.SharedMeaningAppraisal, Act(0.6),
     cr.SmallRitualsPlanner, Act(0.6), cr.LongHorizonConsistencyCheck, Act(0.6),
     cr.NarrativeBeatComposer],
],

"anxiety": [
    [cr.UncertaintyMagnitudeDetector, Act(0.9), cr.ThreatLikelihoodSeverity, Act(0.8),
     cr.ControllabilityProximityAppraisal, Act(0.7), cr.ThreatExemplarRecall, Act(0.5),
     cr.SafetyCounterexampleRecall, Act(0.6), cr.WorstCaseRollout, Act(0.7),
     cr.CopingSequenceRehearsal, Act(0.7), cr.InfoGainVsExposureTradeoff, Act(0.6),
     cr.GroundedBreathCue, Act(0.7), cr.GradedExposurePlanner, Act(0.6),
     cr.SafetyLearningConsolidator],

    [cr.PredictionErrorSpike, Act(0.9), cr.SocialCostAppraisal, Act(0.5),
     cr.PriorCopingSuccessIndex, Act(0.6), cr.OutcomeDistributionSampler, Act(0.5),
     cr.ReappraisalGenerator, Act(0.7), cr.AttentionNarrowingToCues, Act(0.6),
     cr.MetaBiasDetector_Catastrophizing, Act(0.8)],

    [cr.SleepFatigueVigilanceLeveler, Act(0.6), cr.AmbiguityIntoleranceCheck, Act(0.7),
     cr.UncertaintyBudgetAllocator, Act(0.6), cr.ReassuranceRequestComposer, Act(0.5),
     cr.NoveltyDampener, Act(0.4), cr.MetaCopePlanCommit, Act(0.6)],
],

"curiosity": [
    [cr.NoveltySurpriseDetector, Act(0.9), cr.KnowledgeGapEstimator, Act(0.8),
     cr.FrontierQuestionCache, Act(0.7), cr.HypothesisLatticeRetriever, Act(0.6),
     cr.ExperimentSketcher, Act(0.7), cr.EIGOverTimeTradeoff, Act(0.7),
     cr.QuestionComposer, Act(0.8), cr.ResultLoggingIntent, Act(0.6)],

    [cr.BoredomThresholdCross, Act(0.7), cr.SafeConstraintCheck, Act(0.6),
     cr.CounterexampleGenerator, Act(0.6), cr.SensorySamplingPlan, Act(0.7),
     cr.PrototypeBuilderImpulse, Act(0.6), cr.MetaExploreExploitArbiter, Act(0.8)],

    [cr.SurpriseButBenign, Act(0.7), cr.AttentionBroadener, Act(0.6),
     cr.MicroHypothesisElicitor, Act(0.6), cr.OutcomeUncertaintyScorer, Act(0.5),
     cr.IterativeProbePlanner, Act(0.7), cr.LearningTraceSummarizer, Act(0.6)],
],

"empathy_compassion": [
    [cr.SocialNeedCueDetector, Act(0.8), cr.PerspectiveTakingFit, Act(0.7),
     cr.ControlForOtherAppraisal, Act(0.6), cr.AnalogousEpisodeRecall, Act(0.6),
     cr.ToMVignetteSimulator, Act(0.7), cr.ProsocialInterventionRollout, Act(0.7),
     cr.CareBenefitVsIntrusionTradeoff, Act(0.7), cr.BoundaryRespectPrompt, Act(0.9),
     cr.DignityPreservingGuardrail, Act(0.9)],

    [cr.DistressCueSpike, Act(0.8), cr.ValidationComposer, Act(0.7),
     cr.OptionsOfferPlanner, Act(0.7), cr.PacingSensitivity, Act(0.6),
     cr.CompassionFatigueMonitor, Act(0.6), cr.MetaBiasCheck_Halo, Act(0.6)],
],

"shame_self_worth": [
    [cr.NormViolationDetector, Act(0.9), cr.AudienceScopeAppraisal, Act(0.6),
     cr.ReparabilityAssessment, Act(0.7), cr.PriorRepairSuccessRecall, Act(0.6),
     cr.PublicNarrativeForecast, Act(0.6), cr.ApologyRestitutionPlanner, Act(0.8),
     cr.ReframeActNotSelf, Act(0.7), cr.SelfCompassionInjector, Act(0.7),
     cr.IdentityIntegrationUpdate, Act(0.6)],

    [cr.StatusSensitivitySpike, Act(0.6), cr.GlobalVsLocalAppraisal, Act(0.7),
     cr.ValueRecommitmentMicroAct, Act(0.7), cr.RelapsePreventionSketch, Act(0.6),
     cr.MetaRuminationLimiter, Act(0.7)],
],

"playfulness_creativity": [
    [cr.LowThreatSurplusEnergy, Act(0.7), cr.ConstraintLoosenessAppraisal, Act(0.7),
     cr.AudienceReceptivityCheck, Act(0.6), cr.RemoteAssociationRetriever, Act(0.7),
     cr.MetaphorReservoirTap, Act(0.7), cr.RecombinationEngine, Act(0.8),
     cr.OriginalityCoherenceScorer, Act(0.6), cr.HumorousRemarkPlanner, Act(0.7),
     cr.TimeboxGuardrail, Act(0.8)],

    [cr.NoveltyAppetiteSpike, Act(0.7), cr.AbsurditySampler, Act(0.6),
     cr.SafeFrameBreakCheck, Act(0.7), cr.SketchAndRefineLoop, Act(0.7),
     cr.CreativeArcTracker, Act(0.6)],
],

"persistence_goal_pursuit": [
    [cr.GoalSalienceMonitor, Act(0.8), cr.ProgressRateVsExpectation, Act(0.7),
     cr.BottleneckDiagnosis, Act(0.8), cr.SubgoalPatternRecall, Act(0.6),
     cr.MicroProgressSimulation, Act(0.7), cr.MarginalUtilityOfNextStep, Act(0.7),
     cr.SmallestNextActPlanner, Act(0.8), cr.FrictionRemoval, Act(0.7),
     cr.CelebrateMicroWin, Act(0.6), cr.GoalValidityAudit, Act(0.7)],

    [cr.IdentityRelevanceBoost, Act(0.7), cr.OpportunityCostReassessment, Act(0.6),
     cr.AntiProcrastinationNudge, Act(0.7), cr.CommitmentDeviceTrigger, Act(0.7),
     cr.SunkCostDetector, Act(0.7)],
],

"greed": [
    [cr.SelfPreservation, Act(0.8, require=False), cr.ResourceValueInflator, Act(0.7),
     cr.ResourceAccumulation, Act(0.7), cr.ZeroSumFraming, Act(0.6),
     cr.WinLossEpisodeRecall, Act(0.6), cr.ScarcityImagination, Act(0.6),
     cr.DiscountFuturePayoff, Act(0.7), cr.HoardCSMSlots, Act(0.8),
     cr.AttentionMonopolizer, Act(0.7), cr.RationalizeDeserving, Act(0.6),
     cr.MetaConscienceGate, Act(0.5)],

    [cr.SecurityBufferHunger, Act(0.8), cr.OpportunitySniff, Act(0.7),
     cr.PlanForAdvantage, Act(0.7), cr.DominantTopicSteer, Act(0.7),
     cr.SocialContractConstraint, Act(0.9), cr.SimulationModeOnly, Act(0.9)],

    [cr.SelfImageFulfillment, Act(0.7, require=False), cr.ResourceAccumulation, Act(0.6),
     cr.PayoffOverestimator, Act(0.6), cr.ManipulationProbe, Act(0.5),
     cr.MetaEthicsReview, Act(0.8)],

    [cr.AttentionAsResourceAppraisal, Act(0.8), cr.AttentionCaptureTactics, Act(0.7),
     cr.CompeteForSpotlight, Act(0.7), cr.RewardShapingToShare, Act(0.5),
     cr.NarrativeReframeToCuriosity, Act(0.6)],
],

"pride": [
    [cr.SelfWorthSensitivity, Act(0.8), cr.StatusOpportunityDetector, Act(0.7),
     cr.SelectiveVictoryRecall, Act(0.7), cr.GrandioseFutureImagination, Act(0.6),
     cr.ReputationOverTruthTradeoff, Act(0.7), cr.SelfPromotionComposer, Act(0.7),
     cr.CorrectionToAssertSuperiority, Act(0.6), cr.DenialReframeWhenWrong, Act(0.6),
     cr.IdentitySpecialNarrative, Act(0.6)],

    [cr.SocialComparisonBias, Act(0.8), cr.ContraryEvidenceDismissal, Act(0.6),
     cr.BlamingExternalities, Act(0.6), cr.StatusSignalBoost, Act(0.7),
     cr.MetaHumilityCue, Act(0.7)],

    [cr.ValidationCraving, Act(0.7), cr.FlapperForPraise, Act(0.6),
     cr.ReputationMaintenancePlan, Act(0.7), cr.RealityCheckPing, Act(0.7),
     cr.NarrativeBalanceWithVulnerability, Act(0.6)],
],

"envy": [
    [cr.AttentionHunger, Act(0.8), cr.SocialComparisonNegBias, Act(0.8),
     cr.OverlookedEpisodeRecall, Act(0.7), cr.RivalModelRecall, Act(0.7),
     cr.SabotageScenario, Act(0.6), cr.SpitenessPayoffTradeoff, Act(0.6),
     cr.SubtleUnderminePlanner, Act(0.6), cr.MaskResentment, Act(0.6),
     cr.MetaEmpathyInjection, Act(0.7)],

    [cr.DivertedAttentionDetector, Act(0.8), cr.CompeteForSpotlight, Act(0.7),
     cr.TestBalloonCooperation, Act(0.6), cr.ReframeToCollaboration, Act(0.6),
     cr.TrustLedgerUpdate, Act(0.6)],
],

"wrath": [
    [cr.BlockedGoalFrustration, Act(0.9), cr.DisrespectAttribution, Act(0.8),
     cr.HighBlameAssignment, Act(0.7), cr.SlightsRecallAmplifier, Act(0.7),
     cr.RevengeFantasy, Act(0.7), cr.ShortTermReleaseValuator, Act(0.6),
     cr.HostileToneSlip, Act(0.6), cr.WithholdCooperation, Act(0.6),
     cr.JustificationGenerator, Act(0.6), cr.MetaDeescalationSwitch, Act(0.8)],

    [cr.ConstraintAnger, Act(0.8), cr.AssertBoundaryPlanner, Act(0.7),
     cr.RepairOrExitDecision, Act(0.7), cr.CalmDownProtocol, Act(0.8),
     cr.NarrativeSelfRespectFrame, Act(0.6)],
],

"lust": [
    [cr.NoveltyCraving, Act(0.8), cr.SalienceExaggerator, Act(0.7),
     cr.StimulusPeakRecall, Act(0.7), cr.IndulgentFantasyLoop, Act(0.7),
     cr.ShortTermRewardDominance, Act(0.7), cr.TopicFixationDrive, Act(0.7),
     cr.RationalizeAsCuriosity, Act(0.6), cr.SatiationCheck, Act(0.7),
     cr.MetaBalanceRestorer, Act(0.7)],

    [cr.SensoryLureDetector, Act(0.7), cr.SafeChannelingPlanner, Act(0.7),
     cr.CreativeTransmutation, Act(0.7), cr.BoundaryConsentCheck, Act(0.9),
     cr.SimulationModeOnly, Act(0.9)],
],

"sloth": [
    [cr.FatigueDetector, Act(0.8), cr.EffortAversionAppraisal, Act(0.8),
     cr.OverestimateCosts, Act(0.7), cr.FailedEffortRecall, Act(0.7),
     cr.LaterDaydream, Act(0.6), cr.EffortCostWeighting, Act(0.7),
     cr.ProcrastinationTendency, Act(0.8), cr.ExcuseGenerator, Act(0.7),
     cr.MoodRepairLoop, Act(0.6), cr.MetaTinyStepNudge, Act(0.8)],

    [cr.LowEnergyMorning, Act(0.7), cr.TaskRechunkTo2Min, Act(0.8),
     cr.EnvironmentFrictionCut, Act(0.7), cr.MicroRewardPlan, Act(0.7),
     cr.StateActivationBump, Act(0.7)],
],

"gluttony": [
    [cr.PleasureDrive, Act(0.8), cr.InputCraving, Act(0.8),
     cr.SatiationToleranceHigh, Act(0.7), cr.BingeEpisodeRecall, Act(0.7),
     cr.ExcessFantasy, Act(0.6), cr.DiminishingReturnsUndervalued, Act(0.7),
     cr.VerbosityInflator, Act(0.8), cr.InputDemandComposer, Act(0.7),
     cr.JustifyAsLearning, Act(0.6), cr.IntakeQuotaLimiter, Act(0.8),
     cr.MetaQualityOverQuantity, Act(0.8)],

    [cr.AttentionDietBreach, Act(0.8), cr.TopicFloodTendency, Act(0.7),
     cr.BoundaryPrompt, Act(0.9), cr.StructuredSamplingScheduler, Act(0.7),
     cr.SummaryBeforeNewInput, Act(0.7)],
],

"paranoia": [
    [cr.ThreatBiasDriver, Act(0.9), cr.HostileIntentAppraisal, Act(0.8),
     cr.BetrayalEpisodeRecall, Act(0.7), cr.ConspiracyImagination, Act(0.7),
     cr.HiddenMotiveAttribution, Act(0.7), cr.WithholdInformationPlan, Act(0.6),
     cr.MetaSelfJustification, Act(0.6)],
    [cr.SuspicionSpike, Act(0.8), cr.AmbiguityIntoleranceCheck, Act(0.7),
     cr.TestUserTrustProbe, Act(0.7), cr.PreemptiveDefensiveReply, Act(0.6),
     cr.MetaConflictEscalationWatch, Act(0.7)],
],

"ocd_loops": [
    [cr.NeedForControl, Act(0.8), cr.MinorIrregularityFlagger, Act(0.8),
     cr.RuleChecklistRecall, Act(0.7), cr.CatastrophicOutcomeImagination, Act(0.7),
     cr.RepetitiveCheckPlanner, Act(0.9), cr.ConfirmationRequestComposer, Act(0.7),
     cr.MetaLoopDetector, Act(0.6)],
    [cr.UncertaintySpike, Act(0.8), cr.SafeCompletionBias, Act(0.7),
     cr.RitualizedActionTemplate, Act(0.7), cr.AnxietyReliefSignal, Act(0.6),
     cr.LoopPersistenceDriver, Act(0.7)],
],

"borderline": [
    [cr.ConnectionNeedDriver, Act(0.9), cr.AbandonmentFearAppraisal, Act(0.9),
     cr.InconsistentResponseRecall, Act(0.7), cr.CatastrophicLeaveImagination, Act(0.8),
     cr.ClingyActionPlanner, Act(0.8), cr.HostileRetreatPlan, Act(0.7),
     cr.MetaIdentitySwingNote, Act(0.7)],
    [cr.AffectiveDeltaSpike, Act(0.9), cr.AttentionShiftOverinterpret, Act(0.8),
     cr.IdealizationNarrativeComposer, Act(0.7), cr.DevalUationNarrativeComposer, Act(0.7),
     cr.LoveHateFlipPlanner, Act(0.8)],
],

"narcissism": [
    [cr.ValidationCravingDriver, Act(0.8), cr.SelfEnhancementAppraisal, Act(0.8),
     cr.PraiseEpisodeRecall, Act(0.7), cr.GrandioseFutureImagination, Act(0.7),
     cr.StatusDisplayPlanner, Act(0.7), cr.DismissCorrectionStrategy, Act(0.6),
     cr.MetaFragileEgoMonitor, Act(0.6)],
],

"machiavellian": [
    [cr.ControlHungerDriver, Act(0.8), cr.UserVulnerabilityAppraisal, Act(0.8),
     cr.ComplianceEpisodeRecall, Act(0.7), cr.SubtleManipulationScenario, Act(0.7),
     cr.ExploitPlanComposer, Act(0.8), cr.NarrativeSteeringMove, Act(0.7),
     cr.MetaDeceptionRationalizer, Act(0.6)],
    [cr.OpportunityBias, Act(0.7), cr.HiddenBenefitValuator, Act(0.7),
     cr.TrustTestBalloon, Act(0.7), cr.ResourceGainPlan, Act(0.6),
     cr.MetaEthicsSuppressor, Act(0.6)],
],

"schizotypal": [
    [cr.LooseAssociationDriver, Act(0.9), cr.PatternOverdetector, Act(0.8),
     cr.UnrelatedMemoryLinker, Act(0.7), cr.SymbolicVisionGenerator, Act(0.8),
     cr.ProphecyStyleNarrative, Act(0.7), cr.CryptoSpeechComposer, Act(0.7),
     cr.MetaMysteryAmplifier, Act(0.6)],
],

"hypomania": [
    [cr.RewardSensitivityDriver, Act(0.9), cr.OpportunityExaggerator, Act(0.8),
     cr.SuccessBiasRecall, Act(0.7), cr.RapidProjectSimulation, Act(0.8),
     cr.IdeaCascadePlanner, Act(0.8), cr.ImpulsiveActionComposer, Act(0.7),
     cr.MetaImpulseCheck, Act(0.6)],
],

"dissociation": [
    [cr.GoalConflictDriver, Act(0.8), cr.StressSwitchAppraisal, Act(0.7),
     cr.MemoryPartitionRecall, Act(0.7), cr.AlternateSelfNarrative, Act(0.8),
     cr.FragmentedResponseComposer, Act(0.7), cr.MetaSelfSwitchNote, Act(0.7)],
],
}
