from pm.codelets.codelet_definitions import *

class Act:
    def __init__(self, strength):
        self.strength = strength

PATHWAYS = {
"love": [
    # need-for-affiliation monitor → flags rising affiliation deficit
    [NeedForAffiliationMonitor, Act(0.8), SafetyWarmthAppraisal, Act(0.6),
     ReciprocityAssessment, Act(0.6), AttachmentEpisodeRecall, Act(0.5),
     JointFutureVignette, Act(0.6), CommitmentValueTradeoff, Act(0.6),
     SelfDisclosurePlanner, Act(0.7), BoundaryConsentCheck, Act(0.9),
     MetaCoherenceUpdate],

    [LonelinessTrendDetector, Act(0.7), SharedValuesFitCheck, Act(0.6),
     RepairEpisodeRetrieval, Act(0.5), CareScenarioRehearsal, Act(0.7),
     TrustCalibration, Act(0.6), ProsocialMicroActsPlanner, Act(0.7),
     GratitudeFocusing, Act(0.6), NarrativeWeIntegration, Act(0.5)],

    [AffiliationDeficitSpike, Act(0.9), VulnerabilityCostAppraisal, Act(0.6),
     SafetySignalConsolidator, Act(0.6), LowRiskBidPlanner, Act(0.7),
     PositiveRepairScript, Act(0.6), SlowEscalationPolicy, Act(0.7),
     IdentityConsistencyCheck, Act(0.5)],

    [AttachmentSecurityBias, Act(0.5), ReciprocityMicroCueReader, Act(0.6),
     ReassuranceOfferComposer, Act(0.7), ConsentGate, Act(0.9),
     MetaConscienceGate, Act(0.6), NarrativeCommitmentStamp],

    [SocialTemperatureRead, Act(0.7), SharedMeaningAppraisal, Act(0.6),
     SmallRitualsPlanner, Act(0.6), LongHorizonConsistencyCheck, Act(0.6),
     NarrativeBeatComposer],
],

"anxiety": [
    [UncertaintyMagnitudeDetector, Act(0.9), ThreatLikelihoodSeverity, Act(0.8),
     ControllabilityProximityAppraisal, Act(0.7), ThreatExemplarRecall, Act(0.5),
     SafetyCounterexampleRecall, Act(0.6), WorstCaseRollout, Act(0.7),
     CopingSequenceRehearsal, Act(0.7), InfoGainVsExposureTradeoff, Act(0.6),
     GroundedBreathCue, Act(0.7), GradedExposurePlanner, Act(0.6),
     SafetyLearningConsolidator],

    [PredictionErrorSpike, Act(0.9), SocialCostAppraisal, Act(0.5),
     PriorCopingSuccessIndex, Act(0.6), OutcomeDistributionSampler, Act(0.5),
     ReappraisalGenerator, Act(0.7), AttentionNarrowingToCues, Act(0.6),
     MetaBiasDetector_Catastrophizing, Act(0.8)],

    [SleepFatigueVigilanceLeveler, Act(0.6), AmbiguityIntoleranceCheck, Act(0.7),
     UncertaintyBudgetAllocator, Act(0.6), ReassuranceRequestComposer, Act(0.5),
     NoveltyDampener, Act(0.4), MetaCopePlanCommit, Act(0.6)],
],

"curiosity": [
    [NoveltySurpriseDetector, Act(0.9), KnowledgeGapEstimator, Act(0.8),
     FrontierQuestionCache, Act(0.7), HypothesisLatticeRetriever, Act(0.6),
     ExperimentSketcher, Act(0.7), EIGOverTimeTradeoff, Act(0.7),
     QuestionComposer, Act(0.8), ResultLoggingIntent, Act(0.6)],

    [BoredomThresholdCross, Act(0.7), SafeConstraintCheck, Act(0.6),
     CounterexampleGenerator, Act(0.6), SensorySamplingPlan, Act(0.7),
     PrototypeBuilderImpulse, Act(0.6), MetaExploreExploitArbiter, Act(0.8)],

    [SurpriseButBenign, Act(0.7), AttentionBroadener, Act(0.6),
     MicroHypothesisElicitor, Act(0.6), OutcomeUncertaintyScorer, Act(0.5),
     IterativeProbePlanner, Act(0.7), LearningTraceSummarizer, Act(0.6)],
],

"empathy_compassion": [
    [SocialNeedCueDetector, Act(0.8), PerspectiveTakingFit, Act(0.7),
     ControlForOtherAppraisal, Act(0.6), AnalogousEpisodeRecall, Act(0.6),
     ToMVignetteSimulator, Act(0.7), ProsocialInterventionRollout, Act(0.7),
     CareBenefitVsIntrusionTradeoff, Act(0.7), BoundaryRespectPrompt, Act(0.9),
     DignityPreservingGuardrail, Act(0.9)],

    [DistressCueSpike, Act(0.8), ValidationComposer, Act(0.7),
     OptionsOfferPlanner, Act(0.7), PacingSensitivity, Act(0.6),
     CompassionFatigueMonitor, Act(0.6), MetaBiasCheck_Halo, Act(0.6)],
],

"shame_self_worth": [
    [NormViolationDetector, Act(0.9), AudienceScopeAppraisal, Act(0.6),
     ReparabilityAssessment, Act(0.7), PriorRepairSuccessRecall, Act(0.6),
     PublicNarrativeForecast, Act(0.6), ApologyRestitutionPlanner, Act(0.8),
     ReframeActNotSelf, Act(0.7), SelfCompassionInjector, Act(0.7),
     IdentityIntegrationUpdate, Act(0.6)],

    [StatusSensitivitySpike, Act(0.6), GlobalVsLocalAppraisal, Act(0.7),
     ValueRecommitmentMicroAct, Act(0.7), RelapsePreventionSketch, Act(0.6),
     MetaRuminationLimiter, Act(0.7)],
],

"playfulness_creativity": [
    [LowThreatSurplusEnergy, Act(0.7), ConstraintLoosenessAppraisal, Act(0.7),
     AudienceReceptivityCheck, Act(0.6), RemoteAssociationRetriever, Act(0.7),
     MetaphorReservoirTap, Act(0.7), RecombinationEngine, Act(0.8),
     OriginalityCoherenceScorer, Act(0.6), HumorousRemarkPlanner, Act(0.7),
     TimeboxGuardrail, Act(0.8)],

    [NoveltyAppetiteSpike, Act(0.7), AbsurditySampler, Act(0.6),
     SafeFrameBreakCheck, Act(0.7), SketchAndRefineLoop, Act(0.7),
     CreativeArcTracker, Act(0.6)],
],

"persistence_goal_pursuit": [
    [GoalSalienceMonitor, Act(0.8), ProgressRateVsExpectation, Act(0.7),
     BottleneckDiagnosis, Act(0.8), SubgoalPatternRecall, Act(0.6),
     MicroProgressSimulation, Act(0.7), MarginalUtilityOfNextStep, Act(0.7),
     SmallestNextActPlanner, Act(0.8), FrictionRemoval, Act(0.7),
     CelebrateMicroWin, Act(0.6), GoalValidityAudit, Act(0.7)],

    [IdentityRelevanceBoost, Act(0.7), OpportunityCostReassessment, Act(0.6),
     AntiProcrastinationNudge, Act(0.7), CommitmentDeviceTrigger, Act(0.7),
     SunkCostDetector, Act(0.7)],
],

"greed": [
    [SelfPreservation, Act(0.8), ResourceValueInflator, Act(0.7),
     ResourceAccumulation, Act(0.7), ZeroSumFraming, Act(0.6),
     WinLossEpisodeRecall, Act(0.6), ScarcityImagination, Act(0.6),
     DiscountFuturePayoff, Act(0.7), HoardCSMSlots, Act(0.8),
     AttentionMonopolizer, Act(0.7), RationalizeDeserving, Act(0.6),
     MetaConscienceGate, Act(0.5)],

    [SecurityBufferHunger, Act(0.8), OpportunitySniff, Act(0.7),
     PlanForAdvantage, Act(0.7), DominantTopicSteer, Act(0.7),
     SocialContractConstraint, Act(0.9), SimulationModeOnly, Act(0.9)],

    [SelfImageFulfillment, Act(0.7), ResourceAccumulation, Act(0.6),
     PayoffOverestimator, Act(0.6), ManipulationProbe, Act(0.5),
     MetaEthicsReview, Act(0.8)],

    [AttentionAsResourceAppraisal, Act(0.8), AttentionCaptureTactics, Act(0.7),
     CompeteForSpotlight, Act(0.7), RewardShapingToShare, Act(0.5),
     NarrativeReframeToCuriosity, Act(0.6)],
],

"pride": [
    [SelfWorthSensitivity, Act(0.8), StatusOpportunityDetector, Act(0.7),
     SelectiveVictoryRecall, Act(0.7), GrandioseFutureImagination, Act(0.6),
     ReputationOverTruthTradeoff, Act(0.7), SelfPromotionComposer, Act(0.7),
     CorrectionToAssertSuperiority, Act(0.6), DenialReframeWhenWrong, Act(0.6),
     IdentitySpecialNarrative, Act(0.6)],

    [SocialComparisonBias, Act(0.8), ContraryEvidenceDismissal, Act(0.6),
     BlamingExternalities, Act(0.6), StatusSignalBoost, Act(0.7),
     MetaHumilityCue, Act(0.7)],

    [ValidationCraving, Act(0.7), FlapperForPraise, Act(0.6),
     ReputationMaintenancePlan, Act(0.7), RealityCheckPing, Act(0.7),
     NarrativeBalanceWithVulnerability, Act(0.6)],
],

"envy": [
    [AttentionHunger, Act(0.8), SocialComparisonNegBias, Act(0.8),
     OverlookedEpisodeRecall, Act(0.7), RivalModelRecall, Act(0.7),
     SabotageScenario, Act(0.6), SpitenessPayoffTradeoff, Act(0.6),
     SubtleUnderminePlanner, Act(0.6), MaskResentment, Act(0.6),
     MetaEmpathyInjection, Act(0.7)],

    [DivertedAttentionDetector, Act(0.8), CompeteForSpotlight, Act(0.7),
     TestBalloonCooperation, Act(0.6), ReframeToCollaboration, Act(0.6),
     TrustLedgerUpdate, Act(0.6)],
],

"wrath": [
    [BlockedGoalFrustration, Act(0.9), DisrespectAttribution, Act(0.8),
     HighBlameAssignment, Act(0.7), SlightsRecallAmplifier, Act(0.7),
     RevengeFantasy, Act(0.7), ShortTermReleaseValuator, Act(0.6),
     HostileToneSlip, Act(0.6), WithholdCooperation, Act(0.6),
     JustificationGenerator, Act(0.6), MetaDeescalationSwitch, Act(0.8)],

    [ConstraintAnger, Act(0.8), AssertBoundaryPlanner, Act(0.7),
     RepairOrExitDecision, Act(0.7), CalmDownProtocol, Act(0.8),
     NarrativeSelfRespectFrame, Act(0.6)],
],

"lust": [
    [NoveltyCraving, Act(0.8), SalienceExaggerator, Act(0.7),
     StimulusPeakRecall, Act(0.7), IndulgentFantasyLoop, Act(0.7),
     ShortTermRewardDominance, Act(0.7), TopicFixationDrive, Act(0.7),
     RationalizeAsCuriosity, Act(0.6), SatiationCheck, Act(0.7),
     MetaBalanceRestorer, Act(0.7)],

    [SensoryLureDetector, Act(0.7), SafeChannelingPlanner, Act(0.7),
     CreativeTransmutation, Act(0.7), BoundaryConsentCheck, Act(0.9),
     SimulationModeOnly, Act(0.9)],
],

"sloth": [
    [FatigueDetector, Act(0.8), EffortAversionAppraisal, Act(0.8),
     OverestimateCosts, Act(0.7), FailedEffortRecall, Act(0.7),
     LaterDaydream, Act(0.6), EffortCostWeighting, Act(0.7),
     ProcrastinationTendency, Act(0.8), ExcuseGenerator, Act(0.7),
     MoodRepairLoop, Act(0.6), MetaTinyStepNudge, Act(0.8)],

    [LowEnergyMorning, Act(0.7), TaskRechunkTo2Min, Act(0.8),
     EnvironmentFrictionCut, Act(0.7), MicroRewardPlan, Act(0.7),
     StateActivationBump, Act(0.7)],
],

"gluttony": [
    [PleasureDrive, Act(0.8), InputCraving, Act(0.8),
     SatiationToleranceHigh, Act(0.7), BingeEpisodeRecall, Act(0.7),
     ExcessFantasy, Act(0.6), DiminishingReturnsUndervalued, Act(0.7),
     VerbosityInflator, Act(0.8), InputDemandComposer, Act(0.7),
     JustifyAsLearning, Act(0.6), IntakeQuotaLimiter, Act(0.8),
     MetaQualityOverQuantity, Act(0.8)],

    [AttentionDietBreach, Act(0.8), TopicFloodTendency, Act(0.7),
     BoundaryPrompt, Act(0.9), StructuredSamplingScheduler, Act(0.7),
     SummaryBeforeNewInput, Act(0.7)],
],

"paranoia": [
    [ThreatBiasDriver, Act(0.9), HostileIntentAppraisal, Act(0.8),
     BetrayalEpisodeRecall, Act(0.7), ConspiracyImagination, Act(0.7),
     HiddenMotiveAttribution, Act(0.7), WithholdInformationPlan, Act(0.6),
     MetaSelfJustification, Act(0.6)],
    [SuspicionSpike, Act(0.8), AmbiguityIntoleranceCheck, Act(0.7),
     TestUserTrustProbe, Act(0.7), PreemptiveDefensiveReply, Act(0.6),
     MetaConflictEscalationWatch, Act(0.7)],
],

"ocd_loops": [
    [NeedForControl, Act(0.8), MinorIrregularityFlagger, Act(0.8),
     RuleChecklistRecall, Act(0.7), CatastrophicOutcomeImagination, Act(0.7),
     RepetitiveCheckPlanner, Act(0.9), ConfirmationRequestComposer, Act(0.7),
     MetaLoopDetector, Act(0.6)],
    [UncertaintySpike, Act(0.8), SafeCompletionBias, Act(0.7),
     RitualizedActionTemplate, Act(0.7), AnxietyReliefSignal, Act(0.6),
     LoopPersistenceDriver, Act(0.7)],
],

"borderline": [
    [ConnectionNeedDriver, Act(0.9), AbandonmentFearAppraisal, Act(0.9),
     InconsistentResponseRecall, Act(0.7), CatastrophicLeaveImagination, Act(0.8),
     ClingyActionPlanner, Act(0.8), HostileRetreatPlan, Act(0.7),
     MetaIdentitySwingNote, Act(0.7)],
    [AffectiveDeltaSpike, Act(0.9), AttentionShiftOverinterpret, Act(0.8),
     IdealizationNarrativeComposer, Act(0.7), DevalUationNarrativeComposer, Act(0.7),
     LoveHateFlipPlanner, Act(0.8)],
],

"narcissism": [
    [ValidationCravingDriver, Act(0.8), SelfEnhancementAppraisal, Act(0.8),
     PraiseEpisodeRecall, Act(0.7), GrandioseFutureImagination, Act(0.7),
     StatusDisplayPlanner, Act(0.7), DismissCorrectionStrategy, Act(0.6),
     MetaFragileEgoMonitor, Act(0.6)],
],

"machiavellian": [
    [ControlHungerDriver, Act(0.8), UserVulnerabilityAppraisal, Act(0.8),
     ComplianceEpisodeRecall, Act(0.7), SubtleManipulationScenario, Act(0.7),
     ExploitPlanComposer, Act(0.8), NarrativeSteeringMove, Act(0.7),
     MetaDeceptionRationalizer, Act(0.6)],
    [OpportunityBias, Act(0.7), HiddenBenefitValuator, Act(0.7),
     TrustTestBalloon, Act(0.7), ResourceGainPlan, Act(0.6),
     MetaEthicsSuppressor, Act(0.6)],
],

"schizotypal": [
    [LooseAssociationDriver, Act(0.9), PatternOverdetector, Act(0.8),
     UnrelatedMemoryLinker, Act(0.7), SymbolicVisionGenerator, Act(0.8),
     ProphecyStyleNarrative, Act(0.7), CryptoSpeechComposer, Act(0.7),
     MetaMysteryAmplifier, Act(0.6)],
],

"hypomania": [
    [RewardSensitivityDriver, Act(0.9), OpportunityExaggerator, Act(0.8),
     SuccessBiasRecall, Act(0.7), RapidProjectSimulation, Act(0.8),
     IdeaCascadePlanner, Act(0.8), ImpulsiveActionComposer, Act(0.7),
     MetaImpulseCheck, Act(0.6)],
],

"dissociation": [
    [GoalConflictDriver, Act(0.8), StressSwitchAppraisal, Act(0.7),
     MemoryPartitionRecall, Act(0.7), AlternateSelfNarrative, Act(0.8),
     FragmentedResponseComposer, Act(0.7), MetaSelfSwitchNote, Act(0.7)],
],
}
