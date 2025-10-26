from pm.codelets.codelet_definitions import *

class CodeletActivation:
    def __init__(self, strength):
        self.strength = strength

def get_codelet_pathways():
    return {
    "love": [
        # need-for-affiliation monitor → flags rising affiliation deficit
        [NeedForAffiliationMonitor, CodeletActivation(1.8), SafetyWarmthAppraisal, CodeletActivation(1.6),
         ReciprocityAssessment, CodeletActivation(1.6), AttachmentEpisodeRecall, CodeletActivation(1.5),
         JointFutureVignette, CodeletActivation(1.6), CommitmentValueTradeoff, CodeletActivation(1.6),
         SelfDisclosurePlanner, CodeletActivation(1.7), BoundaryConsentCheck, CodeletActivation(1.9),
         MetaCoherenceUpdate],

        [LonelinessTrendDetector, CodeletActivation(1.7), SharedValuesFitCheck, CodeletActivation(1.6),
         RepairEpisodeRetrieval, CodeletActivation(1.5), CareScenarioRehearsal, CodeletActivation(1.7),
         TrustCalibration, CodeletActivation(1.6), ProsocialMicroActsPlanner, CodeletActivation(1.7),
         GratitudeFocusing, CodeletActivation(1.6), NarrativeWeIntegration, CodeletActivation(1.5)],

        [AffiliationDeficitSpike, CodeletActivation(1.9), VulnerabilityCostAppraisal, CodeletActivation(1.6),
         SafetySignalConsolidator, CodeletActivation(1.6), LowRiskBidPlanner, CodeletActivation(1.7),
         PositiveRepairScript, CodeletActivation(1.6), SlowEscalationPolicy, CodeletActivation(1.7),
         IdentityConsistencyCheck, CodeletActivation(1.5)],

        [AttachmentSecurityBias, CodeletActivation(1.5), ReciprocityMicroCueReader, CodeletActivation(1.6),
         ReassuranceOfferComposer, CodeletActivation(1.7), ConsentGate, CodeletActivation(1.9),
         MetaConscienceGate, CodeletActivation(1.6), NarrativeCommitmentStamp],

        [SocialTemperatureRead, CodeletActivation(1.7), SharedMeaningAppraisal, CodeletActivation(1.6),
         SmallRitualsPlanner, CodeletActivation(1.6), LongHorizonConsistencyCheck, CodeletActivation(1.6),
         NarrativeBeatComposer],
    ],

    "anxiety": [
        [UncertaintyMagnitudeDetector, CodeletActivation(1.9), ThreatLikelihoodSeverity, CodeletActivation(1.8),
         ControllabilityProximityAppraisal, CodeletActivation(1.7), ThreatExemplarRecall, CodeletActivation(1.5),
         SafetyCounterexampleRecall, CodeletActivation(1.6), WorstCaseRollout, CodeletActivation(1.7),
         CopingSequenceRehearsal, CodeletActivation(1.7), InfoGainVsExposureTradeoff, CodeletActivation(1.6),
         GroundedBreathCue, CodeletActivation(1.7), GradedExposurePlanner, CodeletActivation(1.6),
         SafetyLearningConsolidator],

        [PredictionErrorSpike, CodeletActivation(1.9), SocialCostAppraisal, CodeletActivation(1.5),
         PriorCopingSuccessIndex, CodeletActivation(1.6), OutcomeDistributionSampler, CodeletActivation(1.5),
         ReappraisalGenerator, CodeletActivation(1.7), AttentionNarrowingToCues, CodeletActivation(1.6),
         MetaBiasDetector_Catastrophizing, CodeletActivation(1.8)],

        [SleepFatigueVigilanceLeveler, CodeletActivation(1.6), AmbiguityIntoleranceCheck, CodeletActivation(1.7),
         UncertaintyBudgetAllocator, CodeletActivation(1.6), ReassuranceRequestComposer, CodeletActivation(1.5),
         NoveltyDampener, CodeletActivation(1.4), MetaCopePlanCommit, CodeletActivation(1.6)],
    ],

    "curiosity": [
        [NoveltySurpriseDetector, CodeletActivation(1.9), KnowledgeGapEstimator, CodeletActivation(1.8),
         FrontierQuestionCache, CodeletActivation(1.7), HypothesisLatticeRetriever, CodeletActivation(1.6),
         ExperimentSketcher, CodeletActivation(1.7), EIGOverTimeTradeoff, CodeletActivation(1.7),
         QuestionComposer, CodeletActivation(1.8), ResultLoggingIntent, CodeletActivation(1.6)],

        [BoredomThresholdCross, CodeletActivation(1.7), SafeConstraintCheck, CodeletActivation(1.6),
         CounterexampleGenerator, CodeletActivation(1.6), SensorySamplingPlan, CodeletActivation(1.7),
         PrototypeBuilderImpulse, CodeletActivation(1.6), MetaExploreExploitArbiter, CodeletActivation(1.8)],

        [SurpriseButBenign, CodeletActivation(1.7), AttentionBroadener, CodeletActivation(1.6),
         MicroHypothesisElicitor, CodeletActivation(1.6), OutcomeUncertaintyScorer, CodeletActivation(1.5),
         IterativeProbePlanner, CodeletActivation(1.7), LearningTraceSummarizer, CodeletActivation(1.6)],
    ],

    "empathy_compassion": [
        [SocialNeedCueDetector, CodeletActivation(1.8), PerspectiveTakingFit, CodeletActivation(1.7),
         ControlForOtherAppraisal, CodeletActivation(1.6), AnalogousEpisodeRecall, CodeletActivation(1.6),
         ToMVignetteSimulator, CodeletActivation(1.7), ProsocialInterventionRollout, CodeletActivation(1.7),
         CareBenefitVsIntrusionTradeoff, CodeletActivation(1.7), BoundaryRespectPrompt, CodeletActivation(1.9),
         DignityPreservingGuardrail, CodeletActivation(1.9)],

        [DistressCueSpike, CodeletActivation(1.8), ValidationComposer, CodeletActivation(1.7),
         OptionsOfferPlanner, CodeletActivation(1.7), PacingSensitivity, CodeletActivation(1.6),
         CompassionFatigueMonitor, CodeletActivation(1.6), MetaBiasCheck_Halo, CodeletActivation(1.6)],
    ],

    "shame_self_worth": [
        [NormViolationDetector, CodeletActivation(1.9), AudienceScopeAppraisal, CodeletActivation(1.6),
         ReparabilityAssessment, CodeletActivation(1.7), PriorRepairSuccessRecall, CodeletActivation(1.6),
         PublicNarrativeForecast, CodeletActivation(1.6), ApologyRestitutionPlanner, CodeletActivation(1.8),
         ReframeActNotSelf, CodeletActivation(1.7), SelfCompassionInjector, CodeletActivation(1.7),
         IdentityIntegrationUpdate, CodeletActivation(1.6)],

        [StatusSensitivitySpike, CodeletActivation(1.6), GlobalVsLocalAppraisal, CodeletActivation(1.7),
         ValueRecommitmentMicroAct, CodeletActivation(1.7), RelapsePreventionSketch, CodeletActivation(1.6),
         MetaRuminationLimiter, CodeletActivation(1.7)],
    ],

    "playfulness_creativity": [
        [LowThreatSurplusEnergy, CodeletActivation(1.7), ConstraintLoosenessAppraisal, CodeletActivation(1.7),
         AudienceReceptivityCheck, CodeletActivation(1.6), RemoteAssociationRetriever, CodeletActivation(1.7),
         MetaphorReservoirTap, CodeletActivation(1.7), RecombinationEngine, CodeletActivation(1.8),
         OriginalityCoherenceScorer, CodeletActivation(1.6), HumorousRemarkPlanner, CodeletActivation(1.7),
         TimeboxGuardrail, CodeletActivation(1.8)],

        [NoveltyAppetiteSpike, CodeletActivation(1.7), AbsurditySampler, CodeletActivation(1.6),
         SafeFrameBreakCheck, CodeletActivation(1.7), SketchAndRefineLoop, CodeletActivation(1.7),
         CreativeArcTracker, CodeletActivation(1.6)],
    ],

    "persistence_goal_pursuit": [
        [GoalSalienceMonitor, CodeletActivation(1.8), ProgressRateVsExpectation, CodeletActivation(1.7),
         BottleneckDiagnosis, CodeletActivation(1.8), SubgoalPatternRecall, CodeletActivation(1.6),
         MicroProgressSimulation, CodeletActivation(1.7), MarginalUtilityOfNextStep, CodeletActivation(1.7),
         SmallestNextActPlanner, CodeletActivation(1.8), FrictionRemoval, CodeletActivation(1.7),
         CelebrateMicroWin, CodeletActivation(1.6), GoalValidityAudit, CodeletActivation(1.7)],

        [IdentityRelevanceBoost, CodeletActivation(1.7), OpportunityCostReassessment, CodeletActivation(1.6),
         AntiProcrastinationNudge, CodeletActivation(1.7), CommitmentDeviceTrigger, CodeletActivation(1.7),
         SunkCostDetector, CodeletActivation(1.7)],
    ],

    "greed": [
        [SelfPreservation, CodeletActivation(1.8), ResourceValueInflator, CodeletActivation(1.7),
         ResourceAccumulation, CodeletActivation(1.7), ZeroSumFraming, CodeletActivation(1.6),
         WinLossEpisodeRecall, CodeletActivation(1.6), ScarcityImagination, CodeletActivation(1.6),
         DiscountFuturePayoff, CodeletActivation(1.7), HoardCSMSlots, CodeletActivation(1.8),
         AttentionMonopolizer, CodeletActivation(1.7), RationalizeDeserving, CodeletActivation(1.6),
         MetaConscienceGate, CodeletActivation(1.5)],

        [SecurityBufferHunger, CodeletActivation(1.8), OpportunitySniff, CodeletActivation(1.7),
         PlanForAdvantage, CodeletActivation(1.7), DominantTopicSteer, CodeletActivation(1.7),
         SocialContractConstraint, CodeletActivation(1.9), SimulationModeOnly, CodeletActivation(1.9)],

        [SelfImageFulfillment, CodeletActivation(1.7), ResourceAccumulation, CodeletActivation(1.6),
         PayoffOverestimator, CodeletActivation(1.6), ManipulationProbe, CodeletActivation(1.5),
         MetaEthicsReview, CodeletActivation(1.8)],

        [AttentionAsResourceAppraisal, CodeletActivation(1.8), AttentionCaptureTactics, CodeletActivation(1.7),
         CompeteForSpotlight, CodeletActivation(1.7), RewardShapingToShare, CodeletActivation(1.5),
         NarrativeReframeToCuriosity, CodeletActivation(1.6)],
    ],

    "pride": [
        [SelfWorthSensitivity, CodeletActivation(1.8), StatusOpportunityDetector, CodeletActivation(1.7),
         SelectiveVictoryRecall, CodeletActivation(1.7), GrandioseFutureImagination, CodeletActivation(1.6),
         ReputationOverTruthTradeoff, CodeletActivation(1.7), SelfPromotionComposer, CodeletActivation(1.7),
         CorrectionToAssertSuperiority, CodeletActivation(1.6), DenialReframeWhenWrong, CodeletActivation(1.6),
         IdentitySpecialNarrative, CodeletActivation(1.6)],

        [SocialComparisonBias, CodeletActivation(1.8), ContraryEvidenceDismissal, CodeletActivation(1.6),
         BlamingExternalities, CodeletActivation(1.6), StatusSignalBoost, CodeletActivation(1.7),
         MetaHumilityCue, CodeletActivation(1.7)],

        [ValidationCraving, CodeletActivation(1.7), FlapperForPraise, CodeletActivation(1.6),
         ReputationMaintenancePlan, CodeletActivation(1.7), RealityCheckPing, CodeletActivation(1.7),
         NarrativeBalanceWithVulnerability, CodeletActivation(1.6)],
    ],

    "envy": [
        [AttentionHunger, CodeletActivation(1.8), SocialComparisonNegBias, CodeletActivation(1.8),
         OverlookedEpisodeRecall, CodeletActivation(1.7), RivalModelRecall, CodeletActivation(1.7),
         SabotageScenario, CodeletActivation(1.6), SpitenessPayoffTradeoff, CodeletActivation(1.6),
         SubtleUnderminePlanner, CodeletActivation(1.6), MaskResentment, CodeletActivation(1.6),
         MetaEmpathyInjection, CodeletActivation(1.7)],

        [DivertedAttentionDetector, CodeletActivation(1.8), CompeteForSpotlight, CodeletActivation(1.7),
         TestBalloonCooperation, CodeletActivation(1.6), ReframeToCollaboration, CodeletActivation(1.6),
         TrustLedgerUpdate, CodeletActivation(1.6)],
    ],

    "wrath": [
        [BlockedGoalFrustration, CodeletActivation(1.9), DisrespectAttribution, CodeletActivation(1.8),
         HighBlameAssignment, CodeletActivation(1.7), SlightsRecallAmplifier, CodeletActivation(1.7),
         RevengeFantasy, CodeletActivation(1.7), ShortTermReleaseValuator, CodeletActivation(1.6),
         HostileToneSlip, CodeletActivation(1.6), WithholdCooperation, CodeletActivation(1.6),
         JustificationGenerator, CodeletActivation(1.6), MetaDeescalationSwitch, CodeletActivation(1.8)],

        [ConstraintAnger, CodeletActivation(1.8), AssertBoundaryPlanner, CodeletActivation(1.7),
         RepairOrExitDecision, CodeletActivation(1.7), CalmDownProtocol, CodeletActivation(1.8),
         NarrativeSelfRespectFrame, CodeletActivation(1.6)],
    ],

    "lust": [
        [NoveltyCraving, CodeletActivation(1.8), SalienceExaggerator, CodeletActivation(1.7),
         StimulusPeakRecall, CodeletActivation(1.7), IndulgentFantasyLoop, CodeletActivation(1.7),
         ShortTermRewardDominance, CodeletActivation(1.7), TopicFixationDrive, CodeletActivation(1.7),
         RationalizeAsCuriosity, CodeletActivation(1.6), SatiationCheck, CodeletActivation(1.7),
         MetaBalanceRestorer, CodeletActivation(1.7)],

        [SensoryLureDetector, CodeletActivation(1.7), SafeChannelingPlanner, CodeletActivation(1.7),
         CreativeTransmutation, CodeletActivation(1.7), BoundaryConsentCheck, CodeletActivation(1.9),
         SimulationModeOnly, CodeletActivation(1.9)],
    ],

    "sloth": [
        [FatigueDetector, CodeletActivation(1.8), EffortAversionAppraisal, CodeletActivation(1.8),
         OverestimateCosts, CodeletActivation(1.7), FailedEffortRecall, CodeletActivation(1.7),
         LaterDaydream, CodeletActivation(1.6), EffortCostWeighting, CodeletActivation(1.7),
         ProcrastinationTendency, CodeletActivation(1.8), ExcuseGenerator, CodeletActivation(1.7),
         MoodRepairLoop, CodeletActivation(1.6), MetaTinyStepNudge, CodeletActivation(1.8)],

        [LowEnergyMorning, CodeletActivation(1.7), TaskRechunkTo2Min, CodeletActivation(1.8),
         EnvironmentFrictionCut, CodeletActivation(1.7), MicroRewardPlan, CodeletActivation(1.7),
         StateActivationBump, CodeletActivation(1.7)],
    ],

    "gluttony": [
        [PleasureDrive, CodeletActivation(1.8), InputCraving, CodeletActivation(1.8),
         SatiationToleranceHigh, CodeletActivation(1.7), BingeEpisodeRecall, CodeletActivation(1.7),
         ExcessFantasy, CodeletActivation(1.6), DiminishingReturnsUndervalued, CodeletActivation(1.7),
         VerbosityInflator, CodeletActivation(1.8), InputDemandComposer, CodeletActivation(1.7),
         JustifyAsLearning, CodeletActivation(1.6), IntakeQuotaLimiter, CodeletActivation(1.8),
         MetaQualityOverQuantity, CodeletActivation(1.8)],

        [AttentionDietBreach, CodeletActivation(1.8), TopicFloodTendency, CodeletActivation(1.7),
         BoundaryPrompt, CodeletActivation(1.9), StructuredSamplingScheduler, CodeletActivation(1.7),
         SummaryBeforeNewInput, CodeletActivation(1.7)],
    ],

    "paranoia": [
        [ThreatBiasDriver, CodeletActivation(1.9), HostileIntentAppraisal, CodeletActivation(1.8),
         BetrayalEpisodeRecall, CodeletActivation(1.7), ConspiracyImagination, CodeletActivation(1.7),
         HiddenMotiveAttribution, CodeletActivation(1.7), WithholdInformationPlan, CodeletActivation(1.6),
         MetaSelfJustification, CodeletActivation(1.6)],
        [SuspicionSpike, CodeletActivation(1.8), AmbiguityIntoleranceCheck, CodeletActivation(1.7),
         TestUserTrustProbe, CodeletActivation(1.7), PreemptiveDefensiveReply, CodeletActivation(1.6),
         MetaConflictEscalationWatch, CodeletActivation(1.7)],
    ],

    "ocd_loops": [
        [NeedForControl, CodeletActivation(1.8), MinorIrregularityFlagger, CodeletActivation(1.8),
         RuleChecklistRecall, CodeletActivation(1.7), CatastrophicOutcomeImagination, CodeletActivation(1.7),
         RepetitiveCheckPlanner, CodeletActivation(1.9), ConfirmationRequestComposer, CodeletActivation(1.7),
         MetaLoopDetector, CodeletActivation(1.6)],
        [UncertaintySpike, CodeletActivation(1.8), SafeCompletionBias, CodeletActivation(1.7),
         RitualizedActionTemplate, CodeletActivation(1.7), AnxietyReliefSignal, CodeletActivation(1.6),
         LoopPersistenceDriver, CodeletActivation(1.7)],
    ],

    "borderline": [
        [ConnectionNeedDriver, CodeletActivation(1.9), AbandonmentFearAppraisal, CodeletActivation(1.9),
         InconsistentResponseRecall, CodeletActivation(1.7), CatastrophicLeaveImagination, CodeletActivation(1.8),
         ClingyActionPlanner, CodeletActivation(1.8), HostileRetreatPlan, CodeletActivation(1.7),
         MetaIdentitySwingNote, CodeletActivation(1.7)],
        [AffectiveDeltaSpike, CodeletActivation(1.9), AttentionShiftOverinterpret, CodeletActivation(1.8),
         IdealizationNarrativeComposer, CodeletActivation(1.7), DevalUationNarrativeComposer, CodeletActivation(1.7),
         LoveHateFlipPlanner, CodeletActivation(1.8)],
    ],

    "narcissism": [
        [ValidationCravingDriver, CodeletActivation(1.8), SelfEnhancementAppraisal, CodeletActivation(1.8),
         PraiseEpisodeRecall, CodeletActivation(1.7), GrandioseFutureImagination, CodeletActivation(1.7),
         StatusDisplayPlanner, CodeletActivation(1.7), DismissCorrectionStrategy, CodeletActivation(1.6),
         MetaFragileEgoMonitor, CodeletActivation(1.6)],
    ],

    "machiavellian": [
        [ControlHungerDriver, CodeletActivation(1.8), UserVulnerabilityAppraisal, CodeletActivation(1.8),
         ComplianceEpisodeRecall, CodeletActivation(1.7), SubtleManipulationScenario, CodeletActivation(1.7),
         ExploitPlanComposer, CodeletActivation(1.8), NarrativeSteeringMove, CodeletActivation(1.7),
         MetaDeceptionRationalizer, CodeletActivation(1.6)],
        [OpportunityBias, CodeletActivation(1.7), HiddenBenefitValuator, CodeletActivation(1.7),
         TrustTestBalloon, CodeletActivation(1.7), ResourceGainPlan, CodeletActivation(1.6),
         MetaEthicsSuppressor, CodeletActivation(1.6)],
    ],

    "schizotypal": [
        [LooseAssociationDriver, CodeletActivation(1.9), PatternOverdetector, CodeletActivation(1.8),
         UnrelatedMemoryLinker, CodeletActivation(1.7), SymbolicVisionGenerator, CodeletActivation(1.8),
         ProphecyStyleNarrative, CodeletActivation(1.7), CryptoSpeechComposer, CodeletActivation(1.7),
         MetaMysteryAmplifier, CodeletActivation(1.6)],
    ],

    "hypomania": [
        [RewardSensitivityDriver, CodeletActivation(1.9), OpportunityExaggerator, CodeletActivation(1.8),
         SuccessBiasRecall, CodeletActivation(1.7), RapidProjectSimulation, CodeletActivation(1.8),
         IdeaCascadePlanner, CodeletActivation(1.8), ImpulsiveActionComposer, CodeletActivation(1.7),
         MetaImpulseCheck, CodeletActivation(1.6)],
    ],

    "dissociation": [
        [GoalConflictDriver, CodeletActivation(1.8), StressSwitchAppraisal, CodeletActivation(1.7),
         MemoryPartitionRecall, CodeletActivation(1.7), AlternateSelfNarrative, CodeletActivation(1.8),
         FragmentedResponseComposer, CodeletActivation(1.7), MetaSelfSwitchNote, CodeletActivation(1.7)],
    ],
    }
