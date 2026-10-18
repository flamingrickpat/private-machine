class staticproperty(property):
    def __get__(self, owner_self, owner_cls):
        return self.fget()

class Sampler:
    @staticproperty
    def TempReasoning():
        return 1.0

    @staticproperty
    def TempInstruct():
        return 0.7


# A19F cycle constants. One place holds every tunable number of the cycle.
# Activation values are unitless and lie in [0, 1] unless a comment says otherwise.

# Current situational model (CSM).
CSM_CAPACITY = 128  # Maximum number of items. Prune drops the lowest activation first.
CSM_DECAY = 0.92  # Activation factor for each cycle at the neutral base-level activation.
CSM_MERGE_SIMILARITY = 0.86  # Minimum embedding cosine similarity for a merge into an existing item.
CSM_BOOST = 0.25  # Activation that one merge or endorsement adds.
CSM_MIN_ACTIVATION = 0.1  # Prune removes unprotected items below this activation.
CSM_BASE_ACTIVATION = 0.5  # Neutral base-level activation of a new item. At this value the decay is CSM_DECAY.
CSM_BASE_DECAY_SPREAD = 0.5  # How strongly base-level activation changes the decay exponent. See SituationalModel.decay_step.
CSM_BASE_LEARN_RATE = 0.1  # Part of the remaining distance to 1.0 that one boost adds to base-level activation.
CSM_BASE_FORGET = 0.995  # Factor for each cycle that slowly lowers base-level activation.
CSM_DEFAULT_AFFECT_TERM = 0.5  # Affect term in a snapshot for an item that has no salience (not a percept).
CSM_PRECURSOR_BONUS = 0.3  # Snapshot bonus for an item that a pathway precursor codelet produced.
CSM_SNAPSHOT_NOISE = 0.05  # Maximum absolute snapshot jitter at salience bias 0.
CSM_SNAPSHOT_EXPLOIT = 0.9  # At or above this salience bias, the snapshot sorts and does not sample.

# Global workspace. Token counts use pm.llm.context_window.estimate_tokens.
WORKSPACE_SHARD_TOKENS = 12000  # Content budget of each shard, in tokens.
WORKSPACE_TAIL_TOKENS = 6000  # Separate budget of each shard tail, in tokens.
WORKSPACE_HOLD_BONUS = 0.1  # Activation bonus that keeps an incumbent item against a newcomer.
CCQ_LENGTH = 8  # Number of recent broadcasts that the cycle state keeps.

# Codelets and decisions.
CODELET_OUTPUT_VARIANTS = 1  # Number of output variants that each codelet returns.
DECISION_VARIANTS = 1  # Number of decision variants for each stimulus group.
CODELET_BASE_ACTIVATION = 0.5  # Initial base-level activation of a codelet. Step 5 tunes the activation formula.
CODELET_COOLDOWN_TICKS = 2  # Ticks after a run with a reduced activation score (old CodeletRuntime default).
PATHWAY_SKIP_DECAY = 0.5  # Pathway strength factor for each unregistered codelet that a chain skips.

# Deliberate focus.
FOCUS_ACTIVATION_WEIGHT = 0.3  # Maximum activation that one focus endorsement adds to an item.
FOCUS_MAX_SHARE = 0.3  # Maximum part of a shard budget that deliberate focus can fill.

# Affect: the EMA mental state over feature deltas (D27).
# See pm.records.mental_state.compute_mental_state for the formula.
MENTAL_STATE_REFERENCE_MINUTES = 24 * 60  # Reference time frame at ego strength 0. The half-life is this frame times |1 - ego_strength|.
MENTAL_STATE_SIGMA_FLOOR = 1e-3  # Smallest deviation scale of one axis. It stops a division by zero for an axis without history.
MENTAL_STATE_TANH_GAIN = 1.5  # Soft clip gain. A deviation of two scales reads as tanh(1.5), about 0.9 of the axis range.
EXPECTATION_VALENCE_GAIN = 0.55  # Core valence change for an expectation error of 1.
EXPECTATION_EMOTION_GAIN = 0.35  # Joy (better) or sadness (worse) change for an expectation error of magnitude 1.

# Regulation mode. The appraise phase maps the appraisal and the accepted state to one mode.
THREAT_APPRAISAL_THRESHOLD = 0.6  # Appraised bodily threat at which danger preempts contemplation.
AVERSIVE_REACTIVE_THRESHOLD = 0.75  # Appraised aversive pressure that makes regulation reactive.
FEAR_REACTIVE_THRESHOLD = 0.55  # Fear that makes regulation reactive.
ANGER_REACTIVE_THRESHOLD = 0.6  # Anger that makes regulation reactive.
SADNESS_REFLECTIVE_THRESHOLD = 0.5  # Sadness that makes regulation reflective.
WITHDRAWAL_REFLECTIVE_THRESHOLD = 0.55  # Disengage-and-silence pressure that makes regulation reflective.

# Main cognition retries a cycle after exhausted model format repairs.
# The delay starts here and doubles after each failed cycle, up to the cap. Seconds.
CYCLE_RETRY_DELAY_SECONDS = 5.0
CYCLE_RETRY_MAX_DELAY_SECONDS = 300.0

# Feelings in words (workflows.main_cognition.affect.describe_mental_state).
# Intensity words: a value at or above a threshold gets that word.
FEELING_STRONG = 0.7  # "strong"
FEELING_MODERATE = 0.4  # "moderate"
FEELING_MILD = 0.15  # "mild"
FEELING_HINT = 0.05  # "a hint of". Below this value the text does not name the feeling.
FEELING_MAX_EMOTIONS = 3  # Number of strongest emotions that the text names.
FEELING_MAX_NEEDS = 2  # Number of most pressing needs that the text names.
FEELING_MAX_RELATIONSHIP = 2  # Number of strongest relationship axes that the text names. Each needs at least FEELING_MILD.
NEED_PRESSING = 0.4  # A supply need below this value, or the closeness drive above 1 minus it, is pressing.
MOOD_VALENCE_THRESHOLD = 0.15  # Core valence at which the mood reads as good (+) or low (-) spirits.
MOOD_AROUSAL_KEYED = 0.6  # Core arousal that reads as "keyed up".
MOOD_AROUSAL_ALERT = 0.3  # Core arousal that reads as "alert".
MOOD_DOMINANCE_THRESHOLD = 0.3  # Core dominance that reads as sure (+) or out of depth (-).
BEHAVIOR_PRESSING = 0.5  # Expression tendency at which the text names the urge.
ATTENTION_DRIFT_THRESHOLD = 0.2  # Valence focus bias at which attention drifts toward threat (-) or reward (+).
