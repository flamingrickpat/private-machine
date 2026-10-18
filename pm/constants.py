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
