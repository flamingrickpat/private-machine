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
WORKSPACE_MIN_RATING = 0.6  # Lowest entry priority of a newcomer or an incumbent. Reserved places ignore it. Raised from 0.5 after the A19F step 6 replays: ratings ran 0.75-0.95 and everything entered.
WORKSPACE_HOME_BONUS = 0.05  # Priority bonus in the home shard of a record: the shard of the entity it is about, or the target shard of its structure.
WORKSPACE_TAIL_CANDIDATES = 96  # Newest stream records that the host offers for each shard tail.
CCQ_LENGTH = 8  # Number of recent broadcasts that the cycle state keeps.
DIALOGUE_THOUGHT_LIMIT = 12  # Most private thoughts that a reply transcript shows. Only thoughts newer than its oldest dialogue line count.

# Structure builders (port of _build_structures_get_coalitions) and attention into shards
# (port of _simulate_attention_on_coalitions in private-machine-head/pm/ghosts/ghost_lida.py).
COALITION_ITEMS_PER_CLUSTER = 8  # The clustering makes one cluster for each this many CSM items (old value).
COALITION_MAX_CLUSTERS = 6  # Upper bound of clusters, so the attention prompt stays focused.
COALITION_AUX_RATING_FACTOR = 0.03  # Weight of the normalized auxiliary rating. It only breaks ties (old value).
COALITION_RECENCY_DECAY = 0.95  # Auxiliary rating factor for each tick since an item entered the CSM (old value).
STRUCTURE_VIEW_TOKENS = 3000  # Token budget of the CSM view that the structure binder reads.
STRUCTURE_BINDER_MAX = 5  # Most structures that one binder call returns.
ATTENTION_LINE_CHARS = 400  # Characters of one record line in the attention rating prompt.
ATTENTION_APERTURE_THRESHOLD = 0.5  # Mental aperture beyond which the rating asks for narrow (-) or broad (+) focus (old value).
ATTENTION_WINNERS_NARROW = 1  # Structures per shard that can win at a narrow aperture. The old code let one coalition win.
ATTENTION_WINNERS_DEFAULT = 2  # Structures per shard that can win at a medium aperture.
ATTENTION_WINNERS_BROAD = 3  # Structures per shard that can win at a broad aperture.

# Codelets and decisions.
CODELET_OUTPUT_VARIANTS = 1  # Number of output variants that each codelet returns.
DECISION_VARIANTS = 1  # Number of decision variants for each stimulus group.
CODELET_BASE_ACTIVATION = 0.5  # Initial base-level activation of a codelet.
CODELET_COOLDOWN_TICKS = 2  # Ticks after a run with a reduced activation score (old CodeletRuntime default).
PATHWAY_SKIP_DECAY = 0.5  # Pathway strength factor for each unregistered codelet that a chain skips.
PATHWAY_BOOST_SCALE = 0.25  # Stored activation that a pathway step of strength 1.0 adds. Old strengths are 1.4 to 1.9.

# Codelet activation score (port of ActivationFactors in private-machine-head/pm/codelets/codelet.py).
# score = W_ACTIVATION * stored + W_STIMULUS * stimulus + W_STATE * state + W_VECTOR * cosine
#         + W_BASE * base - W_RECENCY * recency + W_RANDOM * noise, then the cooldown factor, clamped to [0, 1].
CODELET_W_ACTIVATION = 1.0  # Weight of the stored activation (selector ratings and pathway boosts).
CODELET_W_STIMULUS = 0.35  # Weight of the stimulus match (1 when a current stimulus kind fits the codelet).
CODELET_W_STATE = 0.2  # Weight of the match between the mental state and the codelet families.
CODELET_W_VECTOR = 0.25  # Weight of the cosine similarity of signature and situation embeddings.
CODELET_W_BASE = 0.2  # Weight of the base-level activation (attentional learning).
CODELET_W_RECENCY = 0.05  # Weight of the recency penalty 1 / (1 + ticks since the last run).
CODELET_W_RANDOM = 0.05  # Weight of the uniform noise in [0, 1) that prevents deadlocks.
CODELET_COOLDOWN_FACTOR = 0.25  # Score factor while a codelet is in its cooldown.
CODELET_ACTIVATION_DECAY = 0.5  # Factor for each cycle on the stored activation.
CODELET_MAX_ACTIVATION = 1.0  # Upper bound of the stored activation.
CODELET_BASE_LEARN_RATE = 0.1  # Part of the distance to 1.0 that a workspace entry adds to base-level activation.
CODELET_BASE_FORGET = 0.995  # Factor for each cycle that slowly lowers base-level activation.
CODELET_SELECTOR_WEIGHT = 0.5  # Stored activation that a selector rating of 1.0 adds.
CODELET_SELECTOR_CANDIDATES = 12  # Number of top-scored codelets that the LLM selector rates.
CODELET_ECHO_SIMILARITY = 0.5  # Embedding cosine at which an earlier line of the same speaker echoes the newest line. story_continuation replay: the four deferrals scored 0.51-0.59, most unrelated lines 0.36-0.40, one unrelated line 0.51. The selector reads the lines and judges them, so a false echo costs little.
CODELET_BUDGET_DEFAULT = 4  # Codelet runs of a cycle when triage sets no budget.
CODELET_BUDGET_MAX = 12  # Most codelet runs that triage can grant to one cycle.
CODELET_WAVES = 2  # Waves of one cycle. A later wave sees the CSM after the earlier waves.
CODELET_VIEW_TOKENS = 1500  # Token budget of one codelet's view of the CSM.
CODELET_PERCEPT_TYPES = 3  # Percept types that one percept conversion call offers.
CODELET_PERCEPT_TEMPERATURE = 0.2  # Randomness of the percept type sample: 0 is the top types, 1 is weighted random.
CODELET_TEXT_ATTEMPTS = 10  # Attempts for one prose call that returns empty text.
CODELET_AGENT_ATTEMPTS = 3  # Sessions for one agent codelet whose answer fails validation.
CODELET_AGENT_TIMEOUT_SECONDS = 600  # Deadline of one agent codelet session.

# What the retrieve phase puts into the CSM, with the endorser "retrieval".
CSM_STIMULUS_ACTIVATION = 1.0  # Activation of a current stimulus.
CSM_DIALOGUE_ACTIVATION = 0.6  # Activation of one of the last dialogue turns.
CSM_SELECTED_MEMORY_ACTIVATION = 0.7  # Activation of a memory that the retrieve phase selected.
CSM_CANDIDATE_ACTIVATION = 0.4  # Activation of a retrieved memory candidate that was not selected.
CSM_RECENT_DIALOGUE_TURNS = 8  # Number of dialogue turns before the stimuli that the host supplies.
CSM_NARRATIVE_ACTIVATION = 0.5  # Activation of a narrative of the self or of a group entity.
CSM_GOAL_ACTIVATION = 0.5  # Activation of an active goal.

# Expectations (port of _generate_expectations_for_action and _appraise_stimulus_check_expectation
# in private-machine-head/pm/ghosts/ghost_lida.py, commit 7101933).
EXPECTATION_COUNT = 2  # Expectations that the route phase records for each routed action (old expectation_generation_count).
EXPECTATION_CHECK_LIMIT = 5  # Most open expectations that one appraisal checks (old retrieval_limit_expectations).
EXPECTATION_RELEVANCE_DECAY = 0.8  # Recency factor for each tick since the expectation was recorded (old value).
EXPECTATION_MIN_RECENCY = 0.05  # An expectation with a lower recency factor is too old to check (old value).
EXPECTATION_MIN_RELEVANCE = 0.1  # Lowest relevance score of a checked expectation (old value).
EXPECTATION_SAME_SIMILARITY = 0.92  # Cosine similarity at which a new expectation replaces an open one. Old: distance 0.08, with an inverted test.
EXPECTATION_MIN_CHANGE = 0.05  # Fulfillment change below which an outcome has no emotional effect (old value).
EXPECTATION_VALENCE_MULT = 0.6  # Valence impact of a fulfillment change of 1 for a wanted valence of 1 (old value).
EXPECTATION_ANXIETY_MULT = 0.4  # Fear impact multiplier of a fulfillment change (old anxiety multiplier).
EXPECTATION_FEAR_BASE = 0.5  # Fear falls by this part of a fulfillment rise: a met expectation confirms predictability (old value).
EXPECTATION_STRONG_VALENCE = 0.3  # Wanted valence beyond which the fear modulation below applies (old value).
EXPECTATION_FEAR_DISAPPOINT = 0.3  # Extra fear when a wanted outcome fails (old value).
EXPECTATION_FEAR_CONFIRMED = 0.7  # Extra fear when a feared outcome comes true (old value).
EXPECTATION_FEAR_RELIEF = 0.4  # Fear drop when a feared outcome fails (old value).
EXPECTATION_OPEN_LIMIT = 16  # Newest pending expectations that the host offers to one cycle.
EXPECTATION_SUPPORTED = 0.8  # Fulfillment at or above which an expectation counts as supported.
EXPECTATION_CONTRADICTED = 0.2  # A direct stimulus with a fulfillment score at or below this value contradicts it.
EXPECTATION_DIRECT = 0.5  # Directness at or above which a stimulus can contradict an expectation.
CSM_EXPECTATION_ACTIVATION = 0.5  # CSM activation of an open expectation, with the endorser "expectation".

# Deliberate focus.
FOCUS_ACTIVATION_WEIGHT = 0.3  # Maximum activation that one focus endorsement adds to an item.
FOCUS_MAX_SHARE = 0.3  # Maximum part of a shard budget that deliberate focus can fill.

# Affect: the EMA mental state over feature deltas (D27).
# See pm.records.mental_state.compute_mental_state for the formula.
MENTAL_STATE_REFERENCE_MINUTES = 24 * 60  # EMA half-life of the mental state. Ego-strength scaling is disabled for now.
MENTAL_STATE_SIGMA_FLOOR = 0.35  # Smallest deviation scale of one axis. A small floor lets one small delta pin the readout at a bound. ALPHA RUN: raised from 0.25 to soften the readout (see the alpha block below). Tune with replays.
MENTAL_STATE_TANH_GAIN = 1.0  # Soft clip gain. A deviation of two scales reads as tanh(1.0), about 0.76 of the axis range. Tune with replays.
EXPECTATION_VALENCE_GAIN = 0.55  # Core valence change for an expectation error of 1.
EXPECTATION_EMOTION_GAIN = 0.35  # Joy (better) or sadness (worse) change for an expectation error of magnitude 1.

# ############################################################################
# ##                                                                        ##
# ##   !!!  ALPHA RUN ONLY: TEMPORARY FAILSAFE CLAMPS. FIX PROPERLY LATER  !!!
# ##                                                                        ##
# ##   These limits are hard failsafes for the first owner run with the     ##
# ##   real persona. They hide the symptom, not the cause. In the A19F      ##
# ##   review run, one stimulus moved core valence from +0.78 to -1.0 in    ##
# ##   one tick, and the boredom codelet jumped to 0.95 on a tease.         ##
# ##   The real fix is a tuned affect design (D27: sigma floor, gain,       ##
# ##   half-life, appraisal calibration) and a calibrated codelet           ##
# ##   selector. When that is done, DELETE this block and every use of      ##
# ##   the ALPHA_* names (grep ALPHA_).                                     ##
# ##                                                                        ##
# ############################################################################
ALPHA_MAX_STIMULUS_DELTA = 1.0 / 3.0  # ALPHA ONLY. Largest absolute delta of one record (stimulus appraisal, percept, expectation) on one EMA axis. The EMA reader clamps every sample, old stored ones too.
ALPHA_MAX_STATE_STEP = 1.0 / 3.0  # ALPHA ONLY. Largest change of one mental state axis (core affect, every emotion including boredom, needs, relationship) from one cycle to the next.
ALPHA_MAX_RATING_STEP = 1.0 / 3.0  # ALPHA ONLY. Largest change of the codelet selector rating of one codelet (for example BoredomThresholdCross) from its previous rating.
# ############################################################################

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

# Web chat (pm.app.web_chat). The server binds to loopback only.
WEB_CHAT_HOST = "127.0.0.1"  # Bind address. Other machines cannot connect.
WEB_CHAT_PORT = 8770  # Default TCP port. `--web-port` changes it.
WEB_CHAT_POLL_MS = 1500  # Milliseconds between two browser polls for new messages.
WEB_CHAT_MAX_BODY_BYTES = 64_000  # Largest accepted POST body, in bytes.
WEB_CHAT_MAX_NAME_CHARS = 64  # Longest accepted login name, in characters.

# Character import (pm.app.chat_import). Times are naive UTC.
IMPORT_MESSAGE_SPACING_SECONDS = 120  # Synthetic gap between two messages of a log without timestamps.
IMPORT_SESSION_GAP_SECONDS = 3600  # The synthetic history ends this long before the import time, so live chat starts a new session.
IMPORT_SESSION_SPLIT_SECONDS = 2 * 3600  # A silence of at least this long between two rows starts a new session id.
IMPORT_SOURCE_PREFIX = "chat_import"  # Prefix of the provenance source_ref of an imported record.
TRANSFER_EVENT_SOURCE = "memory_transfer"  # override_source of the synthetic transfer event.

# Memory reconsolidation (pm.memory.reconsolidation). Port of DynamicMemoryConsolidator
# in private-machine-head/pm/memory_consolidation.py; the old values are marked.
RECONSOLIDATION_TRIGGER_EVENTS = 60  # Unclustered events that trigger an automatic run (old min_event_memory_consolidation_threshold).
RECONSOLIDATION_MIN_EVENTS = 3  # Fewest unclustered events that a topical run processes (old min_events_for_processing).
RECONSOLIDATION_CONTEXT_WINDOW = 3  # Neighbors on each side in the contextual embedding (old topical_clustering_context_window).
RECONSOLIDATION_MIN_BLOCK = 8  # A topic block of at least twice this size splits into chunks of at least this size (old min_split_up_cluster_size).
RECONSOLIDATION_CHUNK_EVENTS = 400  # Events per clustering window. The pure Python clustering is quadratic in this number.
RECONSOLIDATION_OPEN_TOPIC_MINUTES = 60  # The newest topic block stays open while its last event is younger than this.
RECONSOLIDATION_SUMMARY_TOKEN_LIMIT = 4000  # Raw story tokens up to which a temporal range is summarized from raw events (old hierchical_summary_token_limit).
RECONSOLIDATION_NARRATIVE_MIN_TOKENS = 3000  # New evidence tokens that a narrative refinement needs (old narrative_rolling_tokens_min).
RECONSOLIDATION_NARRATIVE_MAX_TOKENS = 4500  # Upper evidence size of one refinement (old narrative_rolling_tokens_nax). A buffer closes at the mean of min and max.
RECONSOLIDATION_NARRATIVE_FIRST_TOKENS = 300  # Evidence tokens that the first version of a narrative needs. New in the port.
RECONSOLIDATION_DAY_START_HOUR = 5  # A logical day runs from 05:00 to 05:00 (old _SHIFT_HOURS).
RECONSOLIDATION_WORKERS = 4  # Parallel model calls in one stage.
RECONSOLIDATION_EMBED_WORKERS = 8  # Parallel embedding requests.
RECONSOLIDATION_PROVIDER_RETRIES = 3  # Repeats of a call that the provider failed (ModelError). Transport errors retry inside JsonClient.
RECONSOLIDATION_TEMPERATURE = 0.4  # Sampling temperature of the reconsolidation calls.
RECONSOLIDATION_SESSION_GAP_MINUTES = 120  # Contextual embeddings do not blend events across a silence this long. New in the port.
RECONSOLIDATION_COMMIT_GROUP = 16  # Clusters or ranges that one commit holds, so a long run keeps finished work.
# Synthetic transfer records of a character import. {companion_name}, {user_name}, and {date} are filled at import.
TRANSFER_EVENT_TEXT = ("{companion_name} moved from the earlier chat harness into pm by a memory transfer on {date}. "
                       "The earlier conversations with {user_name} were imported from a chat log; inner thoughts and "
                       "feelings of that time were not recorded and are reconstructed only as summaries.")
TRANSFER_SELF_MODEL_TEXT = ("{companion_name} came into pm through a memory transfer on {date}. The conversations with "
                            "{user_name} before that date come from an imported chat log, without recorded inner "
                            "thoughts or feelings.")
TRANSFER_NARRATIVE_LINE = ("On {date}, {companion_name} moved into pm through a memory transfer; the time with "
                           "{user_name} before that is known from the imported chat log, and its inner life only from "
                           "reconstructed summaries.")
# Capabilities of a new character home when `character init` gets no --capabilities file.
DEFAULT_SUPPORTED_CAPABILITIES = (
    "I can talk with {user_name} through text messages in this chat.",
    "I can sort through my memories and sleep on what happened (memory reconsolidation).",
)
DEFAULT_UNSUPPORTED_CAPABILITIES = (
    "I cannot act in the physical world.",
    "I cannot run external actions unless an integrated workflow exists for them.",
)
RECONSOLIDATION_TIME_FORMAT = "%A, %d %B %Y, %H:%M"  # Times in reconsolidation prompts. A numeric day.month.year format made the model read 01.05.26 as 5 January.

# The tool_call workflow and its MCP servers (pm.llm.mcp_config, workflows.tool_call).
TOOL_CALL_SESSION_SECONDS = 1800  # Deadline of one native tool session, in seconds. A timeout can follow a completed external effect.
TOOL_CALL_REQUEST_SECONDS = 600  # Default deadline of one MCP tool request, in seconds.
TOOL_CALL_TURN_LIMIT = 8  # Newest complete GWT turns, up to the source cycle, that the tool session reads.
TOOL_DISCOVERY_SECONDS = 5.0  # Deadline of one MCP tool listing for the route prompt, in seconds (as the Minecraft discovery).
TOOL_DISCOVERY_RETRY_SECONDS = 60.0  # A server that did not answer is asked again after this many seconds, so a down server does not delay every cycle.

# Images in records (pm.records.media, pm.llm.multimodal, pm.memory.media_alt).
MEDIA_MAX_IMAGE_BYTES = 8 * 1024 * 1024  # Largest decoded image that a record accepts, in bytes.
PROMPT_MAX_IMAGES = 8  # Only the last this-many image tags of one prompt become real image parts; earlier tags become alt text.
PROMPT_IMAGE_TOKENS = 1024  # Token estimate for one image part in context measurement. Providers bill images by tiles; this is a planning bound, not a count.
ALT_TEXT_MAX_CHARS = 400  # Longest accepted alt text. A longer answer counts as invalid output and is retried.
ALT_TEXT_ATTEMPTS = 4  # Model calls for one alt text before the error propagates (provider errors and invalid output).
ALT_TEXT_TEMPERATURE = 0.2  # Sampling temperature of the alt-text call.
WEB_CHAT_MAX_IMAGES = 6  # Most images in one web chat message.
WEB_CHAT_MAX_IMAGE_BYTES = 5 * 1024 * 1024  # Largest decoded image in one web chat upload, in bytes.
WEB_CHAT_MAX_IMAGE_BODY_BYTES = 48 * 1024 * 1024  # Largest POST body of a message with images (base64 grows the data by a third).
MINECRAFT_MCP_SERVER = "pm-minecraft"  # Server name in the provenance of a Minecraft screenshot (pm.records.media.MediaProvenance.server).
BODY_EPISODE_TAGGED_IMAGES = 3  # Newest screenshots of a body episode that get a tag in its content. All screenshots stay in the record's media.
