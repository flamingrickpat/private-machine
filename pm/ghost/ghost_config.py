# --- Configuration ---
from pydantic import BaseModel


class GhostConfig(BaseModel):
    companion_name: str
    user_name: str
    universal_character_card: str
    default_decay_factor: float = 0.95
    retrieval_limit_episodic: int = 8
    retrieval_limit_facts: int = 16  # Reduced slightly
    retrieval_limit_features_context: int = 16  # Reduced slightly
    retrieval_limit_features_causal: int = 256  # Reduced slightly
    context_events_similarity_max_tokens: int = 512
    short_term_intent_count: int = 3
    # --- Expectation Config ---
    retrieval_limit_expectations: int = 5  # How many active expectations to check against stimulus
    expectation_relevance_decay: float = 0.8  # Decay factor per tick for expectation relevance (recency)
    expectation_generation_count: int = 2  # How many expectations to try generating per action
    # --- Action Selection Config ---
    min_simulations_per_reply: int = 1
    mid_simulations_per_reply: int = 1
    max_simulations_per_reply: int = 1  # Keep low initially for performance
    importance_threshold_more_sims: float = 0.5  # Stimulus valence abs() or max urgency > this triggers more simulators
    force_assistant: bool = False
    remember_per_category_limit: int = 4
    coalition_aux_rating_factor: float = 0.03
    cognition_delta_factor: float = 0.33
    complex_coalition_rating: bool = False


