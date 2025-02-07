from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from pm.controller import controller
from pm.cosine_clustering.templates import (
    COSINE_CLUSTER_TEMPLATE_WORLD_REACTION,
    COSINE_CLUSTER_TEMPLATE_USER_EMOTIONAL_RESPONSE,
    COSINE_CLUSTER_TEMPLATE_USER_PREFERENCE,
    COSINE_CLUSTER_TEMPLATE_STRATEGIC_ADJUSTMENT,
    COSINE_CLUSTER_TEMPLATE_PERSONALITY_REFINEMENT,
    COSINE_CLUSTER_TEMPLATE_SELF_REALIZATION,
    COSINE_CLUSTER_TEMPLATE_UNDERSTANDING_RICK
)

# Define category hierarchy (Major Category → Subcategories)
TEMPLATE_CATEGORIES = {
    "conversation_external": {
        "world_reaction": COSINE_CLUSTER_TEMPLATE_WORLD_REACTION,
        "user_emotional_response": COSINE_CLUSTER_TEMPLATE_USER_EMOTIONAL_RESPONSE,
        "user_preference": COSINE_CLUSTER_TEMPLATE_USER_PREFERENCE
    },
    "ai_adaptation": {
        "strategic_adjustment": COSINE_CLUSTER_TEMPLATE_STRATEGIC_ADJUSTMENT,
        "personality_refinement": COSINE_CLUSTER_TEMPLATE_PERSONALITY_REFINEMENT
    },
    "ai_cognitive_growth": {
        "self_realization": COSINE_CLUSTER_TEMPLATE_SELF_REALIZATION,
        "understanding_rick": COSINE_CLUSTER_TEMPLATE_UNDERSTANDING_RICK
    }
}


def cluster_text(cause_effect_text, template_dict):
    """Computes cosine similarity scores for all nested categories and returns a structured dictionary."""

    # Get embedding for the input cause-effect pair
    cause_effect_embedding = np.array(controller.get_embedding(cause_effect_text)).reshape(1, -1)

    results = {}

    for major_category, subcategories in template_dict.items():
        results[major_category] = {}  # Initialize major category

        for subcategory, examples in subcategories.items():
            embeddings = np.array([controller.get_embedding(text) for text in examples])

            # Compute cosine similarity with each example in the subcategory
            similarities = cosine_similarity(cause_effect_embedding, embeddings)

            # Average similarity across all examples in the subcategory
            avg_similarity = np.mean(similarities)

            results[major_category][subcategory] = avg_similarity  # Store score

    return results  # Returns a nested dictionary


def get_best_category(nested_scores, threshold=0.0):
    """Finds the highest-rated category from a nested similarity dictionary.

    Args:
        nested_scores (dict): The structured dictionary with similarity scores.
        threshold (float): Minimum score to consider a category valid.

    Returns:
        str: The highest-rated category as 'major_category -> subcategory'.
    """

    best_category = None
    best_score = -1

    for major_category, subcategories in nested_scores.items():
        for subcategory, score in subcategories.items():
            if score > best_score and score >= threshold:
                best_score = score
                best_category = f"{subcategory}"

    return best_category  # Returns the best matching category as a string


if __name__ == '__main__':
    # Test 1
    res = cluster_text("cause: emmy tells rick a corny joke. effect: rick doesn't like it", template_dict=TEMPLATE_CATEGORIES)
    print(get_best_category(res))

    # Test 2
    res = cluster_text("cause: emmy tells rick a corny joke. effect: rick doesn't like it at all and tells emmy he hates stupid jokes", template_dict=TEMPLATE_CATEGORIES)
    print(get_best_category(res))
