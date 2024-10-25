from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

from pm.database.db_model import Message
from pm.controller import controller


def optimized_threshold_search(cosine_dist_matrix, target_clusters, initial_min=0.1, initial_max=1.0, tolerance=0.01,
                               max_iters=50):
    """
    Uses a binary search to find the distance threshold that yields the target number of clusters.

    Parameters:
        cosine_dist_matrix (ndarray): Precomputed cosine distance matrix for clustering.
        target_clusters (int): Desired number of clusters.
        initial_min (float): Minimum possible threshold.
        initial_max (float): Maximum possible threshold.
        tolerance (float): Tolerance for stopping condition when close to target.
        max_iters (int): Maximum number of iterations to prevent infinite loops.

    Returns:
        labels (ndarray): Cluster labels for each point.
        optimal_threshold (float): The threshold that yielded the target cluster count.
    """
    low, high = initial_min, initial_max
    optimal_threshold = (low + high) / 2

    for i in range(max_iters):
        agg_clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=optimal_threshold,
            metric='precomputed',
            linkage='average'
        )
        labels = agg_clustering.fit_predict(cosine_dist_matrix)
        num_clusters = labels.max() + 1  # Cluster count is max label + 1

        # Check if the current solution is within tolerance of the target
        if abs(num_clusters - target_clusters) <= tolerance:
            print(f"Optimal threshold found: {optimal_threshold}")
            return labels, optimal_threshold

        # Adjust threshold range based on whether we have too many or too few clusters
        if num_clusters < target_clusters:
            high = optimal_threshold  # Decrease threshold range
        else:
            low = optimal_threshold  # Increase threshold range

        # Update optimal_threshold to the midpoint of the new range
        optimal_threshold = (low + high) / 2

    # Return result if maximum iterations reached
    print(f"Max iterations reached. Closest threshold: {optimal_threshold}")
    return labels, optimal_threshold

def get_sliding_window_embedded_messages(messages: List[Message], base_window: int = 64) -> List[Message]:
    model = SentenceTransformer('all-MiniLM-L6-v2', trust_remote_code=True)
    def get_embedding_scalar_float_list_temp(t: str):
        return model.encode(t).tolist()
        #return controller.embedder_clustering.get_embedding_scalar_float_list(t)

    for i, msg in enumerate(messages):
        text_before = "\n".join(tmp.full_text for tmp in messages[0:i])
        text_after = "\n".join(tmp.full_text for tmp in messages[i + 1:])

        msg.embedding = get_embedding_scalar_float_list_temp(msg.full_text)

        for j in range(1, 4):
            window = base_window * j
            text = text_before[len(text_before) - window:] + "\n" + msg.full_text + "\n" + text_after[:window]
            tmp = get_embedding_scalar_float_list_temp(text)
            if j == 1:
                msg.embedding_l1 = tmp
            if j == 2:
                msg.embedding_l2 = tmp
            if j == 3:
                msg.embedding_l3 = tmp

        comb = ((np.array(msg.embedding) * 1) +
                  (np.array(msg.embedding_l1) * 0.75) +
                  (np.array(msg.embedding_l2) * 0.5) +
                  (np.array(msg.embedding_l3) * 0.33))

        msg.embedding_combined = comb.tolist()
    return messages


def get_agglomerative_clusters_as_dict(messages: List[Message]) -> Dict[str, float]:
    embeddings = np.array([msg.embedding_combined for msg in messages])

    # Compute the cosine similarity matrix
    cosine_sim_matrix = cosine_similarity(embeddings)

    # Cosine distance = 1 - cosine similarity
    cosine_dist_matrix = 1 - cosine_sim_matrix

    # Apply Agglomerative Clustering with precomputed distance matrix
    agg_clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5, metric='precomputed',
                                             linkage='average')
    labels = agg_clustering.fit_predict(cosine_dist_matrix)

    # Assign clusters back to messages
    res = {}
    for i, msg in enumerate(messages):
        res[msg.id] = labels[i]
    return res