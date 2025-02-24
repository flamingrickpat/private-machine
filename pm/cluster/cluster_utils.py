from typing import List, Dict

import numpy as np
from pydantic import BaseModel, Field
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

from pm.database.tables import Event


def compute_contextual_embeddings(events: List[Event], context_window):
    embeddings = np.array([event.embedding for event in events])
    contextual_embeddings = []
    for i in range(len(embeddings)):
        tk = events[i].token
        token_weight = np.log(1 + tk)
        temporal_weight = 1
        weight = temporal_weight * token_weight

        combined_embedding = embeddings[i] * weight  # Start with the current embedding
        weight_sum = weight  # Include weight for the current message

        # Add context embeddings
        for j in range(max(0, i - context_window), min(len(embeddings), i + context_window + 1)):
            if i == j:
                continue  # Skip the current message itself

            tk = events[j].token
            temporal_weight = 1 / (1 + abs(i - j))
            token_weight = np.log(1 + tk)
            weight = temporal_weight * token_weight

            combined_embedding += weight * embeddings[j]
            weight_sum += weight

        # Normalize by total weight
        combined_embedding /= weight_sum
        contextual_embeddings.append(combined_embedding)

    return np.array(contextual_embeddings)

class ClusterDescription(BaseModel):
    topic_switch_cluster: Dict[int, int] = Field(default_factory=dict)
    conceptual_cluster: Dict[int, int] = Field(default_factory=dict)

def get_optimal_clusters(events: List[Event]) -> ClusterDescription:
    if len(events) < 2:
        return ClusterDescription()

    embeddings = compute_contextual_embeddings(events, 3)
    cosine_sim_matrix = cosine_similarity(embeddings)
    cosine_dist_matrix = 1 - cosine_sim_matrix

    max_metric = -1
    best_threshold = None

    threshold = 4
    while threshold > 0:
        agg_clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, metric='precomputed',
                                                 linkage='average', compute_full_tree=True)
        labels = agg_clustering.fit_predict(cosine_dist_matrix)
        lblcnt = len(np.unique(labels))

        if lblcnt > 1 and lblcnt < len(events):
            metric = silhouette_score(np.abs(cosine_dist_matrix), labels, metric='precomputed')
        else:
            metric = -1

        if metric > max_metric:
            max_metric = metric
            best_threshold = threshold

        threshold -= 0.01

    agg_clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=best_threshold, metric='precomputed',
                                             linkage='average', compute_full_tree=True)
    labels = agg_clustering.fit_predict(cosine_dist_matrix)
    for i in range(1, len(labels) - 1):
        if labels[i] != labels[i - 1] and labels[i] != labels[i + 1]:
            labels[i] = labels[i - 1]

    res = ClusterDescription()

    prev_cluster = -1
    cur_cluster = 0
    for i, event in enumerate(events):
        res.conceptual_cluster[event.id] = labels[i]

        if labels[i] != prev_cluster:
            cur_cluster += 1

        res.topic_switch_cluster[event.id] = cur_cluster
        prev_cluster = labels[i]

    return res