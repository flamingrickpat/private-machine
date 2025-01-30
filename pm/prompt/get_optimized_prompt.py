import itertools
from typing import List
import random

import numpy as np
import torch
from torch import cosine_similarity

from pm.cluster.cluster_utils import get_optimal_clusters
from pm.controller import controller
from pm.database.tables import Event, Cluster, ClusterType, PromptItem, EventCluster
from pm.embedding.embedding import get_cos_sim


def get_optimized_prompt(
    all_events: List[Event],
    all_clusters: List[Cluster],
    all_event_cluster: List[EventCluster],
    n_context_size: int = 1024,
    n_split: float = 0.33
) -> List[PromptItem]:
    """
    Creates an optimized prompt by balancing recent events, context clusters, and temporal summaries.

    Parameters:
        all_events: List of all available events.
        all_clusters: List of all clusters.
        all_event_cluster: List of EventCluster mappings.
        n_context_size: Total token budget for the prompt.
        n_split: Fraction of tokens reserved for recent events (default 50%).

    Returns:
        A sorted list of PromptItem objects optimized for relevance and completeness.
    """
    # Token budget allocation
    recent_token_budget = int(n_context_size * n_split)
    context_token_budget = int(n_context_size * n_split)
    temp_token_budget = int(n_context_size * n_split)

    # Sort events by timestamp (most recent first)
    all_events.sort(key=lambda x: x.timestamp, reverse=True)

    # Step 1: Select most recent events within the budget
    recent_events = []
    recent_event_ids = []
    recent_tokens_used = 0
    for event in all_events:
        if recent_tokens_used + event.token <= recent_token_budget:
            recent_events.append(event.to_prompt_item())
            recent_event_ids.append(event.id)
            recent_tokens_used += event.token
        else:
            break

    # Step 2: Rank clusters by relevance (e.g., cosine similarity with recent events)
    query_embedding = all_events[-1].embedding

    def rank_clusters(clusters: List[Cluster], query_embedding):
        if query_embedding is None:
            return clusters
        for cluster in clusters:
            cluster.similarity = get_cos_sim(query_embedding, cluster.embedding)
        return sorted(clusters, key=lambda x: x.similarity, reverse=True)

    ranked_clusters = rank_clusters(all_clusters, query_embedding)

    # Step 3: Select clusters within the context budget
    selected_clusters = []
    context_tokens_used = 0

    def get_cluster_events(cluster: Cluster):
        return [
            event for event in all_events
            if any(ec.cog_event_id == event.id and ec.con_cluster_id == cluster.id for ec in all_event_cluster)
        ]

    for cluster in ranked_clusters:
        cluster_events = get_cluster_events(cluster)
        cluster_event_ids = np.array([x.id for x in cluster_events])
        recent_event_ids_array = np.array(recent_event_ids)
        if not np.isin(cluster_event_ids, recent_event_ids_array).all():
            cluster_event_tokens = sum(event.token for event in cluster_events)
            if context_tokens_used + cluster_event_tokens <= context_token_budget:
                selected_clusters.extend(event.to_prompt_item() for event in cluster_events)
                #selected_clusters.append(cluster.to_prompt_item())
                context_tokens_used += cluster_event_tokens
            else:
                break


    # Step 4: Fill gaps with temporal summaries (if applicable)
    temporal_summaries = [
        cluster for cluster in ranked_clusters if cluster.type == ClusterType.Temporal
    ]
    for summary in temporal_summaries:
        if context_tokens_used + summary.token <= context_token_budget:
            selected_clusters.append(summary.to_prompt_item())
            context_tokens_used += summary.token
        elif context_tokens_used >= context_token_budget:
            break

    # Combine recent events and clusters
    res = recent_events + selected_clusters

    # Sort by timestamp to maintain chronological order
    res.sort(key=lambda x: x.timestamp)

    return res


def get_optimized_prompt_temp_cluster(
        all_events: List[Event],
        all_clusters: List[Cluster],
        all_event_cluster: List[EventCluster],
        search_string: str,
        n_context_size: int = 2048,
        n_split: float = 0.33
) -> List[PromptItem]:
    """
    Creates an optimized prompt by balancing recent events, context clusters, and temporal summaries.

    Parameters:
        all_events: List of all available events.
        all_clusters: List of all clusters.
        all_event_cluster: List of EventCluster mappings.
        n_context_size: Total token budget for the prompt.
        n_split: Fraction of tokens reserved for recent events (default 50%).

    Returns:
        A sorted list of PromptItem objects optimized for relevance and completeness.
    """
    # Token budget allocation
    recent_token_budget = int(n_context_size * n_split)
    context_token_budget = int(n_context_size * n_split)
    temp_token_budget = int(n_context_size * n_split)

    # Sort events by timestamp (most recent first)
    all_events.sort(key=lambda x: x.timestamp, reverse=True)

    # Step 1: Select most recent events within the budget
    recent_events = []
    recent_event_ids = []
    recent_tokens_used = 0
    for event in all_events:
        if recent_tokens_used + event.token <= recent_token_budget:
            recent_events.extend(event.to_prompt_item())
            recent_event_ids.append(event.id)
            recent_tokens_used += event.token
        else:
            break

    # Step 2: Rank clusters by relevance (e.g., cosine similarity with recent events)
    query_embedding = controller.get_embedding(search_string)

    def rank_clusters(clusters: List[Cluster], qe):
        if qe is None:
            return clusters
        for cluster in clusters:
            cluster.similarity = get_cos_sim(qe, cluster.embedding)
        return sorted(clusters, key=lambda x: x.similarity, reverse=True)


    # Step 2: Remove clusters that are subsets of recent history
    def is_subset(cluster_events, recent_event_ids):
        cluster_event_ids = np.array([event.id for event in cluster_events])
        return np.all(np.isin(cluster_event_ids, list(recent_event_ids)))

    if True:
        if True:
            # Convert all relevant data to NumPy arrays
            event_ids = np.array([event.id for event in all_events])
            cluster_ids = np.array([cluster.id for cluster in all_clusters])
            ec_event_ids = np.array([ec.cog_event_id for ec in all_event_cluster])
            ec_cluster_ids = np.array([ec.con_cluster_id for ec in all_event_cluster])
            recent_event_ids_set = set(recent_event_ids)  # Use a set for fast membership testing

            # Initialize a list for filtered clusters
            filtered_clusters = []

            # Iterate over clusters but minimize operations inside the loop
            for cluster in all_clusters:
                # Find all events linked to the current cluster using NumPy filtering
                relevant_events_mask = ec_cluster_ids == cluster.id
                relevant_event_ids = ec_event_ids[relevant_events_mask]

                # Find the intersection with all event IDs
                cluster_event_ids = event_ids[np.isin(event_ids, relevant_event_ids)]

                # Check if the cluster event IDs are not a subset of recent event IDs
                if not set(cluster_event_ids).issubset(recent_event_ids_set):
                    filtered_clusters.append(cluster)

            # Update the `all_clusters` variable
            all_clusters = filtered_clusters

    ranked_clusters = rank_clusters([c for c in all_clusters if c.type == ClusterType.Topical], query_embedding)

    # Step 3: Select clusters within the context budget
    selected_clusters = []
    context_tokens_used = 0

    def get_cluster_eventsSlow(cluster: Cluster):
        return [
            event for event in all_events
            if any(ec.cog_event_id == event.id and ec.con_cluster_id == cluster.id for ec in all_event_cluster)
        ]

    def get_cluster_events(cluster: Cluster):
        # Find all events linked to the given cluster using NumPy filtering
        relevant_events_mask = ec_cluster_ids == cluster.id
        relevant_event_ids = ec_event_ids[relevant_events_mask]

        # Return the events where their IDs match the relevant event IDs
        return [event for event in all_events if event.id in relevant_event_ids]

    tmp = []
    for cluster in ranked_clusters:
        cluster_events = get_cluster_events(cluster)
        cluster_event_ids = np.array([x.id for x in cluster_events])
        recent_event_ids_array = np.array(recent_event_ids)
        if not np.isin(cluster_event_ids, recent_event_ids_array).all():
            tmp.append(cluster)
    ranked_clusters = tmp

    for cluster in ranked_clusters:
        cluster_events = get_cluster_events(cluster)
        cluster_event_ids = np.array([x.id for x in cluster_events])
        recent_event_ids_array = np.array(recent_event_ids)
        if not np.isin(cluster_event_ids, recent_event_ids_array).all():
            cluster_event_tokens = sum(event.token for event in cluster_events)
            if context_tokens_used + cluster_event_tokens <= context_token_budget:
                selected_clusters.extend(cluster.to_prompt_item([event.to_prompt_item()[0] for event in cluster_events]))
                context_tokens_used += cluster_event_tokens
            else:
                break

    temporal_clusters = []
    temporal_tokens = 0

    temp_cluster = [c for c in all_clusters if c.type == ClusterType.Temporal]
    temp_cluster_map = {x.id: x for x in temp_cluster}
    clusters = get_optimal_clusters(temp_cluster)
    if len(clusters.topic_switch_cluster.keys()) > 0:
        all_clusters = set(clusters.topic_switch_cluster.values())
        mp = {x: [] for x in all_clusters}
        for key, value in clusters.topic_switch_cluster.items():
            mp[value].append(temp_cluster_map[key])

        mp_avg_emb = {}
        mp_sim = {}
        for key in mp.keys():
            emb = mp[key][0].embedding
            for tmp in mp[key][1:]:
                for i in range(len(emb)):
                    emb[i] += tmp.embedding[i]
            for i in range(len(emb)):
                emb[i] /= len(mp[key])
            mp_avg_emb[key] = emb
            mp_sim[key] = get_cos_sim(emb, query_embedding)

        sorted_keys = sorted(mp_sim, key=mp_sim.get)
        while True:
            popped = False
            limit = False
            for key in sorted_keys:
                if len(mp[key]) == 0:
                    continue

                cluster = mp[key].pop(random.randrange(len(mp[key])))
                popped = True

                if temporal_tokens + cluster.token > temp_token_budget:
                    limit = True
                    break
                else:
                    temporal_clusters.extend(cluster.to_prompt_item())
                    temporal_tokens += cluster.token

            if not popped:
                break
            if limit:
                break

    def flatten_comprehension(matrix):
        return [item for row in matrix for item in row]

    # Combine recent events and clusters
    res = recent_events + selected_clusters + temporal_clusters
    res = list(set(res))

    # Sort by timestamp to maintain chronological order
    res.sort(key=lambda x: x.timestamp)

    return res




# relevance = get_cos_sim(all_events[-1].embedding, cluster.embedding)


def create_optimized_promptasdf(
    all_events: List[Event],
    all_clusters: List[Cluster],
    all_event_cluster: List[EventCluster],
    n_context_size: int = 1024,
    n_split: float = 0.5
) -> List[PromptItem]:
    """
    Creates an optimized prompt by balancing recent events, context clusters, and temporal summaries.

    Parameters:
        all_events: List of all available events.
        all_clusters: List of all clusters.
        all_event_cluster: List of EventCluster mappings.
        n_context_size: Total token budget for the prompt.
        n_split: Fraction of tokens reserved for recent events (default 50%).

    Returns:
        A sorted list of PromptItem objects optimized for relevance and completeness.
    """
    import numpy as np
    from scipy.optimize import linprog

    # Token budget allocation
    recent_token_budget = int(n_context_size * n_split)
    context_token_budget = n_context_size - recent_token_budget

    # Sort events by timestamp (most recent first)
    all_events.sort(key=lambda x: x.timestamp, reverse=True)

    # Step 1: Select most recent events within the budget
    recent_events = []
    recent_event_ids = set()
    recent_tokens_used = 0
    for event in all_events:
        if recent_tokens_used + event.token <= recent_token_budget:
            recent_events.append(event.to_prompt_item())
            recent_event_ids.add(event.id)
            recent_tokens_used += event.token
        else:
            break

    # Step 3: Filter out clusters that are subsets of recent events
    def is_subset(cluster_events, recent_event_ids):
        cluster_event_ids = np.array([event.id for event in cluster_events])
        return np.all(np.isin(cluster_event_ids, list(recent_event_ids)))

    def get_cluster_events(cluster: Cluster):
        return [
            event for event in all_events
            if any(ec.cog_event_id == event.id and ec.con_cluster_id == cluster.id for ec in all_event_cluster)
        ]

    # Step 2: Prepare context clusters and summaries for optimization
    items = []
    for cluster in all_clusters:
        cluster_events = get_cluster_events(cluster)
        if not is_subset(cluster_events, recent_event_ids):
            relevance = get_cos_sim(all_events[-1].embedding, cluster.embedding)
            items.append({"type": "cluster", "id": cluster.id, "weight": relevance, "tokens": cluster.token})


    # Define the optimization problem for the context budget
    weights = [item["weight"] for item in items]
    tokens = [item["tokens"] for item in items]
    context_items = []
    if len(items) > 2:
        # Objective function: maximize weights (negative for linprog minimization)
        c = -np.array(weights)

        # Token budget constraint for context
        A = [tokens]
        b = [context_token_budget]

        # Bounds for variables (0 or 1 for inclusion)
        bounds = [(0, 1) for _ in items]

        # Solve the optimization problem
        result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")

        # Step 3: Select clusters based on optimization result
        selected_clusters = [items[i] for i, x in enumerate(result.x) if x > 0.5]

        # Convert selected clusters to PromptItem objects
        for item in selected_clusters:
            if item["type"] == "cluster":
                context_items.append(next(cluster.to_prompt_item() for cluster in all_clusters if cluster.id == item["id"]))

    # Combine recent events and context clusters
    res = recent_events + context_items

    # Sort by timestamp to maintain chronological order
    res.sort(key=lambda x: x.timestamp)

    return res


def create_optimized_prompt(
    all_events: List[Event],
    all_clusters: List[Cluster],
    all_event_cluster: List[EventCluster],
    n_context_size: int = 1024
) -> List[PromptItem]:
    """
    Creates an optimized prompt by balancing recent history, relevant clusters, and temporal summaries.

    Parameters:
        all_events: List of all available events.
        all_clusters: List of all clusters.
        all_event_cluster: List of EventCluster mappings.
        n_context_size: Total token budget for the prompt.

    Returns:
        A sorted list of PromptItem objects optimized for relevance and completeness.
    """
    import numpy as np

    # Token budget allocation (33% for each category)
    recent_token_budget = int(n_context_size * 0.33)
    cluster_token_budget = int(n_context_size * 0.33)
    temporal_token_budget = n_context_size - recent_token_budget - cluster_token_budget

    # Step 1: Select most recent events within the budget
    all_events.sort(key=lambda x: x.timestamp, reverse=True)
    recent_events = []
    recent_event_ids = set()
    recent_tokens_used = 0

    for event in all_events:
        if recent_tokens_used + event.token <= recent_token_budget:
            recent_events.append(event.to_prompt_item())
            recent_event_ids.add(event.id)
            recent_tokens_used += event.token
        else:
            break

    # Step 2: Remove clusters that are subsets of recent history
    def is_subset(cluster_events, recent_event_ids):
        cluster_event_ids = np.array([event.id for event in cluster_events])
        return np.all(np.isin(cluster_event_ids, list(recent_event_ids)))

    filtered_clusters = []

    for cluster in all_clusters:
        cluster_events = [
            event for event in all_events
            if any(ec.cog_event_id == event.id and ec.con_cluster_id == cluster.id for ec in all_event_cluster)
        ]
        if not is_subset(cluster_events, recent_event_ids):
            filtered_clusters.append(cluster)

    # Step 3: Add the most relevant clusters within the budget
    filtered_clusters.sort(key=lambda c: c.embedding.sum(), reverse=True)  # Assuming relevance can be derived from embedding
    relevant_clusters = []
    cluster_tokens_used = 0

    for cluster in filtered_clusters:
        if cluster_tokens_used + cluster.token <= cluster_token_budget:
            relevant_clusters.append(cluster.to_prompt_item())
            cluster_tokens_used += cluster.token
        else:
            break

    # Step 4: Add temporal summaries, starting with years
    temporal_summaries = [cluster for cluster in all_clusters if cluster.type == ClusterType.Temporal]
    temporal_summaries.sort(key=lambda c: c.level)  # Start with higher-level summaries (e.g., years)

    temporal_items = []
    temporal_tokens_used = 0

    for summary in temporal_summaries:
        if temporal_tokens_used + summary.token <= temporal_token_budget:
            temporal_items.append(summary.to_prompt_item())
            temporal_tokens_used += summary.token
        else:
            break

    # Combine recent events, relevant clusters, and temporal summaries
    res = recent_events + relevant_clusters + temporal_items

    # Sort by timestamp to maintain chronological order
    res.sort(key=lambda x: x.timestamp)

    return res
