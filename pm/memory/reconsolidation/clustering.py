"""Topical clustering of an event sequence, in pure Python.

This is a port of the clustering in ``DynamicMemoryConsolidator``
(``private-machine-head/pm/memory_consolidation.py``) and of
``pm/utils/cluster_utils.py``. The old code used numpy and scikit-learn.
The project locks neither, so these functions use lists of floats.

The steps:

1. :func:`contextual_embeddings` blends each event vector with its
   neighbors, so a short reply carries the topic of its context.
2. :func:`average_linkage` builds the dendrogram of the cosine distances
   once (nearest-neighbor chain, average linkage).
3. :func:`optimal_labels` cuts the dendrogram at 91 distance thresholds
   from 1.0 to 0.1 and keeps the cut with the best silhouette score among
   the allowed cluster counts.
4. :func:`smooth_labels` gives a single stray event the label of its
   neighbors.
5. :func:`topic_blocks` turns the labels into consecutive blocks and splits
   long blocks into chunks of at least the minimum size.

Every function is deterministic. Ties break by the lower index.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import BaseModel


class Merge(BaseModel):
    """One merge of the dendrogram: clusters ``left`` and ``right`` form ``merged``.

    Leaf clusters have the ids ``0..n-1``. Merged clusters get the ids
    ``n, n+1, ...`` in creation order.
    """

    distance: float = 0.0
    left: int = 0
    right: int = 0
    merged: int = 0


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def cosine_distances(vectors: Sequence[Sequence[float]]) -> list[list[float]]:
    """Return the matrix ``max(0, 1 - cosine similarity)`` with a zero diagonal.

    A zero vector has distance 1 to every other vector.
    """
    norms = [_norm(vector) for vector in vectors]
    size = len(vectors)
    result = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1, size):
            if norms[i] == 0 or norms[j] == 0:
                distance = 1.0
            else:
                dot = sum(a * b for a, b in zip(vectors[i], vectors[j]))
                distance = max(0.0, 1.0 - dot / (norms[i] * norms[j]))
            result[i][j] = result[j][i] = distance
    return result


def contextual_embeddings(vectors: Sequence[Sequence[float]], tokens: Sequence[int], window: int,
                          sessions: Sequence[int] | None = None) -> list[list[float]]:
    """Blend each vector with up to ``window`` neighbors on each side.

    Port of ``_compute_contextual_embeddings``. The own vector has the weight
    ``log(1 + tokens)``. A neighbor at index distance ``d`` has the weight
    ``log(1 + tokens) / (1 + d)``. The result is the weighted mean.

    ``sessions`` holds a session number for each event. New in the port: a
    neighbor in another session does not count, because a long silence
    usually ends a topic. Without ``sessions`` all events share one session.
    """
    size = len(vectors)
    result: list[list[float]] = []
    for i in range(size):
        weight = math.log(1 + max(1, tokens[i]))
        combined = [value * weight for value in vectors[i]]
        total = weight
        for j in range(max(0, i - window), min(size, i + window + 1)):
            if j == i or (sessions is not None and sessions[j] != sessions[i]):
                continue
            neighbor = math.log(1 + max(1, tokens[j])) / (1 + abs(i - j))
            combined = [value + neighbor * other for value, other in zip(combined, vectors[j])]
            total += neighbor
        result.append([value / total for value in combined])
    return result


def average_linkage(distances: Sequence[Sequence[float]]) -> list[Merge]:
    """Return the average-linkage dendrogram, sorted by merge distance.

    The nearest-neighbor chain finds the same merges as the naive
    algorithm, in O(n^2) time. Average linkage has no inversions, so the
    merges sorted by distance (stable) form a valid merge order: a child
    merge always comes before its parent.
    """
    size = len(distances)
    table: dict[int, dict[int, float]] = {i: {j: distances[i][j] for j in range(size) if j != i} for i in range(size)}
    sizes = {i: 1 for i in range(size)}
    merges: list[Merge] = []
    chain: list[int] = []
    next_id = size
    while len(sizes) > 1:
        if not chain:
            chain.append(min(sizes))
        current = chain[-1]
        previous = chain[-2] if len(chain) > 1 else None
        row = table[current]
        nearest = min(row, key=lambda other: (row[other], other))
        # Prefer the previous chain element on a tie, so the chain ends.
        if previous is not None and row[previous] <= row[nearest]:
            nearest = previous
        if nearest != previous:
            chain.append(nearest)
            continue
        chain.pop()
        chain.pop()
        left, right = sorted((current, nearest))
        merged = next_id
        next_id += 1
        merges.append(Merge(distance=row[nearest], left=left, right=right, merged=merged))
        left_size, right_size = sizes.pop(left), sizes.pop(right)
        new_row: dict[int, float] = {}
        for other in sizes:
            value = (left_size * table[left][other] + right_size * table[right][other]) / (left_size + right_size)
            new_row[other] = value
            del table[other][left]
            del table[other][right]
            table[other][merged] = value
        del table[left]
        del table[right]
        table[merged] = new_row
        sizes[merged] = left_size + right_size
    return sorted(merges, key=lambda merge: merge.distance)


def cut(merges: Sequence[Merge], size: int, threshold: float) -> list[int]:
    """Return the labels of the clusters below ``threshold``.

    Like scikit-learn's ``distance_threshold``, a merge applies only when
    its distance is lower than the threshold. Labels number the clusters in
    the order of their first event.
    """
    members: dict[int, list[int]] = {i: [i] for i in range(size)}
    for merge in merges:
        if merge.distance >= threshold:
            break
        members[merge.merged] = members.pop(merge.left) + members.pop(merge.right)
    labels = [0] * size
    for label, leaves in enumerate(sorted(members.values(), key=min)):
        for leaf in leaves:
            labels[leaf] = label
    return labels


def silhouette(distances: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    """Return the mean silhouette score of a precomputed distance matrix.

    A member of a single-event cluster scores 0, as in scikit-learn.
    """
    clusters: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        clusters.setdefault(label, []).append(index)
    scores = []
    for index, label in enumerate(labels):
        own = clusters[label]
        if len(own) == 1:
            scores.append(0.0)
            continue
        inner = sum(distances[index][other] for other in own if other != index) / (len(own) - 1)
        outer = min(sum(distances[index][other] for other in members) / len(members)
                    for key, members in clusters.items() if key != label)
        scores.append((outer - inner) / max(inner, outer) if max(inner, outer) > 0 else 0.0)
    return sum(scores) / len(scores)


def allowed_cluster_counts(size: int) -> range:
    """Return the cluster counts that the old code accepted for ``size`` events.

    The old rule was ``max(2, n // 20) < k < min(n, n // 8)``, so there are
    on average 8 to 20 events per topic.
    """
    return range(max(2, size // 20) + 1, min(size, size // 8))


def optimal_labels(distances: Sequence[Sequence[float]], fallback: Sequence[int] | None = None) -> list[int]:
    """Return the cluster label of each event, with the best silhouette score.

    Port of ``_get_optimal_clusters``. The thresholds run from 1.0 down to
    0.1 in steps of 0.01. Only allowed cluster counts compete. The first
    best score wins, so a tie keeps the coarser cut.

    New in the port: when no threshold gives an allowed count (fewer than
    about 24 events), the events get the ``fallback`` labels, for example
    their session numbers, or label 0. The old code returned no clusters
    then. :func:`topic_blocks` still splits a long single block.
    """
    size = len(distances)
    default = list(fallback) if fallback is not None else [0] * size
    if size < 2:
        return default
    merges = average_linkage(distances)
    allowed = allowed_cluster_counts(size)
    best: list[int] | None = None
    best_score = -1.1
    seen: set[tuple[int, ...]] = set()
    for step in range(91):
        threshold = round(1.0 - step * 0.01, 10)
        labels = cut(merges, size, threshold)
        key = tuple(labels)
        if key in seen or len(set(labels)) not in allowed:
            continue
        seen.add(key)
        score = silhouette(distances, labels)
        if score > best_score:
            best, best_score = labels, score
    return best if best is not None else default


def smooth_labels(labels: Sequence[int]) -> list[int]:
    """Give a single event between two equal labels the label of its neighbors."""
    result = list(labels)
    for i in range(1, len(result) - 1):
        if result[i] != result[i - 1] and result[i] != result[i + 1] and result[i - 1] == result[i + 1]:
            result[i] = result[i - 1]
    return result


def split_block(items: Sequence[int], min_size: int) -> list[list[int]]:
    """Split a block into chunks of at least ``min_size`` items, in order.

    Port of ``split_block_keep_order``. A block shorter than twice the
    minimum stays whole. The remainder spreads over the first chunks.
    """
    count = len(items)
    if count < 2 * min_size:
        return [list(items)]
    chunks = count // min_size
    sizes = [min_size] * chunks
    for index in range(count - min_size * chunks):
        sizes[index % chunks] += 1
    result, start = [], 0
    for chunk_size in sizes:
        result.append(list(items[start:start + chunk_size]))
        start += chunk_size
    return result


def topic_blocks(labels: Sequence[int], min_size: int, sessions: Sequence[int] | None = None) -> list[list[int]]:
    """Return consecutive runs of equal labels as index blocks.

    Port of ``cluster_blocks_with_min_size``. Each run is split with
    :func:`split_block`. New in the port: a run also ends where the session
    number changes, so one episode never spans a long silence.
    """
    result: list[list[int]] = []
    run: list[int] = []
    for index, label in enumerate(labels):
        new_session = sessions is not None and run and sessions[run[-1]] != sessions[index]
        if run and (labels[run[-1]] != label or new_session):
            result.extend(split_block(run, min_size))
            run = []
        run.append(index)
    if run:
        result.extend(split_block(run, min_size))
    return result


__all__ = ["Merge", "allowed_cluster_counts", "average_linkage", "contextual_embeddings", "cosine_distances",
           "cut", "optimal_labels", "silhouette", "smooth_labels", "split_block", "topic_blocks"]
