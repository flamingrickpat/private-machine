"""Pure clustering functions of memory reconsolidation, with hand-made vectors."""
import itertools
import math
import random
import unittest

from pm.memory.reconsolidation.clustering import (allowed_cluster_counts, average_linkage, contextual_embeddings,
                                                  cosine_distances, cut, optimal_labels, silhouette, smooth_labels,
                                                  split_block, topic_blocks)


def naive_average_linkage(distances):
    """Merge the closest pair again and again. Return the sorted merge distances."""
    clusters = {i: [i] for i in range(len(distances))}
    heights = []
    while len(clusters) > 1:
        best = min(((sum(distances[x][y] for x in clusters[a] for y in clusters[b]) /
                     (len(clusters[a]) * len(clusters[b])), a, b)
                    for a, b in itertools.combinations(sorted(clusters), 2)))
        heights.append(best[0])
        clusters[best[1]] = clusters[best[1]] + clusters.pop(best[2])
    return sorted(heights)


class ClusteringTests(unittest.TestCase):
    def test_context_blends_neighbors_by_token_weight_and_distance(self):
        vectors = [[1.0, 0.0], [0.0, 1.0]]
        result = contextual_embeddings(vectors, [1, 1], 3)
        own, other = math.log(2), math.log(2) / 2
        self.assertAlmostEqual(result[0][0], own / (own + other))
        self.assertAlmostEqual(result[0][1], other / (own + other))

    def test_context_does_not_cross_sessions(self):
        vectors = [[1.0, 0.0], [0.0, 1.0]]
        self.assertEqual(contextual_embeddings(vectors, [3, 3], 3, [0, 1]), vectors)

    def test_nearest_neighbor_chain_matches_the_naive_algorithm(self):
        generator = random.Random(7)
        vectors = [[generator.random() for _ in range(4)] for _ in range(14)]
        distances = cosine_distances(vectors)
        heights = [merge.distance for merge in average_linkage(distances)]
        for got, want in zip(heights, naive_average_linkage(distances)):
            self.assertAlmostEqual(got, want)

    def test_cut_numbers_clusters_by_their_first_event(self):
        distances = cosine_distances([[1, 0], [0, 1], [1, 0.05], [0, 1]])
        self.assertEqual(cut(average_linkage(distances), 4, 0.5), [0, 1, 0, 1])

    def test_optimal_labels_find_two_topics(self):
        generator = random.Random(3)
        vectors = []
        for topic in [0] * 15 + [1] * 15 + [2] * 15:
            base = [0.0] * 6
            base[topic * 2] = 1.0
            vectors.append([value + generator.random() * 0.1 for value in base])
        labels = optimal_labels(cosine_distances(vectors))
        self.assertIn(len(set(labels)), allowed_cluster_counts(45))
        self.assertEqual(len(set(labels[:15])), 1)
        self.assertEqual(len(set(labels[15:30])), 1)
        self.assertNotEqual(labels[0], labels[20])

    def test_too_few_events_give_one_topic_or_the_fallback(self):
        distances = cosine_distances([[1, 0], [0, 1], [1, 1]])
        self.assertEqual(optimal_labels(distances), [0, 0, 0])
        self.assertEqual(optimal_labels(distances, fallback=[0, 0, 1]), [0, 0, 1])

    def test_silhouette_of_separated_clusters_is_high(self):
        distances = cosine_distances([[1, 0], [1, 0.01], [0, 1], [0.01, 1]])
        self.assertGreater(silhouette(distances, [0, 0, 1, 1]), 0.9)
        self.assertEqual(silhouette(distances, [0, 1, 2, 3]), 0.0)

    def test_smoothing_absorbs_a_single_stray_event(self):
        self.assertEqual(smooth_labels([0, 0, 1, 0, 2, 2]), [0, 0, 0, 0, 2, 2])

    def test_blocks_keep_order_and_split_long_runs(self):
        self.assertEqual(split_block(list(range(5)), 3), [list(range(5))])
        self.assertEqual(split_block(list(range(7)), 3), [[0, 1, 2, 3], [4, 5, 6]])
        self.assertEqual(topic_blocks([0, 0, 1, 1, 1, 0], 8), [[0, 1], [2, 3, 4], [5]])
        self.assertEqual(topic_blocks([0, 0, 0, 0], 8, sessions=[0, 0, 1, 1]), [[0, 1], [2, 3]])


if __name__ == "__main__":
    unittest.main()
