from typing import List

import numpy as np

from pm.ghost.ghost_r1 import GhostR1
from pm.model.knoxel_core import KnoxelBase
from pm.utils.emb_utils import cosine_dist


class MemoryInterface:
    @staticmethod
    def sample_knoxels_embedding(knoxel_source: List[KnoxelBase], query_embedding: List[float], limit: int) -> List[KnoxelBase]:
        if not query_embedding or not knoxel_source: return []
        candidates = [k for k in knoxel_source if k.embedding]
        if not candidates: return []
        query_embedding_np = np.array(query_embedding)
        # Use batch embedding retrieval if possible, otherwise loop
        embeddings_np = np.array([k.embedding for k in candidates])

        distances = [cosine_dist(query_embedding_np, emb) for emb in embeddings_np]
        sorted_candidates = sorted(zip(distances, candidates), key=lambda x: x[0])
        return [k for dist, k in sorted_candidates[:limit]]