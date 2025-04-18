from typing import List

import numpy as np
from sentence_transformers.util import cos_sim
from sqlmodel import SQLModel

from pm.controller import controller


def get_embedding(text: str) -> List[float]:
    return controller.get_embedding(text)

def get_cos_sim(a, b) -> float:
    return cos_sim(np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)).tolist()[0][0]

def vector_search(all_entries: List[SQLModel], emb: List[float], n_facts: int) -> List[SQLModel]:
    embs = np.vstack([e.embedding for e in all_entries])
    # cache the normalized embeddings
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs_normed = embs / norms

    # 2) For each query vector `q`, do:
    def top_k_facts(q: np.ndarray, k: int):
        # normalize the query
        q_normed = q / np.linalg.norm(q)
        # cosine similarities = dot product since we’re normalized
        sims = embs_normed.dot(q_normed)            # shape (N,)
        # find the indices of the top k sims
        # np.argpartition is O(N), far faster than a full sort
        idxs = np.argpartition(-sims, k - 1)[:k]     # top-k unordered
        # now sort these k for exact order
        idxs_sorted = idxs[np.argsort(-sims[idxs])]
        # pull out your DB objects
        return [all_entries[i] for i in idxs_sorted]

    q = np.vstack([emb])
    res = top_k_facts(q, n_facts)
    return res

if __name__ == '__main__':
    a = get_embedding("hans")
    b = get_embedding("peter")
    print(get_cos_sim(a, b))