import numpy as np
try:
    from sentence_transformers.util import cos_sim as _st_cos_sim
except Exception:
    _st_cos_sim = None


def cosine_sim(a, b) -> float:
    a_np = np.array(a, dtype=np.float32)
    b_np = np.array(b, dtype=np.float32)
    if a_np.size == 0 or b_np.size == 0 or a_np.shape != b_np.shape:
        return 0.0

    if _st_cos_sim is not None:
        return _st_cos_sim(a_np, b_np).tolist()[0][0]

    denom = float(np.linalg.norm(a_np) * np.linalg.norm(b_np))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a_np, b_np) / denom)

def cosine_dist(a, b) -> float:
    return 1 - cosine_sim(a, b)
