import numpy as np
from sentence_transformers.util import cos_sim


def cosine_pair(a, b) -> float:
    return cos_sim(np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)).tolist()[0][0]
