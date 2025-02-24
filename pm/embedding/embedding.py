from typing import List

import numpy as np
from sentence_transformers.util import cos_sim

from pm.controller import controller


def get_embedding(text: str) -> List[float]:
    return controller.get_embedding(text)

def get_cos_sim(a, b) -> float:
    return cos_sim(np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)).tolist()[0][0]


if __name__ == '__main__':
    a = get_embedding("hans")
    b = get_embedding("peter")
    print(get_cos_sim(a, b))