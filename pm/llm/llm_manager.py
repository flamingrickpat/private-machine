from typing import Dict, Any
from typing import Dict, Any
from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class LlmManager:
    def __init__(self, model_map: Dict[str, Any], test_mode: bool=False):
        self.test_mode = test_mode
        self.model_map = model_map
        self.dimensions = model_map["embedding_dim"]
        self.emb_model = SentenceTransformer(model_map["embedding_model"], truncate_dim=self.dimensions,
                                             local_files_only=True)

    def get_embedding(self, text: str) -> List[float]:
        if self.emb_model is None or self.test_mode:
            return np.random.random((self.dimensions,)).tolist()
        embeddings = self.emb_model.encode([text], show_progress_bar=False)
        embeddings = torch.tensor(embeddings).to('cpu').tolist()[0]
        return embeddings
