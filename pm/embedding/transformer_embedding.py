import copy

import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import numpy as np
import torch
from numpy import dot
from numpy.linalg import norm

from pm.embedding.base_embedding import *

class Category:
    def __init__(self, label: str, content: str):
        self.label = label
        self.content = content
        self.embedding: np.ndarray | None = Field(default=None)
        self.score: float = Field(default=0, ge=0, le=1)

class TransformerEmbedding(Embedding):
    def __init__(self):
        self.model_path = None
        self.context_size = None
        self.output_dimension = None
        self.tokenizer = None
        self.model = None
        self.identifier = None
        self.cache: Dict[str, List[float]] = {}

    def set_model(self, settings: EmbeddingModel):
        if self.identifier is None or settings.identifier != self.identifier:
            self.model_path = settings.path
            self.context_size = settings.context_size
            self.output_dimension = settings.output_dimension
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = model.to(self.device)

    def get_embedding_scalar(self, text: str) -> np.ndarray:
        input_dict = self.tokenizer([text], max_length=self.context_size, padding=True, truncation=True, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in input_dict.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Mean pooling over all tokens (excluding padding)
        attention_mask = inputs['attention_mask']
        output_hidden_state = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(output_hidden_state.size()).float()
        sum_hidden_states = torch.sum(output_hidden_state * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)  # avoid division by zero
        mean_pooled = sum_hidden_states / sum_mask

        # Normalize the embedding
        normalized_embedding = F.normalize(mean_pooled, p=2, dim=1)

        return normalized_embedding[0].cpu().numpy()

    def get_embedding_scalar_float_list(self, text: str) -> List[float]:
        return self.get_embedding_scalar(text).tolist()

    def get_embeddings(self, text_list: List[str]) -> np.ndarray:
        input_dict = self.tokenizer(text_list, max_length=self.context_size, padding=True, truncation=True, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in input_dict.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        embedding = outputs.last_hidden_state[:, 0]
        return embedding.cpu().numpy()

    def categorize(self, text: str, cats: List[Category], max_cats: int = 0) -> List[Category]:
        embedding = self.get_embedding_scalar(text)
        res = copy.deepcopy(cats)
        for cat in res:
            if cat.content in self.cache:
                cat.embedding = self.cache[cat.content]
            else:
                cat.embedding = self.get_embedding_scalar(cat.content)

            a = embedding
            b = cat.embedding
            cat.score = dot(a, b)/(norm(a)*norm(b))

        res.sort(key=lambda x: x.score, reverse=True)
        if max_cats == 0:
            return res
        else:
            return res[:max_cats]
