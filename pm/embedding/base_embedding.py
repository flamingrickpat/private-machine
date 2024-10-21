from abc import ABC,abstractmethod
from pydantic import BaseModel, Field
from typing import List, Union, Dict, Any, Optional
from enum import Enum
import logging

import numpy as np

logger = logging.getLogger(__name__)

class EmbeddingType(int, Enum):
    Transformer = 1

class EmbeddingModel(BaseModel):
    type: EmbeddingType = Field(default=EmbeddingType.Transformer)
    identifier: str
    path: str
    context_size: int = 4096
    output_dimension: int = 768

class Embedding(ABC):
    @abstractmethod
    def set_model(self, settings: EmbeddingModel):
        pass

    @abstractmethod
    def get_embedding_scalar(self, text: str) -> np.ndarray:
        pass

    @abstractmethod
    def get_embeddings(self, text_list: List[str]) -> np.ndarray:
        pass
