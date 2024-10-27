from typing import Dict

from lancedb import LanceDBConnection

from pm.config.config import read_config_file
from pm.embedding.transformer_embedding import TransformerEmbedding
from pm.llm.llamacpp import LlamaCppLlm
from pm.log_utils import setup_logger
from pm.nlp.nlp_spacy import NlpSpacy


class Controller:
    _shared_state = {}  # This is the shared state dictionary

    def __init__(self):
        # Redirect the instance dictionary to the shared state
        self.__dict__ = self._shared_state

        # Initialize only if it's the first instance
        if not hasattr(self, 'initialized'):
            self.config = read_config_file("config.json")
            self.db = LanceDBConnection(self.config.db_path)
            setup_logger("main.log")
            self.llm = LlamaCppLlm(verbose=True)
            self.llm.set_model(self.config.model)
            self.embedder = TransformerEmbedding()
            self.embedder.set_model(self.config.embedding_model)
            self.embedder_clustering = TransformerEmbedding()
            self.embedder_clustering.set_model(self.config.embedding_model_clustering)
            self.nlp = NlpSpacy()
            self.initialized = True  # Prevents re-initialization

    def start(self):
        from pm.database.db_helper import init_db
        init_db()

    def format_str(self, template: str, extra: Dict = None) -> str:
        if extra is None:
            extra = {}

        values = {
            "user_name": self.config.user_name,
            "companion_name": self.config.companion_name
        }
        values.update(extra)
        formatted_string = template.format(**values)
        return formatted_string


controller = Controller()