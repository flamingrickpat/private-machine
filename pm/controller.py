from lancedb import LanceDBConnection

from pm.config.config import read_config_file
from pm.embedding.transformer_embedding import TransformerEmbedding
from pm.llm.llamacpp import LlamaCppLlm
from pm.log_utils import setup_logger


class Controller:
    def __init__(self):
        self.config = read_config_file("config.json")
        self.db = LanceDBConnection(self.config.db_path)
        setup_logger("main.log")
        self.llm = LlamaCppLlm()
        self.llm.set_model(self.config.model)
        self.embedder = TransformerEmbedding()
        self.embedder.set_model(self.config.embedding_model)
        self.embedder_clustering = TransformerEmbedding()
        self.embedder_clustering.set_model(self.config.embedding_model_clustering)

    def start(self):
        from pm.database.db_helper import init_db
        init_db()

    def format_str(self, template: str) -> str:
        values = {
            "user_name": self.config.user_name,
            "companion_name": self.config.companion_name
        }
        formatted_string = template.format(**values)
        return formatted_string


controller = Controller()