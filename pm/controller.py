from collections import defaultdict
from typing import Dict

from lancedb import LanceDBConnection

from pm.config.config import read_config_file
from pm.consts import THOUGHT_SEP
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
            self.llm = LlamaCppLlm(verbose=False)
            self.llm.set_model(self.config.model)
            self.embedder = TransformerEmbedding()
            self.embedder.set_model(self.config.embedding_model)
            self.embedder_clustering = TransformerEmbedding()
            self.embedder_clustering.set_model(self.config.embedding_model_clustering)
            self.nlp = NlpSpacy()
            self.initialized = True  # Prevents re-initialization
            self.global_python_tools = {}

    def start(self):
        from pm.database.db_helper import init_db
        init_db()

    def format_str(self, template: str, extra: Dict = None) -> str:
        if extra is None:
            extra = {}

        values = {
            "user_name": self.config.user_name,
            "companion_name": self.config.companion_name,
            "character_card_story": self.config.character_card_story,
            "character_card_assistant": self.config.character_card_assistant
        }
        values.update(extra)

        for i in range(10):
            for k, v in values.items():
                template = template.replace("{" + k + "}", v)

        return template

    def format_thought(self, message_text: str, story_mode: bool):
        if THOUGHT_SEP in message_text:
            parts = message_text.strip().split(THOUGHT_SEP)
            if story_mode:
                return self.get_thought_string_story(parts[0]) + parts[1]
            else:
                return self.get_thought_string_assistant(parts[0]) + parts[1]
        else:
            return message_text

    def get_thought_string_story(self, thought: str) -> str:
        return (f"{controller.config.companion_name} thinks: {thought}"
                f"I shouldn't think to long, now I'll respond to {controller.config.user_name}!\n\n"
                f"{controller.config.companion_name}: "
                )

    def get_thought_string_assistant(self, thought: str) -> str:
        return f"**{thought}**\n"

controller = Controller()