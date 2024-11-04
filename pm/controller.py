from collections import defaultdict
from typing import Dict, List, Tuple, Type

from lancedb import LanceDBConnection
from pydantic import BaseModel

from pm.config.config import read_config_file
from pm.consts import THOUGHT_SEP
from pm.embedding.transformer_embedding import TransformerEmbedding
from pm.llm.base_llm import LlmPreset, CommonCompSettings, LlmType
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

    def get_thought_string_story(self, thought: str) -> str:
        return (f"{controller.config.companion_name} thinks: {thought}"
                f"I shouldn't think to long, now I'll respond to {controller.config.user_name}!\n\n"
                f"{controller.config.companion_name}: "
                ).strip()

    def get_thought_string_assistant(self, thought: str) -> str:
        return f"Thought: {thought}".strip()

    def get_response_string_story(self, response: str) -> str:
        return f"{controller.config.companion_name}: {response}".strip()

    def get_response_string_assistant(self, response: str) -> str:
        return f"Response: {response}".strip()

    def completion_tool(self,
                        preset: LlmPreset,
                        inp: List[Tuple[str, str]],
                        comp_settings: CommonCompSettings | None = None,
                        tools: List[Type[BaseModel]] = None
                        ) -> (str, List[BaseModel]):
        if comp_settings is None:
            comp_settings = CommonCompSettings()

        if tools is None:
            tools = []

        if len(tools) > 0:
            comp_settings.tools_json += tools

        model = self.config.get_model(preset)
        if model.type == LlmType.LlamaCpp:
            self.llm.set_model(model)
            response = self.llm.completion(prompt=self.llm.convert_langchain_to_raw_string(inp, comp_settings), comp_settings=comp_settings)
            content = response.output_sanitized

            if len(comp_settings.tools_json) > 0:
                from json_repair import repair_json
                content = repair_json(content)

            models = []
            for tool in comp_settings.tools_json:
                try:
                    obj = tool.model_validate_json(content)
                    models.append(obj)
                    break
                except:
                    pass
            return content, models
        else:
            raise Exception("not implemented")

    def completion_text(self,
                        preset: LlmPreset,
                        inp: List[Tuple[str, str]],
                        comp_settings: CommonCompSettings | None = None,
                        ) -> (str, List[BaseModel]):
        content, models = self.completion_tool(preset, inp, comp_settings)
        return content


controller = Controller()