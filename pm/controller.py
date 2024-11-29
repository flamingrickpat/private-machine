import shutil
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Type
import os

from lancedb import LanceDBConnection
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticToolsParser
from pydantic import BaseModel

from pm.config.config import read_config_file
from pm.consts import THOUGHT_SEP
from pm.embedding.transformer_embedding import TransformerEmbedding
from pm.llm.base_llm import LlmPreset, CommonCompSettings, LlmType
from pm.llm.llamacpp import LlamaCppLlm
from pm.llm.llm_openai import get_llm, chat_complete_openai
from pm.log_utils import setup_logger
from pm.nlp.nlp_spacy import NlpSpacy


class Controller:
    _shared_state = {}  # This is the shared state dictionary

    def setup_db_path(self):
        live_db_path = os.path.join(self.config.db_path, "live")

        # Ensure the base db_path directory exists
        os.makedirs(self.config.db_path, exist_ok=True)

        # If "live" directory already exists, back it up
        if os.path.exists(live_db_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_folder_name = f"backup_{timestamp}"
            backup_path = os.path.join(self.config.db_path, backup_folder_name)

            # Copy the current "live" folder to the backup path
            shutil.copytree(live_db_path, backup_path)
            print(f"Existing 'live' directory backed up to: {backup_path}")

        # Initialize the LanceDB connection without modifying the "live" folder
        self.db = LanceDBConnection(live_db_path)

    def __init__(self):
        # Redirect the instance dictionary to the shared state
        self.__dict__ = self._shared_state

        # Initialize only if it's the first instance
        if not hasattr(self, 'initialized'):
            self.config = read_config_file("config.json")
            self.setup_db_path()
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
            llm = get_llm(model, comp_settings)
            if len(tools) > 0:
                llm = llm.bind_tools(tools=tools, tool_choice="required")
            full = chat_complete_openai(llm, inp)
            content = full.content
            models = []
            for tool in comp_settings.tools_json:
                try:
                    obj = tool.model_validate_json(content)
                    models.append(obj)
                    break
                except:
                    pass
            return content, models

    def completion_text(self,
                        preset: LlmPreset,
                        inp: List[Tuple[str, str]],
                        comp_settings: CommonCompSettings | None = None,
                        ) -> (str, List[BaseModel]):
        content, models = self.completion_tool(preset, inp, comp_settings)
        return content


controller = Controller()