from pathlib import Path
from typing import List, Optional, Dict
import json
import re

from pydantic import BaseModel, Field, ValidationError

from pm.llm.base_llm import LlmModel, LlmPreset
from pm.embedding.base_embedding import EmbeddingModel

class MainConfig(BaseModel):
    models: List[LlmModel]
    model_map: Dict[LlmPreset, str]
    embedding_model: EmbeddingModel
    embedding_model_clustering: EmbeddingModel
    db_path: str = "./.lancedb/"
    user_name: str
    companion_name: str
    character_card_assistant: str
    character_card_story: str
    initial_character_memory: List[str] = Field(default_factory=list)
    example_dialog: List[str] = Field(default_factory=list)
    brave_api_key: str | None = Field(default=None)
    tavily_api_key: str | None = Field(default=None)
    telegram_token: str | None = Field(default=None)
    telegram_chat_id: str | None = Field(default=None)

    def get_model(self, preset: LlmPreset) -> LlmModel:
        for model in self.models:
            if model.identifier == self.model_map[preset.value]:
                return model
        raise Exception("model not configured")

class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass


def format_validation_error(error: ValidationError) -> str:
    """Format a ValidationError into a user-friendly string."""
    error_messages = []
    for error in error.errors():
        location = " -> ".join(str(loc) for loc in error['loc'])
        message = f"Error in {location}: {error['msg']}"
        error_messages.append(message)
    return "\n".join(error_messages)


def read_config_file(file_path: str) -> MainConfig:
    try:
        path = Path(file_path)
        if not path.exists():
            raise ConfigError(f"Configuration file not found: {file_path}")

        with open(path, 'r') as file:
            # Remove comments
            content = re.sub(r'^\s*//.*$', '', file.read(), flags=re.MULTILINE)

        try:
            # Parse JSON
            config_dict = json.loads(content)
        except json.JSONDecodeError as e:
            raise ConfigError(f"Invalid JSON in configuration file: {e}")

        # Create and validate Pydantic model
        try:
            return MainConfig(**config_dict)
        except ValidationError as e:
            formatted_error = format_validation_error(e)
            raise ConfigError(f"Configuration validation failed:\n{formatted_error}")

    except ConfigError as e:
        print(f"Configuration Error: {e}")
    except Exception as e:
        print(f"Unexpected error while reading configuration: {e}")
