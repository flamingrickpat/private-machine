from langchain_community.chat_models import ChatLlamaCpp
from pydantic import (
    BaseModel,
    Field,
    model_validator,
)
from typing_extensions import Self


class ChatLlamaCppCustom(ChatLlamaCpp):
    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        return self

    def set_client(self, client):
        self.client = client
