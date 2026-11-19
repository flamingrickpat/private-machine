from pydantic import BaseModel, ConfigDict


class MinecraftPersona(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    companion_name: str = ""
    user_name: str = ""
    character_card: str = ""
    workspace: str = ""
