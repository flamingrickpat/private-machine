from pydantic import BaseModel, ConfigDict


class WorldPosition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    x: float
    y: float
    z: float
