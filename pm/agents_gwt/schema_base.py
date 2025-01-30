from typing import List

from pydantic import BaseModel, Field


class DynamicAgent(BaseModel):
    name: str
    description: str
    goal: str
    prompt: str

class Subsystem(BaseModel):
    name: str = Field(default_factory=str)
    agents: List[DynamicAgent] = Field(default_factory=list)
    description: str = Field(default_factory=str)
    prompt_prefix: str = Field(default_factory=str)
    goal: str = Field(default_factory=str)