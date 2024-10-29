from datetime import datetime
from typing import List, Type, Callable

from pydantic import BaseModel, Field


class AgentMessage(BaseModel):
    created_at: datetime = Field(default_factory=datetime.now)
    text: str
    public: bool = Field(default=True)
    from_tool: bool = Field(default=False)

class Agent(BaseModel):
    name: str
    system_prompt: str = Field(default_factory=str)
    description: str
    goal: str
    task: str = Field(default_factory=str)
    tools: List[Type[BaseModel]] = Field(default_factory=list)
    functions: List[Callable] = Field(default_factory=list)
    messages: List[AgentMessage] = Field(default_factory=list)