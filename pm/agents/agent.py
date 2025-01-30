from datetime import datetime
from typing import List, Type, Callable

from pydantic import BaseModel, Field

from pm.llm.base_llm import CommonCompSettings


class AgentMessage(BaseModel):
    created_at: datetime = Field(default_factory=datetime.utcnow)
    text: str
    name: str
    public: bool = Field(default=True)
    from_tool: bool = Field(default=False)

class Agent(BaseModel):
    id: str
    name: str
    system_prompt: str | None = Field(default=None)
    description: str
    goal: str = Field(default_factory=str)
    task: str = Field(default_factory=str)
    tools: List[Type[BaseModel]] = Field(default_factory=list)
    functions: List[Callable] = Field(default_factory=list)
    messages: List[AgentMessage] = Field(default_factory=list)
    comp_settings: CommonCompSettings | None = Field(default=None)