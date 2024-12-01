from typing import TypedDict, Dict, List

from pydantic import BaseModel, Field


class AgentState(BaseModel):
    next_agent: str = Field(default_factory=str)
    input: str = Field(default_factory=str)
    plan: str = Field(default_factory=str)
    thought: str = Field(default_factory=str)
    output: str = Field(default_factory=str)
    title: str = Field(default_factory=str)
    conversation_id: str = Field(default_factory=str)
    messages: List[Dict[str, str]] = Field(default_factory=list)
    story_mode: bool = Field(default=False)
    task: List[str] =Field(default_factory=list)
    complexity: float = Field(default=0)
    available_tools: List[str] = Field(default_factory=list)
    completion_mode: str = Field(default_factory=str)
    status: int = Field(default_factory=str)