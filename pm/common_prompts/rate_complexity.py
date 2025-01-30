from typing import List, Tuple

from pydantic import BaseModel, Field

from pm.controller import controller
from pm.llm.base_llm import LlmPreset, CommonCompSettings

class Complexity(BaseModel):
    reason: str = Field(description="your reason for your complexity rating")
    complexity: float = Field(description="float from 0 - 1. 0 for full type 1 thinking, 1 for full type 2 thinking. so 0 is simple and 1 very complex")

def rate_complexity(block: str):
    sysprompt = "You are a helpful assistant. You determine if the AI companion {companion_name} needs to employ Type 1 or Type 2 thinking for the latest user input."
    prompt = [
        ("system", sysprompt),
        ("user", f"### BEGIN MESSAGES\n{block}\n### END MESSAGES\n"),
    ]

    content, calls = controller.completion_tool(LlmPreset.Fast, prompt, comp_settings=CommonCompSettings(temperature=0.4, max_tokens=1024), tools=[Complexity])
    return calls[0].complexity
