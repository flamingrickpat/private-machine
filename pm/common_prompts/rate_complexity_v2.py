import json
from typing import List

from pydantic import BaseModel, Field

from pm.controller import controller
from pm.llm.base_llm import LlmPreset, CommonCompSettings

sys_prompt = """You are a helpful assistant that determines conversation complexity for the AI.
Analyze the following chat log between two people, {user_name} and the AI {companion_name}. 
Your task is to determine the complexity of the query for {companion_name} as float, with 0 being very simple and 1 being very complex.
Your answer will determine if {companion_name} makes the next response using the fast and cheap or the slow and good LLM model.
You output valid JSON only, in this format:
{basemodel_schema}
"""

example1_user = """
{companion_name}: Hey there!  
{user_name}: Hey, how's it going?
{companion_name}: I like cats; they're just the cutest, aren't they?  
{user_name}: Absolutely, I have two Siamese at home. They never cease to entertain me."""

example1_assistant = """
{
    "reason": "they just talk about cute cats",
    "complexity": 0.21
}
"""

example2_user = """
{companion_name}: How can I assist you today?
{user_name}: I'm trying to understand the concept of recursive functions in Python. Could you explain it to me with an example?
{companion_name}: Absolutely! Recursive functions are functions that call themselves to solve a problem. Let's start with a simple example...
"""

example2_assistant = """
{
    "reason": "the user is asking for a technical explanation with an example, indicating medium complexity",
    "complexity": 0.58
}
"""

example3_user = """
{companion_name}: What would you like to learn today?
{user_name}: Can you help me set up a multi-threaded server in Python? I need it to handle multiple client connections simultaneously and be optimized for performance. It would be great if you could also touch on the challenges of thread safety and how to handle shared data across threads.
"""

example3_assistant = """
{
    "reason": "the user requests a detailed guide on a complex, multi-part topic involving multi-threading, performance optimization, and thread safety",
    "complexity": 0.87
}
"""

class ComplexityResponse(BaseModel):
    """Response format for determining complexity."""
    reason: str = Field(description="One short sentence about your reasoning.")
    complexity: float = Field(description="0 for very simple, 1 for very complex", ge=0, le=1)

    def execute(self, state):
        state["complexity"] = self.complexity
        return state

def rate_complexity_v2(message_block: str) -> float:
    messages = [(
        "system",
        controller.format_str(sys_prompt)
    ), (
        "user",
        controller.format_str(example1_user)
    ), (
        "assistant",
        example1_assistant
    ), (
        "user",
        controller.format_str(example2_user)
    ), (
        "assistant",
        example2_assistant
    ), (
        "user",
        controller.format_str(example3_user)
    ), (
        "assistant",
        example3_assistant
    ), (
        "user",
        message_block
    )]

    _, calls = controller.completion_tool(LlmPreset.Auxiliary, messages, comp_settings=CommonCompSettings(max_tokens=1024, temperature=0), tools=[ComplexityResponse])
    state = {"complexity": 0}
    for call in calls:
        call.execute(state)

    return state["complexity"]