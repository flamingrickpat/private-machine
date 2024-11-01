import enum
import json
import logging
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from pm.controller import controller
from pm.database.db_model import Message

from pm.llm.base_llm import LlmPreset, CommonCompSettings
from pm.tools.common import tools_list, tool_docs

# Define tool functions with Pydantic models and simple logging for execution
class ToolResponse(BaseModel):
    """Response format for determining tools to use."""
    reason: str = Field(description="One short sentence about your reasoning.")
    tool_names: List[str] = Field(description="List of tool names to call, as strings.")

    def execute(self, state):
        state["tool_names"] = self.tool_names

sys_prompt = """You are a helpful assistant that determines which tools the AI should use for the next action.
Analyze the following chat log between two people, {{user_name}} and the AI {{companion_name}}. 
Your task is to output a JSON object with a reason and a list of tool names (as strings) to use based on the user's query. 
If no tool is needed, the list should be empty.

Available tools:
{tool_docs}

Output valid JSON only in this format:
{json_schema}
"""

# Few-shot examples for tool determination
example1_user = """
{companion_name}: Hello! How can I assist today?
{user_name}: Can you tell me the weather in Paris?
"""

example1_assistant = """
{
    "reason": "The user is asking for weather information in Paris, so the WeatherTool is required.",
    "tool_names": ["WeatherTool"]
}
"""

example2_user = """
{companion_name}: Good morning! What would you like to know?
{user_name}: Could you help me translate 'Hello, how are you?' to French?
"""

example2_assistant = """
{
    "reason": "The user requested a translation, so the TranslateTool is needed for language translation.",
    "tool_names": ["TranslateTool"]
}
"""

example3_user = """
{companion_name}: How can I make your day easier?
{user_name}: Just curious, what’s the result of 45 times 13?
"""

example3_assistant = """
{
    "reason": "The user is asking for a calculation, so the CalculatorTool is appropriate.",
    "tool_names": ["CalculatorTool"]
}
"""

# Determine tools function to analyze chat logs
def determine_tools(messages: List[Message]) -> List[str]:
    message_block = "\n".join([f"{controller.config.companion_name if x.role == 'assistant' else controller.config.user_name}: {x.text}" for x in messages])

    messages = [(
        "system",
        controller.format_str(sys_prompt, extra={"json_schema": json.dumps(ToolResponse.model_json_schema()),
                                                 "tool_docs": tool_docs})
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

    _, calls = controller.completion_tool(LlmPreset.Default, messages, comp_settings=CommonCompSettings(max_tokens=1024, temperature=0.1), tools=[ToolResponse])

    state = {"tool_names": []}
    for call in calls:
        call.execute(state)

    return state["tool_names"]
