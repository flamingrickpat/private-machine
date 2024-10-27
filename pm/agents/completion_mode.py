import enum
import json
from typing import List

from duckdb.duckdb import description
from pydantic import BaseModel, Field

from pm.controller import controller
from pm.database.db_model import Message
from pm.llm.llm import chat_complete
from pm.llm.tools_parser_local import PydanticToolsParserLocal

sys_prompt = """You are a helpful assistant that determines if the AI should use assistant or story mode for the next model.
Analyze the following chat log between two people, {user_name} and the AI {companion_name}. 
Your task is to determine the completion mode of the query for {companion_name} as string, with "story" for story mode and "assistant" for assistant mode.
Your answer will determine if {companion_name} makes the next response using the LLM model trained for assistant style queries with possible tool calls,
or the roleplay LLM model trained for realistic human-on-human style conversation.
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
    "reason": "they just talk about cute cats like regular people",
    "mode": "story"
}
"""

example2_user = """
{companion_name}: What would you like to talk about today?
{user_name}: I could use some help planning my day. I have a meeting in the morning, but I'd also like to find time for some exercise and grocery shopping. Any tips on how to fit it all in?
"""

example2_assistant = """
{
    "reason": "the user is asking for practical advice on daily planning, indicating they want a structured response with guidance",
    "mode": "assistant"
}
"""

example3_user = """
{companion_name}: How can I make your day a little brighter?
{user_name}: Imagine a story where I’m a pirate captain on the high seas, and I just discovered a mysterious treasure map. What happens next?
"""

example3_assistant = """
{
    "reason": "the user is requesting a creative, imaginative story involving a pirate adventure, making it suitable for story mode",
    "mode": "story"
}
"""

class CompletionModeResponseMode(str, enum.Enum):
    Assistant = "assistant"
    Story = "story"


class CompletionModeResponse(BaseModel):
    """Response format for determining completion mode."""
    reason: str = Field(description="One short sentence about your reasoning.")
    mode: CompletionModeResponseMode = Field(description="'story' for story mode, 'assistant' for assistant mode")

    def execute(self, state):
        state["mode"] = self.mode
        return state

def determine_completion_mode(messages: List[Message]) -> CompletionModeResponseMode:
    message_block = "\n".join([f"{controller.config.companion_name if x.role == 'assistant' else controller.config.user_name}: {x.text}" for x in messages])

    messages = [(
        "system",
        controller.format_str(sys_prompt, extra={"basemodel_schema": json.dumps(CompletionModeResponse.model_json_schema())})
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

    llm = controller.llm
    llm_with_tools = llm.get_langchain_model().bind_tools([CompletionModeResponse])

    full = llm_with_tools.invoke(messages)
    calls = PydanticToolsParserLocal(tools=[CompletionModeResponse]).invoke(full)

    state = {"mode": CompletionModeResponseMode.Assistant}
    for call in calls:
        call.execute(state)

    return state["mode"]