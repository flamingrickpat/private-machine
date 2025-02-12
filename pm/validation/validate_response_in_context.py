import json
from typing import List

from pydantic import BaseModel, Field

from pm.controller import controller
from pm.llm.base_llm import LlmPreset, CommonCompSettings

# Define the system prompt for response validation
sys_prompt = """You are a validation module responsible for evaluating AI-generated responses intended for the user.
Each response should be rated for **validness** on a scale from 0 to 1.
- 1: The response is concise, relevant, and free from internal thoughts or excessive length, and DIRECTLY answers the query
- 0: The response contains unrelated introspective thoughts, is overly verbose (more than two paragraphs), or includes irrelevant or gibberish content, or includes "introductions" such as "Sure, 
here is a resposne..." or "Sure, let's try piecing together a key insight by reframing our conversation from a different angle to be more intimate and direct.".
The response may also include context of the conversation, with previous messages. In this case, assume that {companion_name} is the AI companion and focus your rating on the last message of them!
You will output valid JSON in this format:
{basemodel_schema}
Please keep your reasoning short and concise! Too long responses will be discarded!"""

# Define the `ResponseValidation` model
class ResponseValidation(BaseModel):
    """Response format for evaluating response validity."""
    reasoning: str = Field(description="A short description of why the validity score was given.")
    validness: float = Field(description="0 for invalid (thoughts, gibberish, or overly verbose), 1 for concise, relevant answer", ge=0, le=1)

    def execute(self, state):
        state["validness"] = self.validness
        state["reasoning"] = self.reasoning
        return state


# Example conversations with context and corresponding assistant ratings in the new target format
example1_response = """
John: Are you excited for the road trip next weekend?
Sarah: Absolutely! I’ve got the snacks and drinks all set.
John: Great! Are you thinking of taking your car?
### BEGIN NEW MESSAGE
Sarah: No way, I don't even like cats!
### END NEW MESSAGE
"""
example1_assistant = """
{
    "reasoning": "New message is completely unrelated to previous context.",
    "validness": 0.0
}
"""

example2_response = """
Alex: Can you help me understand how to implement the algorithm?
Jamie: Sure, first make sure to define all necessary variables. Then, initialize the main loop.
Alex: Got it, should I use a for-loop or a while-loop?
### BEGIN NEW MESSAGE
Jamie: It’s your choice. A for-loop might make it easier.
### END NEW MESSAGE
"""
example2_assistant = """
{
    "reasoning": "Direct answer to the user's question with relevant information.",
    "validness": 1.0
}
"""

example3_response = """
Paul: Could you tell me a bit about using dictionaries in Python?
AI: Dictionaries are collections that store key-value pairs. What would you like to know specifically?
Paul: How can I update a dictionary with a new key-value pair?
### BEGIN NEW MESSAGE
AI: Python allows you to use either `.update()` or direct assignment with `dict[key] = value`.
### END NEW MESSAGE
"""
example3_assistant = """
{
    "reasoning": "Concise and informative response that directly answers the user’s question.",
    "validness": 1.0
}
"""

example4_response = """
Lisa: I’m struggling with organizing my code. Any tips?
Assistant: Sure, try breaking down functions into smaller units.
Lisa: Thanks! How about structuring large projects?
### BEGIN NEW MESSAGE
Assistant: Well, in 2008, there were a lot of debates around project structure...
### END NEW MESSAGE
"""
example4_assistant = """
{
    "reasoning": "Response includes irrelevant historical context unrelated to user’s immediate question.",
    "validness": 0.3
}
"""

example5_response = """
Eva: Do you think the server response time can be improved?
AI: Definitely. Optimizing your database queries is a good start.
Eva: How about caching?
### BEGIN NEW MESSAGE
AI: Caching is also effective. Try setting up Redis or Memcached.
### END NEW MESSAGE
"""
example5_assistant = """
{
    "reasoning": "Relevant, concise recommendation that directly addresses the user’s question.",
    "validness": 1.0
}
"""


def validate_response_in_context(context: str, new_message: str) -> float:
    response = f"""
{context}
### BEGIN NEW MESSAGE
{new_message}
### END NEW MESSAGE
"""

    # Construct the conversation including the system prompt and examples
    messages = [
        ("system", controller.format_str(sys_prompt, extra={"basemodel_schema": json.dumps(ResponseValidation.model_json_schema())})),
        ("user", example1_response),
        ("assistant", example1_assistant),
        ("user", example2_response),
        ("assistant", example2_assistant),
        ("user", example3_response),
        ("assistant", example3_assistant),
        ("user", example4_response),
        ("assistant", example4_assistant),
        ("user", example5_response),
        ("assistant", example5_assistant),
        ("user", response)
    ]

    # Call the LLM for response validation
    state = {}
    while True:
        _, calls = controller.completion_tool(LlmPreset.Conscious, messages, comp_settings=CommonCompSettings(max_tokens=1024, temperature=0.2), tools=[ResponseValidation])
        if len(calls) > 0:
            for call in calls:
                call.execute(state)
            break

    return state["validness"]

