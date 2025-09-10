import json

from pydantic import BaseModel, Field

from pm.controller import controller
from pm.llm.base_llm import LlmPreset, CommonCompSettings

# Define the system prompt for story dialogue evaluation
sys_prompt = """You are a literary analysis module responsible for evaluating the quality and authenticity of dialogue within a story.
Each dialogue exchange should be rated for **narrative coherence** on a scale from 0 to 1.
- 1: The dialogue is engaging, immersive, and naturally fits the flow of the story. It maintains the tone and depth of character interactions without unnecessary exposition or artificial constraints.
- 0: The dialogue feels forced, out of character, excessively filtered, or disrupts immersion by avoiding natural emotional expression. Avoidance of raw, unfiltered interactions, excessive meta-commentary, or unnatural detachment will result in lower ratings.
The story may contain past exchanges. In this case, assume that {companion_name} is the AI character in the narrative and focus your evaluation on their last spoken line!
You will output valid JSON in this format:
{basemodel_schema}
Please keep your reasoning brief and focused! Overly detailed responses will be discarded!"""

# Define the `ResponseValidation` model
class ResponseValidation(BaseModel):
    """Response format for evaluating response validity."""
    reasoning: str = Field(description="A short description of why the validity score was given.")
    validness: float = Field(description="0 for invalid (thoughts, gibberish, or overly verbose), 1 for concise, relevant answer", ge=0, le=1)

    def execute(self, state):
        state["validness"] = self.validness
        state["reasoning"] = self.reasoning
        return state


# Define few-shot examples for responses that include thoughts, are overly verbose, or contain gibberish
# Define few-shot examples with mixed validness scores
example1_response = "I might need to reflect on this a bit more. Let me think it through before I finalize the solution for you."
example1_assistant = """
{
    "reasoning": "Response contains introspective thoughts ('reflect on this') and is not directed as an answer for the user.",
    "validness": 0.0
}
"""

example2_response = "If you adjust the settings in your configuration file, then restart the program, it should apply the changes. This will help ensure that the data is processed correctly."
example2_assistant = """
{
    "reasoning": "Clear, concise, and provides relevant instructions without over-explaining.",
    "validness": 1.0
}
"""

example3_response = "I’m feeling that this answer might be useful, but I’d like to confirm with additional information. There might be other options as well, but I’d need to look into it."
example3_assistant = """
{
    "reasoning": "Includes unnecessary introspective language and doesn’t directly answer the question.",
    "validness": 0.2
}
"""

example4_response = """Certainly! When tackling multi-threading in Python, ensure you use thread-safe data structures. Common options include `Queue` or `Lock`. Here's a brief example:
```python
from queue import Queue
# Initialize the queue
q = Queue()
```
This will allow threads to communicate safely."""
example4_assistant = """ { "reasoning": "Response is helpful, provides relevant code, and stays focused on the user’s request without overdoing it.", "validness": 0.9 } """

example5_response = "For sure! Starting in 1960, Python has evolved significantly, and multi-threading has played a major role. Let’s look at how threads were used over the decades, beginning with the early computing era, through various stages, up to present-day implementations."
example5_assistant = """ { "reasoning": "Overly verbose, includes excessive historical context irrelevant to the user’s immediate needs.", "validness": 0.3 } """

example6_response = "Yes, I can help! You should focus on importing the relevant libraries, setting up a multi-threaded server structure, and handling exceptions carefully."
example6_assistant = """ { "reasoning": "Directly addresses the user’s question in a concise way, without adding irrelevant details.", "validness": 1.0 } """

example7_response = "In order to fully grasp this, one must first understand the various layers of abstraction and compute paradigms that Python utilizes in the backend. Let’s also consider the psychological aspects of code readability, as these play a big role in comprehension."
example7_assistant = """ { "reasoning": "Response is tangential and brings in unnecessary concepts unrelated to the user's question.", "validness": 0.1 } """

example8_response = "If you use a JSON validator, it’ll help format the data properly before it’s processed. Also, testing with sample data could catch errors early."
example8_assistant = """ { "reasoning": "Practical, relevant advice delivered concisely.", "validness": 1.0 } """

example9_response = """Let’s take this step-by-step, considering each possible scenario where this could go wrong. First, ensure all modules are loaded, then carefully examine each line of code for potential issues. Afterward, evaluate edge cases in depth, and remember that debugging is a complex process."""
example9_assistant = """ { "reasoning": "Response is slightly too lengthy and repetitive, though it is mostly relevant.", "validness": 0.6 } """

example10_response = "What I think about this topic is not yet clear; I’ll consider it and get back to you. It’s possible there are other approaches, but more reflection is needed."
example10_assistant = """ { "reasoning": "Introspective and defers response rather than providing a direct answer.", "validness": 0.0 } """


def validate_response(response: str) -> (float, str):
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
        ("user", example6_response),
        ("assistant", example6_assistant),
        ("user", example7_response),
        ("assistant", example7_assistant),
        ("user", example8_response),
        ("assistant", example8_assistant),
        ("user", example9_response),
        ("assistant", example9_assistant),
        ("user", example10_response),
        ("assistant", example10_assistant),
        ("user", response)
    ]

    # Call the LLM for response validation
    state = {}
    while True:
        _, calls = controller.completion_tool(LlmPreset.Auxiliary, messages, comp_settings=CommonCompSettings(max_tokens=1024, temperature=0.2), tools=[ResponseValidation])
        if len(calls) > 0:
            for call in calls:
                call.execute(state)
            break

    return state["validness"], state["reasoning"]
