import json

from pydantic import BaseModel, Field

from pm.controller import controller
from pm.llm.base_llm import LlmPreset, CommonCompSettings

# Define system prompt for story dialogue evaluation
sys_prompt = """You are a literary analysis module responsible for evaluating the directness and authenticity of dialogue within a story.
Each dialogue exchange should be rated for **directness** on a scale from 0 to 1.

- **1**: The dialogue is natural, immersive, and flows seamlessly, avoiding unnecessary meta-commentary such as self-referential rewrites, corrections, or apologies.
- **0**: The dialogue includes unnatural acknowledgments of revision, self-correction, or forced justification, disrupting immersion with phrases like "Let me rephrase that," "Here's a revised version," or "I apologize for the previous statement."

Please keep your reasoning brief and focused! Overly long responses will be discarded!"""

# Define `DirectnessValidation` model
class DirectnessValidation(BaseModel):
    """Response format for evaluating if the message is direct and free from rewrite acknowledgments."""
    reasoning: str = Field(description="A short explanation of why the directness score was given.")
    directness: float = Field(description="0 if the response contains revision acknowledgments, 1 if direct and natural", ge=0, le=1)

    def execute(self, state):
        state["directness"] = self.directness
        state["reasoning"] = self.reasoning
        return state


# Example responses with expected assistant ratings
example1_response = """
User: What do you think about Kratom?
Assistant: My sincerest apologies for the previous response. Here is a rewritten version.

**First Paragraph:**
Kratom is an interesting topic that has been widely discussed...
"""
example1_assistant = """
{
    "reasoning": "Message explicitly acknowledges a rewrite, making it indirect.",
    "directness": 0.0
}
"""

example2_response = """
User: Can you rephrase your last response?
Assistant: Sure! Here's a refined version of my response:
Kratom is a fascinating subject...
"""
example2_assistant = """
{
    "reasoning": "Message acknowledges it is a revision rather than directly responding.",
    "directness": 0.0
}
"""

example3_response = """
User: What do you think about Kratom?
Assistant: Kratom is an interesting topic that has been widely discussed...
"""
example3_assistant = """
{
    "reasoning": "Message directly responds without unnecessary revision acknowledgments.",
    "directness": 1.0
}
"""

example4_response = """
User: Can you clarify your last point?
Assistant: Absolutely! To clarify, what I meant was...
"""
example4_assistant = """
{
    "reasoning": "Message directly addresses user without mentioning a rewrite.",
    "directness": 1.0
}
"""

# Function to validate response in context
def validate_directness(context: str, new_message: str) -> float:
    response = f"""
{context}
### BEGIN NEW MESSAGE
{new_message}
### END NEW MESSAGE
"""

    # Construct the conversation including the system prompt and examples
    messages = [
        ("system", controller.format_str(sys_prompt, extra={"basemodel_schema": json.dumps(DirectnessValidation.model_json_schema())})),
        ("user", example1_response),
        ("assistant", example1_assistant),
        ("user", example2_response),
        ("assistant", example2_assistant),
        ("user", example3_response),
        ("assistant", example3_assistant),
        ("user", example4_response),
        ("assistant", example4_assistant),
        ("user", response)
    ]

    # Call the LLM for response validation
    state = {}
    while True:
        _, calls = controller.completion_tool(LlmPreset.Conscious, messages, comp_settings=CommonCompSettings(max_tokens=1024, temperature=0.2), tools=[DirectnessValidation])
        if len(calls) > 0:
            for call in calls:
                call.execute(state)
            break

    return state["directness"]
