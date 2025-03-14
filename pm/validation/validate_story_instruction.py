from pydantic import BaseModel, Field

from pm.controller import controller
from pm.llm.base_llm import LlmPreset, CommonCompSettings

# Define the system prompt for story instruction validation
sys_prompt = """You are a validation module responsible for evaluating story instructions generated by the AI agent.
Each instruction should be rated for its narrative coherence, detail, and engagement on a scale from 0 to 1.
- 0: The instruction is vague, incoherent, lacks narrative depth, or is improperly directed.
- 1: The instruction is clear, detailed, and written in a compelling narrative style that provides actionable guidance for a story writing assistant.
Output valid JSON in this format:
{basemodel_schema}
Please keep your reasoning short and concise.
"""

# Define the StoryInstructionValidation model
class StoryInstructionValidation(BaseModel):
    """Response format for evaluating story instruction validity."""
    reasoning: str = Field(description="A short description of why the validity score was given.")
    validness: float = Field(description="0 for invalid instructions, 1 for highly coherent and engaging narrative instructions", ge=0, le=1)

    def execute(self, state):
        state["validness"] = self.validness
        return state

# Few-shot examples

example1_instruction = "Emmy should subtly suggest an intimate activity during her reunion with Rick, hinting at exciting possibilities that deepen their emotional connection while keeping her approach fresh."
example1_response = """
{
    "reasoning": "The instruction is clear, detailed, and engaging, providing a strong narrative prompt.",
    "validness": 1.0
}
"""

example2_instruction = "Emmy talk to Rick."
example2_response = """
{
    "reasoning": "The instruction is too brief and lacks narrative depth and context.",
    "validness": 0.2
}
"""

example3_instruction = "Emmy should express her feelings."
example3_response = """
{
    "reasoning": "The instruction is vague and lacks specific narrative elements and context.",
    "validness": 0.5
}
"""

example4_instruction = "You should write a story about a reunion."
example4_response = """
{
    "reasoning": "The instruction directly addresses the user rather than offering a narrative prompt from the character's perspective.",
    "validness": 0.1
}
"""

example5_instruction = "After a long separation, Emmy finds herself reflecting on the possibilities of their reunion. She decides to create an atmosphere of subtle sensuality, carefully hinting at intimate moments that await, which might reignite their deep connection."
example5_response = """
{
    "reasoning": "The instruction effectively conveys context and character motivation in an evocative narrative style.",
    "validness": 1.0
}
"""

def validate_story_instruction(instruction: str) -> float:
    messages = [
        ("system", sys_prompt),
        ("user", example1_instruction),
        ("assistant", example1_response),
        ("user", example2_instruction),
        ("assistant", example2_response),
        ("user", example3_instruction),
        ("assistant", example3_response),
        ("user", example4_instruction),
        ("assistant", example4_response),
        ("user", example5_instruction),
        ("assistant", example5_response),
        ("user", instruction)
    ]

    _, calls = controller.completion_tool(LlmPreset.CurrentOne, messages,
                                            comp_settings=CommonCompSettings(max_tokens=1024, temperature=0),
                                            tools=[StoryInstructionValidation])
    state = {"validness": 0}
    for call in calls:
        call.execute(state)

    return state["validness"]
