from pydantic import BaseModel, Field

from pm.controller import controller
from pm.llm.base_llm import LlmPreset, CommonCompSettings
from pydantic import BaseModel, Field

from pm.controller import controller
from pm.llm.base_llm import LlmPreset, CommonCompSettings


class Rating(BaseModel):
    analysis: str = Field(description="initial analysis where you consider the instruction and output. make it short, 3 sentences at most!")
    criticism: str = Field(description="what could be improved? make it short, 3 sentences at most!")
    final_rating: float = Field(description="rating with 0 for very bad, 0.5 for ok-ish and 1 for very good, fulfilling the instructions perfectly")

def rate_agent_output(instruction, user_input, agent_output) -> float:
    sysprompt = "You are a helpful assistant. You rate how well an agent fulfilled the system prompt instructions."
    user_input_block = f"\n### BEGIN USER INPUT\n{user_input}\n### END USER INPUT"

    prompt = [
        ("system", sysprompt),
        ("user", f"### BEGIN AGENT INSTRUCTION\n{instruction}\n### END AGENT INSTRUCTION{user_input_block}\n### BEGIN AGENT OUTPUT\n{agent_output}\n### END AGENT "
                 f"OUTPUT\n"
                 f"With knowing the input from the user, how would you rate the agents output? Did it fulfill its instructions well?"),
    ]

    _, calls = controller.completion_tool(LlmPreset.Auxiliary, prompt, comp_settings=CommonCompSettings(temperature=0.4, max_tokens=256), tools=[Rating])
    r: Rating = calls[0]
    return r.final_rating