from pydantic import BaseModel, Field

from pm.character import user_name, companion_name
from pm.controller import controller
from pm.llm.base_llm import LlmPreset


class EmotionalImpact(BaseModel):
    reasoning: str = Field(description="You reason for your rating.")
    emotional_impact: float = Field(description="Ranges from 0 (low emotional impact) to 1 (high emotional impact). Like System 1 and System 2 thinking, but for the purpose of selecting a subsystem "
                                                "in an AI appliaction. If the impact is below 0.5, the regular mode with a duration of 10s will be used. If its above 0.5, the exhaustive mode with a "
                                                "reply time of 10 minutes will be used. Be careful!")


def rate_emotional_impact(context_data, last_input):
    sysprompt = f"""You are an agent in a cognitive architecture that determines the emotional impact of a user input on an AI companion. In our case {user_name} is the user and {companion_name} the AI companion."""

    userprompt = (f"### BEGIN CONTEXT\n{context_data}\n### END CONTEXT\n"
                  f"### BEGIN LAST INPUT \n{last_input}\n### END LAST INPUT\n"
                  f"Based on the last message, how is the emotional impact on {companion_name} from 0 to 1?")

    messages = [
        ("system", sysprompt),
        ("user", userprompt)
    ]

    _, calls = controller.completion_tool(LlmPreset.Fast, messages, tools=[EmotionalImpact])
    return calls[0].emotional_impact
