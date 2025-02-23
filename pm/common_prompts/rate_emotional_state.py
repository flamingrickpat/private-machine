from pydantic import BaseModel, Field

from pm.character import user_name, companion_name
from pm.controller import controller
from pm.ghost.mental_state import EmotionalAxesModel
from pm.llm.base_llm import LlmPreset


def rate_emotional_state(context_data, last_input) -> EmotionalAxesModel:
    sysprompt = f"""You are an agent in a cognitive architecture that determines the emotional impact of a user input on an AI companion. In our case {user_name} is the user and {companion_name} the AI companion."""

    userprompt = (f"### BEGIN CONTEXT\n{context_data}\n### END CONTEXT\n"
                  f"### BEGIN LAST INPUT \n{last_input}\n### END LAST INPUT\n"
                  f"Based on the last message, how is the emotional impact on {companion_name} from 0 to 1?")

    messages = [
        ("system", sysprompt),
        ("user", userprompt)
    ]

    _, calls = controller.completion_tool(LlmPreset.Auxiliary, messages, tools=[EmotionalAxesModel])
    return calls[0]
