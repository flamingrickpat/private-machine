import json
from typing import List, Tuple, Type

from pydantic import BaseModel

from pm.controller import controller
from pm.tools.common import get_json_schemas, tools_list

prompt_conscious_assistant_base = """{character_card}
{example_dialog_insert}
{tool_insert}
{optional_tool_insert}
{thought_insert}
Some knowledge relevant to the current conversation:
{knowledge}
"""

tool_insert = """
You have the following tools available:
{lst_tool_insert}
Call them with valid JSON!
"""

optional_tool_insert = """
You have the following tools available:
{lst_optional_tool_insert}

If you want to use them, please just describe how you would use them.
The specialized JSON agent will convert your thought into a call and return the result. Do not write the JSON directly yourself!
"""

thought_insert = """
This is a complex query, you may use internal thought to think it over before an answer.
User 'Thought:' to start a thought. This will be invisible to the user. Use 'Response:' after that to start your response visible to the user.
Take a deep breath and think it through before making the final response to the user in the 'Response:' part of the message.
"""

example_dialog_insert = """
You talk like this, this is your writing style:
### BEGIN EXAMPLE UTTERANCES
{lst_example_dialog_insert}
### END EXAMPLE UTTERANCES
"""

def build_sys_prompt_conscious_assistant(use_thoughts: bool, tool_list: List[Type[BaseModel]], knowledge: str, optional_tools: List[Type[BaseModel]] = None) -> str:
    if optional_tools is None:
        optional_tools = []

    map = {
        "character_card": controller.config.character_card_assistant,
        "tool_insert": "",
        "optional_tool_insert": "",
        "thought_insert": "",
        "knowledge": knowledge,
        "example_dialog_insert": ""
    }

    if use_thoughts:
        map["thought_insert"] = thought_insert

    if len(tool_list) > 0:
        map["tool_insert"] = tool_insert
        map["lst_tool_insert"] = "\n".join([json.dumps(x.model_json_schema(), indent=2) for x in tool_list])

    if len(optional_tools) > 0:
        map["optional_tool_insert"] = optional_tool_insert
        map["lst_optional_tool_insert"] = "\n".join([f"- {x.__name__}: {x.__doc__}" for x in optional_tools])

    if len(controller.config.example_dialog):
        map["example_dialog_insert"] = example_dialog_insert
        map["lst_example_dialog_insert"] = "\n- ".join(controller.config.example_dialog)

    prompt = controller.format_str(prompt_conscious_assistant_base, extra=map)
    return prompt
