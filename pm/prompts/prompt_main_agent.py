import json
from typing import List, Tuple

from pydantic import BaseModel

from pm.controller import controller
from pm.tools.common import get_json_schemas, tools_list

prompt_conscious_assistant_base = """{character_card}
{tool_insert}
{thought_insert}
Some knowledge relevant to the current conversation:
{knowledge}
"""

tool_insert = """
You have the following tools available:
{insert_tools}

If you want to use them, please just describe how you would use them.
The specialized JSON agent will convert your thought into a call and return the result. Do not write the JSON directly yourself!
"""

thought_insert = """
This is a complex query, you may use internal thought to think it over before an answer.
User 'Thought:' to start a thought. This will be invisible to the user. Use 'Response:' after that to start your response visible to the user.
Take a deep breath and think it through before making the final response to the user in the 'Response:' part of the message.
"""

def build_sys_prompt_conscious_assistant(use_thoughts: bool, tool_list: List[BaseModel], knowledge: str) -> str:
    map = {
        "character_card": controller.config.character_card_assistant,
        "tool_insert": "",
        "thought_insert": "",
        "knowledge": knowledge
    }

    if use_thoughts:
        map["thought_insert"] = thought_insert

    if len(tool_list) > 0:
        map["tool_insert"] = "\n".join([json.dumps(x.model_json_schema(), indent=2) for x in tool_list])

    prompt = controller.format_str(prompt_conscious_assistant_base, extra=map)
    return prompt
