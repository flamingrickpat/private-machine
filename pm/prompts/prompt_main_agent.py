from typing import List, Tuple

from pm.controller import controller
from pm.tools.common import get_json_schemas

prompt_conscious_assistant_base = """{character_card}
{tool_insert}
{thought_insert}
"""

tool_insert = """
{insert_tools}
"""

thought_insert = """
This is a complex query, you may use internal thought to think it over before an answer.
User 'Thought:' to start a thought. This will be invisible to the user. Use 'Response:' after that to start your response visible to the user.
Take a deep breath and think it through before making the final response to the user in the 'Response:' part of the message.
"""

def build_sys_prompt_conscious_assistant(use_thoughts: bool, tool_list: List[str]) -> str:
    map = {
        "character_card": controller.config.character_card_assistant,
        "tool_insert": "",
        "thought_insert": ""
    }

    if use_thoughts:
        map["thought_insert"] = thought_insert

    if len(tool_list) > 0:
        map["tool_insert"] = get_json_schemas(tool_list)

    prompt = controller.format_str(prompt_conscious_assistant_base, extra=map)
    return prompt
