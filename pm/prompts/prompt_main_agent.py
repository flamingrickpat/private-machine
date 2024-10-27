from typing import List, Tuple

from pm.controller import controller
from pm.tools.common import get_json_schemas

prompt_conscious_assistant_base = """{character_card}
{tool_insert}
{thought_insert}
"""

tool_insert = """
You may answer in natural language or call a a tool. The tool calls are in valid JSON, and valid JSON only! 
If you do a JSON tool call, the next user response will be the result of that tool call.
Here are the possible schemas:

### BEGIN TOOL SCHEMAS
{tool_schemas}
### END TOOL SCHEMAS
"""

thought_insert = """
This is a complex query, you may use internal thought to think it over before an answer.
User ** (double asterisk) to start and end a thought. This will be invisible to the user. 
Take a deep breath and think it through before making the final response to the user in the rest of the message.
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

    prompt = prompt_conscious_assistant_base.format_map(map)
    return prompt
