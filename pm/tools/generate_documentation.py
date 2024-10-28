import json
from typing import List, Type, Dict, Callable

from pydantic import BaseModel

from pm.llm.gbnf_grammar import generate_func_description


def create_documentation_json(tools: List[Type[BaseModel]]) -> Dict[str, str]:
    res = {"insert_tools": ""}
    if len(tools) > 0:
        tool_schemas = "\n".join([json.dumps(t.model_json_schema()) for t in tools])
        full_block = (f"Y**Utilize Tools**: Use the following tools when appropriate to achieve your objective:"
                      f"\n{tool_schemas}\n"
                      f"If you call them, use valid JSON and valid JSON only in your whole response!")
        res = {"insert_tools": full_block}
    return res


def create_header_functions(tools: List[Callable]) -> Dict[str, str]:
    return {"insert_function_header": ", ".join([x.__name__ for x in tools])}


def create_documentation_functions(tools: List[Callable]) -> Dict[str, str]:
    res = {"insert_functions": ""}
    if len(tools) > 0:
        func_schemas = "\n".join([generate_func_description([t]) for t in tools])
        full_block = (f"Y**Utilize Functions**: Use the following python functions when appropriate to achieve your objective:"
                      f"\n{func_schemas}\n"
                      f"If you call them, pay close attention to the syntax! The response will be shown to you instantly.")
        res = {"insert_tools": full_block}
    return res
