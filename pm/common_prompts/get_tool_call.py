import enum
import json
import logging
from typing import List, Dict, Any, Type
from pydantic import BaseModel, Field

from pm.character import companion_name, user_name
from pm.controller import controller
from pm.futils.enum_utils import make_enum_from_list
from pm.llm.base_llm import LlmPreset, CommonCompSettings
from pm.tools.common import tools_list, tool_docs, tool_dict, ToolBase

sys_prompt = f"""You are a helpful assistant that converts natural language into valid tool calls."""

# Determine tools function to analyze chat logs
def get_tool_call(message_block: str, tt: Type[ToolBase]) -> ToolBase:
    messages = [(
        "system", sys_prompt
    ), (
        "user",
        message_block
    )]

    _, calls = controller.completion_tool(LlmPreset.Default, messages, comp_settings=CommonCompSettings(max_tokens=1024, temperature=0.1), tools=[tt])
    return calls[0]
