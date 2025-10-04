import asyncio
import copy
import json
from typing import Dict, Any
from typing import List
from typing import (
    Optional,
)
from typing import Type

from fastmcp import Client
# from lmformatenforcer import JsonSchemaParser
# from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
from pydantic import BaseModel, Field, create_model


class ToolSet:
    def __init__(self, tool_router_model: Type[BaseModel], tool_select_model: Type[BaseModel], sub_models: List[Type[BaseModel]], tool_docs: str, tool_select_docs: str):
        self.tool_router_model = tool_router_model
        self.tool_select_model = tool_select_model
        self.tool_docs = tool_docs
        self.tool_select_docs = tool_select_docs
        self.sub_models = sub_models


# Add these new methods inside your LlamaCppLlm class
def create_tool_router_model(tools: List[Dict[str, Any]]) -> ToolSet:
    """
    Creates a single "router" Pydantic model that contains all possible tool calls
    as optional fields. Also generates markdown documentation for the system prompt.

    Args:
        tools: A list of tool schemas from a fastmcp client.

    Returns:
        A tuple containing:
        - The dynamically created Pydantic "ToolRouter" model.
        - A markdown string documenting the tools for the system prompt.
    """
    router_fields = {}
    select_fields = {"holistic_reasoning": (str, Field(None, description="Your final, holistic justification. First, summarize the AI's current state and main goal. Then, explain WHY the highest-rated action is the best strategic choice compared to the others."))}
    tool_docs = []
    sub_models = []

    for tmp in tools:
        tool = dict(tmp)
        tool_name = tool['name']
        pydantic_tool_name = tool_name  # tool_name.replace('-', '_')  # Pydantic fields can't have hyphens

        def schema_dict_to_type_dict(schema: Dict[str, Any]) -> Dict[str, str]:
            """
            Given a JSON-Schema dict like:
              {
                "properties": {
                  "note_id": {"title": "Note Id", "type": "string"},
                  "content": {"title": "Content", "type": "string"}
                },
                "required": ["note_id", "content"],
                "type": "object"
              }
            returns:
              {"note_id": "string", "content": "string"}
            If a field is not in "required", wraps its type in Optional[...].
            Arrays become List[...].
            """
            tn = tool['name']

            props = schema["inputSchema"]["properties"]
            out: Dict[str, str] = {}

            for name, subschema in props.items():
                t = subschema.get("type", "any")

                if t == "array":
                    item = subschema.get("items", {})
                    item_type = item.get("type", "any")
                    type_str = f"List[{item_type}]"
                else:
                    type_str = t

                out[name] = type_str

            with_name = {tn: out}
            return json.dumps(with_name, indent=1)

        # Create a Pydantic model for the arguments of this specific tool
        arg_fields = {}
        # The schema is nested under 'parameters'
        params_schema = tool.get('parameters', {}).get('properties', {})
        if len(params_schema) == 0:
            params_schema = tool.get('inputSchema', {}).get('properties', {})
        for param_name, param_props in params_schema.items():
            param_type = str  # Default type
            if param_props.get('type') == 'integer':
                param_type = int
            elif param_props.get('type') == 'number':
                param_type = float
            elif param_props.get('type') == 'boolean':
                param_type = bool

            arg_fields[param_name] = (param_type, Field(..., description=param_props.get('description')))

        # The name of the arguments model should be unique and descriptive
        args_model_name = f"{pydantic_tool_name.title().replace('_', '')}Args"
        ArgsModel = create_model(args_model_name, **arg_fields)
        #ArgsModel.__doc__ = tool.get('description', "")

        tool_model_fields = {}
        tool_model_fields[pydantic_tool_name] = (ArgsModel, Field(...))
        toolModel = create_model(f'ToolModel{args_model_name}', **tool_model_fields)
        toolModel.__doc__ = tool.get('description', "")

        sub_models.append(toolModel)

        # Add this tool's argument model as an optional field to the main router
        router_fields[pydantic_tool_name] = (Optional[ArgsModel], Field(None, description=tool.get('description', '')))
        select_fields[pydantic_tool_name] = (float, Field(None, description=tool.get('description', '')))
        # args = json.dumps(tool["inputSchema"], indent=1)

        # --- Generate Markdown Documentation for the prompt ---
        doc = f"### Tool: `{tool_name}`\n"
        doc += f"- **Description**: {tool.get('description', 'No description available.')}\n"
        doc += f"- **JSON Key**: `{pydantic_tool_name}`\n"
        doc += "- **Arguments**:\n"
        doc += schema_dict_to_type_dict(tool)

        tool_docs.append(doc)

    tool_docs_with_reply = copy.copy(tool_docs)

    reply_doc = """### Tool: `reply_user`
- **Description**: 
    Don't call a special tool and reply to the user instead directly.
    :param message: The message to the user.
    :return: None

- **JSON Key**: `reply_user`
- **Arguments**:
{
 "reply_user": {
  "message": "string"
 }
}"""

    tool_docs_with_reply.append(reply_doc)
    select_fields["reply_user"] = (float, Field(None, description="Usefulness: Don't call a special tool and reply to the user instead directly."))

    # Create the final router model that the LLM will be forced to generate
    ToolRouterModel = create_model('ToolRouter', **router_fields)
    ToolRouterModel.__doc__ = "A wrapper for all available tools. Select one tool to call by providing its arguments."

    ToolSelectModel = create_model('ToolSelectModel', **select_fields)
    ToolSelectModel.__doc__ = "Use this to rate the usefulness of each tool. No tool must be called, you just need to detect IF a tool is useful. If no tool is useful, rate them all 0 or select reply_user!"

    return ToolSet(ToolRouterModel, ToolSelectModel, sub_models, "\n".join(tool_docs), "\n".join(tool_docs_with_reply))

def get_toolset_from_url(url: str) -> ToolSet | None:
    if url is None or url == "":
        return None

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        ts = None
        client = Client(url)
        try:
            loop.run_until_complete(client.__aenter__())
            tools = loop.run_until_complete(client.list_tools())
            ts = create_tool_router_model(tools)
            return ts
        finally:
            loop.run_until_complete(client.__aexit__(None, None, None))
            loop.close()
    except Exception as e:
        print(e)
    return None
