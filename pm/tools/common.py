import json

from pydantic import BaseModel

from pm.tools.mental_health_tools import SetReminder, AddGoal, ProgressGoal, FinishGoal, ListGoals, SeeGoalDetails, ExtendPersonality, SearchWeb

tools_list = [SetReminder, AddGoal, ProgressGoal, FinishGoal, ListGoals, SeeGoalDetails, ExtendPersonality, SearchWeb]
tool_dict = {}
for tool in tools_list:
    tool_dict[tool.__name__] = tool

# Generate tool documentation dynamically for the system prompt
tool_docs = "\n".join([f"{tool.__name__}: {tool.__doc__}" for tool in tools_list])

def get_json_schemas(tool_list):
    res = {}
    for tool in tools_list:
        if isinstance(tool, str):
            res[tool] = json.dumps(tool_dict[tool].model_json_schema(), indent=2)
        elif isinstance(tool, BaseModel):
            res[tool.__name__] = json.dumps(tool.model_json_schema(), indent=2)
    return res