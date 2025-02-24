import json
from typing import Dict, Any

from pydantic import BaseModel


class ToolBase(BaseModel):
    def execute(self, state: Dict[str, Any]):
        state["output"] = "Dummy Call"

# Fake home automation tools
class LightControlTool(ToolBase):
    """Controls lights in a specific room."""
    room: str
    action: str  # "on" or "off"

    def execute(self, state: Dict[str, Any]):
        s =  f"Lights in '{self.room}' set to status '{self.action}'"
        print(s)
        state["output"] = s


class DummyTools(ToolBase):
    """Use this tool when you have entered tool calling mode as a mistake and don't want to call any tools."""
    dummy_text: str

    def execute(self, state: Dict[str, Any]):
        state["output"] = f"Exiting tool calling mode..."


class ThermostatTool(ToolBase):
    """Adjusts the thermostat temperature."""
    target_temperature: float

    def execute(self, state: Dict[str, Any]):
        print(f"Setting thermostat to {self.target_temperature}°C.")
        state["output"] = f"Thermostat set to {self.target_temperature}°C"


class EnergyConsumptionTool(ToolBase):
    """Fetches the current energy consumption."""

    def execute(self, state: Dict[str, Any]):
        print("Fetching current energy consumption data.")
        state["output"] = "Energy consumption in normal range."


# List of tool instances available for the model
tools_list = [LightControlTool, ThermostatTool, EnergyConsumptionTool, DummyTools]
tool_dict = {}
for tool in tools_list:
    tool_dict[tool.__name__] = tool

tools_list_str = [x.__name__ for x in tools_list]

# Generate tool documentation dynamically for the system prompt
tool_docs = "\n".join([f"{tool.__name__}: {tool.__doc__}" for tool in tools_list])

def get_json_schemas(tool_list):
    return {tool_name: json.dumps(tool_dict[tool_name].model_json_schema(), indent=2) for tool_name in tool_list}