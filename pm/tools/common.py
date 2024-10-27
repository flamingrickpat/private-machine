import enum
import json
import logging
from typing import List, Dict, Any
from pydantic import BaseModel, Field

# Example dummy tool functions
class WeatherTool(BaseModel):
    """Fetches weather data for a specified location."""
    location: str

    def execute(self, state: Dict[str, Any]):
        logging.info(f"Fetching weather for {self.location}")
        return {"status": "success", "data": f"Weather data for {self.location}"}


class TranslateTool(BaseModel):
    """Translates text from one language to another."""
    text: str
    target_language: str

    def execute(self, state: Dict[str, Any]):
        logging.info(f"Translating '{self.text}' to {self.target_language}")
        return {"status": "success", "data": f"Translated text in {self.target_language}"}


class CalculatorTool(BaseModel):
    """Performs a basic arithmetic operation."""
    operation: str
    a: float
    b: float

    def execute(self, state: Dict[str, Any]):
        logging.info(f"Performing {self.operation} with {self.a} and {self.b}")
        return {"status": "success", "data": f"Result of {self.operation} operation"}

# List of tool instances available for the model
tools_list = [WeatherTool, TranslateTool, CalculatorTool]
tool_dict = {}
for tool in tools_list:
    tool_dict[tool.__name__] = tool

# Generate tool documentation dynamically for the system prompt
tool_docs = "\n".join([f"{tool.__name__}: {tool.__doc__}" for tool in tools_list])

def get_json_schemas(tool_list):
    return {tool_name: json.dumps(tool_dict[tool_name].model_json_schema(), indent=2) for tool_name in tool_list}