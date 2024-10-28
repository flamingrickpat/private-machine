from typing import List, Type

from pydantic import BaseModel


class PydanticToolsParserLocal:
    def __init__(self, tools: List[Type[BaseModel]]):
        self.tools = tools

    def invoke(self, full_output):
        content = full_output.content

        res = []
        for tool in self.tools:
            try:
                obj = tool.model_validate_json(content)
                res.append(obj)
                return res
            except:
                pass
        return res

