from typing import Callable, Dict, Type
from pydantic import BaseModel
from qtpynodeeditor import NodeDataModel

global_func_classes: Dict[str, Type[NodeDataModel]] = {}
global_pure_func_classes: Dict[str, Type[NodeDataModel]] = {}
global_pydantic_classes: Dict[str, Type[NodeDataModel]] = {}
global_enum_classes: Dict[str, Type[NodeDataModel]] = {}
global_python_tools: Dict[str, Callable] = {}
global_json_tools: Dict[str, Type[BaseModel]] = {}
global_json_tools_to_diff_schema: Dict[Type[BaseModel], Type[BaseModel]] = {}
global_data_models: Dict[str, Type[NodeDataModel]] = {}
