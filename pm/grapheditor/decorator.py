from typing import Any, Callable, Dict, Type
from pydantic import BaseModel
import logging

try:
    from pm.grapheditor.type_registry import *
    from pm.grapheditor.type_pydantic import PydanticNodeDataModel
    from pm.grapheditor.func_any import FunctionNodeDataModel
    from pm.grapheditor.type_enum import PydanticEnumNodeDataModel
except:
    pass

def graph_function(pure: bool = False):
    def decorator(func: Callable):
        class_name = func.__name__

        # Dynamically create the class
        new_class = type(class_name, (FunctionNodeDataModel,), {})
        new_class.configure(func, pure)

        new_class_no_ui = type(class_name + "NoUi", (FunctionNodeDataModel.NoGuiVersion,), {})
        new_class.NoGuiVersion = new_class_no_ui
        new_class.NoGuiVersion.configure(func, pure)
        if pure:
            global_pure_func_classes[class_name] = new_class
        else:
            global_func_classes[class_name] = new_class

        return func

    return decorator

def llm_python_tool():
    def decorator(func: Callable):
        func_name = func.__name__
        global_python_tools[func_name] = func
        return func

    return decorator


def graph_function_pure():
    return graph_function(pure=True)


def graph_datamodel():
    def decorator(cls: Type[BaseModel]):
        class_name = cls.__name__

        new_class = type(class_name, (PydanticNodeDataModel,), {})

        # Configure the new class with the function and purity
        new_class.configure(cls)

        new_class_no_ui = type(class_name + "NoUi", (PydanticNodeDataModel.NoGuiVersion,), {})
        new_class.NoGuiVersion = new_class_no_ui
        new_class.NoGuiVersion.configure(cls)

        # Store the new class in the global dictionary
        global_pydantic_classes[class_name] = new_class

        return cls

    return decorator

def llm_json_tool():
    def decorator(cls: Type[BaseModel]):
        class_name = cls.__name__
        global_json_tools[class_name] = cls
        return cls

    return decorator

def graph_enum():
    def decorator(cls: Type[BaseModel]):
        class_name = cls.__name__

        new_class = type(class_name, (PydanticEnumNodeDataModel,), {})

        # Configure the new class with the function and purity
        new_class.configure(cls)

        new_class_no_ui = type(class_name + "NoUi", (PydanticEnumNodeDataModel.NoGuiVersion,), {})
        new_class.NoGuiVersion = new_class_no_ui
        new_class.NoGuiVersion.configure(cls)

        # Store the new class in the global dictionary
        global_enum_classes[class_name] = new_class

        return cls

    return decorator

