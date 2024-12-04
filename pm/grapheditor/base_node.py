from typing import Dict, Tuple, List, Any
from functools import wraps
from types import FunctionType, MethodType

from pm.grapheditor.type_registry import global_data_models

class DataUpdater:
    def __init__(self, node):
        self.node = node

    def emit(self, port):
        if port not in self.node.outgoing_connections.keys():
            return

        data = self.node.out_data(port)
        for con in self.node.outgoing_connections[port]:
            node = con[0]
            in_port = con[1]
            p = Port(in_port)
            node.set_in_data(data, p)


class Port:
    def __init__(self, index):
        self.index = index


class BaseNode:
    def __init__(self):
        self.id = None  # Will be set when creating nodes
        self.outgoing_connections: Dict[int, List[Tuple['BaseNode', int]]] = {}
        self.incoming_connections: Dict[int, Tuple['BaseNode', int]] = {}
        self.data_updated = DataUpdater(self)

    def set_in_data(self, data: Any, port: int):
        raise NotImplementedError("Subclasses must implement set_in_data")

    def out_data(self, port: int) -> Any:
        raise NotImplementedError("Subclasses must implement out_data")

    def restore(self, doc: dict):
        return {}

    def on_data_updated(self):
        for port, _ in self.outgoing_connections.items():
            self.data_updated.emit(port)

def dual_class(gui_base_class):
    def decorator(cls):
        # Get the metaclass of the original class
        cls._is_gui = True
        metaclass = type(cls)

        # Create the NoGuiVersion using the same metaclass
        class NoGuiVersion(BaseNode):
            def __init__(self, *args, **kwargs):
                #self._is_gui = False
                super().__init__()
                self._is_gui = False
                cls.__init__(self, *args, **kwargs)

        # Copy all attributes and methods from the original class to NoGuiVersion
        for name, attr in cls.__dict__.items():
            if not name.startswith('__'):
                if isinstance(attr, (FunctionType, staticmethod, classmethod)):
                    # For methods, static methods, and class methods
                    setattr(NoGuiVersion, name, attr)
                elif isinstance(attr, property):
                    # For properties
                    setattr(NoGuiVersion, name, property(attr.fget, attr.fset, attr.fdel, attr.__doc__))
                else:
                    # For other attributes
                    setattr(NoGuiVersion, name, attr)

        # Attach NoGuiVersion to the original class
        cls.NoGuiVersion = NoGuiVersion

        # Set in type registry
        global_data_models[cls.__name__] = cls
        return cls

    return decorator
