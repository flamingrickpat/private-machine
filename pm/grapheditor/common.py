import threading
import copy
from enum import Enum

from qtpynodeeditor import NodeDataType, NodeData
from qtpy.QtGui import QDoubleValidator, QColor

from pydantic import BaseModel
from typing import Type, Any
from typing import Optional, get_origin, get_args, Union, List, Dict

from pm.utils.singleton_utils import singleton

@singleton
class ExceptionPasser:
    def __init__(self):
        self.exception = None

    def set(self, e):
        self.exception = e

    def reset(self):
        self.exception = None

    def has_exception(self) -> bool:
        return self.exception is not None

@singleton
class TickCounter:
    def __init__(self):
        self.tick = 1

    def inc(self):
        self.tick += 1

    def get(self):
        return self.tick

def is_optional_type(tp):
    return get_origin(tp) is Optional or get_origin(tp) is Union

def is_enumerable(tp):
    return get_origin(tp) is list

def is_dict(tp):
    return get_origin(tp) is dict


def get_inner_type(optional_type):
    return get_args(optional_type)[0]


class ClockCycle(NodeData):
    'Node data holding an integer value'
    data_type = NodeDataType("exec", "Exec")

    def __init__(self, clock: int = 0):
        self._clock = clock
        self._lock = threading.RLock()

    @property
    def lock(self):
        return self._lock

    @property
    def clock(self) -> int:
        return self._clock

    def setclock(self, val):
        self._clock = val

    @property
    def data(self) -> int:
        return self._clock

    def inc(self):
        self._clock += 1


def get_node_data_type(field_type: Any) -> NodeDataType:
    if is_optional_type(field_type):
        field_type = get_inner_type(field_type)

    if field_type == "exec":
        return NodeDataType("exec", "Execution")
    elif field_type == type(None):
        return NodeDataType("none", "None")
    elif field_type == str:
        return NodeDataType("string", "String")
    elif field_type == int:
        return NodeDataType("integer", "Integer")
    elif field_type == bool:
        return NodeDataType("boolean", "Boolean")
    elif field_type == float:
        return NodeDataType("decimal", "Decimal")
    if field_type == "any":
        return NodeDataType("any", "Any")
    if field_type == "tool_json":
        return NodeDataType("tool_json", "Tool JSON")
    if field_type == "tool_python":
        return NodeDataType("tool_python", "Tool Python")
    if field_type.__name__ == "list":
        return NodeDataType("any", "Any")
    if field_type.__name__ == "function":
        return NodeDataType("any", "Any")
    else:
        try:
            if field_type.__class__.__name__ == Any.__class__.__name__:
                return NodeDataType("any", "Any")
        except:
            pass

        try:
            if issubclass(field_type, Enum):
                return NodeDataType(field_type.__name__.lower(), field_type.__name__.lower())
        except:
            pass

        try:
            if issubclass(field_type, BaseModel):
                return NodeDataType(field_type.__name__.lower(), field_type.__name__)
        except:
            pass

    if is_enumerable(field_type) or is_dict(field_type):
        return NodeDataType("any", "Any")

    return NodeDataType("any", "Any")
    #raise ValueError(f"Unsupported field type: {field_type}")



class VarData(NodeData):
    def __init__(self, data: Any = None, data_type: NodeDataType = None):
        self._data = data
        if data_type is None and data is not None:
            self._data_type = get_node_data_type(type(data))
        else:
            self._data_type = data_type
        self._lock = threading.RLock()

    @property
    def lock(self):
        return self._lock

    @property
    def data(self) -> Any:
        return self._data

    @property
    def data_type(self) -> NodeDataType:
        return self._data_type

    def set_data(self, val) -> None:
        self._data = val
        return

        if self._data_type is None and self._data is None:
            self._data = val
            self._data_type = get_node_data_type(type(val))
        else:
            if self._data_type.id == get_node_data_type(type(val)).id or self._data_type.id == "none":
                self._data = val
                self._data_type = get_node_data_type(type(val))
            else:
                raise ValueError(f"VarData is set to {self._data_type.id} but new value {val} is of {get_node_data_type(type(val)).id}!")

    def set_data_type(self, dt) -> None:
        self._data_type = dt
        return

        if self._data is None:
            self._data_type = dt
        else:
            if self._data_type is not None:
                if self._data_type.id == dt.id or self._data_type.id == "none":
                    self._data_type = dt
                else:
                    if dt.id != "any":
                        raise ValueError(f"Incompatible data type: {dt}!")
            else:
                self._data_type = dt


    @property
    def as_str(self) -> str:
        if self._data is None:
            return "None"
        else:
            return f"{self._data}"


def to_any_converter(nodedata) -> VarData:
    return VarData(nodedata.data)

def get_from_any_converter(target: NodeDataType):
    if target.id == "boolean":
        def to_bool(nodedata) -> VarData:
            try:
                return VarData(bool(nodedata.data))
            except Exception:
                return VarData(False)
        return to_bool
    if target.id == "integer":
        def to_int(nodedata) -> VarData:
            try:
                return VarData(int(nodedata.data))
            except Exception:
                return VarData(0)
        return to_int
    if target.id == "decimal":
        def to_float(nodedata) -> VarData:
            try:
                return VarData(float(nodedata.data))
            except Exception:
                return VarData(0.0)
        return to_float
    if target.id == "string":
        def to_str(nodedata) -> VarData:
            try:
                return VarData(str(nodedata.data))
            except Exception:
                return VarData("")
        return to_str

    def to_any(nodedata) -> VarData:
        try:
            return VarData(nodedata.data)
        except Exception:
            return VarData(None)

    return to_any

def invert_qcolor(color):
    if not isinstance(color, QColor):
        raise TypeError("Input must be a QColor object")

    # Get the RGB components
    r, g, b = color.red(), color.green(), color.blue()

    # Invert each component
    r, g, b = 255 - r, 255 - g, 255 - b

    # Create and return the new inverted color
    return QColor(r, g, b)

def pastelify(color, pastel_factor=0.5):
    """
    Convert a color to a pastel shade.
    pastel_factor: 0.0 gives the original color, 1.0 gives white.
    """
    r, g, b = color.red(), color.green(), color.blue()
    r = int(r + (255 - r) * pastel_factor)
    g = int(g + (255 - g) * pastel_factor)
    b = int(b + (255 - b) * pastel_factor)
    return QColor(r, g, b)

def style_node(color, nodemodel):
    vibrant_colors = {
        "black": QColor(0, 0, 0),
        "red": QColor(255, 0, 0),
        "green": QColor(0, 255, 0),
        "blue": QColor(0, 0, 255),
        "yellow": QColor(255, 255, 0),
        "cyan": QColor(0, 255, 255),
        "magenta": QColor(255, 0, 255),
        "orange": QColor(255, 165, 0),
        "purple": QColor(128, 0, 128),
    }
    col = pastelify(vibrant_colors.get(color, QColor(255, 255, 255)), 0.4)

    tmp = col.green() + col.red() + col.blue()
    if tmp < (255 * 1.5):
        text_color = QColor(254, 254, 254)
    else:
        text_color = QColor(1, 1, 1)

    nodemodel._style = copy.deepcopy(nodemodel.style)
    nodemodel.style.node.gradient_colors = ((0.8, col), )
    nodemodel.style.node.font_color = text_color
    nodemodel.style.node.font_color_faded = text_color
    nodemodel.style.node.pen_width = 2

class ExecutionException(Exception):
    def __init__(self, original_exception: str):
        self.original_exception = original_exception


def change_base_class_to(base_class):
    def class_rebuilder(cls):
        class NewClass(base_class):
            def __init__(self, *args, **kwargs):
                super(NewClass, self).__init__(*args, **kwargs)

            # Here we add all attributes/methods of the old class to the new class
            for attr_name, attr_value in cls.__dict__.items():
                setattr(NewClass, attr_name, attr_value)

        return NewClass

    return class_rebuilder
