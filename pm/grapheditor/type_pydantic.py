import contextlib
import logging
import threading

from qtpy.QtGui import QDoubleValidator, QColor
from qtpy.QtWidgets import QApplication, QLabel, QLineEdit, QWidget, QHBoxLayout

import qtpynodeeditor as nodeeditor
from qtpynodeeditor import (NodeData, NodeDataModel, NodeDataType,
                            NodeValidationState, Port, PortType, style as style_module)
from qtpynodeeditor.type_converter import TypeConverter

from pydantic import BaseModel, ValidationError
from typing import Type, Any

from pm.grapheditor.base_node import BaseNode, dual_class
from pm.grapheditor.common import get_node_data_type, style_node, VarData, TickCounter

logger = logging.getLogger(__name__)

class PydanticDataOld(NodeData):
    def __init__(self, data: BaseModel):
        self._data = data
        self._lock = threading.RLock()

    @property
    def lock(self):
        return self._lock

    @property
    def data(self) -> BaseModel:
        return self._data

def format_errors(e: ValidationError) -> str:
    error_list = []
    for error in e.errors():
        field = error['loc'][0]
        error_list.append(field)
    return ", ".join(error_list)

def collect_annotations(model_class):
    annotations = {}
    if model_class.__name__ == BaseModel.__name__:
        return annotations
    for base in model_class.__bases__:
        if hasattr(base, '__annotations__') and base.__name__ != BaseModel.__name__:
            annotations.update(collect_annotations(base))
    if hasattr(model_class, '__annotations__'):
        annotations.update(model_class.__annotations__)

    annotations = {k: v for k, v in annotations.items() if not k.startswith('hidden')}
    return annotations

@dual_class(BaseNode)
class PydanticNodeDataModel(NodeDataModel):
    model_class: Type[BaseModel] = None
    data_type = NodeDataType("pydantic", "Pydantic")
    port_caption_visible = True

    @classmethod
    def configure(cls, model_class: Type[BaseModel]):
        cls.name = cls.__name__
        cls.caption = cls.__name__
        cls.model_class = model_class
        cls.num_ports = {
            PortType.input: len(collect_annotations(model_class)),
            PortType.output: 1,
        }

        inp = {}
        for idx, name in enumerate(collect_annotations(model_class).keys()):
            t = get_node_data_type(collect_annotations(model_class)[name])
            inp[idx] = f"{name}: {t.id}" #{collect_annotations(model_class)[name].__name__}"

        cls.port_caption = {
            'input': inp,
            'output': {0: f'Result: {cls.__name__}'}
        }
        cls.data_types = {
            PortType.input: {idx: get_node_data_type(field_type) for idx, field_type in enumerate(
                collect_annotations(model_class).values())},
            PortType.output: {0: get_node_data_type(model_class)},
        }
        cls.data_type = cls.data_types #NodeDataType(model_class.__name__.lower(), model_class.__name__)
        cls.port_caption_visible = {
            'input': {idx: True for idx, name in enumerate(collect_annotations(model_class).keys())},
            'output': {0: 'Result'}
        }

    def __init__(self, style=None, parent=None):
        if self.model_class is None:
            raise ValueError("Pydantic model class must be configured before instantiation.")

        self.caption = self.__class__.name.replace("Node", "")
        self._data = None
        self.inputs = {name: None for name in collect_annotations(self.model_class).keys()}
        self.num_ports = self.__class__.num_ports
        self.port_caption = self.__class__.port_caption
        self.data_types = self.__class__.data_types
        self.port_caption_visible = self.__class__.port_caption_visible
        self.data_type = self.__class__.data_type
        self._validation_state = NodeValidationState.valid
        self._validation_message = ""

        if self._is_gui:
            super().__init__(style=style, parent=parent)
            style_node("orange", self)

    def set_in_data(self, data: NodeData, port):
        field_name = list(collect_annotations(self.model_class).keys())[port.index]
        self.inputs[field_name] = data.data if data is not None else None
        self.try_make_obj()
        self.data_updated.emit(0)

    def out_data(self, port: int) -> NodeData:
        self.try_make_obj()
        return VarData(self._data)

    def validation_state(self) -> NodeValidationState:
        return self._validation_state

    def validation_message(self) -> str:
        return self._validation_message

    def try_make_obj(self):
        try:
            keys = list(self.inputs.keys())
            for k in keys:
                if self.inputs[k] is None:
                    del self.inputs[k]

            self._data = self.model_class(**self.inputs)
            self._validation_state = NodeValidationState.valid
            self._validation_message = ""
        except ValidationError as e:
            tc = TickCounter()
            if tc.get() > 1:
                self._validation_state = NodeValidationState.error
                self._validation_message = f"{format_errors(e)}"
                logger.error(f"Bad fields: {format_errors(e)}")