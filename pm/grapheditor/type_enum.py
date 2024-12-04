import contextlib
import logging
import threading
from enum import Enum

from PyQt5.QtWidgets import QVBoxLayout
from qtpy.QtGui import QDoubleValidator, QColor
from qtpy.QtWidgets import QApplication, QLabel, QLineEdit, QWidget, QHBoxLayout, QComboBox
from PyQt5.QtCore import Qt
import qtpynodeeditor as nodeeditor
from qtpynodeeditor import (NodeData, NodeDataModel, NodeDataType,
                            NodeValidationState, Port, PortType, style as style_module)
from qtpynodeeditor.type_converter import TypeConverter

from pydantic import BaseModel
from typing import Type, Any

from pm.grapheditor.base_node import dual_class, BaseNode
from pm.grapheditor.common import get_node_data_type, style_node, VarData

@dual_class(BaseNode)
class PydanticEnumNodeDataModel(NodeDataModel):
    model_class: Type[Enum] = None
    data_type = NodeDataType("enum", "Enum")
    port_caption_visible = True

    @classmethod
    def configure(cls, model_class: Type[Enum]):
        cls.name = cls.__name__
        cls.caption = cls.__name__
        cls.model_class = model_class
        cls.num_ports = {
            PortType.input: 0,
            PortType.output: 1,
        }

        cls.port_caption = {
            'input': {},
            'output': {0: f'Result: {cls.__name__}'}
        }
        cls.data_types = {
            PortType.input: {},
            PortType.output: {0: get_node_data_type(model_class)},
        }
        cls.data_type = cls.data_types
        cls.port_caption_visible = {
            'input': {},
            'output': {0: True}
        }

    def __init__(self, style=None, parent=None):
        if self.model_class is None:
            raise ValueError("Enum model class must be configured before instantiation.")

        self.caption = self.__class__.name.replace("Node", "")
        self._data = None
        self.num_ports = self.__class__.num_ports
        self.port_caption = self.__class__.port_caption
        self.data_types = self.__class__.data_types
        self.port_caption_visible = self.__class__.port_caption_visible
        self.data_type = self.__class__.data_type

        if self._is_gui:
            super().__init__(style=style, parent=parent)

            self.combo_box = QComboBox()
            for item in self.model_class:
                self.combo_box.addItem(item.name)
            self.combo_box.currentIndexChanged.connect(self.on_enum_changed)
            self.on_enum_changed()

            self._layout = QVBoxLayout()
            self._layout.addWidget(self.combo_box)

            self._container_widget = QWidget()
            self._container_widget.setLayout(self._layout)
            self._container_widget.setAttribute(Qt.WA_NoSystemBackground)

            style_node("cyan", self)



    def embedded_widget(self) -> QWidget:
        return self._container_widget

    def on_enum_changed(self):
        selected_name = self.combo_box.currentText()
        self._data = self.model_class[selected_name]
        self.data_updated.emit(0)

    def out_data(self, port: int) -> NodeData:
        return VarData(self._data)

    def save(self) -> dict:
        doc = super().save()
        if self._data is not None:
            doc['enum_name'] = self._data.name
        return doc

    def restore(self, state: dict):
        if "enum_name" in state:
            self._data = self.model_class[state["enum_name"]]
            self.data_updated.emit(0)

            if self._is_gui:
                self.combo_box.setCurrentText(state["enum_name"])
                self.on_enum_changed()
