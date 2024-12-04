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

from pm.grapheditor.base_node import BaseNode, dual_class
from pm.grapheditor.common import get_node_data_type, style_node, VarData
from pm.grapheditor.decorator import global_json_tools, global_python_tools

@dual_class(BaseNode)
class ToolJsonNodeDataModel(NodeDataModel):
    model_class: Type[Enum] = None
    data_type = get_node_data_type("tool_json")
    port_caption_visible = True

    def __init__(self, style=None, parent=None):
        self.caption = "Tool JSON Selection"
        self.name = "ToolJsonNodeDataModel"

        self.num_ports = {
            PortType.input: 0,
            PortType.output: 1,
        }

        self.port_caption = {
            'input': {},
            'output': {0: f'Result: Tool JSON'}
        }
        self.data_types = {
            PortType.input: {},
            PortType.output: {0: get_node_data_type("tool_json")},
        }
        self.data_type = self.data_types
        self.port_caption_visible = {
            'input': {},
            'output': {0: True}
        }


        self._data = None

        if self._is_gui:
            super().__init__(style=style, parent=parent)

            self.combo_box = QComboBox()
            for k, v in global_json_tools.items():
                self.combo_box.addItem(k)

            self.combo_box.currentIndexChanged.connect(self.on_tool_changed)
            self.on_tool_changed()

            self._layout = QVBoxLayout()
            self._layout.addWidget(self.combo_box)

            self._container_widget = QWidget()
            self._container_widget.setLayout(self._layout)
            self._container_widget.setAttribute(Qt.WA_NoSystemBackground)

            style_node("cyan", self)

    def embedded_widget(self) -> QWidget:
        return self._container_widget

    def on_tool_changed(self):
        selected_name = self.combo_box.currentText()
        self._data = global_json_tools[selected_name]
        self.data_updated.emit(0)

    def out_data(self, port: int) -> NodeData:
        return VarData(self._data)

    def save(self) -> dict:
        doc = super().save()
        if self._data is not None:
            doc['tool_name'] = self._data.__name__
        return doc

    def restore(self, state: dict):
        if "tool_name" in state:
            self._data = global_json_tools[state["tool_name"]]
            self.data_updated.emit(0)

            if self._is_gui:
                self.combo_box.setCurrentText(state["tool_name"])
                self.on_tool_changed()

@dual_class(BaseNode)
class ToolPythonNodeDataModel(NodeDataModel):
    model_class: Type[Enum] = None
    data_type = get_node_data_type("tool_python")
    port_caption_visible = True

    def __init__(self, style=None, parent=None):
        self.caption = "Tool Python Selection"
        self.name = "ToolPythonNodeDataModel"

        self.num_ports = {
            PortType.input: 0,
            PortType.output: 1,
        }

        self.port_caption = {
            'input': {},
            'output': {0: f'Result: Tool Python'}
        }
        self.data_types = {
            PortType.input: {},
            PortType.output: {0: get_node_data_type("tool_python")},
        }
        self.data_type = self.data_types
        self.port_caption_visible = {
            'input': {},
            'output': {0: True}
        }


        self._data = None

        if self._is_gui:
            super().__init__(style=style, parent=parent)

            self.combo_box = QComboBox()
            for k, v in global_python_tools.items():
                self.combo_box.addItem(k)

            self.combo_box.currentIndexChanged.connect(self.on_tool_changed)
            self.on_tool_changed()

            self._layout = QVBoxLayout()
            self._layout.addWidget(self.combo_box)

            self._container_widget = QWidget()
            self._container_widget.setLayout(self._layout)
            self._container_widget.setAttribute(Qt.WA_NoSystemBackground)

            style_node("cyan", self)

    def embedded_widget(self) -> QWidget:
        return self._container_widget

    def on_tool_changed(self):
        selected_name = self.combo_box.currentText()
        self._data = global_python_tools[selected_name]
        self.data_updated.emit(0)

    def out_data(self, port: int) -> NodeData:
        return VarData(self._data)

    def save(self) -> dict:
        doc = super().save()
        if self._data is not None:
            doc['tool_name'] = self._data.__name__
        return doc

    def restore(self, state: dict):
        if "tool_name" in state:
            self._data = global_python_tools[state["tool_name"]]
            self.data_updated.emit(0)

            if self._is_gui:
                self.combo_box.setCurrentText(state["tool_name"])
                self.on_tool_changed()
