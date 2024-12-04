from typing import Any
import threading

from PyQt5.QtWidgets import QTextEdit, QWidget, QVBoxLayout
from qtpynodeeditor import NodeData, NodeDataModel, NodeDataType, PortType, NodeValidationState
from PyQt5.QtCore import Qt

from pm.grapheditor.common import get_node_data_type, style_node, VarData
from pm.grapheditor.base_node import dual_class, BaseNode

@dual_class(BaseNode)
class pretty_print_func_node(NodeDataModel):
    caption = "PrettyPrint"
    caption_visible = False
    num_ports = {PortType.input: 1, PortType.output: 0}
    port_caption = {'input': {0: 'Input'}}
    data_type = get_node_data_type(Any)

    def __init__(self, style=None, parent=None):
        if self._is_gui:
            super().__init__(style=style, parent=parent)
            style_node("cyan", self)

        self._input_data = None

        if self._is_gui:
            # Create the text widget
            self._text_edit = QTextEdit()
            self._text_edit.setReadOnly(True)
            self._text_edit.setMinimumSize(400, 200)

            # Create a layout and add the text widget
            self._layout = QVBoxLayout()
            self._layout.addWidget(self._text_edit)

            # Create a container widget and set the layout
            self._container_widget = QWidget()
            self._container_widget.setLayout(self._layout)
            self._container_widget.setAttribute(Qt.WA_NoSystemBackground)

    """
    def save(self) -> dict:
        doc = super().save()
        if self._input_data:
            doc['data'] = str(self._input_data)
        return doc

    def restore(self, state: dict):
        try:
            value = state["data"]
        except KeyError:
            ...
        else:
            self._input_data = value
            self._text_edit.setPlainText(self._input_data)
    """

    def set_in_data(self, data: NodeData, port: int):
        self._input_data = data.data if data else None
        self._text_edit.setPlainText(str(self._input_data))

    def embedded_widget(self) -> QWidget:
        return self._container_widget
