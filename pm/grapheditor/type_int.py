import contextlib
import logging
import threading

from qtpy.QtGui import QDoubleValidator
from qtpy.QtWidgets import QApplication, QLabel, QLineEdit, QWidget

import qtpynodeeditor as nodeeditor
from qtpynodeeditor import (NodeData, NodeDataModel, NodeDataType,
                            NodeValidationState, Port, PortType)
from qtpynodeeditor.type_converter import TypeConverter

from pm.grapheditor.base_node import BaseNode, dual_class
from pm.grapheditor.common import style_node, VarData, get_node_data_type


class DecimalDataOld(NodeData):
    'Node data holding a decimal (floating point) number'
    data_type = NodeDataType("decimal", "Decimal")

    def __init__(self, number: float = 0.0):
        self._number = number
        self._lock = threading.RLock()

    @property
    def lock(self):
        return self._lock

    @property
    def number(self) -> float:
        'The number data'
        return self._number

    @property
    def data(self) -> float:
        'The number data'
        return self._number

    def number_as_text(self) -> str:
        'Number as a string'
        return '%g' % self._number


class IntegerDataOld(NodeData):
    'Node data holding an integer value'
    data_type = NodeDataType("integer", "Integer")

    def __init__(self, number: int = 0):
        self._number = number
        self._lock = threading.RLock()

    @property
    def lock(self):
        return self._lock

    @property
    def number(self) -> int:
        'The number data'
        return self._number

    @property
    def data(self) -> int:
        'The number data'
        return self._number

    def number_as_text(self) -> str:
        'Number as a string'
        return str(self._number)

@dual_class(BaseNode)
class NumberSourceDataModel(NodeDataModel):
    caption = "NumberSource"
    caption_visible = False
    num_ports = {PortType.input: 0,
                 PortType.output: 1,
                 }
    port_caption = {'output': {0: 'Result'}}
    data_type = get_node_data_type(float)

    def __init__(self, style=None, parent=None):
        if self._is_gui:
            super().__init__(style=style, parent=parent)
            style_node("green", self)

        self._number = VarData(0.0)

        if self._is_gui:
            self._line_edit = QLineEdit()
            self._line_edit.setValidator(QDoubleValidator())
            self._line_edit.setMaximumSize(self._line_edit.sizeHint())
            self._line_edit.textChanged.connect(self.on_text_edited)
            self._line_edit.setText("0.0")

    @property
    def number(self):
        return self._number

    def save(self) -> dict:
        'Add to the JSON dictionary to save the state of the NumberSource'
        doc = super().save()
        if self._number:
            doc['number'] = self._number.data
        return doc

    def restore(self, state: dict):
        'Restore the number from the JSON dictionary'
        try:
            value = float(state["number"])
            self._number.set_data(value)
            self.data_updated.emit(0)

            if self._is_gui:
                self._line_edit.setText(self._number.as_str)
        except Exception:
            pass


    def out_data(self, port: int) -> NodeData:
        return self._number

    def embedded_widget(self) -> QWidget:
        return self._line_edit

    def on_text_edited(self, string: str):
        try:
            number = float(self._line_edit.text())
        except ValueError:
            self._number.set_data(0)
            self.data_updated.emit(0)
        else:
            self._number.set_data(number)
            self.data_updated.emit(0)

@dual_class(BaseNode)
class NumberDisplayModel(NodeDataModel):
    caption = "NumberDisplay"
    data_type = get_node_data_type(float)
    caption_visible = False
    num_ports = {PortType.input: 1,
                 PortType.output: 0,
                 }
    port_caption = {'input': {0: 'Number'}}

    def __init__(self, style=None, parent=None):
        super().__init__(style=style, parent=parent)
        style_node("green", self)

        self._number = VarData(0.0)
        self._label = QLabel()
        self._label.setMargin(3)
        self._validation_state = NodeValidationState.warning
        self._validation_message = 'Uninitialized'

    def set_in_data(self, data: NodeData, port: Port):
        self._number.set_data(data.data if data is not None else 0)
        number_ok = (self._number is not None and
                     self._number.data_type.id in ('decimal', 'integer'))

        if number_ok:
            self._validation_state = NodeValidationState.valid
            self._validation_message = ''
            self._label.setText(self._number.as_str)
        else:
            self._validation_state = NodeValidationState.warning
            self._validation_message = "Missing or incorrect inputs"
            self._label.clear()

        self._label.adjustSize()

    def embedded_widget(self) -> QWidget:
        return self._label

def integer_to_decimal_converter(data: VarData) -> VarData:
    return VarData(float(data.data))

def decimal_to_integer_converter(data: VarData) -> VarData:
    return VarData(int(data.data))