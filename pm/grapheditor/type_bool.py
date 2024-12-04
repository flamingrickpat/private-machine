import contextlib
import logging
import threading

from qtpy.QtGui import QDoubleValidator, QColor
from qtpy.QtWidgets import QApplication, QLabel, QLineEdit, QWidget, QHBoxLayout
from PyQt5.QtWidgets import QCheckBox
import qtpynodeeditor as nodeeditor
from qtpynodeeditor import (NodeData, NodeDataModel, NodeDataType,
                            NodeValidationState, Port, PortType, style as style_module)
from qtpynodeeditor.type_converter import TypeConverter

from pm.grapheditor.base_node import dual_class, BaseNode
from pm.grapheditor.common import style_node, get_node_data_type, VarData

class BooleanDataOld(NodeData):
    'Node data holding a boolean value'
    data_type = NodeDataType("boolean", "Boolean")

    def __init__(self, value: bool = False):
        self._value = value
        self._lock = threading.RLock()

    @property
    def lock(self):
        return self._lock

    @property
    def value(self) -> bool:
        return self._value

    @property
    def data(self) -> bool:
        'The number data'
        return self._value




@dual_class(BaseNode)
class BooleanSourceDataModel(NodeDataModel):
    caption = "BooleanSource"
    caption_visible = False
    num_ports = {PortType.input: 0,
                 PortType.output: 1,
                 }
    port_caption = {'output': {0: 'Result'}}
    data_type = get_node_data_type(bool)

    def __init__(self, style=None, parent=None):
        super().__init__(style=style, parent=parent)
        style_node("red", self)

        self._boolean = VarData(False)

        if self._is_gui:
            self._check_box = QCheckBox()
            self._check_box.setChecked(False)
            self._check_box.stateChanged.connect(self.on_checkbox_toggled)

            # Create a layout and add the widget
            self._layout = QHBoxLayout()
            self._layout.addWidget(self._check_box)

            # Create a container widget and set the layout
            self._container_widget = QWidget()
            self._container_widget.setStyleSheet("background: transparent;")
            self._container_widget.setLayout(self._layout)

    @property
    def boolean(self):
        return self._boolean

    def save(self) -> dict:
        'Add to the JSON dictionary to save the state of the BooleanSource'
        doc = super().save()
        doc['boolean'] = self._boolean.data
        return doc

    def restore(self, state: dict):
        'Restore the boolean from the JSON dictionary'
        try:
            value = state["boolean"]
            self._boolean.set_data(value)
            if self._is_gui:
                self._check_box.setChecked(self._boolean.data)
        except KeyError:
            pass


    def out_data(self, port: int) -> NodeData:
        return self._boolean

    def embedded_widget(self) -> QWidget:
        'The boolean source has a checkbox widget for the user to toggle'
        return self._container_widget

    def on_checkbox_toggled(self, state: int):
        value = state != 0
        self._boolean.set_data(value)
        self.data_updated.emit(0)


@dual_class(BaseNode)
class BooleanDisplayModel(NodeDataModel):
    caption = "BooleanDisplay"
    data_type = get_node_data_type(bool)
    caption_visible = False
    num_ports = {PortType.input: 1,
                 PortType.output: 0,
                 }
    port_caption = {'input': {0: 'Boolean'}}

    def __init__(self, style=None, parent=None):
        if self._is_gui:
            super().__init__(style=style, parent=parent)
            style_node("red", self)

        self._boolean = VarData(False)

        if self._is_gui:
            self._label = QLabel()
            self._label.setMargin(3)
        self._validation_state = NodeValidationState.warning
        self._validation_message = 'Uninitialized'

    def set_in_data(self, data: NodeData, port: Port):
        '''
        New data propagated to the input

        Parameters
        ----------
        data : NodeData
        int : int
        '''
        self._boolean.set_data(data.data if data is not None else False)
        bool_ok = (self._boolean is not None and
                     self._boolean.data_type.id in ('boolean'))

        if bool_ok:
            self._validation_state = NodeValidationState.valid
            self._validation_message = ''
            if self._is_gui:
                self._label.setText("True" if self._boolean.data else "False")
        else:
            self._validation_state = NodeValidationState.warning
            self._validation_message = "Missing or incorrect inputs"
            if self._is_gui:
                self._label.clear()

        if self._is_gui:
            self._label.adjustSize()

    def embedded_widget(self) -> QWidget:
        'The boolean display has a label'
        return self._label
