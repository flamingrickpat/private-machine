import contextlib
import logging
import threading

from PyQt5.QtGui import QFont
from qtpy.QtGui import QDoubleValidator
from qtpy.QtWidgets import QApplication, QLabel, QLineEdit, QWidget

import qtpynodeeditor as nodeeditor
from qtpynodeeditor import (NodeData, NodeDataModel, NodeDataType,
                            NodeValidationState, Port, PortType)
from qtpynodeeditor.type_converter import TypeConverter
from PyQt5.QtWidgets import QWidget, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QDialog, QTextEdit, QDialogButtonBox
from PyQt5.QtCore import Qt

from pm.grapheditor.common import style_node, get_node_data_type, VarData
from pm.grapheditor.code_editor import CodeEditor
from pm.grapheditor.base_node import dual_class, BaseNode

logger = logging.getLogger(__name__)

class CustomTextEdit(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Set Consolas or another monospace font
        font = QFont("Consolas", 10)  # You can change "Consolas" to another monospace font if needed
        self.setFont(font)

        # Disable word wrap
        self.setLineWrapMode(QTextEdit.NoWrap)

        # Always show horizontal and vertical scrollbars
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

    def insertFromMimeData(self, source):
        # Convert rich text to plain text when pasting
        if source.hasText():
            self.insertPlainText(source.text())
        else:
            super().insertFromMimeData(source)
class FullEditorDialog(QDialog):
    def __init__(self, initial_text="", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Full Text Editor")

        self.resize(800, 600)
        self.setWindowTitle("Full Text Editor")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setWindowModality(Qt.ApplicationModal)

        self.text_edit = CodeEditor()
        self.text_edit.setText(initial_text)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addWidget(self.text_edit)
        layout.addWidget(self.button_box)

        self.setLayout(layout)

    def get_text(self):
        return self.text_edit.toPlainText()


class StringDataOld(NodeData):
    'Node data holding a decimal (floating point) number'
    data_type = NodeDataType("string", "String")

    def __init__(self, string: str = ""):
        self._string = string
        self._lock = threading.RLock()

    @property
    def lock(self):
        return self._lock

    @property
    def string(self) -> str:
        return self._string

    @property
    def data(self) -> str:
        return self._string

@dual_class(BaseNode)
class StringSourceDataModel(NodeDataModel):
    caption = "StringSource"
    caption_visible = False
    num_ports = {PortType.input: 0,
                 PortType.output: 1,
                 }
    port_caption = {'output': {0: 'Result'}}
    data_type = get_node_data_type(str)

    def __init__(self, style=None, parent=None):
        if self._is_gui:
            super().__init__(style=style, parent=parent)
            style_node("purple", self)

        self._string = VarData("")

        if self._is_gui:
            self._line_edit = QLineEdit()
            self._line_edit.setMaximumSize(self._line_edit.sizeHint())
            self._line_edit.textChanged.connect(self.on_text_edited)

            self._button = QPushButton("Edit")
            self._button.clicked.connect(self.open_full_editor)

            # Create a layout and add the widgets
            self._layout = QHBoxLayout()
            self._layout.addWidget(self._line_edit)
            self._layout.addWidget(self._button)

            # Create a container widget and set the layout
            self._container_widget = QWidget()
            self._container_widget.setAttribute(Qt.WA_NoSystemBackground)
            #self._container_widget.setStyleSheet("background: transparent;")
            self._container_widget.setLayout(self._layout)

    def save(self) -> dict:
        'Add to the JSON dictionary to save the state of the NumberSource'
        doc = super().save()
        if self._string:
            doc['string'] = self._string.data
        return doc

    def restore(self, state: dict):
        'Restore the number from the JSON dictionary'
        try:
            value = state["string"]
            self._string.set_data(value)
            self.data_updated.emit(0)

            if self._is_gui:
                self._line_edit.setText(self._string.data)
        except Exception:
            pass

    def out_data(self, port: int) -> NodeData:
        return self._string

    def embedded_widget(self) -> QWidget:
        'The number source has a line edit widget for the user to type in'
        return self._container_widget

    def on_text_edited(self, string: str):
        try:
            self._string.set_data(string)
            self.data_updated.emit(0)
        except ValueError:
            self._string.set_data("")
            self.data_updated.emit(0)

    def open_full_editor(self):
        initial_text = self._line_edit.text()
        dialog = FullEditorDialog(initial_text, parent=None) #self._container_widget)
        if dialog.exec_() == QDialog.Accepted:
            updated_text = dialog.get_text()
            self._line_edit.setText(updated_text)

@dual_class(BaseNode)
class StringDisplayModel(NodeDataModel):
    caption = "StringDisplay"
    data_type = get_node_data_type(str)
    caption_visible = False
    num_ports = {PortType.input: 1,
                 PortType.output: 0,
                 }
    port_caption = {'input': {0: 'String'}}

    def __init__(self, style=None, parent=None):
        super().__init__(style=style, parent=parent)
        style_node("purple", self)

        self._string = VarData("")
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
        self._string.set_data(data.data if data is not None else "")
        str_ok = (self._string is not None and self._string.data_type.id in ('string'))

        if str_ok:
            self._validation_state = NodeValidationState.valid
            self._validation_message = ''
            self._label.setText(self._string.data)
        else:
            self._validation_state = NodeValidationState.warning
            self._validation_message = "Missing or incorrect inputs"
            self._label.clear()

        self._label.adjustSize()

    def embedded_widget(self) -> QWidget:
        'The number display has a label'
        return self._label
