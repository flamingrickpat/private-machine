import inspect
import logging
from typing import Callable

from typing import Any
import threading

from PyQt5.QtWidgets import QTextEdit, QWidget, QVBoxLayout, QPushButton, QLineEdit
from qtpynodeeditor import NodeData, NodeDataModel, NodeDataType, PortType, NodeValidationState
from PyQt5.QtCore import Qt

from pm.grapheditor.common import ClockCycle, style_node, ExecutionException, TickCounter, ExceptionPasser
from pm.grapheditor.base_node import dual_class, BaseNode

logger = logging.getLogger(__name__)

@dual_class(BaseNode)
class RootNode(NodeDataModel):
    caption = "Root"
    caption_visible = True
    num_ports = {PortType.input: 0, PortType.output: 1}
    port_caption = {'output': {0: 'Exec Out'}}
    data_type = ClockCycle.data_type

    def __init__(self, style=None, parent=None):
        if self._is_gui:
            super().__init__(style=style, parent=parent)
            style_node("white", self)

        from privatemachine.system.controller import Controller
        con = Controller()
        con.tick_delegate = self.start_execution

        self._clock = ClockCycle(1)
        if self._is_gui:
            self._button = QPushButton("Start Execution")
            self._button.clicked.connect(self.start_execution)

            self._line_edit = QLineEdit()
            self._line_edit.setReadOnly(True)
            self._line_edit.setMaximumSize(self._line_edit.sizeHint())
            self._line_edit.setText(f"{self._clock.clock}")

            self._layout = QVBoxLayout()
            self._layout.addWidget(self._button)
            self._layout.addWidget(self._line_edit)

            self._container_widget = QWidget()
            self._container_widget.setLayout(self._layout)
            self._container_widget.setAttribute(Qt.WA_NoSystemBackground)



    def embedded_widget(self) -> QWidget:
        return self._container_widget

    def start_execution(self):
        tick = TickCounter()
        tick.inc()

        from privatemachine.system.controller import Controller
        con = Controller()
        con.start_execution()

        self._clock.setclock(tick.get())
        if self._is_gui:
            self._line_edit.setText(f"{self._clock.clock}")
        try:
            self.data_updated.emit(0)
        except Exception as e:
            logger.error(f"Halted execution because of {repr(e)}")

        ep = ExceptionPasser()
        if ep.has_exception():
            logger.error(f"Halted execution because of {ep.exception}")
        ep.reset()

    def out_data(self, port) -> NodeData:
        return ClockCycle(self._clock.clock)