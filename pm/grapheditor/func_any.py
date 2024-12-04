import inspect
from typing import Callable

from typing import Any
import threading

from qtpy.QtGui import QDoubleValidator, QColor
from PyQt5.QtWidgets import QTextEdit, QWidget, QVBoxLayout, QPushButton, QLineEdit, QHBoxLayout, QSizePolicy
from qtpynodeeditor import NodeData, NodeDataModel, NodeDataType, PortType, NodeValidationState
from PyQt5.QtCore import Qt

from pm.grapheditor.common import (get_node_data_type, ClockCycle, style_node, VarData, ExecutionException,
                                               TickCounter, ExceptionPasser)
from pm.grapheditor.base_node import dual_class, BaseNode

import logging
logger = logging.getLogger(__name__)

def create_ports_from_signature(func: Callable, offset: int):
    signature = inspect.signature(func)
    arguments = {}
    return_vals = {}
    has_args = False

    for idx, (name, param) in enumerate(signature.parameters.items()):
        if name == "args":
            has_args = True
        else:
            data_type = get_node_data_type(param.annotation)
            arguments[idx + offset] = (name, data_type)

    return_annotation = signature.return_annotation
    if return_annotation is not inspect.Signature.empty and return_annotation is not None:
        data_type = get_node_data_type(return_annotation)
        return_vals[offset] = (f"Result: {data_type.id}", data_type)

    return arguments, return_vals, has_args

DEFAULT_ARGS_COUNT = 8

@dual_class(BaseNode)
class FunctionNodeDataModel(NodeDataModel):
    func: Callable = None
    data_type = ClockCycle.data_type
    port_caption_visible = True
    pure: bool = False

    @classmethod
    def configure(cls, func: Callable, pure: bool = False):
        cls.caption = func.__name__.replace("Node", "")
        cls.name = cls.__name__
        cls.func = func
        cls.has_single_string = False

        extra_pins = 1
        if pure:
            extra_pins = 0
        cls.arguments, cls.return_vals, cls.has_args = create_ports_from_signature(func, extra_pins)
        cls.pure = pure

        tmp = len(cls.arguments)
        if cls.has_args:
            for i in range(DEFAULT_ARGS_COUNT):
                cls.arguments[tmp + i] = (f"Args {i}", get_node_data_type("any"))
        elif tmp > 0:
            cls.has_single_string = cls.arguments[extra_pins][1].id == "string"
            cls.has_single_string_arg_name = cls.arguments[extra_pins][0]

        cls.arg_length_without_args = tmp + extra_pins

        cls.num_ports = {
            PortType.input: len(cls.arguments) + extra_pins,
            PortType.output: len(cls.return_vals) + extra_pins,
        }
        cls.port_caption = {
            'input': {idx: f"{name}: {val.id}" for idx, (name, val) in cls.arguments.items()},
            'output': {idx: name for idx, (name, val) in cls.return_vals.items()},
        }
        if not pure:
            cls.port_caption['input'][0] = "Exec In"
            cls.port_caption['output'][0] = "Exec Out"

        cls.data_types = {
            'input': {idx: val for idx, (name, val) in cls.arguments.items()},
            'output': {idx: val for idx, (name, val) in cls.return_vals.items()},
        }
        if not pure:
            cls.data_types['input'][0] = get_node_data_type("exec")
            cls.data_types['output'][0] = get_node_data_type("exec")
        cls.data_type = cls.data_types

        cls.port_caption_visible = {
            'input': {idx: True for idx in range(len(cls.arguments) + extra_pins)},
            'output': {idx: True for idx in range(len(cls.return_vals) + extra_pins)},
        }
        cls.pure = pure
        cls.extra_pins = extra_pins

    def __init__(self, style=None, parent=None):
        if self.func is None:
            raise ValueError("Function must be configured before instantiation.")

        tick = TickCounter()

        self._clock = ClockCycle(tick.get())
        self.num_ports = self.__class__.num_ports
        self.port_caption = self.__class__.port_caption
        self.port_caption_visible = self.__class__.port_caption_visible
        self.data_type = self.__class__.data_type
        self.visible_args = 0

        if self._is_gui:
            super().__init__(style=style, parent=parent)
            if self.pure:
                style_node("cyan", self)
            else:
                style_node("blue", self)

        self._data = {}
        for i in range(len(self.return_vals.keys())):
            self._data[i + self.extra_pins] = None

        self.inputs = {}
        self._container_widget = None
        self._validation_state = NodeValidationState.valid
        self._validation_message = ""
        self._single_string_val = None

        if self._is_gui and self.__class__.has_args:
            self._minus = QPushButton("-")
            self._minus.clicked.connect(self.remove_arg)

            self._plus = QPushButton("+")
            self._plus.clicked.connect(self.add_arg)

            self._line_edit = QLineEdit()
            self._line_edit.setReadOnly(True)
            self._line_edit.setMaximumSize(self._line_edit.sizeHint())

            self._layout = QHBoxLayout()
            self._layout.addWidget(self._minus)
            self._layout.addWidget(self._line_edit)
            self._layout.addWidget(self._plus)

            self._container_widget = QWidget()
            self._container_widget.setLayout(self._layout)
            self._container_widget.setAttribute(Qt.WA_NoSystemBackground)

            self.mySizePolicy = QSizePolicy()
            self.mySizePolicy.setHorizontalStretch(1)
            self.mySizePolicy.setVerticalPolicy(2)

            self._container_widget.setSizePolicy(self.mySizePolicy)
            self._container_widget.setFixedHeight(60)

            self.visible_args = DEFAULT_ARGS_COUNT
            self.refresh_visible_args()

        if self._is_gui and self.__class__.has_single_string:
            self._line_edit = QLineEdit()
            self._line_edit.setReadOnly(False)
            self._line_edit.setMaximumSize(self._line_edit.sizeHint())
            self._line_edit.textChanged.connect(self.on_text_edited)

            self._layout = QHBoxLayout()
            self._layout.addWidget(self._line_edit)

            self._container_widget = QWidget()
            self._container_widget.setLayout(self._layout)
            self._container_widget.setAttribute(Qt.WA_NoSystemBackground)

            self.mySizePolicy = QSizePolicy()
            self.mySizePolicy.setHorizontalStretch(1)
            self.mySizePolicy.setVerticalPolicy(2)

            self._container_widget.setSizePolicy(self.mySizePolicy)
            self._container_widget.setFixedHeight(60)

    def on_text_edited(self, string: str):
        self._single_string_val = string

    def refresh_visible_args(self):
        if self.__class__.has_args:
            if self._is_gui:
                self._line_edit.setText(f"{self.visible_args}")

            self.num_ports[PortType.input] = self.__class__.arg_length_without_args + self.visible_args

            for i in range(DEFAULT_ARGS_COUNT):
                self.port_caption["input"][self.__class__.arg_length_without_args + i] = f"Arg {i}" if i < self.visible_args else ""

            for i in range(DEFAULT_ARGS_COUNT):
                self.port_caption_visible["input"][self.__class__.arg_length_without_args + i] = True if i < self.visible_args else False

    def remove_arg(self):
        self.visible_args = max(self.visible_args - 1, 1)
        self.refresh_visible_args()

    def add_arg(self):
        self.visible_args = min(self.visible_args + 1, DEFAULT_ARGS_COUNT - 1)
        self.refresh_visible_args()

    def embedded_widget(self) -> QWidget:
        if self._container_widget is not None:
            return self._container_widget

    def validation_state(self) -> NodeValidationState:
        return self._validation_state

    def validation_message(self) -> str:
        return self._validation_message

    def set_in_data(self, data: NodeData, port):
        ep = ExceptionPasser()
        if ep.has_exception():
            return

        if self.pure:
            field_name, _ = self.arguments[port.index]
            self.inputs[field_name] = data
            try:
                self.execute()

                self._validation_state = NodeValidationState.valid
                self._validation_message = ""
                for i in range(len(self.return_vals)):
                    self.data_updated.emit(i)
            except Exception as e:
                tc = TickCounter()
                if tc.get() > 1:
                    logger.error(f"Error in pure {self.func.__name__}: {repr(e)}")
                    self._validation_state = NodeValidationState.error
                    self._validation_message = f"{repr(e)}"
        else:
            if port.index > 0:
                field_name, _ = self.arguments[port.index]
                self.inputs[field_name] = data
            else:
                if data is not None:
                    newclock = data.clock
                    if newclock > self._clock.clock:
                        exception = None
                        try:
                            self._clock = ClockCycle(newclock)
                            self.execute()
                            self._validation_state = NodeValidationState.valid
                            self._validation_message = ""

                            ep = ExceptionPasser()
                            if ep.has_exception():
                                return

                            self.data_updated.emit(0)
                        except Exception as e:
                            logger.error(f"Error in {self.func.__name__}: {repr(e)}")
                            import traceback
                            logger.error(traceback.format_exc())

                            self._validation_state = NodeValidationState.error
                            self._validation_message = f"{repr(e)}"
                            exception = repr(e)

                        if exception is not None:
                            ep = ExceptionPasser()
                            ep.set(exception)
                            return

    def execute(self):
        #args = [self.inputs[field].data for field in list(self.arguments.keys())[:1]]
        kwargs = {}
        for k, v in self.inputs.items():
            kwargs[k] = v.data

        if self.__class__.has_single_string and self._single_string_val is not None and self._single_string_val.strip() != "":
            kwargs[self.__class__.has_single_string_arg_name] = self._single_string_val.strip()

        if self.__class__.has_args:
            args = []
            for k, v in self.inputs.items():
                args.append(v.data)
            result = self.__class__.func(*args)
        else:
            result = self.__class__.func(**kwargs)

        if result is not None:
            if not isinstance(result, tuple):
                result = [result]
            else:
                result = list(result)

            # plus 1 because port 0 is exec
            for i, val in enumerate(result):
                self._data[i + self.extra_pins] = val
                self.data_updated.emit(i + self.extra_pins)

        if not self.pure:
            self.data_updated.emit(0)  # Trigger execution output

    def out_data(self, port: int) -> NodeData:
        if self.pure:
            out = VarData(self._data[port])
            out.set_data_type(self.return_vals[port][1])
            return out
        else:
            if port > 0:
                out = VarData(self._data[port])
                out.set_data_type(self.return_vals[port][1])
                return out
            else:
                return ClockCycle(self._clock.clock)

    def save(self) -> dict:
        doc = super().save()
        doc['visible_args'] = self.visible_args
        doc['single_string_argument'] = ""
        if self.__class__.has_single_string:
            doc['single_string_argument'] = self._line_edit.text()

        return doc

    def restore(self, state: dict):
        if "visible_args" in state:
            self.visible_args = state["visible_args"]
            self.refresh_visible_args()
        if self.__class__.has_single_string and 'single_string_argument' in state:
            self._single_string_val = state['single_string_argument']
            if self._is_gui:
                self._line_edit.setText(state['single_string_argument'])

    """
    def save(self) -> dict:
        doc = super().save()
        if self._data:
            doc['data'] = {field: self.inputs[field].data for field in self.arguments.keys()}
        return doc

    def restore(self, state: dict):
        try:
            value = state["data"]
        except KeyError:
            ...
        else:
            self.inputs = {field: NodeData(value[field]) for field in self.arguments.keys()}
            self.data_updated.emit(0)
    """


