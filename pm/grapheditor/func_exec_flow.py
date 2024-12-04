import inspect
from typing import Callable
import logging
from typing import Any
import threading

from qtpynodeeditor import NodeData, NodeDataModel, NodeDataType, PortType, NodeValidationState

from pm.grapheditor.base_node import BaseNode, dual_class
from pm.grapheditor.common import get_node_data_type, ClockCycle, style_node, VarData, TickCounter, ExceptionPasser


logger = logging.getLogger(__name__)

@dual_class(BaseNode)
class SwitchNode(NodeDataModel):
    port_caption_visible = True
    caption = "Switch"

    num_ports = {
        PortType.input: 2,
        PortType.output: 2
    }

    port_caption = {
        'input': {
            0: "Exec In",
            1: "switch: bool"
        },
        'output': {
            0: "Exec True",
            1: "Exec False"
        },
    }

    data_types = {
        'input': {
            0: get_node_data_type("exec"),
            1: get_node_data_type(bool)
        },
        'output': {
            0: get_node_data_type("exec"),
            1: get_node_data_type("exec")
        },
    }
    data_type = data_types

    def __init__(self, style=None, parent=None):
        tc = TickCounter()
        self._clock = ClockCycle(tc.get())
        self._clock_true = ClockCycle(tc.get())
        self._clock_false = ClockCycle(tc.get())

        if self._is_gui:
            super().__init__(style=style, parent=parent)
            style_node("black", self)

        self._switch = VarData(False)

    def set_in_data(self, data: NodeData, port):
        if port.index > 0:
            self._switch.set_data(data.data if data is not None else False)
        else:
            newclock = data.clock if data is not None else 0
            if newclock > self._clock.clock:
                self._clock = ClockCycle(newclock)

                if self._switch.data:
                    self._clock_true = ClockCycle(newclock)
                    self.data_updated.emit(0)
                else:
                    self._clock_false = ClockCycle(newclock)
                    self.data_updated.emit(1)

    def out_data(self, port: int) -> NodeData:
        if port == 0:
            return ClockCycle(self._clock_true.clock)
        else:
            return ClockCycle(self._clock_false.clock)





@dual_class(BaseNode)
class SequenceNode(NodeDataModel):
    port_caption_visible = True
    caption = "Sequence"

    num_ports = {
        PortType.input: 1,
        PortType.output: 4
    }

    seq_cnt = 4

    port_caption = {
        'input': {
            0: "Exec In",
        },
        'output': {
            0: "Exec 0",
            1: "Exec 1",
            2: "Exec 2",
            3: "Exec 3",
        },
    }

    data_types = {
        'input': {
            0: get_node_data_type("exec"),
        },
        'output': {
            0: get_node_data_type("exec"),
            1: get_node_data_type("exec"),
            2: get_node_data_type("exec"),
            3: get_node_data_type("exec")
        },
    }
    data_type = data_types

    def __init__(self, style=None, parent=None):
        tc = TickCounter()
        self._clock = ClockCycle(tc.get())

        if self._is_gui:
            super().__init__(style=style, parent=parent)
            style_node("black", self)

        self._switch = VarData(False)

    def set_in_data(self, data: NodeData, port):
        newclock = -1 if data is None else data.clock
        if newclock > self._clock.clock:
            self._clock = ClockCycle(newclock)

            self.data_updated.emit(0)
            self.data_updated.emit(1)
            self.data_updated.emit(2)
            self.data_updated.emit(3)

    def out_data(self, port: int) -> NodeData:
        return ClockCycle(self._clock.clock)



@dual_class(BaseNode)
class IfNotNone(NodeDataModel):
    port_caption_visible = True
    caption = "IfNotNone"

    num_ports = {
        PortType.input: 2,
        PortType.output: 2
    }

    port_caption = {
        'input': {
            0: "Exec In",
            1: "switch: Any"
        },
        'output': {
            0: "Exec Is Not None",
            1: "Exec Is None"
        },
    }

    data_types = {
        'input': {
            0: get_node_data_type("exec"),
            1: get_node_data_type("any")
        },
        'output': {
            0: get_node_data_type("exec"),
            1: get_node_data_type("exec")
        },
    }
    data_type = data_types

    def __init__(self, style=None, parent=None):
        tc = TickCounter()
        self._clock = ClockCycle(tc.get())

        self._clock_true = ClockCycle(tc.get())
        self._clock_false = ClockCycle(tc.get())

        if self._is_gui:
            super().__init__(style=style, parent=parent)
            style_node("black", self)

        self._switch = VarData()

    def set_in_data(self, data: NodeData, port):
        if port.index > 0:
            self._switch.set_data(data.data if data is not None else False)
        else:
            newclock = data.clock if data is not None else 0
            if newclock > self._clock.clock:
                self._clock = ClockCycle(newclock)

                if self._switch.data is not None:
                    self._clock_true = ClockCycle(newclock)
                    self.data_updated.emit(0)
                else:
                    self._clock_false = ClockCycle(newclock)
                    self.data_updated.emit(1)

    def out_data(self, port: int) -> NodeData:
        if port == 0:
            return ClockCycle(self._clock_true.clock)
        else:
            return ClockCycle(self._clock_false.clock)


@dual_class(BaseNode)
class ForEachNode(NodeDataModel):
    port_caption_visible = True
    caption = "ForEach"

    num_ports = {
        PortType.input: 2,
        PortType.output: 2
    }

    port_caption = {
        'input': {
            0: "Exec In",
            1: "Enumerable: Any"
        },
        'output': {
            0: "Exec Out",
            1: "Item: Any"
        },
    }

    data_types = {
        'input': {
            0: get_node_data_type("exec"),
            1: get_node_data_type("any")
        },
        'output': {
            0: get_node_data_type("exec"),
            1: get_node_data_type("any")
        },
    }
    data_type = data_types

    def __init__(self, style=None, parent=None):
        tc = TickCounter()
        self._clock = ClockCycle(tc.get())

        if self._is_gui:
            super().__init__(style=style, parent=parent)
            style_node("black", self)

        self._data = VarData(None)
        self._cur_item = None

    def set_in_data(self, data: NodeData, port):
        if port.index > 0:
            self._data.set_data(data.data if data is not None else None)
        else:
            newclock = data.clock if data is not None else 0
            if newclock > self._clock.clock:
                self._clock = ClockCycle(newclock)

                if self._data.data is not None:
                    for item in self._data.data:
                        self._cur_item = item
                        tc = TickCounter()
                        tc.inc()
                        self._clock = ClockCycle(tc.get())

                        self.data_updated.emit(1)
                        self.data_updated.emit(0)

    def out_data(self, port: int) -> NodeData:
        if port == 0:
            return ClockCycle(self._clock.clock)
        else:
            return VarData(self._cur_item)

@dual_class(BaseNode)
class TryCatch(NodeDataModel):
    port_caption_visible = True
    caption = "TryCatch"

    num_ports = {
        PortType.input: 1,
        PortType.output: 3
    }

    seq_cnt = 3

    port_caption = {
        'input': {
            0: "Exec In",
        },
        'output': {
            0: "Exec",
            1: "Except",
            2: "Finally",
        },
    }

    data_types = {
        'input': {
            0: get_node_data_type("exec"),
        },
        'output': {
            0: get_node_data_type("exec"),
            1: get_node_data_type("exec"),
            2: get_node_data_type("exec")
        },
    }
    data_type = data_types

    def __init__(self, style=None, parent=None):
        tc = TickCounter()
        self._clock = ClockCycle(tc.get())

        if self._is_gui:
            super().__init__(style=style, parent=parent)
            style_node("black", self)

        self._switch = VarData(False)

    def set_in_data(self, data: NodeData, port):
        newclock = -1 if data is None else data.clock
        if newclock > self._clock.clock:
            self._clock = ClockCycle(newclock)

            self.data_updated.emit(0)

            has_except = False
            ep = ExceptionPasser()
            if ep.has_exception():
                logger.error(f"Caught exception: {ep.exception}")
                has_except = True
            ep.reset()

            if has_except:
                self.data_updated.emit(1)
            self.data_updated.emit(2)

    def out_data(self, port: int) -> NodeData:
        return ClockCycle(self._clock.clock)
