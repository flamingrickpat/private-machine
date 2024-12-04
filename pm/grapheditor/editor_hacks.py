import copy
import logging
from typing import Any

from PyQt5.QtCore import QRectF, QPointF, Qt
from PyQt5.QtGui import QColor, QBrush
from PyQt5.QtWidgets import QGraphicsItem
from qtpy.QtCore import QRectF
from qtpy.QtGui import (QPainter)
from qtpy.QtWidgets import (QGraphicsView)
from qtpy.QtWidgets import (QStyleOptionGraphicsItem,
                            QWidget)
from qtpynodeeditor import (NodeDataModel, NodeDataType, DataModelRegistry,
                            PortType, NodeStyle, Port, Connection, ConnectionDataTypeFailure)
from qtpynodeeditor.connection_graphics_object import ConnectionGraphicsObject
from qtpynodeeditor.connection_painter import ConnectionPainter
from qtpynodeeditor.flow_view import FlowView
from qtpynodeeditor.node_graphics_object import NodeGraphicsObject
from qtpynodeeditor.type_converter import TypeConverter
from qtpynodeeditor.connection_geometry import ConnectionGeometry
from qtpynodeeditor.node_geometry import NodeGeometry
from qtpynodeeditor.node_state import NodeState
from qtpynodeeditor.node_painter import NodePainter
from qtpynodeeditor.flow_scene import FlowScene
from qtpynodeeditor.node import Node

from pm.grapheditor.common import VarData, get_node_data_type, to_any_converter, get_from_any_converter
from pm.grapheditor.decorator import global_pydantic_classes

logger = logging.getLogger(__name__)

all_node_graphic_objects = []


def should_be_visible(node, area):
    return (area.contains(node.position) or
            area.contains(node.position + QPointF(node.size.width(), node.size.height())) or
            area.contains(node.position + QPointF(0, node.size.height())) or
            area.contains(node.position + QPointF(node.size.width(), 0)))


class ImprovedNodeEditorView(FlowView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.setViewportUpdateMode(QGraphicsView.BoundingRectViewportUpdate)
        self.setCacheMode(QGraphicsView.CacheBackground)
        self.setRenderHint(QPainter.Antialiasing, False)
        self.setRenderHint(QPainter.TextAntialiasing, False)

    def calculate_visible_area(self):
        world_space_rect = self.mapToScene(self.viewport().rect()).boundingRect()
        return world_space_rect

    def paintEvent(self, p):
        visible_area = self.calculate_visible_area()
        added = False
        for item in all_node_graphic_objects:
            if item is None or item.graphics_object is None:
                continue

            if item.graphics_object not in self.items() and should_be_visible(item, visible_area):
                # The item is not in the scene and should be visible, add it
                self._scene.addItem(item.graphics_object)
                item.graphics_object.update()
                added = True
            elif item.graphics_object in self.items() and not should_be_visible(item, visible_area):
                # The item is in the scene but shouldn't be visible, remove it
                self._scene.removeItem(item.graphics_object)
        if added:
            self.scale(1.0001, 1.0001)
        super().paintEvent(p)

    def drawBackground(self, painter: QPainter, r: QRectF):
        """
        drawBackground

        Parameters
        ----------
        painter : QPainter
        r : QRectF
        """
        # Define a dark grey color
        dark_grey = QColor(40, 40, 40)  # RGB values for dark grey

        # Set the brush to the dark grey color
        painter.setBrush(QBrush(dark_grey))

        # Disable the pen (we don't need an outline)
        painter.setPen(Qt.NoPen)

        # Draw a rectangle covering the entire background
        painter.drawRect(r)

def get_type_converter(self, d1: NodeDataType, d2: NodeDataType) -> TypeConverter:
    if d1.id == "any" and d2.id != "any":
        conv = get_from_any_converter(d2)
        if conv is not None:
            return TypeConverter(get_node_data_type(Any), d2, conv)

    if d2.id == "any" and d1.id != "any":
        return TypeConverter(d1, get_node_data_type(Any), to_any_converter)

    class_from = None
    class_to = None
    if d1.name in global_pydantic_classes:
        class_from = global_pydantic_classes[d1.name]
    if d2.name in global_pydantic_classes:
        class_to = global_pydantic_classes[d2.name]
    if class_from is not None and class_to is not None:
        if issubclass(class_from.model_class, class_to.model_class):
            def to_baseclass_converter(nodedata) -> VarData:
                return VarData(nodedata.data, d2)

            return TypeConverter(d1, d2, to_baseclass_converter)

    return self.type_converters.get((d1, d2), None)


DataModelRegistry.get_type_converter = get_type_converter


def categories(self):
    return sorted(self._categories)


DataModelRegistry.categories = categories


def dragEnterEvent(self, event):
    if event.mimeData().hasText():
        event.acceptProposedAction()


FlowView.dragEnterEvent = dragEnterEvent


def dragMoveEvent(self, event):
    if event.mimeData().hasText():
        event.acceptProposedAction()


FlowView.dragMoveEvent = dragMoveEvent


def dropEvent(self, event):
    if event.mimeData().hasText():
        model_name = event.mimeData().text()
        try:
            model, _ = self._scene.registry.get_model_by_name(model_name)
            node = self._scene.create_node(model)
            pos = self.mapToScene(event.pos())
            node.graphics_object.setPos(pos)
            self._scene.node_placed.emit(node)
            event.acceptProposedAction()
        except ValueError:
            logger.error("Model not found: %s", model_name)


FlowView.dropEvent = dropEvent


def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget):
    from qtpynodeeditor.node_painter import NodePainter
    world_space_rect = self._scene.views()[0].mapToScene(self._scene.views()[0].viewport().rect()).boundingRect()

    if not should_be_visible(self._node, world_space_rect):
        return

    painter.setClipRect(option.exposedRect)
    NodePainter.paint(painter, self._node, self._scene,
                      node_style=self._style.node,
                      connection_style=self._style.connection,
                      )



def paint2(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget):
    from qtpynodeeditor.node_painter import NodePainter

    # Get the current view
    view = self._scene.views()[0]
    world_space_rect = view.mapToScene(view.viewport().rect()).boundingRect()

    if not should_be_visible(self._node, world_space_rect):
        return

    painter.setClipRect(option.exposedRect)

    # Calculate the node's bounding rect in view coordinates
    view_rect = view.mapFromScene(self.boundingRect()).boundingRect()

    # Check if the node is smaller than 100x100 pixels in view space
    if view_rect.width() < 100 and view_rect.height() < 100:
        # Hide the embedded widget if it exists
        if hasattr(self, '_proxy_widget') and self._proxy_widget is not None:
            self._proxy_widget.hide()

        # Paint a simplified rectangle
        painter.setPen(Qt.NoPen)
        color = self._style.node.gradient_colors[0][1]
        painter.setBrush(color)
        painter.drawRect(self.boundingRect())
    else:
        # Paint the full node
        if hasattr(self, '_proxy_widget') and self._proxy_widget is not None:
            self._proxy_widget.show()

        NodePainter.paint(painter, self._node, self._scene,
                          node_style=self._style.node,
                          connection_style=self._style.connection)


NodeGraphicsObject.paint = paint2


@property
def bounding_rect(self) -> QRectF:
    return QRectF(self._out, self._in).normalized()


#ConnectionGeometry.bounding_rect = bounding_rect

dt_to_style_map = {}

caption_height_map = {}


@property
def caption_height(self) -> int:
    if not self._model.caption_visible:
        return 0

    name = self._model.caption
    if name in caption_height_map:
        return caption_height_map[name]
    else:
        cached_height = self._bold_font_metrics.boundingRect(name).height()
        caption_height_map[name] = cached_height
        return cached_height


NodeGeometry.caption_height = caption_height


def paint_color_datatype_length(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget):
    def guid_to_pastel_color(guid_str, length):
        import colorsys
        # Use the hash of the UUID to generate a hue
        hash_value = hash(guid_str)
        hue = (hash_value % 360) / 360.0  # Normalize to [0, 1]
        if guid_str == "exec":
            hue = 0
        saturation = 0.8
        value = 0.65  # High value for brightness
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        r, g, b = int(r * 255), int(g * 255), int(b * 255)
        return f"#{r:02x}{g:02x}{b:02x}"

    painter.setClipRect(option.exposedRect)

    style = self._style
    try:
        dt = self._connection.valid_ports['input'].data_type.id
    except:
        dt = "any"
    if dt in dt_to_style_map:
        style = dt_to_style_map[dt]
    else:
        s = self._style
        try:
            col = guid_to_pastel_color(dt, 0)
            style = copy.deepcopy(self._style)
            style.normal_color = QColor(col)
            dt_to_style_map[dt] = style
        except:
            pass
    ConnectionPainter.paint(painter, self._connection, style)


ConnectionGraphicsObject.paint = paint_color_datatype_length

rect_map = {}


@staticmethod
def draw_entry_labels(painter: QPainter, geom: NodeGeometry,
                      state: NodeState, model: NodeDataModel,
                      node_style: NodeStyle):
    metrics = painter.fontMetrics()

    for port in state.ports:
        scene_pos = port.scene_position
        if not port.connections:
            painter.setPen(node_style.font_color_faded)
        else:
            painter.setPen(node_style.font_color)

        display_text = port.display_text
        if display_text in rect_map:
            rect = rect_map[display_text]
        else:
            rect = metrics.boundingRect(display_text)
            rect_map[display_text] = rect
        scene_pos.setY(scene_pos.y() + rect.height() / 4.0)
        if port.port_type == PortType.input:
            scene_pos.setX(5.0)
        elif port.port_type == PortType.output:
            scene_pos.setX(geom.width - 5.0 - rect.width())

        painter.drawText(scene_pos, display_text)


NodePainter.draw_entry_labels = draw_entry_labels


def create_node(self, data_model: NodeDataModel) -> Node:
    with self._new_node_context(data_model.name) as node:
        ngo = NodeGraphicsObject(self, node)
        node.graphics_object = ngo
    all_node_graphic_objects.append(node)
    return node


def restore_node(self, node_json: dict) -> Node:
    with self._new_node_context(node_json["model"]["name"]) as node:
        node.graphics_object = NodeGraphicsObject(self, node)
        node.__setstate__(node_json)
    all_node_graphic_objects.append(node)
    return node


FlowScene.create_node = create_node
FlowScene.restore_node = restore_node


def create_connection(self, port_a: Port, port_b: Port = None, *,
                      converter: TypeConverter = None,
                      check_cycles=True) -> Connection:
    """
    Create a connection

    Parameters
    ----------
    port_a : Port
        The first port, either input or output
    port_b : Port, optional
        The second port, opposite of the type of port_a
    converter : TypeConverter, optional
        The type converter to use for data propagation
    check_cycles : bool, optional
        Ensures that creating the connection would not introduce a cycle

    Returns
    -------
    value : Connection

    Raises
    ------
    NodeConnectionFailure
        If it is not possible to create the connection
    ConnectionDataTypeFailure
        If port data types are not compatible
    """
    if port_a is not None and port_b is not None:
        in_port = port_a if port_a.port_type == PortType.input else port_b
        out_port = port_b if port_a.port_type == PortType.input else port_a
        if in_port.data_type.id != out_port.data_type.id:
            if not converter:
                # If not specified, try to get it from the registry
                converter = self.registry.get_type_converter(out_port.data_type,
                                                             in_port.data_type)
            if (not converter or (converter.type_in != out_port.data_type
                                  or converter.type_out != in_port.data_type)):
                raise ConnectionDataTypeFailure(
                    f'{in_port.data_type} and {out_port.data_type} are not compatible'
                )
    connection = Connection(port_a=port_a, port_b=port_b, style=self._style, converter=converter)
    if port_a is not None:
        port_a.add_connection(connection)

    if port_b is not None:
        port_b.add_connection(connection)

    if port_a and port_b and check_cycles:
        # In the case of a fully-specified connection, ensure adding the
        # connection would not create a cycle in the graph.  For
        # partially-specified connections (i.e., one port only), the
        # validation happens in the NodeConnectionInteraction
        node_a, node_b = port_a.node, port_b.node
        if node_a.has_connection_by_port_type(node_b, port_b.port_type):
            raise Exception(
                f'Connecting {node_a} and {node_b} would introduce a '
                f'cycle in the graph'
            )

    cgo = ConnectionGraphicsObject(self, connection)
    cgo.setFlag(QGraphicsItem.ItemIsFocusable, False)
    cgo.setFlag(QGraphicsItem.ItemIsSelectable, False)
    # after self function connection points are set to node port
    connection.graphics_object = cgo
    self._connections.append(connection)

    if port_a and port_b:
        in_port, out_port = connection.ports
        out_port.node.on_data_updated(out_port)
        self.connection_created.emit(connection)

    return connection


FlowScene.create_connection = create_connection