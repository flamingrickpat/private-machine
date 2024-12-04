import copy
import os.path
import queue
import sys
import logging
import traceback
import math
import contextlib
import logging
import threading
import json
from typing import Any

from PyQt5.QtGui import QDrag, QColor, QFont, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QSplitter, QWidget, QTextEdit, \
    QMenuBar, QAction, QFileDialog, QPushButton, QMenu, QWidgetAction, QTreeWidget, QTreeWidgetItem, QAbstractItemView, QGraphicsScene, QGraphicsItem
from PyQt5.QtCore import pyqtRemoveInputHook, QPoint, QMimeData, pyqtSignal, QObject, QRectF, QRect
import PyQt5
from PyQt5.QtCore import Qt as Qt
from qtpy.QtGui import QDoubleValidator, QTextCursor
from qtpy.QtWidgets import QApplication, QLabel, QLineEdit, QWidget, QDockWidget
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QWidget, QShortcut, QLabel, QApplication, QHBoxLayout
import qtpynodeeditor.node
import qtpynodeeditor as nodeeditor
from qtpynodeeditor import PortType
from qtpynodeeditor.type_converter import TypeConverter
from qtpy.QtGui import QPainter
from qtpy.QtWidgets import QWidget

from pm.grapheditor.editor_hacks import ImprovedNodeEditorView
from pm.grapheditor.common import VarData, get_node_data_type, to_any_converter, get_from_any_converter
from pm.grapheditor.type_int import NumberSourceDataModel, NumberDisplayModel, decimal_to_integer_converter, integer_to_decimal_converter
from pm.grapheditor.type_str import StringDisplayModel, StringSourceDataModel
from pm.grapheditor.type_bool import BooleanSourceDataModel, BooleanDisplayModel
from pm.grapheditor.type_pydantic import PydanticNodeDataModel
from pm.grapheditor.type_tool import ToolJsonNodeDataModel, ToolPythonNodeDataModel
from pm.grapheditor.func_print import pretty_print_func_node
from pm.grapheditor.func_any import FunctionNodeDataModel
from pm.grapheditor.func_root import RootNode
from pm.grapheditor.func_common import concat_string
from pm.grapheditor.func_exec_flow import SwitchNode, SequenceNode, IfNotNone, ForEachNode, TryCatch
from pm.grapheditor.type_registry import global_pydantic_classes, global_func_classes, global_enum_classes, global_pure_func_classes

logger = logging.getLogger(__name__)

TITLE = "private-machine graph editor"
VISUAL_NODE_DICT = {
    "Primitive": [NumberSourceDataModel, StringSourceDataModel, BooleanSourceDataModel],
    # "Primitive Display": [NumberDisplayModel, StringDisplayModel, BooleanDisplayModel],
    "DataModels": list(global_pydantic_classes.values()),
    "ExecFlow": [RootNode, SwitchNode, SequenceNode, IfNotNone, ForEachNode, TryCatch],
    "Functions": list(global_func_classes.values()),
    "Enums": list(global_enum_classes.values()),
    "Pure Functions": [pretty_print_func_node, ] + list(global_pure_func_classes.values()),
    "Tools": [ToolJsonNodeDataModel, ToolPythonNodeDataModel]
}

class EmptyTitleBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(1)  # Set height to 1 pixel to minimize space

class CustomTreeWidget(QTreeWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

    def startDrag(self, supportedActions):
        item = self.currentItem()
        if item and item.data(0, Qt.UserRole) != "skip me":
            drag = QDrag(self)
            mime_data = QMimeData()
            mime_data.setText(item.data(0, Qt.UserRole))
            drag.setMimeData(mime_data)
            drag.exec_(Qt.MoveAction)

class NodeEditorApp(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        #if io_router is not None:
        #    ChatbotInterface.__init__(self, io_router)

        self.scene, self.view, self.nodes = self.init_ui()
        self.view.setAcceptDrops(True)
        self.setCentralWidget(self.view)
        self.create_menu_bar()
        self.create_node_dock()
        self.cur_filename = None
        self.resize(1280, 1024)
        self.refresh_title()

    def refresh_title(self):
        self.setWindowTitle(f"{TITLE} - {'Not saved!' if self.cur_filename is None else self.cur_filename}")

    def init_ui(self):
        registry = nodeeditor.DataModelRegistry()
        for k, v in VISUAL_NODE_DICT.items():
            for m in v:
                registry.register_model(m, category=k, style=None)

        from pm.grapheditor.type_int import decimal_to_integer_converter, integer_to_decimal_converter
        dec_converter = TypeConverter(get_node_data_type(float), get_node_data_type(int), decimal_to_integer_converter)
        int_converter = TypeConverter(get_node_data_type(int), get_node_data_type(float), integer_to_decimal_converter)
        registry.register_type_converter(get_node_data_type(float), get_node_data_type(int), dec_converter)
        registry.register_type_converter(get_node_data_type(int), get_node_data_type(float), int_converter)

        scene = nodeeditor.FlowScene(registry=registry)
        view = ImprovedNodeEditorView(scene)

        return scene, view, []

    def create_menu_bar(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu('File')

        load_action = QAction('Load (Ctrl+L)', self)
        load_action.triggered.connect(self.load_json)
        file_menu.addAction(load_action)

        load_add_action = QAction('Add existing graph', self)
        load_add_action.triggered.connect(self.load_json_additional)
        file_menu.addAction(load_add_action)

        save_action = QAction('Save (Ctrl+S)', self)
        save_action.triggered.connect(self.save_overwrite)

        save_action2 = QAction('Save As', self)
        save_action2.triggered.connect(self.save_as)

        export_action = QAction('Export PNG', self)
        export_action.triggered.connect(self.export_png)

        self.shortcut_load = QShortcut(QKeySequence("Ctrl+L"), self)
        self.shortcut_load.activated.connect(self.load_json)



        self.shortcut_save = QShortcut(QKeySequence("Ctrl+S"), self)
        self.shortcut_save.activated.connect(self.save_overwrite)

        file_menu.addAction(save_action)
        file_menu.addAction(save_action2)
        file_menu.addAction(export_action)

    def chat_format(self):
        self.chat_output.setReadOnly(False)
        self.chat_output.selectAll()
        self.chat_output.setFontPointSize(13)
        tc = self.chat_output.textCursor()
        tc.movePosition(QTextCursor.MoveOperation.End)
        self.chat_output.setTextCursor(tc)
        self.chat_output.setReadOnly(True)

    def create_log_and_chat_interface(self):
        # Create log output
        self.log_output = QTextEdit(self)
        self.log_output.setReadOnly(True)
        self.log_output.setFontFamily("Consolas")
        self.log_output.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)

        log_layout = QVBoxLayout()
        log_layout.addWidget(self.log_output)

        log_widget = QWidget()
        log_widget.setLayout(log_layout)

        log_dock = QDockWidget("Log Output", self)
        log_dock.setTitleBarWidget(EmptyTitleBar(log_dock))
        log_dock.setWidget(log_widget)

        # Create chat interface
        self.chat_output = QTextEdit(self)
        self.chat_format()

        self.chat_input = QLineEdit(self)
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)

        chat_input_layout = QHBoxLayout()
        chat_input_layout.addWidget(self.chat_input)
        chat_input_layout.addWidget(self.send_button)

        chat_layout = QVBoxLayout()
        chat_layout.addWidget(self.chat_output)
        chat_layout.addLayout(chat_input_layout)

        chat_widget = QWidget()
        chat_widget.setLayout(chat_layout)

        chat_dock = QDockWidget("Chat Interface", self)
        chat_dock.setTitleBarWidget(EmptyTitleBar(chat_dock))
        chat_dock.setWidget(chat_widget)

        # Splitter to divide log and chat
        splitter = QSplitter(PyQt5.QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(log_dock)
        splitter.addWidget(chat_dock)

        dock = QDockWidget()
        dock.setTitleBarWidget(EmptyTitleBar(dock))
        dock.setWidget(splitter)
        self.addDockWidget(PyQt5.QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, dock)

        # Redirect logging to the QTextEdit
        class QTextEditLogger(logging.Handler):
            def __init__(self, parent):
                super().__init__()
                self.widget = parent

            def emit(self, record):
                msg = self.format(record)
                self.widget.append(msg)

        log_handler = QTextEditLogger(self.log_output)
        log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        log_handler.setLevel(logging.WARNING)
        logging.getLogger().addHandler(log_handler)

    def send_message(self):
        message = self.chat_input.text()
        self.append_chat_message(message, True)
        con = Controller()
        con.input_queue.put(message)
        con.tick_delegate()
        msgs = []
        while True:
            try:
                msgs.append(con.output_queue.get(block=False))
            except queue.Empty:
                break
        if len(msgs) == 0:
            msg = "No output!"
        else:
            msg = "\n".join(msgs)
        self.append_chat_message(msg, False)

    def append_chat_message(self, message, from_user):
        color = "lightblue" if from_user else "lightgreen"
        self.chat_output.append(f'<p style="background-color:{color};">{message}</p>')
        self.chat_format()
        self.chat_output.verticalScrollBar().setValue(self.chat_output.verticalScrollBar().maximum())

    def create_node_dock(self):
        dock = QDockWidget("Nodes", self)
        dock.setTitleBarWidget(EmptyTitleBar(dock))
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        container = QWidget()
        layout = QVBoxLayout()

        txt_box = QLineEdit()
        txt_box.setPlaceholderText("Filter")
        txt_box.setClearButtonEnabled(True)
        layout.addWidget(txt_box)

        self.tree_view = CustomTreeWidget()
        self.tree_view.setHeaderHidden(True)
        self.tree_view.setDragEnabled(True)
        self.tree_view.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree_view.setDragDropMode(QAbstractItemView.DragOnly)
        layout.addWidget(self.tree_view)

        container.setLayout(layout)
        dock.setWidget(container)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

        self.populate_tree_view()

        txt_box.textChanged.connect(self.filter_tree_view)

    def populate_tree_view(self):
        top_level_items = {}
        for cat in sorted(self.scene.registry.categories()):
            item = QTreeWidgetItem(self.tree_view)
            item.setText(0, cat)
            item.setData(0, Qt.UserRole, "skip me")
            top_level_items[cat] = item

        registry = self.scene.registry
        for model, category in dict(sorted(registry.registered_models_category_association().items())).items():
            parent_item = top_level_items[category]
            item = QTreeWidgetItem(parent_item)
            item.setText(0, model)
            item.setData(0, Qt.UserRole, model)

        self.tree_view.expandAll()

    def filter_tree_view(self, text):
        for i in range(self.tree_view.topLevelItemCount()):
            top_lvl_item = self.tree_view.topLevelItem(i)
            for j in range(top_lvl_item.childCount()):
                child = top_lvl_item.child(j)
                model_name = child.data(0, Qt.UserRole)
                child.setHidden(text not in model_name)

    def load_json(self):
        self.scene.clear_scene()
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Load JSON File", ".", "JSON Files (*.json);;All Files (*)",
                                                   options=options)
        self.cur_filename = file_name
        self.refresh_title()
        self.load_json_from_file(file_name)

    def load_json_additional(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Load JSON File", ".", "JSON Files (*.json);;All Files (*)",
                                                   options=options)
        self.load_json_from_file(file_name)

    def load_json_from_file(self, file_name):
        gos = []
        if file_name:
            with open(file_name, 'r') as file:
                json_data = file.read()
                obj = json.loads(json_data)

                node_dict = {}
                for k, v in obj["nodes"].items():
                    class_name = v["state"]["model"]["name"]

                    cls = None
                    if class_name in globals():
                        cls = globals()[class_name]

                    from pm.grapheditor.get_cls import get_cls
                    cls = get_cls(class_name)
                    if cls is not None:
                        node = self.scene.restore_node(v["state"])
                        node_dict[v["state"]["id"]] = node

                        if v["state"]["id"] != node.id:
                            raise Exception("ID mismatch!")
                        node_dict[node.id] = node
                        gos.append(node.graphics_object)
                    else:
                        logger.error(f"Can't find class {class_name}, skipping node!")

                for k, v in obj["connections"].items():
                    node_a = node_dict.get(v["in_id"], None)
                    node_b = node_dict.get(v["out_id"], None)
                    try:
                        con = self.scene.create_connection_by_index(node_a, v["in_index"], node_b, v["out_index"], None)
                        #con.graphics_object.setCacheMode(QGraphicsItem.ItemCoordinateCache)
                    except Exception as e:
                        logger.error(f"Error creating node: {e}")

        for i in range(4):
            for k, v in self.scene.nodes.items():
                try:
                    tmp: qtpynodeeditor.node.Node = v
                    for idx, port in tmp[PortType.output].items():
                        try:
                            tmp.on_data_updated(port)
                        except Exception as e:
                            pass
                except Exception as e:
                    pass
        self.scene.clearSelection()
        for go in gos:
            #go.setCacheMode(QGraphicsItem.ItemCoordinateCache)
            go.setSelected(True)
        #self.scene.selectionChanged()

    def save_overwrite(self):
        self.save_json(self.cur_filename)

    def save_as(self):
        self.save_json(None)

    def save_json(self, filename: str = None):
        nodes = {}
        for k, v in self.scene.nodes.items():
            state = v.__getstate__()
            nodes[v.id] = {
                "state": state,
            }

        cons = {}
        for con in self.scene.connections:
            cons[con.id] = con.__getstate__()

        all = {
            "nodes": nodes,
            "connections": cons
        }
        js = json.dumps(
            all,
            sort_keys=True,
            indent=4
        )

        if filename is None or not os.path.exists(filename):
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getSaveFileName(self, "Save JSON File", "", "JSON Files (*.json);;All Files (*)", options=options)
        else:
            file_name = self.cur_filename

        if file_name:
            self.cur_filename = file_name
            self.refresh_title()
            with open(file_name, 'w') as file:
                file.write(js)

    def get_scene_extents(self):
        scene = self.scene
        items = scene.items()  # get all items in the scene
        if not items:
            return None
        # Initialize min and max x, y with the first item
        min_x, max_x = items[0].scenePos().x(), items[0].scenePos().x() + items[0].boundingRect().width()
        min_y, max_y = items[0].scenePos().y(), items[0].scenePos().y() + items[0].boundingRect().height()

        # Update min and max x, y for each item in the scene
        for item in items:
            min_x = min(item.scenePos().x(), min_x)
            max_x = max(item.scenePos().x() + item.boundingRect().width(), max_x)
            min_y = min(item.scenePos().y(), min_y)
            max_y = max(item.scenePos().y() + item.boundingRect().height(), max_y)

        return QRectF(min_x, min_y, max_x - min_x, max_y - min_y)
    def export_png(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save PNG File", "", "PNG Files (*.png);;All Files (*)", options=options)
        scale_factor = 2

        scene = self.scene

        rect = self.get_scene_extents()

        adjusted_rect = QRectF(0, 0, abs(rect.x()) + rect.width(), abs(rect.y()) + rect.height())

        image_size = adjusted_rect.size().toSize() * scale_factor
        pixmap = QImage(image_size, QImage.Format_ARGB32)
        pixmap.setDotsPerMeterX(2835 * scale_factor)  # Adjust DPI here 2835 dots per meter is equivalent to 72 DPI
        pixmap.setDotsPerMeterY(2835 * scale_factor)  # Adjust DPI here

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        # move the painter to account for negative coordinates
        painter.translate(abs(rect.x() * scale_factor), abs(rect.y() * scale_factor))

        # Adjust target rect to the translated painter
        source_rect = QRect(QPoint(), image_size)

        # Render scene into pixmap
        scene.render(painter, QRectF(source_rect), rect)

        pixmap.save(file_name, "PNG")
        painter.end()

    def init_system(self):
        pass

    @classmethod
    def run(cls, architecture_path: str = None, direct_chat: bool = False):
        pyqtRemoveInputHook()
        exception_handler = GlobalExceptionHandler()
        sys.excepthook = exception_handler.handler
        QApplication.setAttribute(Qt.AA_UseDesktopOpenGL)
        app = QApplication([])
        app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        font = QFont("Arial", 10)  # Increase the size
        app.setFont(font)

        editor_app = NodeEditorApp()
        editor_app.init_system()
        if direct_chat:
            editor_app.create_log_and_chat_interface()

        if architecture_path is not None:
            editor_app.load_json_from_file(architecture_path)
            editor_app.cur_filename = architecture_path
            editor_app.refresh_title()
        try:
            editor_app.show()
        except Exception as e:
            logger.error(e)
            editor_app.save_overwrite()

        sys.exit(app.exec_())


class GlobalExceptionHandler(QObject):
    exception_caught = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.original_excepthook = sys.excepthook

    def handler(self, exctype, value, tb):
        exception_string = ''.join(traceback.format_exception(exctype, value, tb))
        logging.error(f"Uncaught exception:\n{exception_string}")
        self.exception_caught.emit(str(value))

if __name__ == '__main__':
    NodeEditorApp.run()