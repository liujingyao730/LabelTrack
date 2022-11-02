import numpy as np
import copy

import cv2
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import GUI.shape as guishape
from GUI.color import *
from GUI.fileworker import fileWorker
from GUI.label_dialog import LabelDialog
from GUI.shape import Shape
from GUI.tools import img_cv_to_qt
from GUI.trackworker import trackWorker
from GUI.utils import *

from GUI.label_dialog import LabelDialog
from GUI.rect_dialog import RectifyDialog

CURSOR_DEFAULT = Qt.ArrowCursor
CURSOR_POINT = Qt.PointingHandCursor
CURSOR_DRAW = Qt.CrossCursor
CURSOR_MOVE = Qt.ClosedHandCursor
CURSOR_GRAB = Qt.OpenHandCursor


class canvas(QWidget):
    newShape = pyqtSignal()
    drawingPolygon = pyqtSignal(bool)
    scrollRequest = pyqtSignal(int, int)
    zoomRequest = pyqtSignal(int)

    CREATE, EDIT, CREATE_ROAD = list(range(3))

    epsilon = 11.0

    def __init__(self, *args, **kwargs):
        super(canvas, self).__init__(*args, **kwargs)
        self.img_off = QPointF(0, 0)
        self.ori_pos = None
        self.trackWorker = trackWorker(self)  # 跟踪线程
        self.trackWorker.sinOut.connect(self.update_track_status)
        self.fileWorker = fileWorker(self)  # 导入文件线程
        self.fileWorker.sinOut.connect(self.update_file_status)
        self.fileWorker.finished.connect(self.load_frames)

        self.numFrames = 0
        self.imgFrames = []  # 储存所有图像帧
        self.curFramesId = 1

        self.pixmap = QPixmap()
        self._painter = QPainter()
        self.drawing_line_color = QColor(0, 0, 255)
        self.drawing_rect_color = QColor(0, 0, 255)
        self.line = Shape(line_color=self.drawing_line_color)
        self.lines = []  # used to save multiple kps in a lane
        self.deltaPos = QPointF()
        self.prev_point = QPointF()
        self.prevRightPoint = QPointF()
        self.offsets = QPointF(), QPointF()
        self.scale = 1.0
        self.current = None
        self.mode = self.EDIT
        self.shapeId = 0
        self.selected_shape = None  # save the selected shape here
        self.shapes = []
        self.h_shape = None
        self.h_vertex = None
        self.window = self.parent().window()
        self.label_dialog = LabelDialog(parent=self, list_item=[])
        # TODO: 初始图片
        self.image = img_cv_to_qt(cv2.imread("./GUI/resources/images/MOT.png"))
        self.load_pixmap(QPixmap.fromImage(self.image))

        # Set widget options.
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.WheelFocus)

    def init_frame(self, path):
        # file worker 线程
        self.fileWorker.load_path(path)
        self.fileWorker.start()

    # finish
    def load_frames(self):
        self.trackWorker.load_frames(self.imgFrames)  # 跟踪加载图片帧
        self.numFrames = len(self.imgFrames)  # 获取视频的总帧数
        frame_0 = self.imgFrames[0]
        Qframe_0 = img_cv_to_qt(frame_0)
        self.load_pixmap(QPixmap.fromImage(Qframe_0))

    def change_frame(self, num):
        n = num - 1
        self.curFramesId = num
        frame_n = self.imgFrames[n]
        Qframe_n = img_cv_to_qt(frame_n)
        self.load_pixmap(QPixmap.fromImage(Qframe_n))

    def track_frame(self):
        self.trackWorker.load_frames(self.imgFrames)
        self.trackWorker.load_model(self.window.currentModel)
        self.trackWorker.start()

    def update_track_status(self, message):
        self.window.statusBar.showMessage(message)
        # if re.search(r'.*(0).+/.*', message):
        #     self.frameLabel.setPixmap(QtGui.QPixmap.fromImage(img_cv_to_qt(self.imgFrames[0])))
        self.window.labelTotalFrame.setText(str(self.numFrames))
        self.window.vedioSlider.setMaximum(self.numFrames)
        self.repaint()

    def update_file_status(self, message):
        self.window.statusBar.showMessage(message)

    def transform_pos(self, point):
        """Convert from widget-logical coordinates to painter-logical coordinates."""
        return point / self.scale - self.offset_to_center() - self.img_off

    # These two, along with a call to adjustSize are required for the
    # scroll area.
    def sizeHint(self):
        return self.minimumSizeHint()

    def minimumSizeHint(self):
        if self.pixmap:
            return self.scale * self.pixmap.size()
        return super(canvas, self).minimumSizeHint()

    def offset_to_center(self):
        s = self.scale
        area = super(canvas, self).size()
        w, h = self.pixmap.width() * s, self.pixmap.height() * s
        aw, ah = area.width(), area.height()
        x = (aw - w) / (2 * s) if aw > w else 0
        y = (ah - h) / (2 * s) if ah > h else 0
        return QPointF(x, y)

    def out_of_pixmap(self, p):
        w, h = self.pixmap.width(), self.pixmap.height()
        return not (0 <= p.x() <= w and 0 <= p.y() <= h)

    def finalise(self):
        assert self.current
        if self.current.points[0] == self.current.points[-1]:
            self.current = None
            self.drawingPolygon.emit(False)
            self.update()
            return

        self.shapeId += 1
        if self.mode is self.CREATE_ROAD:
            self.current.label = self.window.default_label
            self.current.auto = 'L'
        else:
            self.current.label = self.window.default_label
        self.current.id = self.shapeId
        self.current.frameId = self.curFramesId
        self.current.score = 1
        if self.drawing():
            self.current.close()
        self.shapes.append(self.current)
        self.current = None
        # self.set_hiding(False)
        self.newShape.emit()
        self.update()

    def update_shape(self, id, frameId, cls_id, tlwh, score, auto=guishape.MOVING_OBJECT):
        detectPos = Shape()
        detectPos.id = id
        detectPos.frameId = frameId
        label = VISDRONE_CLASSES[cls_id]
        detectPos.label = label
        detectPos.score = score
        detectPos.auto = auto
        generate_line_color, generate_fill_color = generate_color_by_text(
            detectPos.label)
        self.set_shape_label(detectPos, detectPos.label,
                             detectPos.id, generate_line_color, generate_fill_color)
        leftTop = QPointF(tlwh[0], tlwh[1])
        rightTop = QPointF(tlwh[0] + tlwh[2], tlwh[1])
        rightDown = QPointF(tlwh[0] + tlwh[2], tlwh[1] + tlwh[3])
        leftDown = QPointF(tlwh[0], tlwh[1] + tlwh[3])
        pointPos = [leftTop, rightTop, rightDown, leftDown]
        for pos in pointPos:
            if self.out_of_pixmap(pos):
                size = self.pixmap.size()
                clipped_x = min(max(0, pos.x()), size.width())
                clipped_y = min(max(0, pos.y()), size.height())
                pos = QPointF(clipped_x, clipped_y)
            detectPos.add_point(pos)

        detectPos.close()
        self.shapes.append(detectPos)
        detectPos = None
        # self.set_hiding(False)
        self.newShape.emit()
        self.update()

    def delete_shape(self):
        self.current = None
        self.shapeId = 0
        self.selected_shape = None  # save the selected shape here
        self.shapes = []
        self.update()
        self.repaint()

    def load_pixmap(self, pixmap):
        self.pixmap = pixmap
        # self.shapes = []
        self.repaint()

    def drawing(self):
        return self.mode == self.CREATE

    def drawing_road(self):
        return self.mode == self.CREATE_ROAD

    def handle_drawing(self, pos):
        """after press the left button, need to update the label info in the bbox.

        Args:
            pos (_type_): _description_
        """
        if self.drawing():
            if self.current and self.current.reach_max_points() is False:
                init_pos = self.current[0]
                min_x = init_pos.x()
                min_y = init_pos.y()
                target_pos = self.line[1]
                max_x = target_pos.x()
                max_y = target_pos.y()

                self.current.add_point(QPointF(max_x, min_y))
                self.current.add_point(target_pos)
                self.current.add_point(QPointF(min_x, max_y))

                self.finalise()
            elif not self.out_of_pixmap(pos):
                self.current = Shape()
                self.current.add_point(pos)
                self.line.points = [pos, pos]
                # self.set_hiding()
                self.drawingPolygon.emit(True)
                self.update()

        elif self.drawing_road():
            if self.current:
                # the other three points.
                target_pos = self.line[1]
                self.lines.append(copy.deepcopy(self.line))
                self.current.add_point(target_pos)
                self.update()
                self.line[0] = self.line[1]

            elif not self.out_of_pixmap(pos):
                # the left-up point
                self.current = Shape()
                self.current.add_point(pos)
                self.line.points = [pos, pos]
                # self.set_hiding()
                self.drawingPolygon.emit(True)
                self.update()

    def set_editing(self, edit=True):
        """set the canvas status to EDIT.

        warning

        Args:
            edit (bool, optional): If the canvas status should be true. Defaults to True.
        """
        self.mode = self.EDIT if edit else self.CREATE
        if not edit:  # Create
            print(
                'this set_editing should only be used to set EDIT status, so edit should always be true.')
            self.un_highlight()
            self.de_select_shape()
        self.prev_point = QPointF()
        # self.repaint()

    def set_create(self):
        self.mode = self.CREATE
        self.lines = []
        self.line = Shape(line_color=self.drawing_line_color)
        self.un_highlight()
        self.de_select_shape()
        self.prev_point = QPointF()
        self.repaint()

    def set_create_road(self):
        self.mode = self.CREATE_ROAD
        self.lines = []
        self.line = Shape(line_color=self.drawing_line_color)
        self.un_highlight()
        self.de_select_shape()
        self.prev_point = QPointF()
        self.repaint()

    def current_cursor(self):
        cursor = QApplication.overrideCursor()
        if cursor is not None:
            cursor = cursor.shape()
        return cursor

    def override_cursor(self, cursor):
        self._cursor = cursor
        if self.current_cursor() is None:
            QApplication.setOverrideCursor(cursor)
        else:
            QApplication.changeOverrideCursor(cursor)

    def select_shape(self, shape):
        self.de_select_shape()
        shape.selected = True
        self.selected_shape = shape
        # self.set_hiding()
        # self.selectionChanged.emit(True)
        self.update()

    def bounded_move_vertex(self, pos):
        index, shape = self.h_vertex, self.h_shape
        point = shape[index]
        if self.out_of_pixmap(pos):
            size = self.pixmap.size()
            clipped_x = min(max(0, pos.x()), size.width())
            clipped_y = min(max(0, pos.y()), size.height())
            pos = QPointF(clipped_x, clipped_y)

        shift_pos = pos - point

        shape.move_vertex_by(index, shift_pos)

        left_index = (index + 1) % 4
        right_index = (index + 3) % 4
        left_shift = None
        right_shift = None
        if index % 2 == 0:
            right_shift = QPointF(shift_pos.x(), 0)
            left_shift = QPointF(0, shift_pos.y())
        else:
            left_shift = QPointF(shift_pos.x(), 0)
            right_shift = QPointF(0, shift_pos.y())
        shape.move_vertex_by(right_index, right_shift)
        shape.move_vertex_by(left_index, left_shift)

    def bounded_move_shape(self, shape, pos):
        if self.out_of_pixmap(pos):
            return False  # No need to move
        o1 = pos + self.offsets[0]
        if self.out_of_pixmap(o1):
            pos -= QPointF(min(0, o1.x()), min(0, o1.y()))
        o2 = pos + self.offsets[1]
        if self.out_of_pixmap(o2):
            pos += QPointF(min(0, self.pixmap.width() - o2.x()),
                           min(0, self.pixmap.height() - o2.y()))
        # The next line tracks the new position of the cursor
        # relative to the shape, but also results in making it
        # a bit "shaky" when nearing the border and allows it to
        # go outside of the shape's area for some reason. XXX
        # self.calculateOffsets(self.selectedShape, pos)
        dp = pos - self.prev_point
        if dp:
            shape.move_by(dp)
            self.prev_point = pos
            return True
        return False

    def un_highlight(self, shape=None):
        if shape == None or shape == self.h_shape:
            if self.h_shape:
                self.h_shape.highlight_clear()
            self.h_vertex = self.h_shape = None

    def de_select_shape(self):
        if self.selected_shape:
            self.selected_shape.selected = False
            self.selected_shape = None
            # self.set_hiding(False)
            # self.selectionChanged.emit(False)
            self.update()

    def delete_selected(self):
        if self.selected_shape:
            shape = self.selected_shape
            self.un_highlight(shape)
            self.shapes.remove(self.selected_shape)
            self.selected_shape = None
            self.update()
            return shape

    def select_shape_point(self, point):
        """Select the first shape created which contains this point."""
        self.de_select_shape()
        if self.selected_vertex():  # A vertex is marked for selection.
            index, shape = self.h_vertex, self.h_shape
            shape.highlight_vertex(index, shape.MOVE_VERTEX)
            self.select_shape(shape)
            return self.h_vertex
        for shape in reversed(self.shapes):
            if shape.frameId == self.curFramesId or shape.auto == guishape.STATIONARY_OBJECT:
                if shape.contains_point(point):
                    self.select_shape(shape)
                    self.calculate_offsets(shape, point)
                    return self.selected_shape
        return None

    def selected_vertex(self):
        return self.h_vertex is not None

    def calculate_offsets(self, shape, point):
        rect = shape.bounding_rect()
        x1 = rect.x() - point.x()
        y1 = rect.y() - point.y()
        x2 = (rect.x() + rect.width()) - point.x()
        y2 = (rect.y() + rect.height()) - point.y()
        self.offsets = QPointF(x1, y1), QPointF(x2, y2)

    def set_last_label(self, text, line_color=None, fill_color=None):
        """Define the attributes like text or color of the latest drawn shape.

        Args:
            text (string): the label of the bbox
            line_color (QColor, optional): the color of the bbox line. Defaults to None.
            fill_color (Qcolor, optional): the color of the bbox inside. Defaults to None.

        Returns:
            shape (shape): the new shape with attributes.
        """
        assert text
        self.shapes[-1].label = text
        if line_color:
            self.shapes[-1].line_color = line_color

        if fill_color:
            self.shapes[-1].fill_color = fill_color

        return self.shapes[-1]

    def set_shape_label(self, shape, text, id, line_color=None, fill_color=None):
        shape.label = text
        shape.id = id
        if line_color:
            shape.line_color = line_color

        if fill_color:
            shape.fill_color = fill_color

        return shape

    def paintEvent(self, event):
        if not self.pixmap:
            return super(canvas, self).paintEvent(event)
        p = self._painter
        p.begin(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.HighQualityAntialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)

        p.scale(self.scale, self.scale)
        p.translate(self.offset_to_center() + self.img_off)

        p.drawPixmap(QPointF(0, 0), self.pixmap)

        Shape.scale = self.scale
        # Shape.label_font_size = self.label_font_size
        # 画矩形
        for shape in self.shapes:
            # if (shape.selected or not self._hide_background) and self.isVisible(shape):
            #     shape.fill = shape.selected or shape == self.h_shape
            #     shape.paint(p)
            if shape.frameId == self.curFramesId or shape.auto == guishape.STATIONARY_OBJECT:
                shape.fill = shape.selected or shape == self.h_shape  # 是否填充
                shape._highlight_point = shape == self.h_shape
                shape.paint(p)

        if self.mode == self.CREATE:
            # 拖拽时显示矩形
            if self.current is not None and len(self.line) == 2:
                left_top = self.line[0]
                right_bottom = self.line[1]
                rect_width = right_bottom.x() - left_top.x()
                rect_height = right_bottom.y() - left_top.y()
                p.setPen(self.drawing_rect_color)
                brush = QBrush(Qt.Dense7Pattern)
                p.setBrush(brush)
                p.drawRect(int(left_top.x()), int(left_top.y()),
                           int(rect_width), int(rect_height))
        elif self.mode == self.CREATE_ROAD:
            # 拖拽时显示直线
            if self.current is not None and len(self.lines) is not 0:
                for i, lane in enumerate(self.lines):
                    if len(lane) == 2:
                        start = lane[0]
                        end = lane[1]
                        # TODO use another color
                        p.setPen(self.drawing_rect_color)
                        brush = QBrush(Qt.Dense7Pattern)
                        p.setBrush(brush)
                        p.drawLine(int(start.x()), int(start.y()),
                                   int(end.x()), int(end.y()))

        # 十字参考线
        if self.drawing() and not self.prev_point.isNull() and not self.out_of_pixmap(self.prev_point):
            p.setPen(QColor(41, 121, 255))
            p.drawLine(int(self.prev_point.x()), 0, int(
                self.prev_point.x()), int(self.pixmap.height()))
            p.drawLine(0, int(self.prev_point.y()), int(
                self.pixmap.width()), int(self.prev_point.y()))

        p.end()

    # TODO: 边界, self.window.label_coordinates
    def mouseMoveEvent(self, ev):
        """Update line with last point and current coordinates."""
        pos = self.transform_pos(ev.pos())

        # Update coordinates in status bar if image is opened
        if self.numFrames:
            self.window.label_coordinates.setText(
                'X: %d; Y: %d' % (pos.x(), pos.y()))

        if self.drawing() or self.drawing_road():  # create mode
            self.override_cursor(CURSOR_DRAW)

            if self.current:
                # Display annotation width and height while drawing
                current_width = abs(self.current[0].x() - pos.x())
                current_height = abs(self.current[0].y() - pos.y())
                self.window.label_coordinates.setText(
                    'Width: %d, Height: %d / X: %d; Y: %d' % (current_width, current_height, pos.x(), pos.y()))

                color = self.drawing_line_color
                if self.out_of_pixmap(pos):
                    # Don't allow the user to draw outside the pixmap.
                    # Clip the coordinates to 0 or max,
                    # if they are outside the range [0, max]
                    size = self.pixmap.size()
                    clipped_x = min(max(0, pos.x()), size.width())
                    clipped_y = min(max(0, pos.y()), size.height())
                    pos = QPointF(clipped_x, clipped_y)

                self.line[1] = pos
                self.line.line_color = color
                self.prev_point = QPointF()
                self.current.highlight_clear()
            else:
                self.prev_point = pos
            self.repaint()
            return

        # Polygon/Vertex moving.
        if Qt.LeftButton & ev.buttons():
            if self.selected_vertex():
                self.bounded_move_vertex(pos)
                # self.shapeMoved.emit()
                self.repaint()

                # Display annotation width and height while moving vertex
                point1 = self.h_shape[1]
                point3 = self.h_shape[3]
                current_width = abs(point1.x() - point3.x())
                current_height = abs(point1.y() - point3.y())
                self.window.label_coordinates.setText(
                    'Width: %d, Height: %d / X: %d; Y: %d' % (current_width, current_height, pos.x(), pos.y()))

            elif self.selected_shape and self.prev_point:
                self.override_cursor(CURSOR_MOVE)
                self.bounded_move_shape(self.selected_shape, pos)
                self.repaint()

                # Display annotation width and height while moving shape
                point1 = self.selected_shape[1]
                point3 = self.selected_shape[3]
                current_width = abs(point1.x() - point3.x())
                current_height = abs(point1.y() - point3.y())
                self.window.label_coordinates.setText(
                    'Width: %d, Height: %d / X: %d; Y: %d' % (current_width, current_height, pos.x(), pos.y()))
            else:
                temp_pos = pos + self.img_off
                if self.ori_pos is not None:
                    temp_pos = pos + self.img_off
                    self.img_off += temp_pos - self.ori_pos
                self.ori_pos = temp_pos
                self.repaint()

            return

        # pixmap moving
        if Qt.RightButton & ev.buttons():
            delta_x = pos.x() - self.pan_initial_pos.x()
            delta_y = pos.y() - self.pan_initial_pos.y()
            self.scrollRequest.emit(delta_x, Qt.Horizontal)
            self.scrollRequest.emit(delta_y, Qt.Vertical)
            self.update()

            return

        # Just hovering over the canvas, 2 possibilities:
        # - Highlight shapes
        # - Highlight vertex
        # Update shape/vertex fill and tooltip value accordingly.
        self.setToolTip("Image")
        for shape in reversed([s for s in self.shapes]):
            # Look for a nearby vertex to highlight. If that fails,
            # check if we happen to be inside a shape.
            if shape.frameId == self.curFramesId or shape.auto == guishape.STATIONARY_OBJECT:
                index = shape.nearest_vertex(pos, self.epsilon)
                if index is not None:
                    if self.selected_vertex():
                        self.h_shape.highlight_clear()
                    self.h_vertex, self.h_shape = index, shape
                    shape.highlight_vertex(index, shape.MOVE_VERTEX)
                    self.override_cursor(CURSOR_POINT)
                    self.setToolTip("Click & drag to move point")
                    self.setStatusTip(self.toolTip())
                    self.update()
                    break

                elif shape.contains_point(pos):
                    if self.selected_vertex():
                        self.h_shape.highlight_clear()
                    self.h_vertex, self.h_shape = None, shape
                    tooltip = str(shape.label) + str(' ') + \
                        str(shape.id) + ' (' + shape.auto + ')'
                    self.setToolTip(tooltip)
                    self.setStatusTip("Click & drag to move rect")
                    self.override_cursor(CURSOR_GRAB)
                    self.update()
                    # Display annotation width and height while hovering inside
                    point1 = self.h_shape[1]
                    point3 = self.h_shape[3]
                    current_width = abs(point1.x() - point3.x())
                    current_height = abs(point1.y() - point3.y())
                    self.window.label_coordinates.setText(
                        'Width: %d, Height: %d / X: %d; Y: %d' % (current_width, current_height, pos.x(), pos.y()))
                    break

        else:  # Nothing found, clear highlights, reset state.
            if self.h_shape:
                self.h_shape.highlight_clear()
                self.update()
            self.h_vertex, self.h_shape = None, None
            self.override_cursor(CURSOR_DEFAULT)

    def mousePressEvent(self, ev):
        pos = self.transform_pos(ev.pos())
        if ev.button() == Qt.LeftButton:
            if self.drawing() or self.drawing_road():            # start drawing
                self.handle_drawing(pos)
            else:  # not drawing, update the cross reference line.
                selection = self.select_shape_point(pos)
                self.prev_point = pos
                if selection is None:
                    pass
                #     # pan
                #     QApplication.setOverrideCursor(QCursor(Qt.OpenHandCursor))
                #     self.pan_initial_pos = pos
        elif ev.button() == Qt.RightButton:
            self.override_cursor(CURSOR_GRAB)
            self.pan_initial_pos = pos
            if self.mode == self.CREATE_ROAD:
                self.finalise()
                QApplication.restoreOverrideCursor()

        self.update()

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self.ori_pos = None
            pos = self.transform_pos(ev.pos())
            if self.drawing():
                self.handle_drawing(pos)
                QApplication.restoreOverrideCursor()
            else:
                # pan
                QApplication.restoreOverrideCursor()

    def mouseDoubleClickEvent(self, ev):
        # 双击恢复原状
        self.img_off = QPointF(0, 0)
        # 修改标签信息
        if self.selected_shape:
            if self.selected_shape.auto is 'L':
                self.label_dialog = LabelDialog(
                    parent=self, list_item=self.window.roadHint)
            else:
                self.label_dialog = LabelDialog(
                    parent=self, list_item=self.window.labelHint)
            for shape in reversed([s for s in self.shapes]):
                if shape.selected and shape.frameId == self.curFramesId:
                    text, id = self.label_dialog.pop_up(
                        id=shape.id, text=shape.label)
                    if text is not None:
                        generate_line_color, generate_fill_color = generate_color_by_text(
                            text)
                        self.set_shape_label(
                            shape, text, id, generate_line_color, generate_fill_color)
                        break
        self.repaint()

    def wheelEvent(self, ev):
        qt_version = 4 if hasattr(ev, "delta") else 5
        if qt_version == 4:
            if ev.orientation() == Qt.Vertical:
                v_delta = ev.delta()
                h_delta = 0
            else:
                h_delta = ev.delta()
                v_delta = 0
        else:
            delta = ev.angleDelta()
            h_delta = delta.x()
            v_delta = delta.y()

        mods = ev.modifiers()
        if Qt.ControlModifier == int(mods) and v_delta:
            self.zoomRequest.emit(v_delta)
        else:
            v_delta and self.scrollRequest.emit(v_delta, Qt.Vertical)
            h_delta and self.scrollRequest.emit(h_delta, Qt.Horizontal)
        ev.accept()

    def interpolate(self, shape0, shape1):
        generate_line_color, generate_fill_color = generate_color_by_text(
            shape0.label)

        shapes = []
        dis = shape1.frameId - shape0.frameId
        diff = np.subtract(shape1.points, shape0.points)

        for frameId in range(shape0.frameId + 1, shape1.frameId):
            offset = (frameId - shape0.frameId) / dis
            points = shape0.points + diff * offset
            int_shape = Shape()
            int_shape.frameId = frameId
            int_shape.score = shape0.score
            int_shape.auto = shape0.auto
            self.set_shape_label(
                int_shape, shape0.label, shape0.id, generate_line_color, generate_fill_color)
            for pos in points:
                if self.out_of_pixmap(pos):
                    size = self.pixmap.size()
                    clipped_x = min(max(0, pos.x()), size.width())
                    clipped_y = min(max(0, pos.y()), size.height())
                    pos = QPointF(clipped_x, clipped_y)
                int_shape.add_point(pos)
            int_shape.close()
            shapes.append(int_shape)

        return shapes

    def shapeIOU(self, shape0, shape1):
        def intSeg(seg1, seg2):
            if seg1[0] > seg2[0]:
                temp = seg1
                seg1 = seg2
                seg2 = temp
            intLeft = seg2[0]
            intRight = seg2[1]
            if seg1[1] < seg2[1]:
                intRight = seg1[1]
            if seg2[0] > seg1[1]:
                intLeft = None
                intRight = None
            return [intLeft, intRight]

        wh1 = shape0.points[2] - shape0.points[0]
        wh2 = shape1.points[2] - shape1.points[0]
        area1 = wh1.x() * wh1.y()
        area2 = wh2.x() * wh2.y()

        xSegment = intSeg([shape0.points[0].x(), shape0.points[2].x()],
                          [shape1.points[0].x(), shape1.points[2].x()])

        ySegment = intSeg([shape0.points[0].y(), shape0.points[2].y()],
                          [shape1.points[0].y(), shape1.points[2].y()])

        if None in xSegment or None in ySegment:
            return 0

        intersection = (xSegment[1] - xSegment[0]) * \
            (ySegment[1] - ySegment[0])
        union = area1 + area2 - intersection

        return intersection / union

    def rectify_selected(self, iou_thresh=0.85):
        if self.selected_shape:
            dialog = RectifyDialog(parent=self)
            toFrame, targetId, isPadding = dialog.pop_up()

            if toFrame is None or toFrame <= 0:
                return

            rect_shapes = []

            if toFrame < self.selected_shape.frameId and targetId >= 1:
                # 与之前的帧匹配, 首先找到首帧目标
                shape0 = None
                shape1 = self.selected_shape
                for shape in self.shapes:
                    if shape.id == targetId and shape.frameId == toFrame:
                        shape0 = shape
                        break
                if shape0 is None:
                    return

                generate_line_color, generate_fill_color = generate_color_by_text(
                    shape0.label)
                self.set_shape_label(self.shapes[self.shapes.index(shape1)],
                                     shape0.label, shape0.id, generate_line_color, generate_fill_color)

                # 计算中间帧的插值
                rect_shapes = self.interpolate(shape0, shape1)

            elif toFrame > self.selected_shape.frameId:
                shape0 = self.selected_shape
                generate_line_color, generate_fill_color = generate_color_by_text(
                    shape0.label)
                tracker = cv2.legacy.TrackerCSRT_create()
                x, y = shape0.points[0].x(), shape0.points[0].y()
                w = shape0.points[2].x() - x
                h = shape0.points[2].y() - y
                curFrameId = shape0.frameId
                bbox = (x, y, w, h)
                tracker.init(self.imgFrames[curFrameId - 1], bbox)

                while curFrameId < toFrame:
                    curFrameId += 1
                    ok, bbox = tracker.update(self.imgFrames[curFrameId - 1])
                    leftTop = QPointF(bbox[0], bbox[1])
                    rightTop = QPointF(bbox[0] + bbox[2], bbox[1])
                    rightDown = QPointF(bbox[0] + bbox[2], bbox[1] + bbox[3])
                    leftDown = QPointF(bbox[0], bbox[1] + bbox[3])
                    points = [leftTop, rightTop, rightDown, leftDown]
                    track_shape = Shape()
                    track_shape.frameId = curFrameId
                    track_shape.score = shape0.score
                    track_shape.auto = shape0.auto
                    self.set_shape_label(
                        track_shape, shape0.label, shape0.id, generate_line_color, generate_fill_color)
                    for pos in points:
                        if self.out_of_pixmap(pos):
                            size = self.pixmap.size()
                            clipped_x = min(max(0, pos.x()), size.width())
                            clipped_y = min(max(0, pos.y()), size.height())
                            pos = QPointF(clipped_x, clipped_y)
                        track_shape.add_point(pos)
                    track_shape.close()
                    rect_shapes.append(track_shape)

            if len(rect_shapes) > 0:
                ocp_shape = None
                for rect_shape in rect_shapes:
                    frame_shapes = [
                        s for s in self.shapes if s.frameId == rect_shape.frameId]
                    flag = False
                    for s in frame_shapes:
                        # 找到该帧中id和目标id一致的框
                        if s.id == shape0.id:
                            ocp_shape = s
                            if self.shapeIOU(s, rect_shape) > iou_thresh:
                                flag = True
                                if s.label != shape0.label:
                                    ind = self.shapes.index(s)
                                    self.set_shape_label(self.shapes[ind], shape0.label, shape0.id,
                                                         generate_line_color, generate_fill_color)
                            break
                    if flag:
                        continue
                    # 不存在id一致或者id一致的框不符合要求要找到正确的框
                    for s in frame_shapes:
                        if self.shapeIOU(s, rect_shape) > iou_thresh:
                            flag = True
                            ind = self.shapes.index(s)
                            if ocp_shape:
                                ocp_ind = self.shapes.index(ocp_shape)
                                self.shapes[ocp_ind].id = self.shapes[ind].id
                            self.set_shape_label(self.shapes[ind], shape0.label, shape0.id,
                                                 generate_line_color, generate_fill_color)
                            break
                    if flag:
                        continue
                    # 完全找不到匹配的，就直接添加一个这样的框
                    if isPadding and ocp_shape is None:
                        self.shapes.append(rect_shape)
