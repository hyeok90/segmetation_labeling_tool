import math
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox

from shape import Shape
import utils
from sam_thread import SAMPredictionThread

CURSOR_DEFAULT = QtCore.Qt.ArrowCursor
CURSOR_POINT = QtCore.Qt.PointingHandCursor
CURSOR_DRAW = QtCore.Qt.CrossCursor
CURSOR_MOVE = QtCore.Qt.ClosedHandCursor
CURSOR_GRAB = QtCore.Qt.OpenHandCursor

class ImageViewer(QtWidgets.QWidget):
    polygon_selected = QtCore.pyqtSignal(object)
    new_polygon_drawn = QtCore.pyqtSignal(object)

    CREATE, EDIT = 0, 1

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.mode = self.EDIT
        self.shapes = []
        self.shapes_backups = []
        self.num_backups = 10
        self.current = None
        self.selected_shapes = []
        self.line = Shape()
        self.prev_point = QtCore.QPoint()
        self.scale = 1.0
        self.pixmap = QtGui.QPixmap()
        self._painter = QtGui.QPainter()
        self._cursor = CURSOR_DEFAULT
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.WheelFocus)
        self.epsilon = 11.0

        self.h_shape = None
        self.h_vertex = None
        self.moving_shape = False

        self.is_panning = False
        self.pan_start_pos = QtCore.QPoint()
        self.offset = QtCore.QPointF()

        # SAM-related attributes
        self.sam_points = []
        self.sam_labels = []
        self.sam_preview_shapes = []
        self.sam_prediction_thread = None
        self.is_predicting = False

    def store_shapes(self):
        shapes_backup = []
        for shape in self.shapes:
            shapes_backup.append(shape.copy())
        if len(self.shapes_backups) > self.num_backups:
            self.shapes_backups = self.shapes_backups[-self.num_backups - 1 :]
        self.shapes_backups.append(shapes_backup)

    @property
    def is_shape_restorable(self):
        return len(self.shapes_backups) > 1

    def restore_shape(self):
        if not self.is_shape_restorable:
            return
        self.shapes_backups.pop()
        shapes_backup = self.shapes_backups.pop()
        self.shapes = shapes_backup
        self.selected_shapes = []
        for shape in self.shapes:
            shape.selected = False
        self.update()

    def set_editing(self, value=True):
        self.mode = self.EDIT if value else self.CREATE
        if not value:
            self.un_highlight()
            self.deselect_shape()

    def drawing(self):
        return self.mode == self.CREATE

    def editing(self):
        return self.mode == self.EDIT

    def set_image(self, pixmap):
        self.pixmap = pixmap
        self.update()

    def clear_polygons(self):
        self.shapes = []
        self.update()

    def find_shape(self, point):
        for shape in reversed(self.shapes):
            if shape.contains_point(point):
                return shape
        return None

    def select_shape(self, shape, multi_select=False):
        if not multi_select:
            self.deselect_shape()
        
        if shape:
            shape.selected = True
            if shape not in self.selected_shapes:
                self.selected_shapes.append(shape)
            self.polygon_selected.emit(shape)
        
        self.update()

    def deselect_shape(self):
        for shape in self.selected_shapes:
            shape.selected = False
        self.selected_shapes = []
        self.polygon_selected.emit(None)
        self.update()

    def paintEvent(self, event):
        if self.pixmap.isNull():
            return
        p = self._painter
        p.begin(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

        p.translate(self.offset)
        p.scale(self.scale, self.scale)

        p.drawPixmap(0, 0, self.pixmap)

        # Draw SAM preview shapes
        if self.parent.is_sam_mode and self.sam_preview_shapes:
            p.setPen(QtGui.QPen(QtGui.QColor(0, 100, 255, 100)))
            p.setBrush(QtGui.QColor(0, 100, 255, 100))
            for shape in self.sam_preview_shapes:
                poly = QtGui.QPolygonF()
                for pnt in shape.points:
                    poly.append(pnt)
                p.drawPolygon(poly)
            p.setBrush(Qt.NoBrush)

        Shape.scale = self.scale
        for shape in self.shapes:
            shape.paint(p)

        if self.current:
            self.current.paint(p)
            self.line.paint(p)

        # Draw SAM points
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        point_size = 8 / self.scale
        for i, point in enumerate(self.sam_points):
            color = QtGui.QColor(0, 255, 0) if self.sam_labels[i] == 1 else QtGui.QColor(255, 0, 0)
            p.setPen(QtGui.QPen(color, 2 / self.scale))
            p.setBrush(QtGui.QBrush(color))
            p.drawEllipse(point, point_size / 2, point_size / 2)

        p.end()

    def wheelEvent(self, event: QtGui.QWheelEvent):
        old_pos = self.transform_pos(event.pos())
        
        delta = event.angleDelta().y()
        if delta > 0:
            self.scale *= 1.2
        else:
            self.scale /= 1.2
        
        new_pos = self.transform_pos(event.pos())
        
        delta_pos = new_pos - old_pos
        self.offset += delta_pos * self.scale
        
        self.update()

    def mousePressEvent(self, ev: QtGui.QMouseEvent):
        pos = self.transform_pos(ev.pos())

        if self.parent.is_sam_mode and ev.button() == Qt.LeftButton:
            if self.parent.sam_point_mode is not None:
                self.add_sam_point(pos, self.parent.sam_point_mode)
            return # Always consume left clicks in SAM mode

        if ev.button() == Qt.LeftButton:
            if self.drawing():
                self.handle_drawing(pos)
            else:
                if self.h_vertex is not None:
                    self.select_shape(self.h_shape)
                else:
                    shape = self.find_shape(pos)
                    self.select_shape(shape)
                self.prev_point = pos
                self.moving_shape = True

        elif ev.button() == Qt.RightButton:
            if self.drawing() and self.current:
                self.finalise()

        elif ev.button() == Qt.MidButton:
            self.is_panning = True
            self.pan_start_pos = ev.pos()

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent):
        pos = self.transform_pos(ev.pos())

        if self.is_panning:
            delta = ev.pos() - self.pan_start_pos
            self.offset += delta
            self.pan_start_pos = ev.pos()
            self.update()
            return

        if self.drawing() and self.current and not self.parent.is_sam_mode:
            if self.close_enough(pos, self.current.points[0]):
                pos = self.current.points[0]
            self.line.points = [self.current.points[-1], pos]
            self.update()
            return

        if self.editing():
            if self.h_vertex is not None and (ev.buttons() & Qt.LeftButton):
                self.h_shape.move_vertex_by(self.h_vertex, pos - self.prev_point)
                self.prev_point = pos
                self.update()
                return
            
            if self.selected_shapes and (ev.buttons() & Qt.LeftButton):
                dp = pos - self.prev_point
                for shape in self.selected_shapes:
                    shape.move_by(dp)
                self.prev_point = pos
                self.update()
                return

        # Hover logic
        self.un_highlight()
        
        # --- Optimization: Filter shapes by bounding box ---
        candidate_shapes = [s for s in reversed(self.shapes) if s.bounding_rect().contains(pos)]
        
        # Find nearest vertex in candidate shapes
        for shape in candidate_shapes:
            index = shape.nearest_vertex(pos, self.epsilon / self.scale)
            if index is not None:
                self.h_shape = shape
                self.h_vertex = index
                shape.highlight_vertex(index, Shape.MOVE_VERTEX)
                self.update()
                break
        else: # if no vertex found, check for shape containment in candidates
            for shape in candidate_shapes:
                if shape.contains_point(pos):
                    self.h_shape = shape
                    self.update()
                    break

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent):
        if ev.button() == Qt.MidButton:
            self.is_panning = False
        
        if ev.button() == Qt.LeftButton and self.moving_shape:
            self.store_shapes()
            self.moving_shape = False

    def handle_drawing(self, pos):
        if self.current is None:
            self.current = Shape(shape_type='polygon')
            self.current.add_point(pos)
            self.line.points = [pos, pos]
        else:
            if len(self.current.points) > 1 and self.close_enough(pos, self.current.points[0]):
                self.finalise()
            else:
                self.current.add_point(pos)

    def finalise(self):
        if self.current:
            self.current.close()
            self.new_polygon_drawn.emit(self.current)
            self.current = None
            self.update()

    def delete_selected_vertex(self):
        if self.h_shape is None or self.h_vertex is None:
            return False

        shape = self.h_shape
        vertex_index = self.h_vertex

        shape.remove_point(vertex_index)

        # If a polygon becomes too small, remove it completely
        if shape.shape_type == 'polygon' and len(shape.points) < 3:
            self.shapes.remove(shape)
            self.deselect_shape() # In case it was selected

        self.un_highlight()
        self.store_shapes()
        self.update()
        return True


    def cancel_drawing(self):
        self.current = None
        self.line.points = []
        self.clear_sam_points()
        self.repaint()

    def undo_last_point(self):
        if not self.current or not self.current.points:
            return
        
        self.current.pop_point()
        if self.current.points:
            self.line.points = [self.current.points[-1], self.current.points[-1]]
        else:
            self.line.points = []
            self.current = None

        self.repaint()

    def transform_pos(self, point):
        return (point - self.offset) / self.scale

    def fit_to_window(self):
        if self.pixmap.isNull():
            return
        self.scale = min(self.width() / self.pixmap.width(), self.height() / self.pixmap.height())
        self.offset = QtCore.QPointF()
        self.update()

    def un_highlight(self):
        if self.h_shape:
            self.h_shape.highlight_clear()
        self.h_shape = None
        self.h_vertex = None
        self.update()

    def set_draw_mode(self, enabled):
        self.set_editing(not enabled)

    def close_enough(self, p1, p2):
        return utils.distance(p1 - p2) < (self.epsilon / self.scale)

    # --- SAM Methods ---

    def clear_sam_points(self):
        self.sam_points = []
        self.sam_labels = []
        self.sam_preview_shapes = []
        self.repaint()

    def reset_sam_prediction(self):
        self.sam_points = []
        self.sam_labels = []
        self.sam_preview_shapes = []
        self.update()

    def add_sam_point(self, pos, label):
        self.sam_points.append(pos)
        self.sam_labels.append(label)
        self.predict_sam_mask()

    def predict_sam_mask(self):
        if not self.sam_points or not self.parent.sam_predictor or self.is_predicting:
            return

        self.is_predicting = True
        self.parent.statusBar().showMessage("Running SAM prediction...")

        points = np.array([[p.x(), p.y()] for p in self.sam_points])
        labels = np.array(self.sam_labels)
        
        self.sam_prediction_thread = SAMPredictionThread(self.parent.sam_predictor, points, labels)
        self.sam_prediction_thread.prediction_finished.connect(self.on_sam_prediction_finished)
        self.sam_prediction_thread.prediction_failed.connect(self.on_sam_prediction_failed)
        self.sam_prediction_thread.start()

    def on_sam_prediction_finished(self, polygons):
        self.sam_preview_shapes = []
        for poly_points in polygons:
            shape = Shape(shape_type='polygon')
            shape.points = [QtCore.QPointF(p[0], p[1]) for p in poly_points]
            shape.close()
            shape.fill = True
            shape.line_color = QtGui.QColor(0, 100, 255, 100)
            shape.fill_color = QtGui.QColor(0, 100, 255, 100)
            self.sam_preview_shapes.append(shape)
        
        self.is_predicting = False
        self.parent.statusBar().showMessage("SAM prediction finished.", 2000)
        self.update()

    def on_sam_prediction_failed(self, error_msg):
        self.is_predicting = False
        self.parent.statusBar().showMessage("SAM prediction failed.", 5000)
        QMessageBox.critical(self, "SAM Error", f"SAM prediction failed:\n{error_msg}")

    def finalize_sam_shape(self):
        if not self.sam_preview_shapes:
            return []
        
        new_shapes = [s.copy() for s in self.sam_preview_shapes]
        for shape in new_shapes:
            shape.fill = False
            shape.line_color = QtGui.QColor(0, 255, 0, 128) # Default color
        
        # The viewer state is now cleared from the main window after the user confirms the class.
        return new_shapes