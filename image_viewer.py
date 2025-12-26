import math
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtWidgets import QMessageBox

from shape import Shape
import utils
from sam_thread import SAMPredictionThread
from config import Config

from shapely.geometry import Point, Polygon, MultiPolygon, LineString
from shapely.ops import unary_union

CURSOR_DEFAULT = QtCore.Qt.ArrowCursor
CURSOR_POINT = QtCore.Qt.PointingHandCursor
CURSOR_DRAW = QtCore.Qt.CrossCursor
CURSOR_MOVE = QtCore.Qt.ClosedHandCursor
CURSOR_GRAB = QtCore.Qt.OpenHandCursor

class ImageViewer(QtWidgets.QWidget):
    polygon_selected = QtCore.pyqtSignal(object)
    new_polygon_drawn = QtCore.pyqtSignal(object)
    shapes_updated = QtCore.pyqtSignal()

    # Modes
    EDIT = 0
    CREATE_POLY = 1
    CREATE_BRUSH = 2
    CREATE_ERASER = 3

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.mode = self.EDIT
        self.shapes = []
        self.shapes_backups = []
        self.num_backups = 10
        self.current = None # Used for polygon drawing
        self.selected_shapes = []
        self.line = Shape()
        self.prev_point = QtCore.QPoint()
        self.scale = 1.0
        self.pixmap = QtGui.QPixmap()
        self._painter = QtGui.QPainter()
        self._cursor = CURSOR_DEFAULT
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.WheelFocus)
        self.epsilon = 11.0 # Tolerance for mouse hover / vertex selection
        self.simplify_epsilon = Config.DEFAULT_EPSILON # Tolerance for RDP simplification

        self.h_shape = None
        self.h_vertex = None
        self.moving_shape = False

        self.is_panning = False
        self.pan_start_pos = QtCore.QPoint()
        self.offset = QtCore.QPointF()

        # Brush / Eraser attributes
        self.brush_radius = Config.DEFAULT_BRUSH_RADIUS
        self.brush_path_points = [] # Temporary storage for brush trajectory
        self.is_brushing = False
        self.brush_layer = None
        self.brush_painter = None
        self.last_brush_pos = None

        # SAM-related attributes
        self.sam_points = []
        self.sam_labels = []
        self.sam_preview_shapes = []
        self.sam_prediction_thread = None
        self.is_predicting = False
        self.last_predicted_point_count = 0

    def store_shapes(self):
        shapes_backup = []
        for shape in self.shapes:
            shapes_backup.append(shape.copy())
        if len(self.shapes_backups) > self.num_backups:
            self.shapes_backups = self.shapes_backups[-self.num_backups - 1 :]
        self.shapes_backups.append(shapes_backup)

    @property
    def is_shape_restorable(self):
        return len(self.shapes_backups) > 0

    def restore_shape(self):
        if not self.is_shape_restorable:
            return
        shapes_backup = self.shapes_backups.pop()
        self.shapes = shapes_backup
        self.selected_shapes = []
        for shape in self.shapes:
            shape.selected = False
        self.update()
        self.shapes_updated.emit()

    def set_mode(self, mode):
        self.mode = mode
        if mode == self.EDIT:
            self.un_highlight()
            self.deselect_shape()
        elif mode in [self.CREATE_BRUSH, self.CREATE_ERASER]:
            # Do NOT deselect when entering brush/eraser mode
            pass
        else:
            self.deselect_shape() # Usually deselect when starting other tools
        self.update()

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

        if self.mode == self.CREATE_POLY and self.current:
            self.current.paint(p)
            self.line.paint(p)

        # Draw Brush/Eraser Trajectory (Visual feedback)
        if self.is_brushing and self.brush_layer:
            # The brush layer is in scene coordinates (same size as pixmap)
            # but we are already transformed (scaled/translated) so we draw it at 0,0
            p.drawPixmap(0, 0, self.brush_layer)

        # Draw Brush/Eraser Cursor
        if self.mode in [self.CREATE_BRUSH, self.CREATE_ERASER]:
            cursor_pos = self.mapFromGlobal(QtGui.QCursor.pos())
            # Map back to scene coords just for logic, but here we need to draw relative to scene transforms
            # Actually, it's easier to draw the cursor in screen space, but we are inside the transformed painter.
            # So we draw the cursor at the transformed mouse position.
            
            # Since we can't easily get the mouse position inside paintEvent without tracking it,
            # we rely on mouseMoveEvent to trigger updates. 
            # Ideally, we track the last mouse scene position.
            if hasattr(self, 'last_mouse_pos'):
                cursor_color = Qt.green if self.mode == self.CREATE_BRUSH else Qt.red
                p.setPen(QtGui.QPen(cursor_color, 2 / self.scale, Qt.SolidLine))
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(self.last_mouse_pos, self.brush_radius, self.brush_radius)


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
        self.last_mouse_pos = pos

        # SAM Mode
        if self.parent.is_sam_mode:
            if ev.button() == Qt.LeftButton:
                # Left Click -> Positive Point (1)
                self.add_sam_point(pos, 1)
                self.parent.statusBar().showMessage("Added positive point.", 1000)
                return
            elif ev.button() == Qt.RightButton:
                # Right Click -> Negative Point (0)
                self.add_sam_point(pos, 0)
                self.parent.statusBar().showMessage("Added negative point.", 1000)
                return
            # Allow Middle Button to fall through to panning logic below

        # Panning
        if ev.button() == Qt.MidButton:
            self.is_panning = True
            self.pan_start_pos = ev.pos()
            return

        if ev.button() == Qt.LeftButton:
            if self.mode == self.CREATE_POLY:
                self.handle_poly_drawing(pos)
            
            elif self.mode in [self.CREATE_BRUSH, self.CREATE_ERASER]:
                self.is_brushing = True
                self.brush_path_points = [pos]
                self.last_brush_pos = pos
                
                # Initialize temporary layer for smooth rendering
                if not self.pixmap.isNull():
                    self.brush_layer = QtGui.QPixmap(self.pixmap.size())
                    self.brush_layer.fill(Qt.transparent)
                    
                    self.brush_painter = QtGui.QPainter(self.brush_layer)
                    self.brush_painter.setRenderHint(QtGui.QPainter.Antialiasing)
                    self.brush_painter.setCompositionMode(QtGui.QPainter.CompositionMode_Source)
                    
                    color = QtGui.QColor(0, 255, 0, 100) if self.mode == self.CREATE_BRUSH else QtGui.QColor(255, 0, 0, 100)
                    self.brush_painter.setPen(Qt.NoPen)
                    self.brush_painter.setBrush(color)
                    self.brush_painter.drawEllipse(pos, self.brush_radius, self.brush_radius)

                self.update()

            elif self.mode == self.EDIT:
                if self.h_vertex is not None:
                    self.select_shape(self.h_shape)
                else:
                    shape = self.find_shape(pos)
                    self.select_shape(shape)
                self.prev_point = pos
                self.moving_shape = True

        elif ev.button() == Qt.RightButton:
            if self.mode == self.CREATE_POLY and self.current:
                self.finalise_poly()

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent):
        pos = self.transform_pos(ev.pos())
        self.last_mouse_pos = pos

        if self.is_panning:
            delta = ev.pos() - self.pan_start_pos
            self.offset += delta
            self.pan_start_pos = ev.pos()
            self.update()
            return

        if self.mode == self.CREATE_POLY and self.current and not self.parent.is_sam_mode:
            if self.close_enough(pos, self.current.points[0]):
                pos = self.current.points[0]
            self.line.points = [self.current.points[-1], pos]
            self.update()
            return

        if self.mode in [self.CREATE_BRUSH, self.CREATE_ERASER]:
            if self.is_brushing:
                self.brush_path_points.append(pos)
                
                if self.brush_painter and self.last_brush_pos:
                    # Draw a line of circles to fill gaps
                    color = QtGui.QColor(0, 255, 0, 100) if self.mode == self.CREATE_BRUSH else QtGui.QColor(255, 0, 0, 100)
                    pen = QtGui.QPen(color, self.brush_radius * 2)
                    pen.setCapStyle(Qt.RoundCap)
                    self.brush_painter.setPen(pen)
                    self.brush_painter.drawLine(self.last_brush_pos, pos)
                    
                self.last_brush_pos = pos
                
            self.update() # To redraw cursor and layer
            return

        if self.mode == self.EDIT:
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
            candidate_shapes = [s for s in reversed(self.shapes) if s.bounding_rect().contains(pos)]
            for shape in candidate_shapes:
                index = shape.nearest_vertex(pos, self.epsilon / self.scale)
                if index is not None:
                    self.h_shape = shape
                    self.h_vertex = index
                    shape.highlight_vertex(index, Shape.MOVE_VERTEX)
                    self.update()
                    break
            else:
                for shape in candidate_shapes:
                    if shape.contains_point(pos):
                        self.h_shape = shape
                        self.update()
                        break

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent):
        if ev.button() == Qt.MidButton:
            self.is_panning = False
        
        if ev.button() == Qt.LeftButton:
            if self.mode == self.EDIT and self.moving_shape:
                self.store_shapes()
                self.moving_shape = False
            
            elif self.mode in [self.CREATE_BRUSH, self.CREATE_ERASER] and self.is_brushing:
                self.finish_brush_stroke()

    def handle_poly_drawing(self, pos):
        if self.current is None:
            self.current = Shape(shape_type='polygon')
            self.current.add_point(pos)
            self.line.points = [pos, pos]
        else:
            if len(self.current.points) > 1 and self.close_enough(pos, self.current.points[0]):
                self.finalise_poly()
            else:
                self.current.add_point(pos)

    def finalise_poly(self):
        if self.current:
            self.current.close()
            self.new_polygon_drawn.emit(self.current)
            self.current = None
            self.update()

    def finish_brush_stroke(self):
        self.is_brushing = False
        
        # Cleanup painter
        if self.brush_painter:
            self.brush_painter.end()
            self.brush_painter = None
        
        self.brush_layer = None # Release memory
        
        if not self.brush_path_points:
            return

        self.store_shapes()
        
        # 1. Create a unified polygon from the brush stroke
        # Instead of unioning many circles (which creates bumps), we buffer the path as a single line
        if len(self.brush_path_points) < 2:
             # Fallback for single point click
             unified_stroke = Point(self.brush_path_points[0].x(), self.brush_path_points[0].y()).buffer(self.brush_radius)
        else:
             path_coords = [(p.x(), p.y()) for p in self.brush_path_points]
             # Buffer the line string to create a thick stroke
             unified_stroke = LineString(path_coords).buffer(self.brush_radius, cap_style=1, join_style=1)
        
        # Simplify slightly to reduce vertex count
        unified_stroke = unified_stroke.simplify(self.simplify_epsilon, preserve_topology=True)

        if self.mode == self.CREATE_BRUSH:
            # If selection exists, merge. Else, create new.
            target_shape = None
            if len(self.selected_shapes) == 1:
                target_shape = self.selected_shapes[0]
            
            if target_shape:
                # Merge with existing selected shape
                original_poly = self.shape_to_shapely(target_shape)
                
                if not original_poly.is_valid:
                    original_poly = original_poly.buffer(0)
                
                try:
                    merged_poly = unary_union([original_poly, unified_stroke])
                except Exception as e:
                    print(f"Topology error during merge, attempting repair: {e}")
                    original_poly = original_poly.buffer(0)
                    unified_stroke = unified_stroke.buffer(0)
                    merged_poly = unary_union([original_poly, unified_stroke])
                
                if not merged_poly.is_valid:
                    merged_poly = merged_poly.buffer(0)

                self.update_shape_from_shapely(target_shape, merged_poly)
            else:
                # Create NEW shape from the stroke (No selection -> New Instance)
                new_shape = self.shapely_to_shape(unified_stroke)
                if new_shape:
                    self.new_polygon_drawn.emit(new_shape)
        
        elif self.mode == self.CREATE_ERASER:
            # If selection exists, erase from it. If NO selection, do NOT erase anything.
            if not self.selected_shapes:
                return # Protect other instances

            targets = self.selected_shapes
            
            shapes_to_remove = []
            
            for shape in targets:
                original_poly = self.shape_to_shapely(shape)
                if original_poly.is_empty or not original_poly.is_valid:
                    continue
                    
                diff_poly = original_poly.difference(unified_stroke)
                
                # Fix potential topological errors (self-intersections)
                if not diff_poly.is_valid:
                    diff_poly = diff_poly.buffer(0)

                if diff_poly.is_empty:
                    shapes_to_remove.append(shape)
                else:
                    if isinstance(diff_poly, MultiPolygon):
                        # Keep the largest part for the original object
                        parts = list(diff_poly.geoms)
                        parts.sort(key=lambda p: p.area, reverse=True)
                        
                        self.update_shape_from_shapely(shape, parts[0])
                        
                        # Discard smaller parts (parts[1:]) per user request
                    else:
                        self.update_shape_from_shapely(shape, diff_poly)

            for s in shapes_to_remove:
                if s in self.shapes:
                    self.shapes.remove(s)
                # Ensure the shape is also removed from selection, so next brush stroke creates a NEW instance
                if s in self.selected_shapes:
                    self.selected_shapes.remove(s)
            
            # If selection became empty, trigger deselection event
            if not self.selected_shapes:
                self.deselect_shape()

        self.brush_path_points = []
        self.update()
        self.shapes_updated.emit()

    def shape_to_shapely(self, shape):
        # Convert QPointF list to [(x,y)]
        coords = [(p.x(), p.y()) for p in shape.points]
        if len(coords) < 3:
            return Polygon()
        return Polygon(coords)

    def shapely_to_shape(self, poly):
        if poly.is_empty:
            return None
        
        # Handle just Polygon for now. MultiPolygon handled in caller.
        if isinstance(poly, MultiPolygon):
             # Just return largest for safety, though caller should handle
             return self.shapely_to_shape(max(poly.geoms, key=lambda p: p.area))

        xs, ys = poly.exterior.coords.xy
        points = [QPointF(x, y) for x, y in zip(xs, ys)]
        
        # Remove last point if duplicate (Shapely closes loop)
        if len(points) > 0 and points[0] == points[-1]:
            points.pop()
            
        s = Shape(shape_type='polygon')
        s.points = points
        s.close()
        return s

    def update_shape_from_shapely(self, shape, poly):
        # Update existing shape object with new geometry
        # Simplify result to avoid excessive points
        poly = poly.simplify(self.simplify_epsilon, preserve_topology=True)
        
        if isinstance(poly, MultiPolygon):
             poly = max(poly.geoms, key=lambda p: p.area)
             
        xs, ys = poly.exterior.coords.xy
        points = [QPointF(x, y) for x, y in zip(xs, ys)]
        if len(points) > 0 and points[0] == points[-1]:
            points.pop()
        shape.points = points

    def delete_selected_vertex(self):
        if self.h_shape is None or self.h_vertex is None:
            return False

        shape = self.h_shape
        vertex_index = self.h_vertex

        shape.remove_point(vertex_index)

        if shape.shape_type == 'polygon' and len(shape.points) < 3:
            self.shapes.remove(shape)
            self.deselect_shape()

        self.un_highlight()
        self.store_shapes()
        self.update()
        self.shapes_updated.emit()
        return True

    def cancel_drawing(self):
        self.current = None
        self.line.points = []
        self.clear_sam_points()
        self.brush_path_points = []
        self.is_brushing = False
        
        if self.brush_painter:
            self.brush_painter.end()
            self.brush_painter = None
        self.brush_layer = None
        
        self.update()

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

    def close_enough(self, p1, p2):
        return utils.distance(p1, p2) < (self.epsilon / self.scale)

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
        self.last_predicted_point_count = len(self.sam_points)
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

        # Check if new points were added while predicting
        if len(self.sam_points) != self.last_predicted_point_count:
            self.predict_sam_mask()

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
        return new_shapes