import os
import sys
import shutil
import cv2
import numpy as np
from PyQt5 import QtGui
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QAction, QFileDialog, 
    QListWidget, QMessageBox, QDockWidget, QListWidgetItem, QInputDialog, QLabel, QMenu, QDialog, QDialogButtonBox, QProgressDialog
)
from PyQt5.QtGui import QPixmap, QIcon, QColor, QImage
from PyQt5.QtCore import Qt, QPointF
from yolo_predictor import RealYOLOPredictor
from sam_predictor import SAMPredictor
from image_viewer import ImageViewer
from shape import Shape
from utils import load_yolo_labels, save_yolo_labels
from training_dialog import TrainingDialog
from training_thread import TrainingThread
from encoder_thread import ImageEncoderThread
from inference_dialog import InferenceDialog
from class_manager_dialog import ClassManagerDialog
from class_selector_dialog import ClassSelectorDialog
from class_specification_dialog import ClassSpecificationDialog

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("YOLOv11-seg Active Learning Tool")
        self.setGeometry(100, 100, 1200, 800)
        
        self.model = None
        self.model_path = None
        self.image_paths = []
        self.current_image_index = -1
        self.current_image_rgb = None
        self.class_names = []
        self.color_map = []

        # Directory for temporary labels, created within the project folder
        self.temp_dir = os.path.join(os.getcwd(), ".temp_labels")
        self.source_label_dir = None

        # SAM Predictor
        self.sam_predictor = None
        self.is_sam_mode = False
        self.sam_point_mode = None # 1 for positive, 0 for negative
        self.encoder_thread = None
        self.is_encoding = False
        encoder_path = "sam/sam_encoder.onnx"
        decoder_path = "sam/sam_decoder.onnx"
        if os.path.exists(encoder_path) and os.path.exists(decoder_path):
            try:
                self.sam_predictor = SAMPredictor(encoder_path, decoder_path)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load SAM model: {e}")
        else:
            print("SAM models (sam_encoder.onnx, sam_decoder.onnx) not found. SAM mode will be disabled.")


        self.viewer = ImageViewer(self)
        self.setCentralWidget(self.viewer)
        
        self.create_actions()
        self.create_menu_bar()
        self.create_tool_bar()
        self.create_docks()
        self.create_status_bar()
        
        self.set_actions_enabled(False)
        self.open_folder_action.setEnabled(True)

        self.viewer.polygon_selected.connect(self.on_polygon_selected)
        self.viewer.new_polygon_drawn.connect(self.on_new_polygon_drawn)

    def create_actions(self):
        self.open_folder_action = QAction(QIcon.fromTheme("folder-open"), "&Open Image Folder", self)
        self.open_folder_action.triggered.connect(self.open_folder)

        self.upload_label_action = QAction(QIcon.fromTheme("document-import"), "Upload Labels...", self)
        self.upload_label_action.triggered.connect(self.upload_labels)

        self.run_inference_action = QAction(QIcon.fromTheme("system-run"), "Run Model Inference...", self)
        self.run_inference_action.triggered.connect(self.show_inference_dialog)

        self.export_action = QAction(QIcon.fromTheme("document-send"), "&Export", self)
        self.export_action.triggered.connect(self.export_files)

        self.train_action = QAction(QIcon.fromTheme("system-run"), "&Train", self)
        self.train_action.triggered.connect(self.open_training_dialog)
        
        self.save_labels_action = QAction(QIcon.fromTheme("document-save"), "&Save Labels", self)
        self.save_labels_action.triggered.connect(self.save_current_labels)
        
        self.prev_image_action = QAction(QIcon.fromTheme("go-previous"), "&Previous Image", self)
        self.prev_image_action.triggered.connect(self.prev_image)

        self.next_image_action = QAction(QIcon.fromTheme("go-next"), "&Next Image", self)
        self.next_image_action.triggered.connect(self.next_image)

        self.draw_poly_action = QAction(QIcon.fromTheme("edit-draw"), "Draw Polygon", self)
        self.draw_poly_action.setCheckable(True)
        self.draw_poly_action.toggled.connect(self.toggle_draw_mode)

        self.draw_sam_action = QAction(QIcon.fromTheme("applications-science"), "SAM Draw", self)
        self.draw_sam_action.setCheckable(True)
        self.draw_sam_action.toggled.connect(self.toggle_sam_mode)
        if not self.sam_predictor:
            self.draw_sam_action.setEnabled(False)
        
        self.fit_window_action = QAction(QIcon.fromTheme("zoom-fit-best"), "Fit to Window", self)
        self.fit_window_action.triggered.connect(self.viewer.fit_to_window)

        self.undo_action = QAction(QIcon.fromTheme("edit-undo"), "&Undo", self)
        self.undo_action.triggered.connect(self.undo_shape)

        self.manage_classes_action = QAction(QIcon.fromTheme("document-properties"), "Class Manager...", self)
        self.manage_classes_action.triggered.connect(self.open_class_manager)

        self.class_specification_action = QAction(QIcon.fromTheme("system-search"), "Class Specification...", self)
        self.class_specification_action.triggered.connect(self.run_class_specification)

    def set_navigation_enabled(self, enabled):
        self.prev_image_action.setEnabled(enabled)
        self.next_image_action.setEnabled(enabled)
        self.save_labels_action.setEnabled(enabled)
        self.undo_action.setEnabled(enabled)
        self.open_folder_action.setEnabled(enabled)
        self.file_list_widget.setEnabled(enabled)

    def create_menu_bar(self):
        menu_bar = self.menuBar()
        
        file_menu = menu_bar.addMenu("&File")
        file_menu.addAction(self.open_folder_action)
        file_menu.addAction(self.upload_label_action)
        file_menu.addAction(self.save_labels_action)
        file_menu.addSeparator()
        file_menu.addAction(self.run_inference_action)
        file_menu.addSeparator()
        file_menu.addAction(self.export_action)
        file_menu.addSeparator()
        file_menu.addAction(self.train_action)

        edit_menu = menu_bar.addMenu("&Edit")
        edit_menu.addAction(self.undo_action)
        edit_menu.addSeparator()
        edit_menu.addAction(self.manage_classes_action)
        edit_menu.addSeparator()
        edit_menu.addAction(self.class_specification_action)

        view_menu = menu_bar.addMenu("&View")
        view_menu.addAction(self.prev_image_action)
        view_menu.addAction(self.next_image_action)
        view_menu.addAction(self.fit_window_action)

    def create_tool_bar(self):
        tool_bar = self.addToolBar("Main ToolBar")
        tool_bar.addAction(self.draw_poly_action)
        tool_bar.addAction(self.draw_sam_action)
        tool_bar.addSeparator()
        tool_bar.addAction(self.fit_window_action)

    def create_docks(self):
        file_list_dock = QDockWidget("File List", self)
        self.file_list_widget = QListWidget()
        self.file_list_widget.itemClicked.connect(self.on_file_item_clicked)
        self.file_list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.file_list_widget.customContextMenuRequested.connect(self.show_file_list_context_menu)
        file_list_dock.setWidget(self.file_list_widget)
        file_list_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.LeftDockWidgetArea, file_list_dock)
        
        class_list_dock = QDockWidget("Class List", self)
        self.class_list_widget = QListWidget()
        class_list_dock.setWidget(self.class_list_widget)
        class_list_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.LeftDockWidgetArea, class_list_dock)

        instance_list_dock = QDockWidget("Instance List", self)
        self.instance_list_widget = QListWidget()
        self.instance_list_widget.itemClicked.connect(self.on_instance_item_clicked)
        self.instance_list_widget.itemDoubleClicked.connect(self.on_instance_double_clicked)
        instance_list_dock.setWidget(self.instance_list_widget)
        instance_list_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.RightDockWidgetArea, instance_list_dock)

    def create_status_bar(self):
        self.statusBar().showMessage("Ready")
        self.conf_label = QLabel("Avg. Confidence: N/A")
        self.statusBar().addPermanentWidget(self.conf_label)

    def set_actions_enabled(self, enabled):
        self.open_folder_action.setEnabled(enabled)
        self.upload_label_action.setEnabled(enabled)
        self.run_inference_action.setEnabled(enabled)
        self.export_action.setEnabled(enabled)
        self.train_action.setEnabled(enabled)
        self.save_labels_action.setEnabled(enabled)
        self.prev_image_action.setEnabled(enabled)
        self.next_image_action.setEnabled(enabled)
        self.draw_poly_action.setEnabled(enabled)
        if self.sam_predictor:
            self.draw_sam_action.setEnabled(enabled)
        self.fit_window_action.setEnabled(enabled)
        self.undo_action.setEnabled(enabled)
        self.manage_classes_action.setEnabled(enabled)
        self.class_specification_action.setEnabled(enabled)

    def open_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Open Image Folder")
        if folder_path:
            # Reset the state before loading new folder
            self.current_image_index = -1

            if os.path.isdir(os.path.join(folder_path, "images")):
                folder_path = os.path.join(folder_path, "images")

            self.image_paths = []
            self.file_list_widget.clear()
            
            image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            labels_dir = os.path.join(os.path.dirname(folder_path), "labels")
            os.makedirs(labels_dir, exist_ok=True)
            
            for img_file in image_files:
                img_path = os.path.join(folder_path, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Error reading image {img_path}")
                    continue
                img_h, img_w = img.shape[:2]
                self.image_paths.append((img_path, (img_w, img_h)))

            self.file_list_widget.addItems([os.path.basename(p) for p, d in self.image_paths])
            
            if self.image_paths:
                self.current_image_index = 0
                self._show_image_only(0)
                self.file_list_widget.setCurrentRow(0)
                self.set_actions_enabled(True)
                self.run_inference_action.setEnabled(True)
                self.train_action.setEnabled(True)
            else:
                self.clear_viewer()

    def show_inference_dialog(self):
        if not self.image_paths:
            QMessageBox.warning(self, "Warning", "Please open an image folder first.")
            return

        dialog = InferenceDialog(self.image_paths, self)
        if dialog.exec_() == QDialog.Accepted:
            selected_images = dialog.get_selected_images()
            model_path = dialog.get_selected_model()

            if not model_path:
                QMessageBox.warning(self, "Warning", "Please select a model for inference.")
                return
            
            if not selected_images:
                QMessageBox.warning(self, "Warning", "Please select at least one image for inference.")
                return

            self.run_inference_on_selection(selected_images, model_path)

    def run_inference_on_selection(self, selected_images, model_path):
        try:
            predictor = RealYOLOPredictor(model_path)
            model_class_names = [predictor.get_class_names()[i] for i in sorted(predictor.get_class_names().keys())]
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {e}")
            return

        # Add new classes to the main list
        new_classes_added = False
        for name in model_class_names:
            if name not in self.class_names:
                self.class_names.append(name)
                new_classes_added = True

        if new_classes_added:
            self.rebuild_color_map_and_refresh_ui()

        progress_dialog = QProgressDialog("Running inference...", "Cancel", 0, len(selected_images), self)
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setWindowTitle("Model Inference")
        
        for i, img_path in enumerate(selected_images):
            progress_dialog.setValue(i)
            progress_dialog.setLabelText(f"Processing {os.path.basename(img_path)}...")
            QApplication.processEvents()

            if progress_dialog.wasCanceled():
                break
            
            self.perform_inference(img_path, predictor, model_class_names) # Use model-specific class names for saving

        progress_dialog.setValue(len(image_paths))
        QMessageBox.information(self, "Complete", "Inference process finished.")

        # Always reload the current view to reflect any possible changes
        # (e.g., new class colors, or if the current image itself was updated).
        self.load_labels_for_current_image()

    def perform_inference(self, image_path, predictor, class_names):
        try:
            # Find corresponding image dimensions from self.image_paths
            img_dims = next((dims for path, dims in self.image_paths if path == image_path), None)
            if not img_dims:
                print(f"Could not find dimensions for image {image_path}")
                return

            img_w, img_h = img_dims
            
            instances, _, _ = predictor.predict_and_optimize(image_path)
            shapes_to_save = []
            for inst in instances:
                class_id, polygon_data, conf = inst
                class_name = class_names[class_id]
                shape = Shape(label=class_name, shape_type='polygon', score=conf)
                shape.points = [QPointF(p[0], p[1]) for p in polygon_data]
                shape.close()
                shapes_to_save.append(shape)
            
            labels_dir = os.path.join(self.temp_dir, "labels")
            os.makedirs(labels_dir, exist_ok=True)
            txt_path = os.path.join(labels_dir, os.path.splitext(os.path.basename(image_path))[0] + ".txt")

            save_yolo_labels(txt_path, shapes_to_save, img_w, img_h, class_names)
        except Exception as e:
            print(f"Failed to run inference on {os.path.basename(image_path)}: {e}")

    def open_training_dialog(self):
        dialog = TrainingDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            params = dialog.get_parameters()
            if not params['data']:
                QMessageBox.warning(self, "Warning", "Dataset YAML file is required.")
                return

            self.training_thread = TrainingThread(self.model, params)
            self.training_thread.training_finished.connect(self.on_training_finished)
            self.training_thread.training_failed.connect(self.on_training_failed)
            self.training_thread.start()
            
            self.train_action.setEnabled(False)
            self.statusBar().showMessage("Training started... Logs will be shown in the console.")

    def on_training_failed(self, error_msg):
        QMessageBox.critical(self, "Training Failed", error_msg)
        self.statusBar().showMessage("Training failed.", 5000)
        self.train_action.setEnabled(True)

    def on_training_finished(self, results):
        self.train_action.setEnabled(True)
        try:
            best_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')
            if os.path.exists(best_model_path):
                self.model_path = best_model_path # The new model is now our current model path.
                self.model = RealYOLOPredictor(self.model_path) # Load it
                QMessageBox.information(self, "Training Complete", f"Model has been updated to the newly trained version:\n{self.model_path}")
                self.statusBar().showMessage("Training complete. New model is now active.", 5000)
                
                # Update the class list based on the new model
                class_map = self.model.get_class_names()
                new_class_names = [class_map[i] for i in sorted(class_map.keys())]

                # Check for new classes and add them
                new_classes_added = False
                for name in new_class_names:
                    if name not in self.class_names:
                        self.class_names.append(name)
                        new_classes_added = True

                if new_classes_added:
                    self.rebuild_color_map_and_refresh_ui()

                # Also reload the current image to apply any potential new class colors
                if self.current_image_index != -1:
                    self.load_image_by_index(self.current_image_index, save_previous=False)
            else:
                raise FileNotFoundError("best.pt not found in results directory.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model after training: {e}")
            self.statusBar().showMessage("Error loading new model.", 5000)

    def open_class_manager(self):
        if not self.class_names and not self.image_paths:
            reply = QMessageBox.question(self, "No Classes", 
                                         "There are no classes yet. Would you like to add some?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply == QMessageBox.No:
                return

        old_names = self.class_names[:]
        dialog = ClassManagerDialog(old_names, self)
        
        if dialog.exec_() == QDialog.Accepted:
            new_names = dialog.get_final_class_names()

            if old_names == new_names:
                return

            self.viewer.store_shapes() # Save state for undo before any modifications

            removed_names = sorted(list(set(old_names) - set(new_names)))
            added_names = sorted(list(set(new_names) - set(old_names)))
            
            rename_map = {}
            
            # --- Heuristic to find renames ---
            if len(removed_names) == 1 and len(added_names) == 1:
                old, new = removed_names[0], added_names[0]
                rename_map[old] = new
                print(f"Detected rename: {old} -> {new}")
                # Apply the rename
                for shape in self.viewer.shapes:
                    if shape.label == old:
                        shape.label = new
                # Remove from lists so they are not treated as deletion/addition
                removed_names.clear()
                added_names.clear()

            # --- Handle Deletions ---
            if removed_names:
                shapes_to_delete = [s for s in self.viewer.shapes if s.label in removed_names]
                
                if shapes_to_delete:
                    reply = QMessageBox.question(self, "Delete Class in Use",
                                                 f"The class(es) '{', '.join(removed_names)}' are used by "
                                                 f"{len(shapes_to_delete)} instance(s) in the current image.\n\n"
                                                 "Do you want to permanently delete these instances as well?",
                                                 QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                    if reply == QMessageBox.Yes:
                        self.viewer.shapes = [s for s in self.viewer.shapes if s.label not in removed_names]
                    else:
                        QMessageBox.information(self, "Cancelled", "Class management operation has been cancelled.")
                        self.viewer.restore_shape() # Restore the initial state
                        return # Abort
            
            # --- Final Update ---
            self.class_names = new_names
            self.rebuild_color_map_and_refresh_ui()
            self.populate_instance_list() # To reflect name changes and deletions
            self.statusBar().showMessage("Class list updated.", 3000)

    def upload_labels(self):
        # 1. Select class name file
        class_file, _ = QFileDialog.getOpenFileName(self, "Select Class Name File (e.g., labels.txt)", "", "Text Files (*.txt)")
        if not class_file:
            return
        
        try:
            with open(class_file, 'r') as f:
                class_names = [line.strip() for line in f if line.strip()]
            self.class_names = class_names
            self.rebuild_color_map_and_refresh_ui()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read class file: {e}")
            return

        # 2. Select label directory
        label_dir = QFileDialog.getExistingDirectory(self, "Select Folder Containing Label Files")
        if not label_dir:
            self.source_label_dir = None
        else:
            self.source_label_dir = label_dir

        # 3. Clear all existing temporary labels to force a full reload
        temp_labels_path = os.path.join(self.temp_dir, "labels")
        if os.path.exists(temp_labels_path):
            progress_dialog = QProgressDialog("Clearing old temporary labels...", "Cancel", 0, len(self.image_paths), self)
            progress_dialog.setWindowModality(Qt.WindowModal)
            progress_dialog.setWindowTitle("Uploading Labels")

            for i, (img_path, _) in enumerate(self.image_paths):
                progress_dialog.setValue(i)
                QApplication.processEvents()
                if progress_dialog.wasCanceled():
                    break

                txt_file = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
                temp_label_path_file = os.path.join(temp_labels_path, txt_file)
                if os.path.exists(temp_label_path_file):
                    os.remove(temp_label_path_file)
            
            progress_dialog.setValue(len(self.image_paths))

        # 4. Reload labels for the current image from the new source.
        self.load_labels_for_current_image()
        self.statusBar().showMessage("Labels uploaded and temporary cache cleared.", 3000)

    def load_labels_for_current_image(self):
        if not self.class_names:
            # If no classes are defined, we cannot load labels. Clear and return.
            self.viewer.shapes = []
            self.viewer.store_shapes()
            self.populate_instance_list()
            self.viewer.update()
            return

        if self.current_image_index == -1:
            self.viewer.clear_polygons()
            self.populate_instance_list()
            self.viewer.update()
            return

        img_path, (img_w, img_h) = self.image_paths[self.current_image_index]
        txt_file = os.path.splitext(os.path.basename(img_path))[0] + ".txt"

        temp_label_path = os.path.join(self.temp_dir, "labels", txt_file)
        
        source_label_path = None
        if self.source_label_dir:
            source_label_path = os.path.join(self.source_label_dir, txt_file)

        label_path_to_load = None
        # Default behavior: prioritize temporary (edited) labels over source labels.
        if os.path.exists(temp_label_path):
            label_path_to_load = temp_label_path
        elif source_label_path and os.path.exists(source_label_path):
            label_path_to_load = source_label_path
        
        shapes = []
        if label_path_to_load:
            try:
                shapes = load_yolo_labels(label_path_to_load, img_w, img_h, self.class_names)
            except Exception as e:
                print(f"Error loading label file {label_path_to_load}: {e}")

        self.viewer.shapes = shapes
        self.viewer.store_shapes()
        self.populate_instance_list()
        
        scores = [s.score for s in self.viewer.shapes if s.score is not None]
        avg_conf = sum(scores) / len(scores) if scores else 0.0
        self.conf_label.setText(f"Avg. Confidence: {avg_conf:.2f}")

        self.viewer.update()

    def _show_image_only(self, index):
        """Loads and displays an image without its labels."""
        img_path, (img_w, img_h) = self.image_paths[index]

        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            QMessageBox.warning(self, "Error", f"Failed to load image with OpenCV: {img_path}")
            return False

        self.current_image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width, channel = self.current_image_rgb.shape
        bytes_per_line = 3 * width
        q_image = QImage(self.current_image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        if self.sam_predictor:
            self.viewer.clear_sam_points()

        self.viewer.clear_polygons()
        self.instance_list_widget.clear()
        self.viewer.set_image(pixmap)
        self.viewer.fit_to_window()
        return True

    def load_image_by_index(self, index, save_previous=True):
        if not (0 <= index < len(self.image_paths)):
            return

        if save_previous:
            self.save_current_labels()

        if self.draw_sam_action.isChecked():
            self.draw_sam_action.setChecked(False)
            self.toggle_sam_mode(False)
        
        self.current_image_index = index
        self.file_list_widget.setCurrentRow(index)
        
        if self._show_image_only(index):
            self.load_labels_for_current_image()
        
        self.viewer.update()

    def clear_viewer(self):
        self.viewer.clear_polygons()
        self.viewer.set_image(QPixmap())
        self.instance_list_widget.clear()
        self.conf_label.setText("Avg. Confidence: N/A")
        self.set_actions_enabled(False)
        self.open_folder_action.setEnabled(True)

    def populate_instance_list(self):
        self.instance_list_widget.clear()
        for i, shape in enumerate(self.viewer.shapes):
            score_text = f"({shape.score:.2f})" if shape.score is not None else ""
            item = QListWidgetItem(f"[{i}] {shape.label} {score_text}")
            item.setData(Qt.UserRole, i)
            
            try:
                if shape.label in self.class_names:
                    class_index = self.class_names.index(shape.label)
                    color = self.color_map[class_index % len(self.color_map)]
                    item.setForeground(color)
                    shape.line_color = color
            except (ValueError, IndexError):
                pass # Ignore if class name not in list or color map mismatch

            self.instance_list_widget.addItem(item)

    def rebuild_color_map_and_refresh_ui(self):
        """Rebuilds the color map from the class names and refreshes the UI."""
        self.color_map = []
        if not self.class_names:
            self.refresh_class_list()
            return
            
        hue_step = 360.0 / len(self.class_names)
        for i in range(len(self.class_names)):
            color = QColor.fromHsv(int(i * hue_step), 200, 200)
            self.color_map.append(color)
        
        self.refresh_class_list()

    def refresh_class_list(self):
        self.class_list_widget.clear()
        for i, name in enumerate(self.class_names):
            item = QListWidgetItem(name)
            try:
                if i < len(self.color_map):
                    color = self.color_map[i]
                    item.setForeground(color)
            except IndexError:
                pass
            self.class_list_widget.addItem(item)

    def on_file_item_clicked(self, item):
        index = self.file_list_widget.row(item)
        self.load_image_by_index(index)

    def on_instance_item_clicked(self, item):
        instance_id = item.data(Qt.UserRole)
        shape = self.viewer.shapes[instance_id]
        self.viewer.select_shape(shape)

    def on_polygon_selected(self, shape):
        if not shape:
            self.instance_list_widget.clearSelection()
            return
        instance_id = self.viewer.shapes.index(shape)
        for i in range(self.instance_list_widget.count()):
            item = self.instance_list_widget.item(i)
            if item.data(Qt.UserRole) == instance_id:
                item.setSelected(True)
                break

    def prev_image(self):
        if self.current_image_index > 0:
            self.load_image_by_index(self.current_image_index - 1)

    def next_image(self):
        if self.current_image_index < len(self.image_paths) - 1:
            self.load_image_by_index(self.current_image_index + 1)

    def toggle_draw_mode(self, checked):
        self.viewer.set_draw_mode(checked)
        self.set_navigation_enabled(not checked) # Disable navigation when drawing
        if checked:
            # If SAM mode is on, turn it off.
            if self.draw_sam_action.isChecked():
                self.draw_sam_action.setChecked(False)
                self.toggle_sam_mode(False) # This will handle cleanup

    def toggle_sam_mode(self, checked):
        self.set_navigation_enabled(not checked) # Disable navigation when SAM is active or preparing
        if checked:
            if self.is_encoding or self.current_image_index < 0:
                self.draw_sam_action.setChecked(False)
                self.set_navigation_enabled(True) # Re-enable if we bail out
                return

            current_img_path = self.image_paths[self.current_image_index][0]
            if self.sam_predictor.current_filename != current_img_path:
                self.is_encoding = True
                
                progress_dialog = QProgressDialog("Processing image for SAM...", "Cancel", 0, 0, self)
                progress_dialog.setWindowModality(Qt.WindowModal)
                progress_dialog.setWindowTitle("Preparing SAM")

                self.encoder_thread = ImageEncoderThread(self.sam_predictor, self.current_image_rgb, current_img_path)
                self.encoder_thread.encoding_finished.connect(progress_dialog.accept)
                self.encoder_thread.encoding_failed.connect(progress_dialog.reject)
                self.encoder_thread.encoding_finished.connect(self.on_sam_encoding_finished)
                self.encoder_thread.encoding_failed.connect(self.on_sam_encoding_failed)
                self.encoder_thread.start()

                if progress_dialog.exec_() == QDialog.Rejected:
                    self.encoder_thread.terminate() # Should be used with caution
                    self.on_sam_encoding_failed("Encoding cancelled or failed.")
                    return
            else:
                # Already encoded, just enable the mode
                self.on_sam_encoding_finished()
        else:
            self.is_sam_mode = False
            self.sam_point_mode = None
            self.viewer.set_editing(True)
            self.viewer.clear_sam_points()
            self.statusBar().showMessage("Ready", 2000)

    def on_sam_encoding_finished(self):
        self.is_encoding = False
        self.draw_sam_action.setEnabled(True)
        print("SAM processing complete.")
        self.statusBar().showMessage("SAM processing complete. Press 'Q' for positive points, 'E' for negative.", 5000)

        # Proceed to enable SAM mode
        self.is_sam_mode = True
        self.viewer.set_editing(False)
        if self.draw_poly_action.isChecked():
            self.draw_poly_action.setChecked(False)

    def on_sam_encoding_failed(self, error_msg):
        self.is_encoding = False
        self.draw_sam_action.setEnabled(True)
        self.draw_sam_action.setChecked(False)
        self.set_navigation_enabled(True) # Re-enable navigation on failure
        self.statusBar().showMessage("SAM model loading failed.", 5000)
        QMessageBox.critical(self, "SAM Encoding Error", f"Failed to prepare SAM model:\n{error_msg}")

    def on_new_polygon_drawn(self, shape):
        self.draw_poly_action.setChecked(False)
        self.toggle_draw_mode(False)
        self.viewer.store_shapes()

        dialog = ClassSelectorDialog(self.class_names, self)
        if dialog.exec_() == QDialog.Accepted:
            class_name = dialog.get_selected_class()
            
            if not class_name: # Should not happen if dialog is accepted, but as a safeguard
                self.viewer.restore_shape()
                self.viewer.update()
                return

            # Check if it's a new class
            if class_name not in self.class_names:
                self.class_names.append(class_name)
                self.rebuild_color_map_and_refresh_ui()

            # Assign the class and add the shape
            shape.label = class_name
            shape.score = 1.0
            self.viewer.shapes.append(shape)
            self.populate_instance_list()
            self.viewer.update()
            self.viewer.store_shapes() # Store the final state with the new shape
        else:
            # User cancelled, so remove the shape
            self.viewer.restore_shape()
            self.viewer.update()
        
    def save_current_labels(self):
        if self.current_image_index == -1:
            return

        img_path, (img_w, img_h) = self.image_paths[self.current_image_index]
        
        txt_file = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        
        # Save to a temporary directory instead of the source
        labels_dir = os.path.join(self.temp_dir, "labels")
        os.makedirs(labels_dir, exist_ok=True)
        txt_path = os.path.join(labels_dir, txt_file)

        save_yolo_labels(txt_path, self.viewer.shapes, img_w, img_h, self.class_names)
        self.statusBar().showMessage(f"Saved labels for {os.path.basename(img_path)} to temp.", 2000)

    def export_files(self):
        if not self.image_paths:
            QMessageBox.warning(self, "Warning", "No images to export.")
            return

        # 1. Get the filter class list file
        class_filter_path, _ = QFileDialog.getOpenFileName(self, "Select Filter Class File (e.g., classes.txt)", "", "Text Files (*.txt)")
        if not class_filter_path:
            return

        try:
            with open(class_filter_path, 'r') as f:
                valid_export_classes = [line.strip() for line in f if line.strip()]
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read class filter file: {e}")
            return

        if not valid_export_classes:
            QMessageBox.warning(self, "Warning", "Class filter file is empty. No labels will be exported.")

        # 2. Get destination directories
        dest_images_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory for Images")
        if not dest_images_dir:
            return

        dest_labels_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory for Labels")
        if not dest_labels_dir:
            return
        
        # 3. Process and export each file
        progress_dialog = QProgressDialog("Exporting files...", "Cancel", 0, len(self.image_paths), self)
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setWindowTitle("Exporting")

        exported_labels_count = 0
        for i, (source_img_path, (img_w, img_h)) in enumerate(self.image_paths):
            progress_dialog.setValue(i)
            QApplication.processEvents()
            if progress_dialog.wasCanceled():
                break

            img_filename = os.path.basename(source_img_path)
            txt_filename = os.path.splitext(img_filename)[0] + ".txt"

            # a. Copy image file
            try:
                shutil.copy(source_img_path, dest_images_dir)
            except Exception as e:
                print(f"Failed to copy image {img_filename}: {e}")
                continue # Skip to next image

            # b. Find the correct label file to use (temp > source)
            temp_label_path = os.path.join(self.temp_dir, "labels", txt_filename)
            source_label_path = os.path.join(self.source_label_dir, txt_filename) if self.source_label_dir else None
            
            label_path_to_process = None
            if os.path.exists(temp_label_path):
                label_path_to_process = temp_label_path
            elif source_label_path and os.path.exists(source_label_path):
                label_path_to_process = source_label_path

            # c. If a label exists, filter and write it
            if label_path_to_process:
                try:
                    # Read shapes using the application's current full class list
                    all_shapes = load_yolo_labels(label_path_to_process, img_w, img_h, self.class_names)
                    
                    # Filter shapes based on the valid_export_classes list
                    shapes_to_export = [s for s in all_shapes if s.label in valid_export_classes]

                    if shapes_to_export:
                        dest_txt_path = os.path.join(dest_labels_dir, txt_filename)
                        # Write the filtered shapes, mapping class names to the new filtered index
                        save_yolo_labels(dest_txt_path, shapes_to_export, img_w, img_h, valid_export_classes)
                        exported_labels_count += 1

                except Exception as e:
                    print(f"Failed to process or write label for {img_filename}: {e}")

        progress_dialog.setValue(len(self.image_paths))
        QMessageBox.information(self, "Export Complete", 
                                f"Exported {len(self.image_paths)} images and {exported_labels_count} label files.")

    def run_class_specification(self):
        if not self.image_paths:
            QMessageBox.warning(self, "Warning", "Please open an image folder first.")
            return
        if not self.class_names:
            QMessageBox.warning(self, "Warning", "There are no classes to specify. Please add or upload labels first.")
            return

        dialog = ClassSpecificationDialog(self.class_names, self.image_paths, self)
        if dialog.exec_() != QDialog.Accepted:
            return
        
        options = dialog.get_selected_options()
        target_classes = options["target_classes"]
        model_path = options["model_path"]
        image_paths = options["image_paths"]

        if not all([target_classes, model_path, image_paths]):
            QMessageBox.warning(self, "Warning", "All fields are required: Target Class(es), Model, and at least one Image.")
            return

        try:
            refinement_model = RealYOLOPredictor(model_path)
            refinement_model_classes = [refinement_model.get_class_names()[i] for i in sorted(refinement_model.get_class_names().keys())]
            
            # Eagerly add any newly discovered classes to the main list to prevent data loss on save.
            new_classes_added = False
            for name in refinement_model_classes:
                if name not in self.class_names:
                    self.class_names.append(name)
                    new_classes_added = True
            if new_classes_added:
                self.rebuild_color_map_and_refresh_ui()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load refinement model: {e}")
            return

        progress_dialog = QProgressDialog("Running Class Specification...", "Cancel", 0, len(image_paths), self)
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setWindowTitle("Class Specification")

        for i, img_path in enumerate(image_paths):
            progress_dialog.setValue(i)
            progress_dialog.setLabelText(f"Processing {os.path.basename(img_path)}...")
            QApplication.processEvents()
            if progress_dialog.wasCanceled():
                break
            
            img_dims = next((dims for path, dims in self.image_paths if path == img_path), None)
            if not img_dims:
                continue

            self.perform_specification_on_image(img_path, img_dims[0], img_dims[1], target_classes, refinement_model, refinement_model_classes)
        
        progress_dialog.setValue(len(image_paths))
        QMessageBox.information(self, "Complete", "Class specification process finished.")
        
        # Reload the current image to show changes if it was processed
        if self.image_paths[self.current_image_index][0] in image_paths:
            self.load_labels_for_current_image()

    def perform_specification_on_image(self, image_path, img_w, img_h, target_classes, refinement_model, refinement_model_classes):
        applied_labels = set()
        txt_filename = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
        temp_label_path = os.path.join(self.temp_dir, "labels", txt_filename)
        source_label_path = os.path.join(self.source_label_dir, txt_filename) if self.source_label_dir else None
        
        label_path_to_process = None
        if os.path.exists(temp_label_path):
            label_path_to_process = temp_label_path
        elif source_label_path and os.path.exists(source_label_path):
            label_path_to_process = source_label_path

        if not label_path_to_process:
            return applied_labels

        try:
            original_image = cv2.imread(image_path)
            all_shapes = load_yolo_labels(label_path_to_process, img_w, img_h, self.class_names)
            
            final_shapes = []
            # padding = 20  <-- Removed fixed padding

            for shape in all_shapes:
                if shape.label not in target_classes:
                    final_shapes.append(shape)
                    continue

                rect = shape.bounding_rect().toRect()
                
                # Dynamic padding: Add 50% of width/height on each side (approx 2x context)
                w = rect.width()
                h = rect.height()
                pad_x = int(w * 0.5)
                pad_y = int(h * 0.5)
                
                x1 = max(0, rect.left() - pad_x)
                y1 = max(0, rect.top() - pad_y)
                x2 = min(img_w, rect.right() + pad_x)
                y2 = min(img_h, rect.bottom() + pad_y)
                
                cropped_image = original_image[y1:y2, x1:x2]
                
                if cropped_image.size == 0:
                    final_shapes.append(shape)
                    continue

                temp_crop_path = os.path.join(self.temp_dir, "crop.jpg")
                cv2.imwrite(temp_crop_path, cropped_image)

                # Use lower confidence (0.15) for refinement to capture subtle detections
                instances, _, _ = refinement_model.predict_and_optimize(temp_crop_path, conf=0.15)
                
                if not instances:
                    final_shapes.append(shape)
                    continue

                best_instance = max(instances, key=lambda item: item[2])
                new_class_id = best_instance[0]
                new_conf = best_instance[2]
                new_label = refinement_model_classes[new_class_id]

                shape.label = new_label
                shape.score = new_conf
                final_shapes.append(shape)
                applied_labels.add(new_label)

            temp_labels_dir = os.path.join(self.temp_dir, "labels")
            os.makedirs(temp_labels_dir, exist_ok=True)
            save_yolo_labels(temp_label_path, final_shapes, img_w, img_h, self.class_names)

        except Exception as e:
            print(f"Failed during specification for {os.path.basename(image_path)}: {e}")
        
        return applied_labels

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()

        # Block all input if SAM is encoding
        if self.is_encoding:
            event.accept()
            return

        # --- Global Shortcuts ---
        if key == Qt.Key_Escape:
            self.cancel_all_modes()
            event.accept()
            return

        # --- Drawing Mode Shortcuts ---
        if self.draw_poly_action.isChecked():
            if key == Qt.Key_Backspace:
                self.viewer.undo_last_point()
            # Block other conflicting shortcuts
            elif key in [Qt.Key_A, Qt.Key_D, Qt.Key_S]:
                self.statusBar().showMessage("Finish or cancel drawing before changing images or modes.", 2000)
            elif key == Qt.Key_W:
                 self.draw_poly_action.setChecked(False) # Toggle off
            event.accept()
            return

        # --- SAM Mode Shortcuts ---
        if self.is_sam_mode:
            if key == Qt.Key_Q:
                self.sam_point_mode = 1
                self.statusBar().showMessage("Adding positive points (Left-click to add).")
            elif key == Qt.Key_E:
                self.sam_point_mode = 0
                self.statusBar().showMessage("Adding negative points (Left-click to add).")
            elif key == Qt.Key_G:
                self.viewer.reset_sam_prediction()
                self.statusBar().showMessage("SAM points cleared.", 2000)
            elif key == Qt.Key_F:
                new_shapes = self.viewer.finalize_sam_shape()
                if new_shapes:
                    dialog = ClassSelectorDialog(self.class_names, self)
                    if dialog.exec_() == QDialog.Accepted:
                        class_name = dialog.get_selected_class()
                        
                        if not class_name:
                             self.viewer.clear_sam_points()
                             self.viewer.update()
                             return

                        # Check if it's a new class and update UI
                        if class_name not in self.class_names:
                            self.class_names.append(class_name)
                            self.rebuild_color_map_and_refresh_ui()

                        # Assign class to all new shapes from SAM
                        self.viewer.store_shapes()
                        for shape in new_shapes:
                            shape.label = class_name
                            shape.score = 1.0
                            self.viewer.shapes.append(shape)
                        
                        self.populate_instance_list()
                        self.viewer.store_shapes()
                
                # Clean up viewer state regardless
                self.viewer.clear_sam_points()
                self.viewer.update()
            elif key == Qt.Key_S:
                self.draw_sam_action.setChecked(False) # Toggle off
            # Block other conflicting shortcuts
            elif key in [Qt.Key_A, Qt.Key_D, Qt.Key_W]:
                 self.statusBar().showMessage("Finish or cancel SAM mode before changing images or modes.", 2000)
            event.accept()
            return

        # --- Normal / Edit Mode Shortcuts ---
        if modifiers == Qt.ControlModifier and key == Qt.Key_S:
            self.save_labels_action.trigger()

        elif modifiers == Qt.ControlModifier and key == Qt.Key_Z:
            self.undo_action.trigger()

        elif key == Qt.Key_A:
            self.prev_image_action.trigger()

        elif key == Qt.Key_D:
            self.next_image_action.trigger()
        
        elif key == Qt.Key_W:
            self.draw_poly_action.setChecked(True)

        elif key == Qt.Key_S:
            if self.draw_sam_action.isEnabled():
                self.draw_sam_action.setChecked(True)

        elif key == Qt.Key_Delete or key == Qt.Key_Backspace:
            if self.viewer.delete_selected_vertex():
                self.populate_instance_list()
            elif self.viewer.selected_shapes:
                reply = QMessageBox.question(self, "Delete", "Delete selected instances?",
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply == QMessageBox.Yes:
                    self.delete_selected_instances()
        else:
            super().keyPressEvent(event)
            return
        
        event.accept()

    def cancel_all_modes(self):
        # Deactivate drawing modes
        if self.draw_poly_action.isChecked():
            self.draw_poly_action.setChecked(False)
            self.toggle_draw_mode(False)

        if self.draw_sam_action.isChecked():
            self.draw_sam_action.setChecked(False)
            self.toggle_sam_mode(False)
        
        self.sam_point_mode = None
        # Clear any unfinished drawings in the viewer
        self.viewer.cancel_drawing()
        
        # Ensure viewer is in edit mode
        self.viewer.set_editing(True)
        
        self.statusBar().showMessage("Drawing cancelled", 2000)

    def delete_selected_instances(self):
        self.viewer.store_shapes()
        for shape in self.viewer.selected_shapes:
            self.viewer.shapes.remove(shape)
        self.viewer.deselect_shape()
        self.populate_instance_list()
        self.viewer.update()
        
    def on_instance_double_clicked(self, item):
        instance_id = item.data(Qt.UserRole)
        shape = self.viewer.shapes[instance_id]
        self.change_instance_class(shape)
        
    def change_instance_class(self, shape):
        self.viewer.store_shapes()
        current_class_name = shape.label
        class_name, ok = QInputDialog.getItem(self, "Select Class", "Class:", self.class_names, 
                                            self.class_names.index(current_class_name), False)
        if ok and class_name and class_name != current_class_name:
            shape.label = class_name
            self.populate_instance_list()
            self.viewer.update()
            self.viewer.store_shapes()

    def undo_shape(self):
        self.viewer.restore_shape()
        self.populate_instance_list()

    def show_file_list_context_menu(self, pos):
        item = self.file_list_widget.itemAt(pos)
        if item is None:
            return

        context_menu = QMenu(self)
        delete_action = context_menu.addAction("Delete")
        action = context_menu.exec_(self.file_list_widget.mapToGlobal(pos))

        if action == delete_action:
            self.delete_selected_image()

    def delete_selected_image(self):
        selected_items = self.file_list_widget.selectedItems()
        if not selected_items:
            return

        item = selected_items[0]
        index = self.file_list_widget.row(item)
        
        reply = QMessageBox.question(self, "Delete Image", 
                                     f"Are you sure you want to permanently delete {item.text()} and its label?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            img_path, _ = self.image_paths[index]
            txt_file = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
            labels_dir = os.path.join(os.path.dirname(os.path.dirname(img_path)), "labels")
            txt_path = os.path.join(labels_dir, txt_file)

            try:
                if os.path.exists(img_path):
                    os.remove(img_path)
                if os.path.exists(txt_path):
                    os.remove(txt_path)
                
                # Remove from data structure and UI
                del self.image_paths[index]
                self.file_list_widget.takeItem(index)

                self.statusBar().showMessage(f"Deleted {os.path.basename(img_path)}", 3000)

                # Update viewer
                if not self.image_paths:
                    self.clear_viewer()
                    self.current_image_index = -1
                elif index == self.current_image_index:
                    # If the deleted image was the current one, load the next one or previous one
                    new_index = min(index, len(self.image_paths) - 1)
                    if new_index < 0:
                        self.clear_viewer()
                    else:
                        self.load_image_by_index(new_index, save_previous=False)
                elif index < self.current_image_index:
                    # If an image before the current one is deleted, the index shifts
                    self.current_image_index -= 1
                    self.file_list_widget.setCurrentRow(self.current_image_index)


            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete files: {e}")

    def closeEvent(self, event):
        """Clean up temporary files on exit."""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                print(f"Cleared temporary directory: {self.temp_dir}")
        except Exception as e:
            print(f"Error cleaning up temporary directory: {e}")
        finally:
            event.accept()
            
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
