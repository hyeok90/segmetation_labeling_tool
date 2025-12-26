import os
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QListWidget, QListWidgetItem, 
    QCheckBox, QDialogButtonBox, QFileDialog, QLineEdit, 
    QPushButton, QGroupBox, QFormLayout, QDoubleSpinBox
)
from PyQt5.QtCore import Qt

class InferenceDialog(QDialog):
    def __init__(self, image_paths, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Run Inference on Images")
        self.layout = QVBoxLayout(self)
        self.setMinimumSize(400, 600)

        # --- Model Selection ---
        self.model_group = QGroupBox("Model Selection")
        self.model_layout = QFormLayout(self.model_group)
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setReadOnly(True)
        self.model_path_edit.setPlaceholderText("Select a model file...")
        self.model_path_edit.textChanged.connect(self.validate_inputs)
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_model)
        self.model_layout.addRow("Model Path:", self.browse_button)
        self.model_layout.addRow(self.model_path_edit)
        self.layout.addWidget(self.model_group)

        # --- Settings ---
        self.settings_group = QGroupBox("Inference Settings")
        self.settings_layout = QFormLayout(self.settings_group)
        
        self.epsilon_spin = QDoubleSpinBox()
        self.epsilon_spin.setRange(0.0, 10.0)
        self.epsilon_spin.setSingleStep(0.1)
        self.epsilon_spin.setValue(1.0)
        
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.25) # Default YOLO conf
        
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.01, 1.0)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(0.7) # Default YOLO NMS IoU
        self.iou_spin.setToolTip("Lower value = stricter NMS (fewer overlaps)")

        self.settings_layout.addRow("Confidence Threshold:", self.conf_spin)
        self.settings_layout.addRow("NMS IoU Threshold:", self.iou_spin)
        self.settings_layout.addRow("Polygon Simplify (Epsilon):", self.epsilon_spin)
        
        self.layout.addWidget(self.settings_group)

        # --- Image Selection ---
        self.image_group = QGroupBox("Image Selection")
        self.image_layout = QVBoxLayout(self.image_group)
        
        self.image_paths = [path for path, _ in image_paths]
        
        self.list_widget = QListWidget()
        for path in self.image_paths:
            item = QListWidgetItem(os.path.basename(path))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.list_widget.addItem(item)
        
        self.list_widget.itemChanged.connect(self.validate_inputs)

        self.select_all_checkbox = QCheckBox("Select All")
        self.select_all_checkbox.stateChanged.connect(self.toggle_select_all)
        
        self.image_layout.addWidget(self.select_all_checkbox)
        self.image_layout.addWidget(self.list_widget)
        self.layout.addWidget(self.image_group)
        
        # --- Dialog Buttons ---
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.button_box.button(QDialogButtonBox.Ok).setEnabled(False) # Default disabled
        self.layout.addWidget(self.button_box)

    def validate_inputs(self):
        model_valid = bool(self.model_path_edit.text().strip())
        
        images_valid = False
        for i in range(self.list_widget.count()):
            if self.list_widget.item(i).checkState() == Qt.Checked:
                images_valid = True
                break
        
        self.button_box.button(QDialogButtonBox.Ok).setEnabled(model_valid and images_valid)

    def browse_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model for Inference", "", "PyTorch Models (*.pt);;All Files (*)")
        if file_path:
            self.model_path_edit.setText(file_path)

    def get_selected_model(self):
        return self.model_path_edit.text()

    def get_epsilon(self):
        return self.epsilon_spin.value()

    def get_conf(self):
        return self.conf_spin.value()

    def get_iou(self):
        return self.iou_spin.value()

    def toggle_select_all(self, state):
        check_state = Qt.Checked if state == Qt.Checked else Qt.Unchecked
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(check_state)
            
    def get_selected_images(self):
        selected_images = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.Checked:
                selected_images.append(self.image_paths[i])
        return selected_images
