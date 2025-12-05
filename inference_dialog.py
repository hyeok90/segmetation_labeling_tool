import os
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QListWidget, QListWidgetItem, 
    QCheckBox, QDialogButtonBox, QFileDialog, QLineEdit, 
    QPushButton, QGroupBox, QFormLayout
)
from PyQt5.QtCore import Qt

class InferenceDialog(QDialog):
    def __init__(self, image_paths, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Run Inference on Images")
        self.layout = QVBoxLayout(self)
        self.setMinimumSize(400, 500)

        # --- Model Selection ---
        self.model_group = QGroupBox("Model Selection")
        self.model_layout = QFormLayout(self.model_group)
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setReadOnly(True)
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_model)
        self.model_layout.addRow("Model Path:", self.browse_button)
        self.model_layout.addRow(self.model_path_edit)
        self.layout.addWidget(self.model_group)

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
            
        self.select_all_checkbox = QCheckBox("Select All")
        self.select_all_checkbox.stateChanged.connect(self.toggle_select_all)
        
        self.image_layout.addWidget(self.select_all_checkbox)
        self.image_layout.addWidget(self.list_widget)
        self.layout.addWidget(self.image_group)
        
        # --- Dialog Buttons ---
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)

    def browse_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model for Inference", "", "PyTorch Models (*.pt)")
        if file_path:
            self.model_path_edit.setText(file_path)

    def get_selected_model(self):
        return self.model_path_edit.text()

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
