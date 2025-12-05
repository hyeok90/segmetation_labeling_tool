from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem, QComboBox,
    QPushButton, QDialogButtonBox, QLabel, QLineEdit, QFileDialog, QGroupBox,
    QCheckBox
)
from PyQt5.QtCore import Qt
import os

class ClassSpecificationDialog(QDialog):
    def __init__(self, class_names, image_paths, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Class Specification")
        self.setMinimumSize(500, 600)

        self.image_paths = [path for path, _ in image_paths]
        self.selected_class = None
        self.model_path = None

        # --- Layout ---
        layout = QVBoxLayout(self)

        # --- Target Class Selection ---
        class_group = QGroupBox("1. Select Target Class(es)")
        class_layout = QVBoxLayout(class_group)
        self.class_list = QListWidget()
        self.class_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.class_list.addItems(class_names)
        class_layout.addWidget(QLabel("Specify polygons with these classes (use Ctrl or Shift to select multiple):"))
        class_layout.addWidget(self.class_list)
        layout.addWidget(class_group)

        # --- Model Selection ---
        model_group = QGroupBox("2. Select Refinement Model")
        model_layout = QHBoxLayout(model_group)
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setReadOnly(True)
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_model)
        model_layout.addWidget(self.model_path_edit)
        model_layout.addWidget(browse_button)
        layout.addWidget(model_group)

        # --- Image Selection ---
        image_group = QGroupBox("3. Select Images to Apply Specification")
        image_layout = QVBoxLayout(image_group)
        self.list_widget = QListWidget()
        for path in self.image_paths:
            item = QListWidgetItem(os.path.basename(path))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.list_widget.addItem(item)
        
        select_all_checkbox = QCheckBox("Select All")
        select_all_checkbox.stateChanged.connect(self.toggle_select_all)
        image_layout.addWidget(select_all_checkbox)
        image_layout.addWidget(self.list_widget)
        layout.addWidget(image_group)
        
        # --- Dialog Buttons ---
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.button(QDialogButtonBox.Ok).setText("Run Specification")
        layout.addWidget(self.button_box)

        # --- Connections ---
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

    def browse_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Refinement Model", "", "PyTorch Models (*.pt)")
        if file_path:
            self.model_path_edit.setText(file_path)

    def toggle_select_all(self, state):
        check_state = Qt.Checked if state == Qt.Checked else Qt.Unchecked
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(check_state)
            
    def get_selected_options(self):
        selected_images = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.Checked:
                selected_images.append(self.image_paths[i])
        
        selected_classes = [item.text() for item in self.class_list.selectedItems()]

        return {
            "target_classes": selected_classes,
            "model_path": self.model_path_edit.text(),
            "image_paths": selected_images
        }
