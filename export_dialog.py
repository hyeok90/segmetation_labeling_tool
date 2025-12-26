from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QDialogButtonBox, QLabel, QCheckBox, QFileDialog, QLineEdit, QGroupBox
)
from PyQt5.QtCore import Qt

class ExportDialog(QDialog):
    def __init__(self, all_class_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Dataset")
        self.setMinimumSize(400, 500)
        self.all_class_names = all_class_names

        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # 1. Output Directory Selection
        dir_group = QGroupBox("Output Directory")
        dir_layout = QHBoxLayout()
        self.dir_edit = QLineEdit()
        self.dir_edit.setPlaceholderText("Select destination folder...")
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self.browse_directory)
        dir_layout.addWidget(self.dir_edit)
        dir_layout.addWidget(self.browse_btn)
        dir_group.setLayout(dir_layout)
        layout.addWidget(dir_group)

        # 2. Options
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout()
        self.export_images_check = QCheckBox("Export Images")
        self.export_images_check.setChecked(False) # Default to labels only
        options_layout.addWidget(self.export_images_check)
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        # 3. Class Selection & Ordering
        class_group = QGroupBox("Select Classes to Export (Drag to Reorder)")
        class_layout = QVBoxLayout()
        self.class_list = QListWidget()
        self.class_list.setDragDropMode(QListWidget.InternalMove)
        
        for name in self.all_class_names:
            item = QListWidgetItem(name)
            item.setCheckState(Qt.Checked)
            self.class_list.addItem(item)
            
        class_layout.addWidget(self.class_list)
        
        # Select All / None buttons
        btn_layout = QHBoxLayout()
        self.select_all_btn = QPushButton("Select All")
        self.select_none_btn = QPushButton("Select None")
        self.select_all_btn.clicked.connect(self.select_all)
        self.select_none_btn.clicked.connect(self.select_none)
        btn_layout.addWidget(self.select_all_btn)
        btn_layout.addWidget(self.select_none_btn)
        class_layout.addLayout(btn_layout)
        
        class_group.setLayout(class_layout)
        layout.addWidget(class_group)

        # 4. Dialog Buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def browse_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.dir_edit.setText(directory)

    def select_all(self):
        for i in range(self.class_list.count()):
            self.class_list.item(i).setCheckState(Qt.Checked)

    def select_none(self):
        for i in range(self.class_list.count()):
            self.class_list.item(i).setCheckState(Qt.Unchecked)

    def get_export_data(self):
        """
        Returns:
            output_dir (str): Path to export directory
            export_images (bool): Whether to export images
            ordered_classes (list): List of class names in desired order (only checked ones)
        """
        output_dir = self.dir_edit.text().strip()
        export_images = self.export_images_check.isChecked()
        
        ordered_classes = []
        for i in range(self.class_list.count()):
            item = self.class_list.item(i)
            if item.checkState() == Qt.Checked:
                ordered_classes.append(item.text())
                
        return output_dir, export_images, ordered_classes
