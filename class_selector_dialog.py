from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QLineEdit,
    QPushButton, QDialogButtonBox, QLabel, QMessageBox
)

class ClassSelectorDialog(QDialog):
    def __init__(self, class_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select or Create Class")
        self.setMinimumSize(350, 400)

        self.class_names = class_names
        self.selected_class = None

        # --- Widgets ---
        self.instructions_label = QLabel("Double-click to select an existing class, or create a new one below.")
        
        self.list_widget = QListWidget()
        self.list_widget.addItems(self.class_names)
        
        self.new_class_edit = QLineEdit()
        self.new_class_edit.setPlaceholderText("Enter new class name...")
        
        self.create_button = QPushButton("Create and Select")
        
        self.button_box = QDialogButtonBox(QDialogButtonBox.Cancel)

        # --- Layout ---
        layout = QVBoxLayout(self)
        layout.addWidget(self.instructions_label)
        layout.addWidget(self.list_widget)

        create_layout = QHBoxLayout()
        create_layout.addWidget(self.new_class_edit)
        create_layout.addWidget(self.create_button)
        
        layout.addLayout(create_layout)
        layout.addWidget(self.button_box)

        # --- Connections ---
        self.list_widget.itemDoubleClicked.connect(self.on_item_double_clicked)
        self.create_button.clicked.connect(self.on_create_clicked)
        self.new_class_edit.returnPressed.connect(self.on_create_clicked) # Allow pressing Enter
        self.button_box.rejected.connect(self.reject)

    def on_item_double_clicked(self, item):
        self.selected_class = item.text()
        self.accept()

    def on_create_clicked(self):
        new_class = self.new_class_edit.text().strip()
        if not new_class:
            QMessageBox.warning(self, "Warning", "Class name cannot be empty.")
            return
        
        # Although the main window handles logic, a quick check here improves UX
        if new_class in self.class_names:
            reply = QMessageBox.question(self, "Class Exists", 
                                         f"Class '{new_class}' already exists. Do you want to select it?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply == QMessageBox.No:
                return
        
        self.selected_class = new_class
        self.accept()

    def get_selected_class(self):
        return self.selected_class
