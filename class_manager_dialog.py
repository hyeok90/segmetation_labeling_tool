from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, 
    QPushButton, QDialogButtonBox, QInputDialog, QMessageBox
)

class ClassManagerDialog(QDialog):
    def __init__(self, class_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Class Manager")
        self.setMinimumSize(300, 400)

        # Make a copy to work with
        self.class_names = class_names[:]

        # --- Widgets ---
        self.list_widget = QListWidget()
        self.list_widget.addItems(self.class_names)

        self.add_button = QPushButton("Add...")
        self.rename_button = QPushButton("Rename...")
        self.delete_button = QPushButton("Delete")

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)

        # --- Layout ---
        layout = QVBoxLayout(self) 
        
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.rename_button)
        button_layout.addWidget(self.delete_button)

        layout.addLayout(button_layout)
        layout.addWidget(self.list_widget)
        layout.addWidget(self.button_box)

        # --- Connections ---
        self.add_button.clicked.connect(self.add_class)
        self.rename_button.clicked.connect(self.rename_class)
        self.delete_button.clicked.connect(self.delete_class)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

    def add_class(self):
        new_class, ok = QInputDialog.getText(self, "Add Class", "Enter new class name:")
        if ok and new_class:
            new_class = new_class.strip()
            if not new_class:
                QMessageBox.warning(self, "Warning", "Class name cannot be empty.")
                return
            if new_class in self.class_names:
                QMessageBox.warning(self, "Warning", f"Class '{new_class}' already exists.")
                return
            
            self.class_names.append(new_class)
            self.list_widget.addItem(new_class)

    def rename_class(self):
        selected_item = self.list_widget.currentItem()
        if not selected_item:
            QMessageBox.warning(self, "Warning", "Please select a class to rename.")
            return

        old_name = selected_item.text()
        new_name, ok = QInputDialog.getText(self, "Rename Class", "Enter new name:", text=old_name)

        if ok and new_name:
            new_name = new_name.strip()
            if not new_name:
                QMessageBox.warning(self, "Warning", "Class name cannot be empty.")
                return
            if new_name != old_name and new_name in self.class_names:
                QMessageBox.warning(self, "Warning", f"Class '{new_name}' already exists.")
                return

            # Update internal list
            index = self.class_names.index(old_name)
            self.class_names[index] = new_name
            # Update list widget
            selected_item.setText(new_name)

    def delete_class(self):
        selected_item = self.list_widget.currentItem()
        if not selected_item:
            QMessageBox.warning(self, "Warning", "Please select a class to delete.")
            return

        reply = QMessageBox.question(self, "Delete Class", 
                                     f"Are you sure you want to delete '{selected_item.text()}'?\nThis might affect existing labels.",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            name_to_delete = selected_item.text()
            self.class_names.remove(name_to_delete)
            self.list_widget.takeItem(self.list_widget.row(selected_item))

    def get_final_class_names(self):
        return self.class_names
