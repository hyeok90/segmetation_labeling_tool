from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit, QDialogButtonBox, 
    QFileDialog, QPushButton, QGroupBox, QSpinBox, QDoubleSpinBox, 
    QGridLayout, QLabel, QComboBox, QHBoxLayout
)
from yaml_creator_dialog import YamlCreatorDialog

class TrainingDialog(QDialog):
    def __init__(self, current_model_path=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Train Model")
        self.setMinimumWidth(600)

        self.layout = QVBoxLayout(self)

        # Model Selection
        self.model_group = QGroupBox("Model")
        self.model_layout = QFormLayout(self.model_group)
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Select a base model (e.g. yolov11n-seg.pt)...")
        self.model_path_edit.textChanged.connect(self.validate_inputs)
        if current_model_path:
            self.model_path_edit.setText(current_model_path)
        
        self.model_browse_btn = QPushButton("Browse...")
        self.model_browse_btn.clicked.connect(self.browse_model)
        
        self.model_layout.addRow("Base Model:", self.model_browse_btn)
        self.model_layout.addRow("", self.model_path_edit)
        self.layout.addWidget(self.model_group)

        # Data selection
        self.data_group = QGroupBox("Data")
        self.data_layout = QFormLayout(self.data_group)
        self.yaml_path_edit = QLineEdit()
        self.yaml_path_edit.setPlaceholderText("Select a dataset YAML file...")
        self.yaml_path_edit.textChanged.connect(self.validate_inputs)
        
        # YAML Buttons layout
        self.yaml_buttons_layout = QHBoxLayout()
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_yaml)
        self.create_yaml_button = QPushButton("Create New...")
        self.create_yaml_button.clicked.connect(self.create_new_yaml)
        self.yaml_buttons_layout.addWidget(self.browse_button)
        self.yaml_buttons_layout.addWidget(self.create_yaml_button)
        
        self.data_layout.addRow("Dataset YAML:", self.yaml_buttons_layout)
        self.data_layout.addRow("", self.yaml_path_edit)
        self.layout.addWidget(self.data_group)

        # Training parameters
        self.training_group = QGroupBox("Training Parameters")
        self.training_layout = QGridLayout(self.training_group)
        
        self.epochs_spinbox = QSpinBox()
        self.epochs_spinbox.setRange(1, 10000)
        self.epochs_spinbox.setValue(100)
        self.batch_spinbox = QSpinBox()
        self.batch_spinbox.setRange(-1, 256)
        self.batch_spinbox.setValue(16)

        self.imgsz_spinbox = QSpinBox()
        self.imgsz_spinbox.setRange(32, 4096)
        self.imgsz_spinbox.setSingleStep(32)
        self.imgsz_spinbox.setValue(640)

        self.lr0_dspinbox = QDoubleSpinBox()
        self.lr0_dspinbox.setDecimals(5)
        self.lr0_dspinbox.setRange(0.0, 1.0)
        self.lr0_dspinbox.setSingleStep(0.001)
        self.lr0_dspinbox.setValue(0.01)
        self.lrf_dspinbox = QDoubleSpinBox()
        self.lrf_dspinbox.setDecimals(5)
        self.lrf_dspinbox.setRange(0.0, 1.0)
        self.lrf_dspinbox.setSingleStep(0.001)
        self.lrf_dspinbox.setValue(0.01)
        self.patience_spinbox = QSpinBox()
        self.patience_spinbox.setRange(0, 1000)
        self.patience_spinbox.setValue(50)

        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(['auto', 'SGD', 'Adam', 'AdamW'])

        self.training_layout.addWidget(QLabel("Epochs:"), 0, 0)
        self.training_layout.addWidget(self.epochs_spinbox, 0, 1)
        self.training_layout.addWidget(QLabel("Batch Size:"), 0, 2)
        self.training_layout.addWidget(self.batch_spinbox, 0, 3)

        self.training_layout.addWidget(QLabel("Image Size:"), 1, 0)
        self.training_layout.addWidget(self.imgsz_spinbox, 1, 1)

        self.training_layout.addWidget(QLabel("Optimizer:"), 2, 0)
        self.training_layout.addWidget(self.optimizer_combo, 2, 1)
        self.training_layout.addWidget(QLabel("Initial LR (lr0):"), 2, 2)
        self.training_layout.addWidget(self.lr0_dspinbox, 2, 3)
        self.training_layout.addWidget(QLabel("Final LR (lrf):"), 2, 4)
        self.training_layout.addWidget(self.lrf_dspinbox, 2, 5)

        self.training_layout.addWidget(QLabel("Patience:"), 3, 0)
        self.training_layout.addWidget(self.patience_spinbox, 3, 1)
        self.layout.addWidget(self.training_group)

        # Augmentation parameters
        self.aug_group = QGroupBox("Augmentation Parameters")
        self.aug_main_layout = QVBoxLayout(self.aug_group)

        # Geometry Group
        self.geometry_group = QGroupBox("Geometry")
        self.geometry_layout = QGridLayout(self.geometry_group)
        self.degrees_dspinbox = QDoubleSpinBox()
        self.degrees_dspinbox.setRange(0.0, 180.0)
        self.degrees_dspinbox.setValue(0.0)
        self.translate_dspinbox = QDoubleSpinBox()
        self.translate_dspinbox.setRange(0.0, 1.0)
        self.translate_dspinbox.setValue(0.1)
        self.scale_dspinbox = QDoubleSpinBox()
        self.scale_dspinbox.setRange(0.0, 1.0)
        self.scale_dspinbox.setValue(0.5)
        self.shear_dspinbox = QDoubleSpinBox()
        self.shear_dspinbox.setRange(0.0, 90.0)
        self.shear_dspinbox.setValue(0.0)
        self.geometry_layout.addWidget(QLabel("Degrees:"), 0, 0)
        self.geometry_layout.addWidget(self.degrees_dspinbox, 0, 1)
        self.geometry_layout.addWidget(QLabel("Translate:"), 0, 2)
        self.geometry_layout.addWidget(self.translate_dspinbox, 0, 3)
        self.geometry_layout.addWidget(QLabel("Scale:"), 1, 0)
        self.geometry_layout.addWidget(self.scale_dspinbox, 1, 1)
        self.geometry_layout.addWidget(QLabel("Shear:"), 1, 2)
        self.geometry_layout.addWidget(self.shear_dspinbox, 1, 3)
        self.aug_main_layout.addWidget(self.geometry_group)

        # Color Group
        self.color_group = QGroupBox("Color")
        self.color_layout = QGridLayout(self.color_group)
        self.hsv_h_dspinbox = QDoubleSpinBox()
        self.hsv_h_dspinbox.setRange(0.0, 1.0)
        self.hsv_h_dspinbox.setValue(0.015)
        self.hsv_s_dspinbox = QDoubleSpinBox()
        self.hsv_s_dspinbox.setRange(0.0, 1.0)
        self.hsv_s_dspinbox.setValue(0.7)
        self.hsv_v_dspinbox = QDoubleSpinBox()
        self.hsv_v_dspinbox.setRange(0.0, 1.0)
        self.hsv_v_dspinbox.setValue(0.4)
        self.color_layout.addWidget(QLabel("HSV-Hue:"), 0, 0)
        self.color_layout.addWidget(self.hsv_h_dspinbox, 0, 1)
        self.color_layout.addWidget(QLabel("HSV-Saturation:"), 0, 2)
        self.color_layout.addWidget(self.hsv_s_dspinbox, 0, 3)
        self.color_layout.addWidget(QLabel("HSV-Value:"), 0, 4)
        self.color_layout.addWidget(self.hsv_v_dspinbox, 0, 5)
        self.aug_main_layout.addWidget(self.color_group)

        # Other Augmentations Group
        self.other_aug_group = QGroupBox("Other Augmentations")
        self.other_aug_layout = QGridLayout(self.other_aug_group)
        self.flipud_dspinbox = QDoubleSpinBox()
        self.flipud_dspinbox.setRange(0.0, 1.0)
        self.flipud_dspinbox.setValue(0.0)
        self.fliplr_dspinbox = QDoubleSpinBox()
        self.fliplr_dspinbox.setRange(0.0, 1.0)
        self.fliplr_dspinbox.setValue(0.5)
        self.mosaic_dspinbox = QDoubleSpinBox()
        self.mosaic_dspinbox.setRange(0.0, 1.0)
        self.mosaic_dspinbox.setValue(1.0)
        self.mixup_dspinbox = QDoubleSpinBox()
        self.mixup_dspinbox.setRange(0.0, 1.0)
        self.mixup_dspinbox.setValue(0.0)
        self.copy_paste_dspinbox = QDoubleSpinBox()
        self.copy_paste_dspinbox.setRange(0.0, 1.0)
        self.copy_paste_dspinbox.setValue(0.0)
        self.other_aug_layout.addWidget(QLabel("Flip U/D:"), 0, 0)
        self.other_aug_layout.addWidget(self.flipud_dspinbox, 0, 1)
        self.other_aug_layout.addWidget(QLabel("Flip L/R:"), 0, 2)
        self.other_aug_layout.addWidget(self.fliplr_dspinbox, 0, 3)
        self.other_aug_layout.addWidget(QLabel("Mosaic:"), 1, 0)
        self.other_aug_layout.addWidget(self.mosaic_dspinbox, 1, 1)
        self.other_aug_layout.addWidget(QLabel("MixUp:"), 1, 2)
        self.other_aug_layout.addWidget(self.mixup_dspinbox, 1, 3)
        self.other_aug_layout.addWidget(QLabel("Copy-Paste:"), 2, 0)
        self.other_aug_layout.addWidget(self.copy_paste_dspinbox, 2, 1)
        self.aug_main_layout.addWidget(self.other_aug_group)

        self.layout.addWidget(self.aug_group)

        # Dialog buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.button_box.button(QDialogButtonBox.Ok).setText("Start Training")
        self.button_box.button(QDialogButtonBox.Ok).setEnabled(False) # Default disabled
        self.layout.addWidget(self.button_box)

    def validate_inputs(self):
        is_valid = bool(self.yaml_path_edit.text().strip()) and bool(self.model_path_edit.text().strip())
        self.button_box.button(QDialogButtonBox.Ok).setEnabled(is_valid)

    def browse_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Base Model", "", "PyTorch Models (*.pt);;All Files (*)")
        if file_path:
            self.model_path_edit.setText(file_path)
            self.validate_inputs()

    def browse_yaml(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Dataset YAML File", "", "YAML Files (*.yaml *.yml)")
        if file_path:
            self.yaml_path_edit.setText(file_path)
            self.validate_inputs()

    def create_new_yaml(self):
        dialog = YamlCreatorDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            created_path = dialog.get_created_file_path()
            if created_path:
                self.yaml_path_edit.setText(created_path)
                self.validate_inputs()

    def get_parameters(self):
        return {
            'model_path': self.model_path_edit.text(),
            'data': self.yaml_path_edit.text(),
            'epochs': self.epochs_spinbox.value(),
            'imgsz': self.imgsz_spinbox.value(),
            'batch': self.batch_spinbox.value(),
            'lr0': self.lr0_dspinbox.value(),
            'lrf': self.lrf_dspinbox.value(),
            'patience': self.patience_spinbox.value(),
            'optimizer': self.optimizer_combo.currentText(),
            'degrees': self.degrees_dspinbox.value(),
            'translate': self.translate_dspinbox.value(),
            'scale': self.scale_dspinbox.value(),
            'shear': self.shear_dspinbox.value(),
            'hsv_h': self.hsv_h_dspinbox.value(),
            'hsv_s': self.hsv_s_dspinbox.value(),
            'hsv_v': self.hsv_v_dspinbox.value(),
            'flipud': self.flipud_dspinbox.value(),
            'fliplr': self.fliplr_dspinbox.value(),
            'mosaic': self.mosaic_dspinbox.value(),
            'mixup': self.mixup_dspinbox.value(),
            'copy_paste': self.copy_paste_dspinbox.value(),
        }
