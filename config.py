import os
from PyQt5.QtGui import QColor

class Config:
    # App Info
    APP_NAME = "YOLOv11-seg Active Learning Tool"
    WINDOW_WIDTH = 1200
    WINDOW_HEIGHT = 800
    
    # Paths
    TEMP_DIR_NAME = ".temp_labels"
    SAM_ENCODER_PATH = os.path.join("sam", "sam_encoder.onnx")
    SAM_DECODER_PATH = os.path.join("sam", "sam_decoder.onnx")
    SAM2_CHECKPOINT_PATH = os.path.join("models", "sam2.1_hiera_large.pt")
    
    # Defaults
    DEFAULT_EPSILON = 1.0
    DEFAULT_BRUSH_RADIUS = 20
    DEFAULT_BRUSH_MAX = 100
    
    # Colors
    SAM_PREVIEW_COLOR = QColor(0, 100, 255, 100)
    SAM_POSITIVE_COLOR = QColor(0, 255, 0)
    SAM_NEGATIVE_COLOR = QColor(255, 0, 0)
    
    @staticmethod
    def get_temp_dir():
        return os.path.join(os.getcwd(), Config.TEMP_DIR_NAME)
