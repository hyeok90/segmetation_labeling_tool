import traceback
from PyQt5.QtCore import QThread, pyqtSignal

class ImageEncoderThread(QThread):
    encoding_finished = pyqtSignal()
    encoding_failed = pyqtSignal(str)

    def __init__(self, predictor, image_rgb, img_path, parent=None):
        super().__init__(parent)
        self.predictor = predictor
        self.image_rgb = image_rgb
        self.img_path = img_path

    def run(self):
        try:
            self.predictor.set_image(self.image_rgb, self.img_path)
            self.encoding_finished.emit()
        except Exception:
            exc_str = traceback.format_exc()
            self.encoding_failed.emit(exc_str)
