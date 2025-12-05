import traceback
from PyQt5.QtCore import QThread, pyqtSignal

class SAMPredictionThread(QThread):
    prediction_finished = pyqtSignal(object)
    prediction_failed = pyqtSignal(str)

    def __init__(self, predictor, points, labels, parent=None):
        super().__init__(parent)
        self.predictor = predictor
        self.points = points
        self.labels = labels

    def run(self):
        try:
            polygons = self.predictor.predict(self.points, self.labels)
            self.prediction_finished.emit(polygons)
        except Exception:
            exc_str = traceback.format_exc()
            self.prediction_failed.emit(exc_str)
