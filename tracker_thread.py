import traceback
import numpy as np
import cv2
from PyQt5.QtCore import QThread, pyqtSignal, QPointF
from rdp import rdp
from shape import Shape
from sam2_tracker import SAM2Tracker

class SAM2TrackerThread(QThread):
    tracking_finished = pyqtSignal(dict) # Returns {frame_idx: [Shape, ...]}
    tracking_failed = pyqtSignal(str)
    progress_update = pyqtSignal(int, int) # current, total

    def __init__(self, tracker: SAM2Tracker, image_paths, start_frame_idx, initial_shapes, class_names, epsilon=1.0, parent=None):
        super().__init__(parent)
        self.tracker = tracker
        self.image_paths = image_paths
        self.start_frame_idx = start_frame_idx
        self.initial_shapes = initial_shapes # List of Shape objects on the start frame
        self.class_names = class_names
        self.epsilon = epsilon
        self.is_running = True

    def run(self):
        try:
            # 1. Initialize Video
            # We assume image_paths are absolute paths
            print("Initializing SAM 2 state...")
            self.tracker.initialize_video(self.image_paths)

            # 2. Add Initial Masks
            # We need to convert Shapes to binary masks
            img = cv2.imread(self.image_paths[self.start_frame_idx])
            h, w = img.shape[:2]

            for i, shape in enumerate(self.initial_shapes):
                mask = np.zeros((h, w), dtype=np.uint8)
                points = np.array([(p.x(), p.y()) for p in shape.points], dtype=np.int32)
                cv2.fillPoly(mask, [points], 1)
                
                # Use index 'i+1' as object ID (SAM 2 uses positive integers)
                # But we need to map obj_id back to label later.
                # We'll assume the order is preserved or use a dict.
                self.tracker.add_initial_mask(self.start_frame_idx, i + 1, mask.astype(bool))

            # 3. Propagate
            print("Starting propagation...")
            results = {} # frame_idx -> list of Shapes
            
            # Count frames for progress
            total_frames = len(self.image_paths)
            
            for frame_idx, obj_ids, masks in self.tracker.propagate():
                if not self.is_running:
                    break
                
                self.progress_update.emit(frame_idx + 1, total_frames)
                
                frame_shapes = []
                # masks array shape: (N, H, W) corresponding to obj_ids
                for j, obj_id in enumerate(obj_ids):
                    mask = masks[j]
                    
                    # Convert to Polygon
                    mask_uint8 = mask.astype(np.uint8) * 255
                    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours:
                        if len(contour) < 3:
                            continue
                        
                        simplified = rdp(contour.reshape(-1, 2), epsilon=self.epsilon)
                        if len(simplified) < 3:
                            continue
                            
                        # Recover the label from obj_id
                        # obj_id corresponds to self.initial_shapes[obj_id - 1]
                        original_shape = self.initial_shapes[obj_id - 1]
                        
                        new_shape = Shape(label=original_shape.label, shape_type='polygon')
                        new_shape.points = [QPointF(float(p[0]), float(p[1])) for p in simplified]
                        new_shape.close()
                        new_shape.score = 1.0 # SAM 2 doesn't give easy confidence score here
                        
                        frame_shapes.append(new_shape)
                
                results[frame_idx] = frame_shapes

            if self.is_running:
                self.tracking_finished.emit(results)
            else:
                self.tracking_failed.emit("Tracking cancelled.")

        except Exception:
            self.tracking_failed.emit(traceback.format_exc())

    def stop(self):
        self.is_running = False
