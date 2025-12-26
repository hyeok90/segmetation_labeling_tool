import cv2
import numpy as np
import threading
from rdp import rdp
from ultralytics import YOLO
import torch

class RealYOLOPredictor:
    def __init__(self, model_path):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"Initializing model on device: {self.device}")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.lock = threading.Lock()

    def get_class_names(self):
        with self.lock:
            if self.model and self.model.names:
                return self.model.names
            return {}

    def predict_and_optimize(self, img_path, conf=0.25, iou=0.7, epsilon=1.0):
        with self.lock:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error: Could not read image {img_path}")
                return [], (0, 0), 0.0
                
            img_h, img_w = img.shape[:2]
            
            # Pass IoU (NMS threshold) to model
            results = self.model(img, imgsz=1280, conf=conf, iou=iou, device=self.device, retina_masks=True)

            if not results or results[0].masks is None:
                return [], (img_w, img_h), 0.0

            instances = []
            total_conf = 0
            num_insts = 0

            for i, mask in enumerate(results[0].masks):
                if mask.xyn is None or len(mask.xyn) == 0:
                    continue

                polygon_points_normalized = mask.xyn[0]
                
                polygon_points = []
                for p_norm in polygon_points_normalized:
                    x_abs = p_norm[0] * img_w
                    y_abs = p_norm[1] * img_h
                    polygon_points.append([x_abs, y_abs])

                if epsilon > 0:
                    polygon_points = rdp(polygon_points, epsilon=epsilon)

                if len(polygon_points) < 3:
                    continue
                    
                class_id = int(results[0].boxes.cls[i].cpu().numpy())
                conf = float(results[0].boxes.conf[i].cpu().numpy())
                
                instances.append((class_id, polygon_points, conf))
                
                if conf > 0:
                    total_conf += conf
                    num_insts += 1
            
            avg_conf = total_conf / num_insts if num_insts > 0 else 0.0
            
            return instances, (img_w, img_h), avg_conf

    def train(self, **kwargs):
        with self.lock:
            return self.model.train(**kwargs)
