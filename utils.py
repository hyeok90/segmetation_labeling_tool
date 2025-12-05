import os
import math
from PyQt5.QtCore import QPointF

def distance(p):
    """Distance between two points"""
    return math.sqrt(p.x() * p.x() + p.y() * p.y())

def load_yolo_labels(txt_path, img_w, img_h, class_names):
    from shape import Shape
    shapes = []
    if not os.path.exists(txt_path):
        return shapes

    with open(txt_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
            
        try:
            class_id = int(parts[0])
            class_name = class_names[class_id]
            
            # Check if confidence score is present
            if len(parts) % 2 == 0:
                score = float(parts[-1])
                polygon_parts = parts[1:-1]
            else:
                score = 1.0
                polygon_parts = parts[1:]

            polygon_coords = []
            for j in range(0, len(polygon_parts), 2):
                x_norm = float(polygon_parts[j])
                y_norm = float(polygon_parts[j+1])
                
                x_abs = x_norm * img_w
                y_abs = y_norm * img_h
                polygon_coords.append(QPointF(x_abs, y_abs))
            
            shape = Shape(label=class_name, shape_type='polygon', score=score)
            shape.points = polygon_coords
            shape.close()
            shapes.append(shape)
        except Exception as e:
            print(f"Error parsing line {line}: {e}")

    return shapes

def save_yolo_labels(txt_path, shapes, img_w, img_h, class_names):
    lines = []
    for shape in shapes:
        class_name = shape.label
        if class_name not in class_names:
            continue
        class_id = class_names.index(class_name)
        
        normalized_coords = []
        for pt in shape.points:
            x_abs = pt.x()
            y_abs = pt.y()

            x_norm = max(0.0, min(1.0, x_abs / img_w))
            y_norm = max(0.0, min(1.0, y_abs / img_h))
            normalized_coords.append(f"{x_norm:.6f}")
            normalized_coords.append(f"{y_norm:.6f}")
        
        if normalized_coords:
            line_parts = [str(class_id)] + normalized_coords
            lines.append(" ".join(line_parts))
    
    with open(txt_path, 'w') as f:
        f.write("\n".join(lines))