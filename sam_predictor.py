import os
import cv2
import numpy as np
import onnxruntime
from rdp import rdp

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.order = []

    def get(self, key: str):
        if key in self.cache:
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        return None

    def put(self, key: str, value):
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.capacity:
            oldest_key = self.order.pop(0)
            del self.cache[oldest_key]
        self.cache[key] = value
        self.order.append(key)

class SegmentAnythingONNX:
    def __init__(self, encoder_path, decoder_path, device='cpu'):
        self.device = device
        self.encoder = onnxruntime.InferenceSession(encoder_path, providers=['CPUExecutionProvider' if self.device == 'cpu' else 'CUDAExecutionProvider'])
        self.decoder = onnxruntime.InferenceSession(decoder_path, providers=['CPUExecutionProvider' if self.device == 'cpu' else 'CUDAExecutionProvider'])
        self.input_size = 1024

    def encode(self, image: np.ndarray):
        h, w, _ = image.shape
        long_side = max(h, w)
        scale = self.input_size / long_side
        new_h, new_w = int(h * scale), int(w * scale)
        
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded_image = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        padded_image[:new_h, :new_w] = resized_image

        input_tensor = self.preprocess(padded_image)
        
        input_feed = {self.encoder.get_inputs()[0].name: input_tensor}
        embedding = self.encoder.run(None, input_feed)[0]
        
        return embedding, (h, w)

    def predict_masks(self, image_embedding, original_image_size, points, labels):
        h, w = original_image_size
        long_side = max(h, w)
        scale = self.input_size / long_side
        
        scaled_points = points * scale
        
        input_point = np.array(scaled_points, dtype=np.float32)
        input_label = np.array(labels, dtype=np.float32)

        onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :].astype(np.float32)
        onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)

        input_feed = {
            'image_embeddings': image_embedding,
            'point_coords': onnx_coord,
            'point_labels': onnx_label,
            'mask_input': np.zeros((1, 1, 256, 256), dtype=np.float32),
            'has_mask_input': np.zeros(1, dtype=np.float32),
            'orig_im_size': np.array(original_image_size, dtype=np.float32)
        }

        masks, _, _ = self.decoder.run(None, input_feed)
        return masks

    def preprocess(self, image: np.ndarray):
        pixel_mean = np.array([123.675, 116.28, 103.53]).reshape(1, 1, -1)
        pixel_std = np.array([58.395, 57.12, 57.375]).reshape(1, 1, -1)
        
        image = (image - pixel_mean) / pixel_std
        image = image.astype(np.float32)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)
        return image

class SAMPredictor:
    def __init__(self, encoder_path, decoder_path, device='cpu'):
        self.model = SegmentAnythingONNX(encoder_path, decoder_path, device)
        self.image_embedding_cache = LRUCache(10)
        self.current_image_embedding = None
        self.current_image_shape = None
        self.current_filename = None
        self.epsilon = 1.5  # For RDP

    def set_image(self, image: np.ndarray, filename: str):
        if filename == self.current_filename:
            return

        self.current_filename = filename
        cached_embedding = self.image_embedding_cache.get(filename)
        if cached_embedding:
            self.current_image_embedding, self.current_image_shape = cached_embedding
        else:
            self.current_image_embedding, self.current_image_shape = self.model.encode(image)
            self.image_embedding_cache.put(filename, (self.current_image_embedding, self.current_image_shape))

    def predict(self, points: np.ndarray, labels: np.ndarray):
        if self.current_image_embedding is None:
            return []

        # The ONNX model uses the 'orig_im_size' parameter to resize the mask internally.
        # The output 'masks' should already be at the full, original image resolution.
        masks = self.model.predict_masks(self.current_image_embedding, self.current_image_shape, points, labels)
        
        # We take the first mask, assuming it's the most likely one.
        mask = masks[0, 0, :, :]
        
        # Post-process the full-resolution mask to get polygons. No more scaling is needed.
        polygons = self.post_process(mask)

        return polygons

    def post_process(self, mask: np.ndarray):
        # This method now expects a full-resolution mask.
        mask[mask > 0.0] = 255
        mask[mask <= 0.0] = 0
        mask = mask.astype(np.uint8)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        polygons = []
        for contour in contours:
            if len(contour) < 3:
                continue
            
            # Apply RDP to simplify the polygon
            simplified_contour = rdp(contour.reshape(-1, 2), epsilon=self.epsilon)
            
            if len(simplified_contour) < 3:
                continue

            polygons.append(simplified_contour)
            
        return polygons
