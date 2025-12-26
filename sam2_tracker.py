import os
import torch
import numpy as np
import cv2
from typing import List, Tuple
from shapely.geometry import Polygon

# Try importing SAM 2. If not found, we handle it gracefully in the main window.
try:
    from sam2.build_sam import build_sam2_video_predictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    print("SAM 2 library not found. Video segmentation features will be disabled.")

class SAM2Tracker:
    def __init__(self, model_cfg, checkpoint_path, device='cuda'):
        if not SAM2_AVAILABLE:
            raise ImportError("SAM 2 library is not installed.")
        
        self.device = device
        # Ensure CUDA is available if requested
        if self.device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU for SAM 2.")
            self.device = 'cpu'

        print(f"Loading SAM 2 model: {checkpoint_path} on {self.device}")
        self.predictor = build_sam2_video_predictor(model_cfg, checkpoint_path, device=self.device)
        self.inference_state = None

    def initialize_video(self, image_paths: List[str]):
        """
        Initializes the video inference state with a list of image paths.
        """
        # SAM 2 API allows passing a list of frame paths (ensure they are sorted/ordered correctly)
        self.inference_state = self.predictor.init_state(video_path=None, frame_names=None, img_paths=image_paths)
        self.predictor.reset_state(self.inference_state)

    def add_initial_mask(self, frame_idx: int, obj_id: int, mask: np.ndarray):
        """
        Adds an initial mask prompt to a specific frame.
        mask: Binary mask (H, W) where True/1 is the object.
        """
        # SAM 2 expects masks (1, H, W) or (N, H, W) logits?
        # add_new_mask expects a binary mask or logits.
        # We need to verify the exact API. Usually it takes a binary mask.
        
        # Check if mask needs to be bool or float
        if mask.dtype != bool:
            mask = mask > 0

        self.predictor.add_new_mask(
            inference_state=self.inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            mask=mask
        )

    def propagate(self):
        """
        Propagates the masks through the video.
        Yields (frame_idx, obj_ids, masks)
        """
        # propagate_in_video returns a generator
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
            # out_mask_logits is (N, H, W)
            # Convert logits to binary masks
            masks = (out_mask_logits > 0.0).cpu().numpy()
            yield out_frame_idx, out_obj_ids, masks

    def reset(self):
        if self.inference_state:
            self.predictor.reset_state(self.inference_state)
