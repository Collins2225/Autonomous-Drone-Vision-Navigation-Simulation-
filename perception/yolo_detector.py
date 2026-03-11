"""
perception/yolo_detector.py
============================
[RESEARCH EXTENSION] Semantic obstacle detection with YOLOv8.

MENTOR NOTE — Why Semantic Perception?
  Our depth-based obstacle detector is CLASS-AGNOSTIC — it sees "something
  close" but doesn't know WHAT it is. Semantic detection adds class labels:
  
  Semantic awareness enables class-specific behaviors:
    - Person detected → stop completely and wait (safety-critical)
    - Tree detected   → fly over (if altitude allows)
    - Building wall   → full avoidance
    - Parked car      → slow, go around
  
  This is how commercial drones like DJI implement intelligent flight modes.
  Research in semantic navigation is very active (2023-present).
  
  YOLO (You Only Look Once) family is the standard baseline for real-time
  object detection. YOLOv8 from Ultralytics runs at ~50fps on a modern GPU,
  or ~5fps on CPU for the nano model.
  
  Research Reference:
    Redmon & Farhadi, "YOLOv3: An Incremental Improvement" (2018)
    Jocher et al., "YOLOv8" (Ultralytics, 2023)

Requirements:
    pip install ultralytics
"""

import numpy as np
from dataclasses import dataclass
from typing import List

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


# ──────────────────────────────────────────────
# DATA STRUCTURES
# ──────────────────────────────────────────────

@dataclass
class DetectedObject:
    """Single YOLO detection result."""
    class_name: str         # e.g., "person", "car", "tree"
    confidence: float       # 0.0 to 1.0
    bbox: tuple             # (x1, y1, x2, y2) in pixels
    center_x: float         # pixel column of center
    center_y: float         # pixel row of center
    estimated_distance: float = 0.0  # from depth image, if available


# ──────────────────────────────────────────────
# YOLO DETECTOR
# ──────────────────────────────────────────────

class SemanticObstacleDetector:
    """
    YOLO-based semantic detector with depth fusion.
    
    MENTOR NOTE — Sensor Fusion:
      We FUSE two sources:
        1. YOLO bounding box    → WHAT is the obstacle + WHERE (image coords)
        2. Depth image          → HOW FAR is it (metric distance)
      
      This is a simple form of sensor fusion (2D det + depth).
      Advanced fusion uses Kalman filters or learned fusion networks.
    """

    # Classes that are safety-critical → drone should stop
    SAFETY_CRITICAL_CLASSES = {"person", "bicycle", "dog", "cat", "horse"}

    # Classes to fly over if possible
    FLYOVER_CLASSES = {"car", "truck", "bus"}

    def __init__(self, model_size: str = "n", confidence_threshold: float = 0.4):
        """
        Args:
            model_size: 'n' (nano), 's' (small), 'm' (medium), 'l' (large)
                        Nano: fastest, least accurate. Large: slowest, best.
            confidence_threshold: minimum detection confidence to report
        """
        if not YOLO_AVAILABLE:
            raise ImportError(
                "Ultralytics YOLO not found. Install: pip install ultralytics"
            )

        self.conf_threshold = confidence_threshold
        model_name = f"yolov8{model_size}.pt"
        print(f"[YOLODetector] Loading {model_name}...")
        self.model = YOLO(model_name)
        print("[YOLODetector] Model loaded ✓")

    def detect(
        self,
        rgb: np.ndarray,
        depth: np.ndarray = None
    ) -> List[DetectedObject]:
        """
        Run YOLO detection and optionally fuse with depth.
        
        Args:
            rgb:   H×W×3 uint8 image (BGR or RGB, YOLO handles both)
            depth: H×W float32 depth image (optional, for distance estimation)
        
        Returns:
            List of DetectedObject, sorted by distance (closest first)
        """
        # Run YOLO inference
        results = self.model(
            rgb,
            conf=self.conf_threshold,
            verbose=False   # Suppress YOLO's verbose output
        )

        detections = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                # Extract bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                # Estimate distance from depth image at bbox center
                estimated_dist = 100.0  # default: "far away"
                if depth is not None:
                    estimated_dist = self._estimate_distance_from_depth(
                        depth, int(x1), int(y1), int(x2), int(y2)
                    )

                detections.append(DetectedObject(
                    class_name=cls_name,
                    confidence=conf,
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    center_x=cx,
                    center_y=cy,
                    estimated_distance=estimated_dist
                ))

        # Sort by distance
        detections.sort(key=lambda d: d.estimated_distance)
        return detections

    def _estimate_distance_from_depth(
        self,
        depth: np.ndarray,
        x1: int, y1: int, x2: int, y2: int
    ) -> float:
        """
        Get robust distance estimate for a bounding box region.
        
        MENTOR NOTE — Why median depth in the CENTER of the bbox?
          The bbox includes background pixels at edges.
          We use the CENTER 50% of the box and take the MEDIAN to:
          1. Avoid background contamination at bbox edges
          2. Use median (not mean) for robustness against outlier depths
        
          Research improvement: use the LOWER PERCENTILE of depths in the box.
          The closest pixels are most likely the actual object surface.
        """
        # Take center 50% of bounding box
        h = y2 - y1
        w = x2 - x1
        cy1 = y1 + h // 4
        cy2 = y2 - h // 4
        cx1 = x1 + w // 4
        cx2 = x2 - w // 4

        # Clamp to image bounds
        H, W = depth.shape
        cy1, cy2 = max(0, cy1), min(H, cy2)
        cx1, cx2 = max(0, cx1), min(W, cx2)

        if cy1 >= cy2 or cx1 >= cx2:
            return 100.0

        roi = depth[cy1:cy2, cx1:cx2]
        valid = roi[np.isfinite(roi) & (roi > 0)]

        if len(valid) == 0:
            return 100.0

        # 20th percentile: biased toward closest part of the object
        return float(np.percentile(valid, 20))

    def has_safety_critical(self, detections: List[DetectedObject]) -> bool:
        """Returns True if any safety-critical class detected nearby (<5m)."""
        for det in detections:
            if (det.class_name in self.SAFETY_CRITICAL_CLASSES
                    and det.estimated_distance < 5.0):
                return True
        return False

    def draw_detections(
        self, frame: np.ndarray, detections: List[DetectedObject]
    ) -> np.ndarray:
        """Draw bounding boxes and labels on frame for visualization."""
        import cv2
        canvas = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            is_critical = det.class_name in self.SAFETY_CRITICAL_CLASSES

            color = (0, 0, 255) if is_critical else (0, 200, 0)  # BGR
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            label = f"{det.class_name} {det.confidence:.0%} {det.estimated_distance:.1f}m"
            cv2.putText(canvas, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        return canvas
