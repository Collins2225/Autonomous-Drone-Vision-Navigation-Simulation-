"""
perception/yolo_detector.py
============================
YOLOv8 semantic obstacle detector with depth fusion.

Detects WHAT obstacles are (person, car, wall) and HOW FAR they are.
Runs every N frames to preserve CPU for the main perception loop.

MENTOR NOTE:
  YOLO runs at ~5fps on CPU with the nano model.
  Our main depth loop runs at 8Hz.
  By running YOLO every 3rd frame (YOLO_RUN_EVERY_N=3),
  we get ~2.5fps YOLO detections without slowing the main loop.
  This is called "frame skipping" — standard in real-time robotics.
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Optional

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

from config.settings import (
    YOLO_MODEL,
    YOLO_CONFIDENCE,
    YOLO_RUN_EVERY_N,
    YOLO_CLASS_BEHAVIORS,
    YOLO_STOP_DISTANCE,
    YOLO_SLOW_DISTANCE,
    IMAGE_WIDTH,
    IMAGE_HEIGHT
)


# ──────────────────────────────────────────────
# DATA STRUCTURE
# ──────────────────────────────────────────────

@dataclass
class YOLODetection:
    """Single YOLO detection result with depth fusion."""
    class_name:  str         # e.g. "person", "car"
    confidence:  float       # 0.0 to 1.0
    bbox:        tuple       # (x1, y1, x2, y2) pixels
    center_x:    float       # pixel column of center
    center_y:    float       # pixel row of center
    distance:    float       # meters from depth image
    behavior:    str         # "STOP", "AVOID", "SLOW", "IGNORE"
    is_critical: bool        # True if behavior == STOP


@dataclass
class YOLOResult:
    """Full YOLO processing result for one frame."""
    detections:       List[YOLODetection]
    should_stop:      bool   # True if safety-critical object nearby
    should_slow:      bool   # True if any obstacle within slow range
    closest_critical: Optional[YOLODetection]
    frame_count:      int    # which frame this was computed on


# ──────────────────────────────────────────────
# YOLO DETECTOR
# ──────────────────────────────────────────────

class YOLOObstacleDetector:
    """
    Runs YOLOv8 detection every N frames and fuses with depth image.

    Usage:
        detector = YOLOObstacleDetector()
        result = detector.process(rgb_frame, depth_image, frame_number)
        if result.should_stop:
            # emergency stop
        for det in result.detections:
            print(det.class_name, det.distance)
    """

    def __init__(self):
        if not YOLO_AVAILABLE:
            raise ImportError(
                "Ultralytics not found. Run: pip install ultralytics"
            )

        print(f"[YOLO] Loading {YOLO_MODEL}...")
        print("[YOLO] First load downloads weights (~6MB) — one time only.")
        self.model = YOLO(YOLO_MODEL)
        print("[YOLO] Model loaded ✓")

        # Cache last result so we can return it on skipped frames
        self._last_result: Optional[YOLOResult] = None
        self._frame_count = 0

    # ──────────────────────────────────────────────
    # MAIN PROCESS METHOD
    # ──────────────────────────────────────────────

    def process(
        self,
        rgb:   np.ndarray,
        depth: np.ndarray,
        frame_number: int = 0
    ) -> Optional[YOLOResult]:
        """
        Run YOLO detection with frame skipping.

        Args:
            rgb:          H x W x 3 uint8 image
            depth:        H x W float32 depth in meters
            frame_number: current loop iteration number

        Returns:
            YOLOResult if this frame was processed
            Last cached YOLOResult if frame was skipped
            None if no result available yet
        """
        self._frame_count += 1

        # Frame skipping — only run YOLO every N frames
        if frame_number % YOLO_RUN_EVERY_N != 0:
            return self._last_result

        # Run YOLO inference
        detections = self._run_inference(rgb, depth)

        # Analyze results
        should_stop      = False
        should_slow      = False
        closest_critical = None
        min_critical_dist = float('inf')

        for det in detections:
            if det.behavior == "STOP" and det.distance < YOLO_STOP_DISTANCE:
                should_stop = True
                if det.distance < min_critical_dist:
                    min_critical_dist = det.distance
                    closest_critical  = det

            if det.distance < YOLO_SLOW_DISTANCE:
                should_slow = True

        result = YOLOResult(
            detections=detections,
            should_stop=should_stop,
            should_slow=should_slow,
            closest_critical=closest_critical,
            frame_count=frame_number
        )

        self._last_result = result
        return result

    # ──────────────────────────────────────────────
    # INFERENCE
    # ──────────────────────────────────────────────

    def _run_inference(
        self,
        rgb:   np.ndarray,
        depth: np.ndarray
    ) -> List[YOLODetection]:
        """Run YOLO and fuse detections with depth data."""

        results = self.model(
            rgb,
            conf=YOLO_CONFIDENCE,
            verbose=False
        )

        detections = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf     = float(box.conf[0])
                cls_id   = int(box.cls[0])
                cls_name = self.model.names[cls_id]

                cx = float((x1 + x2) / 2.0)
                cy = float((y1 + y2) / 2.0)

                # Get distance from depth image
                dist = self._get_distance(
                    depth,
                    int(x1), int(y1),
                    int(x2), int(y2)
                )

                # Determine behavior from class
                behavior    = YOLO_CLASS_BEHAVIORS.get(cls_name, "AVOID")
                is_critical = behavior == "STOP"

                detections.append(YOLODetection(
                    class_name=cls_name,
                    confidence=conf,
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    center_x=cx,
                    center_y=cy,
                    distance=dist,
                    behavior=behavior,
                    is_critical=is_critical
                ))

        # Sort by distance — closest first
        detections.sort(key=lambda d: d.distance)
        return detections

    def _get_distance(
        self,
        depth: np.ndarray,
        x1: int, y1: int,
        x2: int, y2: int
    ) -> float:
        """
        Get robust distance for a bounding box region.
        Uses center 50% of bbox and 20th percentile depth.

        MENTOR NOTE:
          20th percentile = biased toward closest part of object.
          This gives us the distance to the FRONT FACE of the obstacle
          rather than an average that includes background pixels.
        """
        h = y2 - y1
        w = x2 - x1

        # Center 50% of bbox
        cy1 = max(0, y1 + h // 4)
        cy2 = min(IMAGE_HEIGHT, y2 - h // 4)
        cx1 = max(0, x1 + w // 4)
        cx2 = min(IMAGE_WIDTH,  x2 - w // 4)

        if cy1 >= cy2 or cx1 >= cx2:
            return 100.0

        roi   = depth[cy1:cy2, cx1:cx2]
        valid = roi[np.isfinite(roi) & (roi > 0) & (roi < 999.0)]

        if len(valid) == 0:
            return 100.0

        return float(np.percentile(valid, 20))

    # ──────────────────────────────────────────────
    # VISUALIZATION
    # ──────────────────────────────────────────────

    def draw(
        self,
        frame:  np.ndarray,
        result: YOLOResult
    ) -> np.ndarray:
        """
        Draw YOLO bounding boxes on frame.

        Color coding:
          RED    = STOP class (person, animal) — safety critical
          ORANGE = AVOID class (car, truck)
          GREEN  = other detections
        """
        if result is None:
            return frame

        canvas = frame.copy()

        color_map = {
            "STOP":   (0,   0,   255),   # red
            "AVOID":  (0,   140, 255),   # orange
            "SLOW":   (0,   200, 200),   # yellow
            "IGNORE": (100, 100, 100),   # gray
        }

        for det in result.detections:
            x1, y1, x2, y2 = det.bbox
            color = color_map.get(det.behavior, (0, 200, 0))

            # Bounding box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            # Label with class, confidence and distance
            label = f"{det.class_name} {det.confidence:.0%} {det.distance:.1f}m"
            label_y = max(y1 - 5, 10)
            cv2.putText(
                canvas, label, (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
            )

            # Critical warning
            if det.is_critical and det.distance < YOLO_STOP_DISTANCE:
                cv2.putText(
                    canvas, "! STOP !", (x1, y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
                )

        # Overall status
        if result.should_stop:
            cv2.putText(
                canvas, "EMERGENCY STOP", (5, IMAGE_HEIGHT - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )
        elif result.should_slow:
            cv2.putText(
                canvas, "SLOWING", (5, IMAGE_HEIGHT - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1
            )

        return canvas
