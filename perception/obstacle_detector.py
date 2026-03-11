"""
perception/obstacle_detector.py
================================
Converts raw camera images into a structured obstacle representation.

MENTOR NOTE — The Perception Pipeline:
  Raw pixels are useless to a navigation algorithm. The perception module's
  job is to answer ONE question clearly: "Where are the obstacles relative
  to the drone, and how close are they?"

  Our pipeline:
    Depth Image (float32, meters)
      │
      ├─ [1] Threshold → Binary Danger Mask (obstacle pixels = 255)
      │
      ├─ [2] Morphology → Clean Mask (remove noise, fill gaps)
      │
      ├─ [3] Grid Analysis → 3×3 occupancy grid (which sectors blocked?)
      │
      └─ [4] Centroid → Closest obstacle position (x, y, distance)

  This produces an ObstacleMap — the clean output handed to Navigation.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import cv2

from config.settings import (
    OBSTACLE_DISTANCE_THRESHOLD,
    MORPH_KERNEL_SIZE,
    GRID_COLS, GRID_ROWS,
    OBSTACLE_CELL_THRESHOLD,
    IMAGE_WIDTH, IMAGE_HEIGHT
)


# ──────────────────────────────────────────────
# DATA STRUCTURES
# ──────────────────────────────────────────────

@dataclass
class ObstacleInfo:
    """
    Represents a single detected obstacle.
    
    image_x, image_y: pixel center of the obstacle in the camera frame
    distance:         estimated distance in meters (from depth image)
    direction:        normalized direction vector [left/right, up/down]
                      in image space. (-1,0) = full left, (1,0) = full right
    """
    image_x: float         # pixel column (0 = left, WIDTH = right)
    image_y: float         # pixel row (0 = top, HEIGHT = bottom)
    distance: float        # meters
    direction: np.ndarray  # shape (2,) normalized vector in image space


@dataclass
class ObstacleMap:
    """
    Full perception output for one frame.
    
    Fields:
      danger_mask:    binary image (H×W uint8), 255 where obstacle detected
      grid:           3×3 bool array, True = obstacle present in that sector
      obstacles:      list of detected ObstacleInfo (up to 5 closest)
      closest:        the single closest ObstacleInfo, or None if clear
      is_clear:       True if no obstacles in danger zone
    """
    danger_mask: np.ndarray                          # H×W uint8
    grid: np.ndarray                                 # GRID_ROWS × GRID_COLS bool
    obstacles: List[ObstacleInfo] = field(default_factory=list)
    closest: Optional[ObstacleInfo] = None
    is_clear: bool = True


# ──────────────────────────────────────────────
# MAIN DETECTOR CLASS
# ──────────────────────────────────────────────

class ObstacleDetector:
    """
    Converts depth + RGB images into a structured ObstacleMap.
    
    Usage:
        detector = ObstacleDetector()
        images = sim_client.get_images()
        obs_map = detector.process(images['depth'], images['rgb'])
    """

    def __init__(self):
        # Pre-build the morphological kernel (avoids recreating every frame)
        k = MORPH_KERNEL_SIZE
        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (k, k)
        )

        # Pre-compute grid cell boundaries for efficiency
        self._cell_h = IMAGE_HEIGHT // GRID_ROWS
        self._cell_w = IMAGE_WIDTH  // GRID_COLS

    # ──────────────────────────────────────────────
    # PUBLIC INTERFACE
    # ──────────────────────────────────────────────

    def process(self, depth: np.ndarray, rgb: np.ndarray = None) -> ObstacleMap:
        """
        Full perception pipeline: depth image → ObstacleMap.
        
        Args:
            depth: float32 array of shape (H, W), values in meters
            rgb:   optional uint8 array (H, W, 3), used for debug overlay only
        
        Returns:
            ObstacleMap with all fields populated
        """
        # Step 1: Threshold depth image to find danger zone
        danger_mask = self._threshold_depth(depth)

        # Step 2: Morphological operations — remove noise, fill gaps
        clean_mask = self._clean_mask(danger_mask)

        # Step 3: Analyze spatial grid
        grid = self._compute_grid(clean_mask)

        # Step 4: Find individual obstacle centroids
        obstacles = self._find_obstacle_centroids(clean_mask, depth)

        # Step 5: Package into ObstacleMap
        closest = min(obstacles, key=lambda o: o.distance) if obstacles else None
        is_clear = len(obstacles) == 0

        return ObstacleMap(
            danger_mask=clean_mask,
            grid=grid,
            obstacles=obstacles,
            closest=closest,
            is_clear=is_clear
        )

    # ──────────────────────────────────────────────
    # STEP 1: DEPTH THRESHOLDING
    # ──────────────────────────────────────────────

    def _threshold_depth(self, depth: np.ndarray) -> np.ndarray:
        """
        Create a binary mask where pixels are 255 if obstacle is within
        OBSTACLE_DISTANCE_THRESHOLD meters.
        
        MENTOR NOTE:
          We also clip infinity values (AirSim returns inf for sky/empty pixels)
          and zero values (invalid sensor readings). Only valid, close depths
          become obstacle pixels.
          
          np.inf pixels → set to large value so they DON'T become obstacles.
          Zero pixels    → also treated as invalid (skip them).
        """
        # Replace invalid values with a safe "far away" distance
        valid_depth = depth.copy()
        valid_depth[~np.isfinite(valid_depth)] = 999.0
        valid_depth[valid_depth <= 0] = 999.0

        # Binary threshold: 255 where depth < threshold, else 0
        mask = np.zeros_like(valid_depth, dtype=np.uint8)
        mask[valid_depth < OBSTACLE_DISTANCE_THRESHOLD] = 255

        return mask

    # ──────────────────────────────────────────────
    # STEP 2: MORPHOLOGICAL CLEANING
    # ──────────────────────────────────────────────

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Remove noise and fill gaps in the obstacle mask.
        
        MENTOR NOTE — Why Morphology?
          Depth sensors are noisy. A single real obstacle might produce a 
          scattered, holey mask. Morphological operations fix this:
          
          OPEN  (erode then dilate) = removes small isolated noise blobs
          CLOSE (dilate then erode) = fills small holes inside real obstacles
          
          Order matters: we open first to kill noise, then close to solidify
          the remaining real obstacles.
        """
        # Opening: kill isolated noise pixels smaller than kernel
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._morph_kernel)

        # Closing: fill gaps and holes in real obstacles
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self._morph_kernel)

        return closed

    # ──────────────────────────────────────────────
    # STEP 3: SPATIAL GRID ANALYSIS
    # ──────────────────────────────────────────────

    def _compute_grid(self, mask: np.ndarray) -> np.ndarray:
        """
        Divide the image into a GRID_ROWS × GRID_COLS grid.
        Mark each cell True if obstacle pixel fraction > threshold.
        
        MENTOR NOTE:
          A 3×3 grid gives us 9 spatial zones:
            [TL][TC][TR]
            [ML][MC][MR]
            [BL][BC][BR]
          
          The navigation module uses this grid to decide:
          - "obstacle in center" → must evade left OR right
          - "obstacle left"      → prefer going right
          - etc.
          
          This is much faster than per-pixel reasoning for the nav module.
        """
        grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=bool)

        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                # Extract cell region from mask
                y_start = row * self._cell_h
                y_end   = (row + 1) * self._cell_h
                x_start = col * self._cell_w
                x_end   = (col + 1) * self._cell_w

                cell = mask[y_start:y_end, x_start:x_end]

                # Count obstacle pixels as fraction of cell size
                obstacle_fraction = np.sum(cell > 0) / cell.size

                if obstacle_fraction > OBSTACLE_CELL_THRESHOLD:
                    grid[row, col] = True

        return grid

    # ──────────────────────────────────────────────
    # STEP 4: OBSTACLE CENTROID DETECTION
    # ──────────────────────────────────────────────

    def _find_obstacle_centroids(
        self,
        mask: np.ndarray,
        depth: np.ndarray,
        max_obstacles: int = 3
    ) -> List[ObstacleInfo]:
        """
        Find up to max_obstacles individual obstacles via connected components.
        
        For each blob, compute:
          - Centroid pixel location
          - Median depth (more robust than mean against noisy depth values)
          - Normalized direction from image center
        
        MENTOR NOTE:
          cv2.connectedComponentsWithStats is one of the most useful OpenCV
          functions in robotics. It labels connected blobs and gives you:
            - area (pixels)
            - bounding box
            - centroid
          
          We use the centroid to compute direction, and index back into the
          original depth image to get real distance.
        """
        if np.sum(mask) == 0:
            return []

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        # Label 0 = background, skip it
        obstacles = []
        image_cx = IMAGE_WIDTH  / 2.0
        image_cy = IMAGE_HEIGHT / 2.0

        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area < 50:   # Ignore tiny blobs (likely noise that survived morphology)
                continue

            cx, cy = centroids[label_id]

            # Get all depth values in this blob
            blob_pixels = depth[labels == label_id]
            valid_depths = blob_pixels[np.isfinite(blob_pixels) & (blob_pixels > 0)]

            if len(valid_depths) == 0:
                continue

            # Median depth = distance to this obstacle (robust to outliers)
            dist = float(np.median(valid_depths))

            # Normalized direction: how far left/right, up/down in image space
            # Range: [-1, 1] where 0 = center
            dir_x = (cx - image_cx) / image_cx
            dir_y = (cy - image_cy) / image_cy
            direction = np.array([dir_x, dir_y], dtype=np.float32)

            obstacles.append(ObstacleInfo(
                image_x=float(cx),
                image_y=float(cy),
                distance=dist,
                direction=direction
            ))

        # Sort by distance (closest first) and return top N
        obstacles.sort(key=lambda o: o.distance)
        return obstacles[:max_obstacles]

    # ──────────────────────────────────────────────
    # UTILITY
    # ──────────────────────────────────────────────

    def grid_summary(self, grid: np.ndarray) -> str:
        """Human-readable grid string for logging."""
        rows = []
        for row in range(GRID_ROWS):
            cells = []
            for col in range(GRID_COLS):
                cells.append("█" if grid[row, col] else "·")
            rows.append(" ".join(cells))
        return "\n".join(rows)
