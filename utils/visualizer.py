"""
utils/visualizer.py
====================
Real-time debug visualization window for the autonomous navigation system.

MENTOR NOTE — Why Visualize?
  In robotics research, visualization is not optional — it's how you debug.
  When your drone behaves unexpectedly, you need to see:
    1. What the camera sees
    2. What the perception module detected
    3. What the navigation module decided
    4. What commands were sent
  
  All of this in real-time, in one window. This is your "cockpit HUD".
  
  We use OpenCV's imshow for simplicity. For research-grade visualization,
  consider RViz (ROS), rerun.io, or custom Dash/Plotly dashboards.
"""

import numpy as np
import cv2
from typing import Optional

from config.settings import (
    IMAGE_WIDTH, IMAGE_HEIGHT,
    GRID_COLS, GRID_ROWS,
    VISUALIZER_SCALE, VIZ_WINDOW_NAME,
    OBSTACLE_DISTANCE_THRESHOLD
)
from perception.obstacle_detector import ObstacleMap
from navigation.potential_field import MotionCommand


# Color palette (BGR for OpenCV)
COLOR_GREEN  = (50, 220, 50)
COLOR_RED    = (50, 50, 220)
COLOR_YELLOW = (0, 220, 220)
COLOR_CYAN   = (220, 200, 0)
COLOR_WHITE  = (255, 255, 255)
COLOR_GRAY   = (128, 128, 128)
COLOR_ORANGE = (0, 140, 255)
COLOR_DARK   = (30, 30, 30)


class DroneVisualizer:
    """
    Draws a composite debug HUD over the drone's camera view.
    
    Layout:
    ┌─────────────────────────────────────┐
    │  RGB Camera Feed + Obstacle Overlay  │
    │  + Grid overlay + Obstacle markers  │
    ├──────────────┬──────────────────────┤
    │ Depth Map    │   Status Panel       │
    │ (colorized)  │   Velocity arrows    │
    │              │   Grid status        │
    └──────────────┴──────────────────────┘
    """

    def __init__(self):
        self._frame_count = 0
        self._fps_timer = []

    def render(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        obs_map: ObstacleMap,
        cmd: MotionCommand,
        loop_hz: float = 0.0
    ):
        """
        Render the HUD and display it. Call once per control loop iteration.
        
        Args:
            rgb:      H×W×3 uint8 RGB camera image
            depth:    H×W float32 depth image (meters)
            obs_map:  ObstacleMap from perception
            cmd:      MotionCommand from navigation/control
            loop_hz:  current loop frequency for FPS display
        """
        # Build the top panel: camera + overlays
        top_panel = self._draw_camera_panel(rgb, obs_map)

        # Build the bottom panel: depth + status
        depth_vis = self._colorize_depth(depth)
        # Resize depth to half width to match layout
        depth_vis = cv2.resize(depth_vis, (IMAGE_WIDTH // 2, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
        status_panel = self._draw_status_panel(obs_map, cmd, loop_hz)
        bottom_panel = np.hstack([depth_vis, status_panel])

        # Stack top and bottom
        composite = np.vstack([top_panel, bottom_panel])

        # Scale up for readability
        h, w = composite.shape[:2]
        display = cv2.resize(
            composite,
            (w * VISUALIZER_SCALE, h * VISUALIZER_SCALE),
            interpolation=cv2.INTER_NEAREST
        )

        cv2.imshow(VIZ_WINDOW_NAME, display)
        cv2.waitKey(1)  # Non-blocking — processes GUI events

        self._frame_count += 1

    # ──────────────────────────────────────────────
    # CAMERA PANEL
    # ──────────────────────────────────────────────

    def _draw_camera_panel(self, rgb: np.ndarray, obs_map: ObstacleMap) -> np.ndarray:
        """Draw RGB camera feed with obstacle overlays."""
        canvas = rgb.copy()

        # Draw danger zone overlay (semi-transparent red)
        if obs_map.danger_mask is not None and np.any(obs_map.danger_mask):
            red_overlay = np.zeros_like(canvas)
            red_overlay[obs_map.danger_mask > 0] = (0, 0, 180)  # BGR red
            canvas = cv2.addWeighted(canvas, 1.0, red_overlay, 0.4, 0)

        # Draw grid lines
        canvas = self._draw_grid_overlay(canvas, obs_map.grid)

        # Draw obstacle centroids
        for obs in obs_map.obstacles:
            cx, cy = int(obs.image_x), int(obs.image_y)

            # Circle at centroid
            color = COLOR_RED if obs.distance < 2.0 else COLOR_YELLOW
            cv2.circle(canvas, (cx, cy), 8, color, 2)

            # Distance label
            label = f"{obs.distance:.1f}m"
            cv2.putText(canvas, label, (cx + 10, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Crosshair at image center
        cx, cy = IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2
        cv2.line(canvas, (cx - 15, cy), (cx + 15, cy), COLOR_CYAN, 1)
        cv2.line(canvas, (cx, cy - 15), (cx, cy + 15), COLOR_CYAN, 1)

        # Status text
        status_text = "CLEAR" if obs_map.is_clear else "OBSTACLE"
        status_color = COLOR_GREEN if obs_map.is_clear else COLOR_RED
        cv2.putText(canvas, status_text, (5, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

        return canvas

    def _draw_grid_overlay(self, canvas: np.ndarray, grid: np.ndarray) -> np.ndarray:
        """Draw 3×3 spatial grid with obstacle highlighting."""
        cell_h = IMAGE_HEIGHT // GRID_ROWS
        cell_w = IMAGE_WIDTH  // GRID_COLS

        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                x1 = col * cell_w
                y1 = row * cell_h
                x2 = x1 + cell_w
                y2 = y1 + cell_h

                if grid[row, col]:
                    # Obstacle in this cell — red border
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_RED, 1)
                else:
                    # Clear cell — subtle gray border
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_GRAY, 1)

        return canvas

    # ──────────────────────────────────────────────
    # DEPTH PANEL
    # ──────────────────────────────────────────────

    def _colorize_depth(self, depth: np.ndarray) -> np.ndarray:
        """
        Convert float32 depth map to a colorized BGR image.
        
        Near = red/hot, Far = blue/cool (JET colormap).
        Clamp to [0, OBSTACLE_DISTANCE_THRESHOLD * 1.5] meters for dynamic range.
        """
        # Ensure depth has correct dimensions
        if depth.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
            depth = cv2.resize(depth, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
        
        max_range = OBSTACLE_DISTANCE_THRESHOLD * 1.5

        # Normalize to [0, 255]
        depth_clipped = np.clip(depth, 0, max_range)
        depth_clipped[~np.isfinite(depth_clipped)] = max_range
        depth_norm = (depth_clipped / max_range * 255).astype(np.uint8)

        # Apply JET colormap: near = red, far = blue
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

        # Label
        cv2.putText(depth_color, "DEPTH", (5, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_WHITE, 1)
        cv2.putText(depth_color, f"max={max_range:.0f}m", (5, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_GRAY, 1)

        return depth_color

    # ──────────────────────────────────────────────
    # STATUS PANEL
    # ──────────────────────────────────────────────

    def _draw_status_panel(
        self, obs_map: ObstacleMap, cmd: MotionCommand, loop_hz: float
    ) -> np.ndarray:
        """Draw a text + arrow status panel showing velocity commands and grid."""
        # Status panel should be half width so bottom_panel matches top_panel width
        panel = np.full(
            (IMAGE_HEIGHT, IMAGE_WIDTH // 2, 3), 20, dtype=np.uint8  # dark background
        )

        # Title
        cv2.putText(panel, "NAV STATUS", (5, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_CYAN, 1)

        # Loop frequency
        cv2.putText(panel, f"Loop: {loop_hz:.1f} Hz", (5, 32),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_WHITE, 1)

        # Velocity commands
        y = 55
        cv2.putText(panel, "Velocity Commands:", (5, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_GRAY, 1)
        y += 15

        def vel_bar(label, val, max_val, y_pos, color):
            cv2.putText(panel, f"{label}: {val:+.2f}", (5, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_WHITE, 1)
            bar_len = int(abs(val) / max_val * 60)
            bar_x = 90 if val >= 0 else 90 - bar_len
            cv2.rectangle(panel, (bar_x, y_pos - 8), (bar_x + bar_len, y_pos - 1), color, -1)

        vel_bar("Vx (fwd)", cmd.vx, 4.0, y,    COLOR_GREEN);  y += 14
        vel_bar("Vy (lat)", cmd.vy, 2.0, y,    COLOR_YELLOW); y += 14
        vel_bar("Vz (vert)", cmd.vz, 1.0, y,   COLOR_CYAN);   y += 20

        # Grid visualization
        cv2.putText(panel, "Sector Grid:", (5, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_GRAY, 1)
        y += 14

        cell_size = 18
        grid_labels = [["TL","TC","TR"],["ML","MC","MR"],["BL","BC","BR"]]
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                gx = 10 + col * (cell_size + 3)
                gy = y + row * (cell_size + 3)
                color = COLOR_RED if obs_map.grid[row, col] else (40, 70, 40)
                cv2.rectangle(panel, (gx, gy), (gx + cell_size, gy + cell_size), color, -1)
                cv2.putText(panel, grid_labels[row][col], (gx + 1, gy + 11),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.28, COLOR_WHITE, 1)

        y += GRID_ROWS * (cell_size + 3) + 12

        # State flags
        if cmd.is_escaping:
            cv2.putText(panel, "⚠ ESCAPE MANEUVER", (5, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_ORANGE, 1)
            y += 14
        if cmd.goal_reached:
            cv2.putText(panel, "✓ GOAL REACHED", (5, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_GREEN, 1)

        # Closest obstacle info
        if obs_map.closest:
            y += 20
            dist = obs_map.closest.distance
            cv2.putText(panel, f"Closest: {dist:.1f}m", (5, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                       COLOR_RED if dist < 3.0 else COLOR_YELLOW, 1)

        return panel

    def close(self):
        """Close the visualizer window."""
        cv2.destroyAllWindows()
