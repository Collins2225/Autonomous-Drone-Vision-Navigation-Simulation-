"""
utils/visualizer.py
====================
Real-time debug visualization window for the autonomous navigation system.
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
        top_panel = self._draw_camera_panel(rgb, obs_map)

        depth_vis = self._colorize_depth(depth)
        depth_vis = cv2.resize(depth_vis, (IMAGE_WIDTH // 2, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)

        status_panel = self._draw_status_panel(obs_map, cmd, loop_hz)
        bottom_panel = np.hstack([depth_vis, status_panel])

        composite = np.vstack([top_panel, bottom_panel])

        h, w = composite.shape[:2]
        display = cv2.resize(
            composite,
            (w * VISUALIZER_SCALE, h * VISUALIZER_SCALE),
            interpolation=cv2.INTER_NEAREST
        )

        cv2.imshow(VIZ_WINDOW_NAME, display)
        cv2.waitKey(1)

        self._frame_count += 1

    # ──────────────────────────────────────────────
    # CAMERA PANEL
    # ──────────────────────────────────────────────

    def _draw_camera_panel(self, rgb: np.ndarray, obs_map: ObstacleMap) -> np.ndarray:
        canvas = rgb.copy()

        # FIXED: Resize danger mask to match RGB resolution
        if obs_map.danger_mask is not None and np.any(obs_map.danger_mask):
            mask_resized = cv2.resize(
                obs_map.danger_mask.astype(np.uint8),
                (rgb.shape[1], rgb.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

            red_overlay = np.zeros_like(canvas)
            red_overlay[mask_resized > 0] = (0, 0, 180)
            canvas = cv2.addWeighted(canvas, 1.0, red_overlay, 0.4, 0)

        # Draw grid overlay
        canvas = self._draw_grid_overlay(canvas, obs_map.grid)

        # Draw obstacle centroids
        for obs in obs_map.obstacles:
            cx, cy = int(obs.image_x), int(obs.image_y)
            color = COLOR_RED if obs.distance < 2.0 else COLOR_YELLOW

            cv2.circle(canvas, (cx, cy), 8, color, 2)
            cv2.putText(canvas, f"{obs.distance:.1f}m", (cx + 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Crosshair
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
        cell_h = IMAGE_HEIGHT // GRID_ROWS
        cell_w = IMAGE_WIDTH  // GRID_COLS

        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                x1 = col * cell_w
                y1 = row * cell_h
                x2 = x1 + cell_w
                y2 = y1 + cell_h

                color = COLOR_RED if grid[row, col] else COLOR_GRAY
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)

        return canvas

    # ──────────────────────────────────────────────
    # DEPTH PANEL
    # ──────────────────────────────────────────────

    def _colorize_depth(self, depth: np.ndarray) -> np.ndarray:
        if depth.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
            depth = cv2.resize(depth, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)

        max_range = OBSTACLE_DISTANCE_THRESHOLD * 1.5

        depth_clipped = np.clip(depth, 0, max_range)
        depth_clipped[~np.isfinite(depth_clipped)] = max_range
        depth_norm = (depth_clipped / max_range * 255).astype(np.uint8)

        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

        cv2.putText(depth_color, "DEPTH", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_WHITE, 1)
        cv2.putText(depth_color, f"max={max_range:.0f}m", (5, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_GRAY, 1)

        return depth_color

    # ──────────────────────────────────────────────
    # STATUS PANEL
    # ──────────────────────────────────────────────

    def _draw_status_panel(self, obs_map: ObstacleMap, cmd: MotionCommand, loop_hz: float) -> np.ndarray:
        panel = np.full((IMAGE_HEIGHT, IMAGE_WIDTH // 2, 3), 20, dtype=np.uint8)

        cv2.putText(panel, "NAV STATUS", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_CYAN, 1)
        cv2.putText(panel, f"Loop: {loop_hz:.1f} Hz", (5, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_WHITE, 1)

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

        vel_bar("Vx (fwd)", cmd.vx, 4.0, y, COLOR_GREEN);  y += 14
        vel_bar("Vy (lat)", cmd.vy, 2.0, y, COLOR_YELLOW); y += 14
        vel_bar("Vz (vert)", cmd.vz, 1.0, y, COLOR_CYAN);  y += 20

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

        if cmd.is_escaping:
            cv2.putText(panel, "⚠ ESCAPE MANEUVER", (5, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_ORANGE, 1)
            y += 14

        if cmd.goal_reached:
            cv2.putText(panel, "✓ GOAL REACHED", (5, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_GREEN, 1)

        if obs_map.closest:
            y += 20
            dist = obs_map.closest.distance
            cv2.putText(panel, f"Closest: {dist:.1f}m", (5, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        COLOR_RED if dist < 3.0 else COLOR_YELLOW, 1)

        return panel

    def close(self):
        cv2.destroyAllWindows()
