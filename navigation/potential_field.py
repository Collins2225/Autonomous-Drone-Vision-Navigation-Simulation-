"""
navigation/potential_field.py
==============================
Translates obstacle maps into motion vectors using Artificial Potential Fields.

MENTOR NOTE — Why APF?
  Artificial Potential Fields (Khatib, 1986) model navigation as physics:
  
  - The GOAL exerts an ATTRACTIVE force (like gravity pulling a ball)
  - Each OBSTACLE exerts a REPULSIVE force (like magnets with same poles)
  - The drone follows the NET FORCE vector
  
  Mathematically:
    F_total = F_attractive + Σ F_repulsive_i
  
  F_attractive = k_att * (goal_position - drone_position)
  F_repulsive  = k_rep * (1/dist - 1/range) * (1/dist²) * unit_vector_away
  
  This gives smooth, continuous trajectories that naturally avoid obstacles
  while heading toward the goal — no discrete state machine needed.
  
  LIMITATION: Local minima — APF can get "stuck" in a pocket surrounded by
  obstacles where all forces cancel out. We detect and escape this situation.
  
  RESEARCH REFERENCE:
    Khatib, O. (1986). Real-time obstacle avoidance for manipulators and
    mobile robots. IJRR, 5(1), 90-98.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
import time

from config.settings import (
    APF_GOAL_GAIN, APF_GOAL_THRESHOLD,
    APF_REPULSION_GAIN, APF_REPULSION_RANGE,
    APF_STUCK_THRESHOLD, APF_STUCK_FRAMES, APF_ESCAPE_MAGNITUDE,
    CRUISE_SPEED, MAX_VX, MAX_VY, MAX_VZ,
    IMAGE_WIDTH, IMAGE_HEIGHT
)
from perception.obstacle_detector import ObstacleMap


# ──────────────────────────────────────────────
# DATA STRUCTURES
# ──────────────────────────────────────────────

@dataclass
class MotionCommand:
    """
    Navigation output: desired velocity in drone body/world frame.
    
    Fields:
      vx: forward velocity (m/s, positive = forward/North)
      vy: lateral velocity (m/s, positive = right/East)
      vz: vertical velocity (m/s, positive = DOWN — use negatives to climb!)
      yaw_rate: desired yaw rate (deg/s, positive = clockwise from above)
      is_escaping: True if currently in local-minima escape maneuver
      goal_reached: True if drone has arrived at waypoint
    """
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw_rate: float = 0.0
    is_escaping: bool = False
    goal_reached: bool = False


# ──────────────────────────────────────────────
# MAIN NAVIGATOR
# ──────────────────────────────────────────────

class PotentialFieldNavigator:
    """
    APF-based navigator. Receives obstacle maps, outputs velocity commands.
    
    The goal is expressed in RELATIVE 2D image space:
      [0, 0] = center of image = "fly straight ahead"
      [-1, 0] = go left
      [1, 0]  = go right
    
    For 3D waypoint navigation, extend with a global planner that feeds
    waypoints in world coordinates and projects them into the drone's
    local frame.
    """

    def __init__(self):
        # Current goal in image-space: [0,0] = straight ahead by default
        self.goal_image: np.ndarray = np.array([0.0, 0.0], dtype=np.float32)
        self.goal_distance: float = 100.0  # meters to goal (use GPS/odometry)

        # Local minima detection
        self._low_speed_frames: int = 0
        self._escape_vector: Optional[np.ndarray] = None
        self._escape_frames_remaining: int = 0

        # Navigation state
        self._last_command_time: float = time.time()

    def set_goal(self, image_x_norm: float, image_y_norm: float, distance_m: float):
        """
        Set a new navigation goal.
        
        Args:
            image_x_norm: normalized image x (-1=left, 0=center, 1=right)
            image_y_norm: normalized image y (-1=top, 0=center, 1=bottom)
            distance_m:   estimated distance to goal in meters
        """
        self.goal_image = np.array([image_x_norm, image_y_norm], dtype=np.float32)
        self.goal_distance = distance_m

    def compute(self, obs_map: ObstacleMap, current_speed: float = 0.0) -> MotionCommand:
        """
        Main navigation computation.
        
        Args:
            obs_map:       ObstacleMap from the perception module
            current_speed: drone's current forward speed (m/s) for stuck detection
        
        Returns:
            MotionCommand with desired velocities
        """
        # ── Check goal reached ──
        if self.goal_distance < APF_GOAL_THRESHOLD:
            return MotionCommand(goal_reached=True)

        # ── Check for local minima / stuck situation ──
        if self._escape_frames_remaining > 0:
            return self._execute_escape()

        self._update_stuck_detector(current_speed)

        if self._low_speed_frames >= APF_STUCK_FRAMES and not obs_map.is_clear:
            self._trigger_escape(obs_map)
            return self._execute_escape()

        # ── Compute APF forces ──
        f_attractive = self._attractive_force()
        f_repulsive  = self._repulsive_force(obs_map)
        f_total      = f_attractive + f_repulsive

        # ── Convert force vector → velocity command ──
        cmd = self._force_to_velocity(f_total, obs_map)

        return cmd

    # ──────────────────────────────────────────────
    # APF FORCE COMPUTATIONS
    # ──────────────────────────────────────────────

    def _attractive_force(self) -> np.ndarray:
        """
        Compute attractive force pulling drone toward goal.
        
        F_att = k_att * (goal - current_position)
        
        In image space, current position is [0,0] (image center = forward direction).
        Goal is the target direction vector.
        
        MENTOR NOTE:
          We use a LINEAR attractive potential (not quadratic) because:
          1. It produces constant-magnitude attraction regardless of distance
          2. Less likely to cause overshooting at close range
          3. Simpler to tune
          
          For global navigation, switch to quadratic to get stronger pull at
          greater distances.
        """
        # Direction from current image center to goal
        goal_dir = self.goal_image - np.array([0.0, 0.0])
        norm = np.linalg.norm(goal_dir)

        if norm < 1e-6:
            # Already aligned with goal direction
            return np.array([APF_GOAL_GAIN, 0.0])  # just go forward

        # Normalize and scale
        return APF_GOAL_GAIN * (goal_dir / norm)

    def _repulsive_force(self, obs_map: ObstacleMap) -> np.ndarray:
        """
        Compute repulsive forces from all detected obstacles.
        
        F_rep_i = k_rep * (1/dist_i - 1/range) * (1/dist_i²) * unit_away_i
        
        This force is:
          - Zero when obstacle is beyond APF_REPULSION_RANGE
          - Grows as 1/d² as obstacle approaches (very strong at close range)
          - Directed AWAY from the obstacle
        
        MENTOR NOTE:
          We work in a MIXED coordinate system here:
          - Lateral avoidance (left/right) uses image-space direction vectors
          - Distance uses real metric values from depth image
          
          This hybrid approach avoids needing camera calibration for the basic
          case. For research quality, project image coordinates → world
          coordinates using the camera's intrinsic matrix.
        """
        f_total = np.zeros(2, dtype=np.float32)

        if not obs_map.obstacles:
            return f_total

        for obs in obs_map.obstacles:
            dist = max(obs.distance, 0.3)   # clamp to avoid division by zero

            if dist > APF_REPULSION_RANGE:
                continue  # This obstacle is too far to matter

            # Repulsion magnitude: (1/d - 1/range) / d²
            magnitude = APF_REPULSION_GAIN * (
                (1.0 / dist - 1.0 / APF_REPULSION_RANGE) / (dist * dist)
            )

            # Direction: away from obstacle (opposite to obs.direction)
            # obs.direction points FROM image center TO obstacle
            # We want force pointing AWAY from obstacle → negate
            away_direction = -obs.direction
            d_norm = np.linalg.norm(away_direction)
            if d_norm > 1e-6:
                away_direction /= d_norm

            f_total += magnitude * away_direction

        return f_total

    # ──────────────────────────────────────────────
    # FORCE → VELOCITY CONVERSION
    # ──────────────────────────────────────────────

    def _force_to_velocity(
        self, force: np.ndarray, obs_map: ObstacleMap
    ) -> MotionCommand:
        """
        Map the 2D APF force vector to 3D velocity commands.
        
        Image space force: [lateral, vertical] → world space: [vx, vy, vz]
        
        MENTOR NOTE — Coordinate Mapping:
          Image X (left/right) → World Y (East/West lateral)
          Image Y (up/down)    → World Z (Down/Up altitude)
          
          Forward speed (vx) is a separate term:
            - If path is clear → cruise at CRUISE_SPEED
            - If obstacle ahead → slow down proportionally
            - If imminent collision → stop / back up
        """
        # Extract lateral (image-X) and vertical (image-Y) force components
        lateral_force  = float(force[0])  # positive = go right in image
        vertical_force = float(force[1])  # positive = go down in image (avoid above)

        # ── Forward speed: reduce when obstacle is directly ahead ──
        center_blocked = obs_map.grid[1, 1]  # center cell of grid
        if obs_map.closest is not None:
            proximity_factor = max(0.0, min(1.0,
                (obs_map.closest.distance - 1.0) / (APF_REPULSION_RANGE - 1.0)
            ))
        else:
            proximity_factor = 1.0

        if center_blocked:
            vx = CRUISE_SPEED * proximity_factor * 0.3   # heavily slow down
        else:
            vx = CRUISE_SPEED * proximity_factor

        # ── Lateral velocity: proportional to lateral APF force ──
        vy = np.clip(lateral_force * 3.0, -MAX_VY, MAX_VY)

        # ── Vertical velocity: use vertical force for altitude adjustment ──
        # Negative vertical force (obstacle below) → climb (negative vz in NED)
        vz = np.clip(-vertical_force * 0.5, -MAX_VZ, MAX_VZ)

        return MotionCommand(vx=vx, vy=vy, vz=vz)

    # ──────────────────────────────────────────────
    # LOCAL MINIMA ESCAPE
    # ──────────────────────────────────────────────

    def _update_stuck_detector(self, current_speed: float):
        """Track how many consecutive frames the drone has been moving slowly."""
        if current_speed < APF_STUCK_THRESHOLD:
            self._low_speed_frames += 1
        else:
            self._low_speed_frames = 0

    def _trigger_escape(self, obs_map: ObstacleMap):
        """
        Generate a random escape vector to break out of a local minimum.
        
        MENTOR NOTE:
          Local minima are APF's Achilles heel. The classic fix is to add
          random perturbation ("random walk" escape). More sophisticated
          approaches include:
          
          1. Virtual obstacles — add a repulsive force at the current position
             to prevent re-entering the same trap
          2. Tangential force — rotate the repulsive force 90°
          3. Navigation functions — mathematically guaranteed APF variant
             (Rimon & Koditschek, 1992) that has no local minima, but
             requires full map knowledge
          4. RRT* — switch to sampling-based planning when APF gets stuck
          
          For now, random perturbation works well in practice.
        """
        print("[Navigation] ⚠ Local minimum detected — triggering escape maneuver")

        # Choose escape direction opposite to most populated obstacle side
        left_blocked  = obs_map.grid[1, 0]  # middle-left
        right_blocked = obs_map.grid[1, 2]  # middle-right

        if left_blocked and not right_blocked:
            escape_y = APF_ESCAPE_MAGNITUDE   # go right
        elif right_blocked and not left_blocked:
            escape_y = -APF_ESCAPE_MAGNITUDE  # go left
        else:
            # Both blocked or neither — pick random direction
            escape_y = APF_ESCAPE_MAGNITUDE * np.random.choice([-1.0, 1.0])

        self._escape_vector = np.array([escape_y], dtype=float)
        self._escape_frames_remaining = 15  # maintain escape for 15 frames
        self._low_speed_frames = 0

    def _execute_escape(self) -> MotionCommand:
        """Execute current escape maneuver."""
        self._escape_frames_remaining -= 1
        vy = float(self._escape_vector[0]) if self._escape_vector is not None else 0.0
        return MotionCommand(
            vx=0.3,        # slow forward during escape
            vy=vy,
            vz=-0.3,       # slight climb during escape (often helps)
            is_escaping=True
        )
