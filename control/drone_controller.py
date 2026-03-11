"""
control/drone_controller.py
============================
Translates MotionCommands into AirSim velocity API calls.

MENTOR NOTE — The Control Layer's Role:
  Separation of concerns is crucial in robotics systems:
  
  Navigation says: "I want to go 2 m/s forward and 1 m/s right"
  Control asks:    "Are those velocities safe? Within limits? Properly timed?"
  
  The control layer is responsible for:
    1. Safety clipping (enforce velocity limits)
    2. Smooth command ramping (avoid jerky motion → motor stress)
    3. Altitude hold (maintain cruise altitude independent of obstacles)
    4. Emergency stop
    5. Telemetry logging
  
  In a real drone, the control layer would also interface with a PID
  position/velocity controller. AirSim's SimpleFlight handles the inner
  PID loop for us, so we just send high-level velocity commands.

COORDINATE REMINDER:
  AirSim NED: +X=North(forward), +Y=East(right), +Z=Down
  Our vx = forward, vy = right, vz = DOWN (negative = climb)
"""

import time
import numpy as np
from collections import deque

from config.settings import (
    MAX_VX, MAX_VY, MAX_VZ,
    TAKEOFF_ALTITUDE, CONTROL_DT,
    VELOCITY_GAIN
)
from navigation.potential_field import MotionCommand
from simulation.airsim_client import DroneSimClient


class DroneController:
    """
    Receives MotionCommands and sends them to AirSim with safety checks.
    
    Also handles:
      - Altitude hold (keeps drone at cruise altitude unless avoiding vertically)
      - Emergency stop trigger
      - Smooth velocity ramping (exponential moving average filter)
    """

    def __init__(self, sim_client: DroneSimClient):
        self.sim = sim_client
        self._emergency_stop = False

        # Smoothing buffers for velocity commands
        # Exponential Moving Average: filtered = alpha * new + (1-alpha) * old
        self._SMOOTH_ALPHA = 0.4    # 0 = very smooth/slow, 1 = no smoothing
        self._vx_smooth = 0.0
        self._vy_smooth = 0.0
        self._vz_smooth = 0.0

        # Telemetry log (last 100 commands)
        self._telemetry = deque(maxlen=100)

        # Altitude controller state
        self._target_altitude_z = TAKEOFF_ALTITUDE   # NED Z (negative = up)

    # ──────────────────────────────────────────────
    # PUBLIC INTERFACE
    # ──────────────────────────────────────────────

    def execute(self, cmd: MotionCommand) -> bool:
        """
        Main method: take a MotionCommand and send it to the drone.
        
        Args:
            cmd: MotionCommand from the navigation module
        
        Returns:
            True if command was sent, False if emergency stop triggered
        """
        if self._emergency_stop:
            self._send_hover()
            return False

        if cmd.goal_reached:
            print("[Control] 🎯 Goal reached — hovering.")
            self._send_hover()
            return True

        # Step 1: Apply safety limits
        safe_cmd = self._apply_limits(cmd)

        # Step 2: Altitude correction
        # If the nav command has minimal vertical component, hold altitude
        altitude_vz = self._altitude_hold_correction()

        # Blend: if navigation wants significant vertical movement, use nav's vz
        # otherwise use altitude hold
        if abs(safe_cmd.vz) > 0.2:
            final_vz = safe_cmd.vz    # nav wants vertical movement (avoiding)
        else:
            final_vz = altitude_vz   # hold altitude

        # Step 3: Smooth commands (prevents jerky motion)
        svx, svy, svz = self._smooth(safe_cmd.vx, safe_cmd.vy, final_vz)

        # Step 4: Final safety clip after smoothing
        svx = np.clip(svx, -MAX_VX, MAX_VX)
        svy = np.clip(svy, -MAX_VY, MAX_VY)
        svz = np.clip(svz, -MAX_VZ, MAX_VZ)

        # Step 5: Send to AirSim
        self.sim.set_velocity(svx, svy, svz)

        # Step 6: Log telemetry
        self._log(svx, svy, svz, cmd.is_escaping)

        return True

    def emergency_stop(self):
        """
        Immediately halt the drone.
        
        MENTOR NOTE:
          Always implement an emergency stop. In hardware testing, this is
          mapped to a physical killswitch. In simulation, it's a software flag.
          Your autonomous loop should check keyboard input or ROS topic for
          an E-stop signal every cycle.
        """
        print("[Control] 🛑 EMERGENCY STOP")
        self._emergency_stop = True
        self._send_hover()

    def reset_emergency_stop(self):
        """Re-enable after emergency stop."""
        self._emergency_stop = False
        self._vx_smooth = 0.0
        self._vy_smooth = 0.0
        self._vz_smooth = 0.0

    # ──────────────────────────────────────────────
    # SAFETY & SMOOTHING
    # ──────────────────────────────────────────────

    def _apply_limits(self, cmd: MotionCommand) -> MotionCommand:
        """
        Clip all velocities to configured maximums.
        
        MENTOR NOTE:
          Never send unlimited velocity commands to a drone.
          Even in simulation, runaway velocity commands cause instability
          and make debugging very difficult. Always enforce hard limits here,
          separate from any limits in the navigation module (defense in depth).
        """
        return MotionCommand(
            vx=np.clip(cmd.vx, -MAX_VX, MAX_VX),
            vy=np.clip(cmd.vy, -MAX_VY, MAX_VY),
            vz=np.clip(cmd.vz, -MAX_VZ, MAX_VZ),
            is_escaping=cmd.is_escaping,
            goal_reached=cmd.goal_reached
        )

    def _altitude_hold_correction(self) -> float:
        """
        Compute a vz correction to maintain cruise altitude.
        
        Gets current Z from AirSim state, computes error, applies proportional
        control to correct it.
        
        MENTOR NOTE — Proportional Altitude Hold:
          This is a P-controller (proportional only):
            error = target_z - current_z
            vz_correction = Kp * error
          
          For production, use a PID controller (add integral to eliminate
          steady-state error, and derivative to damp oscillations).
          AirSim's SimpleFlight has its own inner-loop PID, so our outer-loop
          P-only controller is usually sufficient.
        """
        state = self.sim.get_state()
        current_z = state["position"][2]   # NED Z (negative = above ground)

        # Error in NED: positive error → drone is too high → send positive vz (go down)
        error = self._target_altitude_z - current_z
        Kp = 0.5
        vz_correction = np.clip(Kp * error, -MAX_VZ, MAX_VZ)

        return vz_correction

    def _smooth(self, vx: float, vy: float, vz: float):
        """
        Apply exponential moving average to soften command changes.
        
        MENTOR NOTE — Why smooth commands?
          Sudden velocity changes (from 0 → full speed in one step) cause:
          1. Mechanical stress on motors and ESCs
          2. Motion blur in camera (hurts perception)
          3. Attitude oscillations that destabilize the drone
          
          EMA smoothing: value_new = α * input + (1-α) * value_old
          Low α = heavy smoothing (laggy but stable)
          High α = light smoothing (responsive but jittery)
          
          Tune α based on your control loop rate and drone dynamics.
        """
        a = self._SMOOTH_ALPHA
        self._vx_smooth = a * vx + (1 - a) * self._vx_smooth
        self._vy_smooth = a * vy + (1 - a) * self._vy_smooth
        self._vz_smooth = a * vz + (1 - a) * self._vz_smooth
        return self._vx_smooth, self._vy_smooth, self._vz_smooth

    def _send_hover(self):
        """Send zero-velocity command → drone holds position."""
        self._vx_smooth *= 0.5  # ramp down smoothly
        self._vy_smooth *= 0.5
        self._vz_smooth *= 0.5
        self.sim.set_velocity(
            self._vx_smooth,
            self._vy_smooth,
            self._vz_smooth
        )

    # ──────────────────────────────────────────────
    # TELEMETRY
    # ──────────────────────────────────────────────

    def _log(self, vx: float, vy: float, vz: float, escaping: bool):
        """Log command to telemetry buffer."""
        self._telemetry.append({
            "t": time.time(),
            "vx": vx, "vy": vy, "vz": vz,
            "escaping": escaping
        })

    def get_telemetry_summary(self) -> dict:
        """
        Return summary statistics from telemetry buffer.
        Useful for performance analysis after a flight.
        """
        if not self._telemetry:
            return {}

        vxs = [t["vx"] for t in self._telemetry]
        vys = [t["vy"] for t in self._telemetry]
        escape_count = sum(1 for t in self._telemetry if t["escaping"])

        return {
            "mean_forward_speed": np.mean(vxs),
            "max_lateral_speed":  np.max(np.abs(vys)),
            "escape_maneuvers":   escape_count,
            "total_commands":     len(self._telemetry)
        }
