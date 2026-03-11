"""
simulation/airsim_client.py
===========================
Wraps the raw AirSim Python API into a clean interface.

MENTOR NOTE:
  The "Facade" design pattern — wrapping a complex external API in a simpler
  interface — is essential in robotics. If AirSim changes its API, you only
  fix THIS file, not every module that uses it. It also makes unit testing
  easier: swap this with a MockClient and test the rest of the system offline.

  AirSim uses NED coordinates:
    +X = North (forward)
    +Y = East  (right)
    +Z = Down  (NEGATIVE means up!)
  
  Velocity commands are in the drone's LOCAL body frame when using
  moveByVelocityBodyFrameAsync, or WORLD frame with moveByVelocityAsync.
  We use world frame for simplicity.
"""

import time
import numpy as np
import cv2

try:
    import airsim
    AIRSIM_AVAILABLE = True
except ImportError:
    AIRSIM_AVAILABLE = False
    print("[SimClient] WARNING: 'airsim' package not found. Using mock mode.")

from config.settings import (
    AIRSIM_IP, VEHICLE_NAME,
    DEPTH_CAMERA_NAME, RGB_CAMERA_NAME,
    IMAGE_WIDTH, IMAGE_HEIGHT,
    TAKEOFF_ALTITUDE, CONTROL_DT
)


class DroneSimClient:
    """
    Clean interface to AirSim for our drone.
    
    Handles: connection, takeoff, image capture, velocity commands, landing.
    Falls back to mock data if AirSim is not available (for offline testing).
    """

    def __init__(self, mock_mode: bool = False):
        self.mock_mode = mock_mode or not AIRSIM_AVAILABLE
        self.client = None
        self._mock_frame_counter = 0

        if not self.mock_mode:
            self._connect()
        else:
            print("[SimClient] Running in MOCK mode — no AirSim connection needed.")

    # ──────────────────────────────────────────────
    # CONNECTION & ARMING
    # ──────────────────────────────────────────────

    def _connect(self):
        """Establish connection to AirSim and enable API control."""
        print(f"[SimClient] Connecting to AirSim at {AIRSIM_IP}...")
        self.client = airsim.MultirotorClient(ip=AIRSIM_IP)

        # confirmConnection() prints drone state — useful for debugging
        self.client.confirmConnection()

        # enableApiControl MUST be called before any command
        self.client.enableApiControl(True, VEHICLE_NAME)

        # arm() "powers on" the motors
        self.client.armDisarm(True, VEHICLE_NAME)

        print("[SimClient] Connected and armed ✓")

    def takeoff(self):
        """
        Hover to target altitude.
        
        MENTOR NOTE:
          AirSim's takeoffAsync() only goes to ~1m. For a specific altitude,
          we use moveToZAsync() which commands a Z position in NED.
          Remember: negative Z = higher altitude!
        """
        if self.mock_mode:
            print("[SimClient] MOCK takeoff complete.")
            return

        print(f"[SimClient] Taking off to altitude {abs(TAKEOFF_ALTITUDE)}m...")
        self.client.takeoffAsync(vehicle_name=VEHICLE_NAME).join()

        # Move to desired altitude and wait (join = blocking)
        self.client.moveToZAsync(
            TAKEOFF_ALTITUDE,
            velocity=2.0,
            vehicle_name=VEHICLE_NAME
        ).join()

        # Small hover pause for stability
        time.sleep(1.0)
        print("[SimClient] Takeoff complete ✓")

    def land_and_disarm(self):
        """Safely land the drone and release API control."""
        if self.mock_mode:
            print("[SimClient] MOCK landing complete.")
            return

        print("[SimClient] Landing...")
        self.client.landAsync(vehicle_name=VEHICLE_NAME).join()
        self.client.armDisarm(False, VEHICLE_NAME)
        self.client.enableApiControl(False, VEHICLE_NAME)
        print("[SimClient] Landed and disarmed ✓")

    # ──────────────────────────────────────────────
    # IMAGE CAPTURE
    # ──────────────────────────────────────────────

    def get_images(self) -> dict:
        """
        Capture RGB and Depth images simultaneously.
        
        Returns:
            dict with keys 'rgb' (H×W×3 uint8) and 'depth' (H×W float32, meters)
        
        MENTOR NOTE:
          We request BOTH images in a SINGLE API call using a list of
          ImageRequest objects. This is important — if you call them separately,
          they won't be time-synchronized. Simultaneous capture = same timestamp.
          
          Depth image type 2 = DepthPlanar (perpendicular distance from camera plane)
          vs type 1 = DepthPerspective (distance along ray). Planar is better for
          computing lateral obstacle position.
        """
        if self.mock_mode:
            return self._get_mock_images()

        requests = [
            airsim.ImageRequest(
                RGB_CAMERA_NAME,
                airsim.ImageType.Scene,
                pixels_as_float=False,
                compress=False
            ),
            airsim.ImageRequest(
                DEPTH_CAMERA_NAME,
                airsim.ImageType.DepthPlanar,
                pixels_as_float=True,  # returns float32 in meters
                compress=False
            ),
        ]

        responses = self.client.simGetImages(requests, vehicle_name=VEHICLE_NAME)

        # ── Decode RGB ──
        rgb_response = responses[0]
        rgb_1d = np.frombuffer(rgb_response.image_data_uint8, dtype=np.uint8)
        rgb = rgb_1d.reshape(rgb_response.height, rgb_response.width, 3)

        # ── Decode Depth ──
        depth_response = responses[1]
        depth = np.array(depth_response.image_data_float, dtype=np.float32)
        depth = depth.reshape(depth_response.height, depth_response.width)

        return {"rgb": rgb, "depth": depth}

    def _get_mock_images(self) -> dict:
        """
        Generate synthetic test images for offline development.
        
        MENTOR NOTE:
          Always build mock/stub interfaces early. This lets you develop and
          test perception & navigation logic WITHOUT needing AirSim running.
          Real robotics research spends >50% of time in simulation, so fast
          iteration in mock mode accelerates development significantly.
        """
        self._mock_frame_counter += 1
        t = self._mock_frame_counter * 0.1

        # Synthetic RGB: gradient with a moving "obstacle" rectangle
        rgb = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        rgb[:, :, 2] = 50   # slight blue sky

        # Animate an obstacle moving left-right
        obs_x = int(IMAGE_WIDTH * 0.5 + 80 * np.sin(t))
        obs_y = int(IMAGE_HEIGHT * 0.4)
        cv2.rectangle(rgb, (obs_x - 30, obs_y - 40), (obs_x + 30, obs_y + 40),
                      (100, 80, 60), -1)

        # Synthetic Depth: far background (20m), close where obstacle is
        depth = np.full((IMAGE_HEIGHT, IMAGE_WIDTH), 20.0, dtype=np.float32)
        depth[obs_y - 40:obs_y + 40, obs_x - 30:obs_x + 30] = 2.5  # 2.5m away

        return {"rgb": rgb, "depth": depth}

    # ──────────────────────────────────────────────
    # VELOCITY COMMANDS
    # ──────────────────────────────────────────────

    def set_velocity(self, vx: float, vy: float, vz: float, duration: float = None):
        """
        Command drone velocity in WORLD frame (NED).
        
        Args:
            vx: forward velocity (m/s, positive = North)
            vy: lateral velocity (m/s, positive = East / right)
            vz: vertical velocity (m/s, positive = DOWN — use negatives to climb!)
            duration: how long to apply (default = one control loop tick)
        
        MENTOR NOTE:
          moveByVelocityAsync is NON-BLOCKING by default.
          We pass duration = one tick, then immediately call it again next loop.
          This effectively gives us continuous velocity control at our loop rate.
          
          Yaw mode ZERO_YAW_RATE keeps current heading (no spinning).
          Switch to FORWARD to auto-align with velocity direction.
        """
        if self.mock_mode:
            return  # In mock mode, we just acknowledge the command silently

        if duration is None:
            duration = CONTROL_DT * 1.5  # slight overlap to prevent hover gaps

        self.client.moveByVelocityAsync(
            vx, vy, vz,
            duration=duration,
            yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0),
            vehicle_name=VEHICLE_NAME
        )

    def hover(self):
        """Command zero velocity — drone holds position."""
        self.set_velocity(0.0, 0.0, 0.0, duration=2.0)

    def get_state(self) -> dict:
        """
        Get current drone position, velocity, and orientation.
        
        Returns a simplified state dict (not the full AirSim KinematicsState).
        """
        if self.mock_mode:
            return {
                "position": np.array([0.0, 0.0, TAKEOFF_ALTITUDE]),
                "velocity": np.array([1.0, 0.0, 0.0]),
                "yaw_deg": 0.0
            }

        state = self.client.getMultirotorState(vehicle_name=VEHICLE_NAME)
        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity

        # Convert quaternion → Euler angles for readable yaw
        orientation = state.kinematics_estimated.orientation
        _, _, yaw = airsim.to_eularian_angles(orientation)

        return {
            "position": np.array([pos.x_val, pos.y_val, pos.z_val]),
            "velocity": np.array([vel.x_val, vel.y_val, vel.z_val]),
            "yaw_deg": np.degrees(yaw)
        }
