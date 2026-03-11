"""
simulation/airsim_client.py
============================
AirSim connection wrapper — rewritten from scratch.
Handles connection, takeoff, image capture, velocity commands, landing.
"""

import time
import numpy as np
import cv2

try:
    import airsim
    AIRSIM_AVAILABLE = True
except ImportError:
    AIRSIM_AVAILABLE = False
    print("[SimClient] WARNING: airsim package not found. Using mock mode.")

from config.settings import (
    AIRSIM_IP,
    VEHICLE_NAME,
    DEPTH_CAMERA_NAME,
    RGB_CAMERA_NAME,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    TAKEOFF_ALTITUDE,
    CONTROL_DT
)


class DroneSimClient:
    """
    Clean interface to AirSim for our drone.
    Handles connection, takeoff, image capture,
    velocity commands and landing.
    Falls back to mock data if AirSim is not available.
    """

    def __init__(self, mock_mode: bool = False):
        self.mock_mode = mock_mode or not AIRSIM_AVAILABLE
        self.client = None
        self._mock_frame_counter = 0

        if not self.mock_mode:
            self._connect()
        else:
            print("[SimClient] Running in MOCK mode.")

    # ──────────────────────────────────────────────
    # CONNECTION
    # ──────────────────────────────────────────────

    def _connect(self):
        """Connect to AirSim and enable API control."""
        print(f"[SimClient] Connecting to AirSim at {AIRSIM_IP}...")
        self.client = airsim.MultirotorClient(ip=AIRSIM_IP)
        self.client.confirmConnection()
        self.client.enableApiControl(True, VEHICLE_NAME)
        self.client.armDisarm(True, VEHICLE_NAME)
        print("[SimClient] Connected and armed ✓")

    # ──────────────────────────────────────────────
    # TAKEOFF AND LANDING
    # ──────────────────────────────────────────────

    def takeoff(self):
        """Take off and hover at target altitude."""
        if self.mock_mode:
            print("[SimClient] MOCK takeoff complete.")
            return

        print(f"[SimClient] Taking off to altitude {abs(TAKEOFF_ALTITUDE)}m...")
        self.client.takeoffAsync(vehicle_name=VEHICLE_NAME).join()
        self.client.moveToZAsync(
            TAKEOFF_ALTITUDE,
            velocity=2.0,
            vehicle_name=VEHICLE_NAME
        ).join()
        time.sleep(1.0)
        print("[SimClient] Takeoff complete ✓")

    def land_and_disarm(self):
        """Land the drone and release API control."""
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
        Capture RGB and Depth images from AirSim simultaneously.

        Returns:
            dict with:
              'rgb'   -> numpy array shape (IMAGE_HEIGHT, IMAGE_WIDTH, 3) uint8
              'depth' -> numpy array shape (IMAGE_HEIGHT, IMAGE_WIDTH)    float32
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
                pixels_as_float=True,
                compress=False
            ),
        ]

        responses = self.client.simGetImages(
            requests,
            vehicle_name=VEHICLE_NAME
        )

        rgb   = self._decode_rgb(responses[0])
        depth = self._decode_depth(responses[1])

        return {"rgb": rgb, "depth": depth}

    def _decode_rgb(self, response) -> np.ndarray:
        """
        Decode RGB image response from AirSim.
        Reshapes using ACTUAL response dimensions then
        resizes to configured IMAGE_WIDTH x IMAGE_HEIGHT.
        """
        raw = np.frombuffer(response.image_data_uint8, dtype=np.uint8)

        h = response.height
        w = response.width
        expected_size = h * w * 3
        actual_size   = len(raw)

        if actual_size == expected_size:
            img = raw.reshape(h, w, 3)
        else:
            print(f"[SimClient] RGB size mismatch: got {actual_size}, expected {expected_size}")
            pixels = actual_size // 3
            side   = int(np.sqrt(pixels))
            img    = raw[:side * side * 3].reshape(side, side, 3)

        img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
        return img

    def _decode_depth(self, response) -> np.ndarray:
        """
        Decode depth image response from AirSim.
        Reshapes using ACTUAL response dimensions then
        resizes to configured IMAGE_WIDTH x IMAGE_HEIGHT.
        """
        raw = np.array(response.image_data_float, dtype=np.float32)

        h = response.height
        w = response.width
        expected_size = h * w
        actual_size   = len(raw)

        if actual_size == expected_size:
            depth = raw.reshape(h, w)
        else:
            print(f"[SimClient] Depth size mismatch: got {actual_size}, expected {expected_size}")
            side  = int(np.sqrt(actual_size))
            depth = raw[:side * side].reshape(side, side)

        depth[~np.isfinite(depth)] = 999.0
        depth[depth <= 0]          = 999.0

        depth = cv2.resize(depth, (IMAGE_WIDTH, IMAGE_HEIGHT))
        return depth

    # ──────────────────────────────────────────────
    # MOCK IMAGE GENERATOR
    # ──────────────────────────────────────────────

    def _get_mock_images(self) -> dict:
        """Generate synthetic test images for offline development."""
        self._mock_frame_counter += 1
        t = self._mock_frame_counter * 0.1

        rgb = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        rgb[:, :, 2] = 50

        obs_x = int(IMAGE_WIDTH  * 0.5 + 60 * np.sin(t))
        obs_y = int(IMAGE_HEIGHT * 0.4)
        cv2.rectangle(
            rgb,
            (obs_x - 25, obs_y - 30),
            (obs_x + 25, obs_y + 30),
            (100, 80, 60), -1
        )

        depth = np.full((IMAGE_HEIGHT, IMAGE_WIDTH), 20.0, dtype=np.float32)
        depth[obs_y - 30:obs_y + 30, obs_x - 25:obs_x + 25] = 2.5

        return {"rgb": rgb, "depth": depth}

    # ──────────────────────────────────────────────
    # VELOCITY COMMANDS
    # ──────────────────────────────────────────────

    def set_velocity(self, vx: float, vy: float, vz: float, duration: float = None):
        """
        Command drone velocity in world frame NED coordinates.

        Args:
            vx: forward velocity m/s  (positive = forward)
            vy: lateral velocity m/s  (positive = right)
            vz: vertical velocity m/s (positive = DOWN, negative = UP)
            duration: seconds to apply this command
        """
        if self.mock_mode:
            return

        if duration is None:
            duration = CONTROL_DT * 1.5

        self.client.moveByVelocityAsync(
            vx, vy, vz,
            duration=duration,
            yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0),
            vehicle_name=VEHICLE_NAME
        )

    def hover(self):
        """Command zero velocity — drone holds current position."""
        self.set_velocity(0.0, 0.0, 0.0, duration=2.0)

    # ──────────────────────────────────────────────
    # STATE
    # ──────────────────────────────────────────────

    def get_state(self) -> dict:
        """
        Get current drone position, velocity and orientation.

        Returns:
            dict with keys:
              'position' -> np.array [x, y, z] in meters NED
              'velocity' -> np.array [vx, vy, vz] in m/s
              'yaw_deg'  -> float current heading in degrees
        """
        if self.mock_mode:
            return {
                "position": np.array([0.0, 0.0, TAKEOFF_ALTITUDE]),
                "velocity": np.array([1.0, 0.0, 0.0]),
                "yaw_deg":  0.0
            }

        state = self.client.getMultirotorState(vehicle_name=VEHICLE_NAME)
        pos   = state.kinematics_estimated.position
        vel   = state.kinematics_estimated.linear_velocity
        ori   = state.kinematics_estimated.orientation

        _, _, yaw = airsim.to_eularian_angles(ori)

        return {
            "position": np.array([pos.x_val, pos.y_val, pos.z_val]),
            "velocity": np.array([vel.x_val, vel.y_val, vel.z_val]),
            "yaw_deg":  float(np.degrees(yaw))
        }