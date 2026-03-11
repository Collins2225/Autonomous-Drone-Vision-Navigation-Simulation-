"""
config/settings.py
==================
Central configuration for the entire drone navigation system.

MENTOR NOTE:
  Keeping ALL tunable parameters in ONE place is a research best-practice.
  When you run experiments, you only need to change this file — not hunt
  through multiple source files. This pattern is also called a "config object"
  and is standard in ML research codebases (e.g., Hydra, OmegaConf).
"""

# ─────────────────────────────────────────────
# SIMULATION SETTINGS
# ─────────────────────────────────────────────
AIRSIM_IP = "127.0.0.1"          # AirSim server IP (localhost)
VEHICLE_NAME = "Drone1"           # Must match settings.json

# Camera names (must match AirSim settings.json)
DEPTH_CAMERA_NAME = "front_depth"
RGB_CAMERA_NAME   = "front_rgb"
IMAGE_WIDTH  = 160
IMAGE_HEIGHT = 120

# ─────────────────────────────────────────────
# FLIGHT PARAMETERS
# ─────────────────────────────────────────────
TAKEOFF_ALTITUDE = -3.0   # NED: negative = UP. 3 meters above ground.
CRUISE_SPEED     = 3.5    # m/s forward cruise speed
MAX_YAW_RATE     = 30.0   # degrees/second maximum yaw rate
HOVER_THROTTLE   = 0.5    # Not used in SimpleFlight, but good to document

# ─────────────────────────────────────────────
# PERCEPTION PARAMETERS
# ─────────────────────────────────────────────
# Depth image: pixels closer than this (in meters) are "dangerous"
OBSTACLE_DISTANCE_THRESHOLD = 15.0   # meters

# Morphological kernel size for noise removal in depth mask
MORPH_KERNEL_SIZE = 5

# Divide the image into a grid for spatial obstacle awareness
# e.g., 3x3 = 9 zones: left/center/right × top/middle/bottom
GRID_COLS = 3
GRID_ROWS = 3

# What fraction of a grid cell must be "obstacle pixels" to flag it
OBSTACLE_CELL_THRESHOLD = 0.15   # 15% of pixels in a cell = obstacle

# ─────────────────────────────────────────────
# NAVIGATION PARAMETERS (Artificial Potential Field)
# ─────────────────────────────────────────────
# Attractive force toward goal
APF_GOAL_GAIN      = 3.0    # How strongly the goal pulls
APF_GOAL_THRESHOLD = 0.5    # Distance (m) at which we consider goal reached

# Repulsive force from obstacles
APF_REPULSION_GAIN    = 10.0   # How strongly obstacles push
APF_REPULSION_RANGE   = 10.0   # Only obstacles within this range repel (meters)

# Local minima escape: if stuck, add random perturbation
APF_STUCK_THRESHOLD   = 0.3   # Speed below which we consider "stuck" (m/s)
APF_STUCK_FRAMES      = 20    # Frames at low speed before triggering escape
APF_ESCAPE_MAGNITUDE  = 1.5   # Magnitude of escape perturbation

# ─────────────────────────────────────────────
# CONTROL PARAMETERS
# ─────────────────────────────────────────────
CONTROL_LOOP_HZ   = 4        # Target loop frequency
CONTROL_DT        = 0.25

# Velocity command limits (m/s)
MAX_VX =  4.0   # max forward/backward
MAX_VY =  3.0   # max lateral (strafe)
MAX_VZ =  1.0   # max vertical (NED: positive = DOWN, negative = UP)

# PID-like gain for translating APF vector → velocity commands
VELOCITY_GAIN = 2.0

# ─────────────────────────────────────────────
# VISUALIZER SETTINGS
# ─────────────────────────────────────────────
VISUALIZER_ENABLED  = True
VISUALIZER_SCALE    = 2      # Upscale factor for display window
VIZ_WINDOW_NAME     = "Drone Nav — Debug View"
