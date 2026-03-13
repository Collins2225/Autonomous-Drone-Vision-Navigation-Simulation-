"""
config/settings.py
==================
Optimized configuration — performance tuned + YOLO enabled.
"""

# ─────────────────────────────────────────────
# SIMULATION SETTINGS
# ─────────────────────────────────────────────
AIRSIM_IP    = "127.0.0.1"
VEHICLE_NAME = "Drone1"

DEPTH_CAMERA_NAME = "front_depth"
RGB_CAMERA_NAME   = "front_rgb"
IMAGE_WIDTH  = 160
IMAGE_HEIGHT = 120

# ─────────────────────────────────────────────
# FLIGHT PARAMETERS
# ─────────────────────────────────────────────
TAKEOFF_ALTITUDE = -3.0
CRUISE_SPEED     = 2.5
MAX_YAW_RATE     = 30.0
HOVER_THROTTLE   = 0.5

# ─────────────────────────────────────────────
# PERCEPTION PARAMETERS
# ─────────────────────────────────────────────
OBSTACLE_DISTANCE_THRESHOLD = 10.0
MORPH_KERNEL_SIZE            = 5
GRID_COLS                    = 3
GRID_ROWS                    = 3
OBSTACLE_CELL_THRESHOLD      = 0.15

# ─────────────────────────────────────────────
# NAVIGATION PARAMETERS
# ─────────────────────────────────────────────
APF_GOAL_GAIN         = 3.0
APF_GOAL_THRESHOLD    = 0.5
APF_REPULSION_GAIN    = 5.0
APF_REPULSION_RANGE   = 10.0
APF_STUCK_THRESHOLD   = 0.3
APF_STUCK_FRAMES      = 20
APF_ESCAPE_MAGNITUDE  = 2.5

# ─────────────────────────────────────────────
# CONTROL PARAMETERS
# ─────────────────────────────────────────────
CONTROL_LOOP_HZ = 8
CONTROL_DT      = 1.0 / 8

MAX_VX = 4.0
MAX_VY = 3.0
MAX_VZ = 1.0

VELOCITY_GAIN = 2.0

# ─────────────────────────────────────────────
# YOLO SETTINGS
# ─────────────────────────────────────────────
YOLO_ENABLED     = True
YOLO_MODEL       = "yolov8n.pt"
YOLO_CONFIDENCE  = 0.4
YOLO_RUN_EVERY_N = 3

YOLO_CLASS_BEHAVIORS = {
    "person":   "STOP",
    "dog":      "STOP",
    "cat":      "STOP",
    "bicycle":  "STOP",
    "car":      "AVOID",
    "truck":    "AVOID",
    "bus":      "AVOID",
}

YOLO_STOP_DISTANCE = 5.0
YOLO_SLOW_DISTANCE = 8.0

# ─────────────────────────────────────────────
# VISUALIZER SETTINGS
# ─────────────────────────────────────────────
VISUALIZER_ENABLED = True
VISUALIZER_SCALE   = 2
VIZ_WINDOW_NAME    = "Drone Nav — Debug View"
