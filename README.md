# 🚁 Autonomous Drone Vision Navigation System
### A Research-Level Tutorial — Mentored by a Senior Robotics Engineer

---

## Table of Contents
1. [Project Overview & Architecture](#architecture)
2. [Core Concepts Explained](#concepts)
3. [Setup & Installation](#setup)
4. [Module Walkthrough](#modules)
5. [Running the System](#running)
6. [Research-Level Extensions](#research)

---

## 1. Project Architecture

```
drone_nav/
├── config/
│   └── settings.py          # All tunable parameters in one place
├── perception/
│   └── obstacle_detector.py # Camera → obstacle map
├── navigation/
│   └── potential_field.py   # Obstacle map → motion vector
├── control/
│   └── drone_controller.py  # Motion vector → AirSim commands
├── simulation/
│   └── airsim_client.py     # AirSim connection wrapper
├── utils/
│   └── visualizer.py        # Real-time debug overlay
├── main.py                  # Main autonomous loop
└── README.md
```

**Data Flow:**
```
AirSim Camera → [Perception] → ObstacleMap
ObstacleMap   → [Navigation] → MotionVector
MotionVector  → [Control]    → AirSim Commands
                                     ↓
                              Drone moves in sim
```

---

## 2. Core Concepts Explained

### 🧠 Concept 1: Optical Flow vs Depth-Based Obstacle Detection
We use **depth image** from AirSim's stereo camera. A depth image encodes
distance-to-surface per pixel. Nearby obstacles appear bright (or dark,
depending on encoding). We threshold this image to extract a danger mask.

Why not RGB only? Because colour tells you "what" is there, not "how far".
Depth tells you "how far" directly — perfect for reactive avoidance.

### 🧠 Concept 2: Potential Field Navigation
Imagine the drone as a ball on a landscape:
- The **goal** is a valley (attractive force pulls you in)
- Each **obstacle** is a hill (repulsive force pushes you away)
- The drone rolls downhill — sum of all forces = motion vector

This is the APF (Artificial Potential Field) method, first formalized by
Khatib (1986) and still used in research today because it's fast and elegant.

**Known limitation:** APF can get stuck in local minima (a valley surrounded
by hills with no path out). We handle this with a random perturbation escape.

### 🧠 Concept 3: Control Loop Timing
Real-time robotics requires careful timing. Our loop:
1. Capture image (I/O bound, ~50ms in AirSim)
2. Process image (CPU bound, ~10-30ms with OpenCV)
3. Compute navigation (CPU, <1ms for APF)
4. Send command (network, ~5ms to AirSim)

We target 10 Hz (100ms/cycle). Faster = more responsive but more CPU.
Slower = laggy but cheaper. 10Hz is a good research starting point.

### 🧠 Concept 4: AirSim Coordinate System
AirSim uses NED (North-East-Down):
- X = forward (North)
- Y = right (East)
- Z = DOWN (negative = up!)
This is standard in aerospace. Don't confuse with ROS (which uses ENU).

---

## 3. Setup & Installation

### Step 1: Install AirSim
```bash
# Download from: https://github.com/microsoft/AirSim/releases
# Extract and run the Blocks environment executable
```

### Step 2: Configure AirSim
Place `settings.json` in `~/Documents/AirSim/`:
```json
{
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "Vehicles": {
    "Drone1": {
      "VehicleType": "SimpleFlight",
      "AutoCreate": true,
      "Cameras": {
        "front_depth": {
          "CaptureSettings": [{"ImageType": 2, "Width": 320, "Height": 240}],
          "X": 0.5, "Y": 0, "Z": -0.1,
          "Pitch": 0, "Roll": 0, "Yaw": 0
        },
        "front_rgb": {
          "CaptureSettings": [{"ImageType": 0, "Width": 320, "Height": 240}],
          "X": 0.5, "Y": 0, "Z": -0.1
        }
      }
    }
  }
}
```

### Step 3: Install Python Dependencies
```bash
pip install airsim opencv-python numpy msgpack-rpc-python
pip install torch torchvision   # optional, for deep learning module
pip install ultralytics          # optional, for YOLO
```

---

## 4. Module Walkthrough
See each file for inline explanations of every design decision.

---

## 5. Running the System

```bash
# 1. Launch AirSim (Blocks environment)
# 2. In terminal:
cd drone_nav
python main.py

# Debug mode (shows visualizer window):
python main.py --debug

# Dry run (no AirSim, uses mock data):
python main.py --mock
```

---

## 6. Research-Level Extensions

### 🔬 Extension A: SLAM (Simultaneous Localization and Mapping)
Current system is purely reactive — it has no memory of where it's been.
SLAM builds a map while tracking position. Use **ORB-SLAM3** or **RTAB-Map**.
AirSim provides ground-truth pose for evaluation.
```
Read: Mur-Artal et al., "ORB-SLAM3" (2021), IEEE T-RO
```

### 🔬 Extension B: Deep Reinforcement Learning
Replace the APF navigation with a neural network trained via RL.
The agent learns to fly by trial-and-error in simulation.
Use **PPO** (Proximal Policy Optimization) with AirSim as the environment.
```
Read: Loquercio et al., "Learning High-Speed Flight" (2021), Science Robotics
```

### 🔬 Extension C: Monocular Depth Estimation
No depth camera? Estimate depth from a single RGB image using:
- **MiDaS** (Intel) — pretrained, fast, runs on CPU
- **DPT** (Dense Prediction Transformer) — more accurate
This turns any cheap camera into a pseudo-depth sensor.
```
pip install timm
# See: https://github.com/isl-org/MiDaS
```

### 🔬 Extension D: YOLO Object Detection
Detect specific obstacle *types* (people, cars, trees) and apply
different avoidance behaviors per class. Use YOLOv8:
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model(frame)
```

### 🔬 Extension E: Path Planning (Global)
APF = local/reactive. For long-range navigation, combine with:
- **A*** or **RRT*** for global path planning
- Use the AirSim voxel map or a pre-built occupancy grid
This gives you the full autonomous navigation stack used in real UAVs.
