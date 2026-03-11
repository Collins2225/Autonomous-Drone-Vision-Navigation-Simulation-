# 📁 Final Project Structure (Flat Organization)

## Documents Folder Layout

```
C:\Users\Collins\Documents\
│
├── AirSim\                          ← AirSim simulator settings
│   └── settings.json                ← Simulator configuration
│
├── Autonomous Drone Vision Navigation(Simulation)\    ← Main project folder
│   │
│   ├── main.py                      ← Entry point — run this to fly drone
│   ├── README.md                    ← Project documentation
│   ├── PROJECT_STRUCTURE.md         ← Structure guide
│   │
│   ├── config\                      ← Configuration module
│   │   ├── __init__.py
│   │   ├── settings.py              ← All tunable parameters (MOST IMPORTANT FILE!)
│   │   └── airsim_settings.json     ← Backup of AirSim config
│   │
│   ├── perception\                  ← Computer Vision & Obstacle Detection
│   │   ├── __init__.py
│   │   ├── obstacle_detector.py     ← Depth image → obstacle map (MAIN)
│   │   ├── depth_estimator.py       ← Optional: MiDaS mono depth estimation
│   │   └── yolo_detector.py         ← Optional: YOLO semantic detection
│   │
│   ├── navigation\                  ← Path Planning & Motion Planning
│   │   ├── __init__.py
│   │   └── potential_field.py       ← Artificial Potential Field algorithm
│   │
│   ├── control\                     ← Drone Control & Safety
│   │   ├── __init__.py
│   │   └── drone_controller.py      ← Velocity commands + safety checks
│   │
│   ├── simulation\                  ← AirSim Simulator Interface
│   │   ├── __init__.py
│   │   └── airsim_client.py         ← Connection wrapper (Facade pattern)
│   │
│   ├── utils\                       ← Utilities & Visualization
│   │   ├── __init__.py
│   │   └── visualizer.py            ← Real-time debug HUD
│   │
│   └── venv\                        ← Python virtual environment (auto-generated)
│       └── (auto-generated files)
│
└── AirSim Blocks Simulator\         ← Where Blocks.exe is installed
    └── WindowsNoEditor\
        ├── Blocks.exe               ← Launch this to start simulator
        └── (other sim files)
```

---

## Key Files Explained

### 🚀 Entry Point
- **main.py** - The autonomy loop (SENSE → THINK → ACT)
  - Read this first to understand the system flow
  - Run with `python main.py`

### ⚙️ Configuration (config/)
- **settings.py** - Central location for ALL parameters
  - Obstacle detection thresholds
  - Navigation gains
  - Control limits
  - FPS and timing
  - Image dimensions
  
### 👁️ Perception (perception/)
- **obstacle_detector.py** - Main perception pipeline
  - Converts depth images to obstacle map
  - Uses morphological operations
  - Detects obstacles in 3×3 grid
  
- **depth_estimator.py** - (Optional research extension)
  - MiDaS mono depth from RGB camera
  
- **yolo_detector.py** - (Optional research extension)
  - YOLOv8 semantic object detection
  - Enables class-aware navigation

### 🗺️ Navigation (navigation/)
- **potential_field.py** - Motion planning algorithm
  - Artificial Potential Fields (APF)
  - Attractive force to goal
  - Repulsive force from obstacles
  - Local minima escape

### 🎮 Control (control/)
- **drone_controller.py** - High-level drone commands
  - Converts motion vectors to velocity commands
  - Safety checks and velocity limiting
  - Altitude hold
  - Telemetry logging

### 🖥️ Simulation (simulation/)
- **airsim_client.py** - AirSim API wrapper
  - Clean facade over raw AirSim API
  - Handles connection/disconnection
  - Image capture
  - Velocity command sending
  - Mock mode for offline testing

### 🔧 Utils (utils/)
- **visualizer.py** - Real-time debug display
  - Shows camera feed
  - Displays detected obstacles
  - Shows motion vectors
  - Interactive debugging

---

## How to Use

### Run the Drone
```bash
python main.py              # Live mode with AirSim
python main.py --mock       # Offline mode (no AirSim needed)
python main.py --debug      # Show visualizer window
```

### Modify Behavior
1. **Change parameters** → Edit `config/settings.py`
2. **Change detection** → Edit `perception/obstacle_detector.py`
3. **Change navigation** → Edit `navigation/potential_field.py`
4. **Change simulator settings** → Edit `config/airsim_settings.json`

### Add Features
- New perception → Add to `perception/`
- New navigation → Add to `navigation/`
- New visualization → Add to `utils/`

---

## Import Pattern

All modules use relative imports:
```python
# ✅ CORRECT (flat structure)
from config.settings import CONTROL_LOOP_HZ
from perception.obstacle_detector import ObstacleMap
from navigation.potential_field import PotentialFieldNavigator
from control.drone_controller import DroneController
from utils.visualizer import DroneVisualizer
```

---

## Next Steps

✅ Project organized in flat structure\
✅ All imports fixed\
✅ Python packages configured\
⏭️ Run `python main.py --debug` to test\
⏭️ Adjust `config/settings.py` for your environment
