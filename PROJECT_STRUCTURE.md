# 📊 Project Structure Guide

## Overview
This document describes the organized structure of the Autonomous Drone Vision Navigation system.

---

## Directory Organization

```
autonomous-drone-nav/
│
├── config/                      # Configuration files
│   ├── settings.py             # Main parameters (tunable settings)
│   └── airsim_settings.json    # AirSim simulator configuration
│
├── modules/                     # Core system modules
│   │
│   ├── perception/             # Computer Vision & Obstacle Detection
│   │   ├── obstacle_detector.py   # Generic obstacle detection
│   │   ├── yolo_detector.py       # YOLO-based detection
│   │   └── depth_estimator.py     # Depth estimation from images
│   │
│   ├── navigation/             # Path Planning & Motion Planning
│   │   └── potential_field.py     # Potential field algorithm
│   │
│   ├── control/                # Drone Control
│   │   └── drone_controller.py    # Motion command generation
│   │
│   ├── simulation/             # AirSim Simulator Interface
│   │   └── airsim_client.py       # Bridge to AirSim simulator
│   │
│   └── utils/                  # Utility Functions
│       └── visualizer.py          # Debug visualization
│
├── docs/                        # Documentation
│   └── README.md               # Project documentation
│
└── main.py                      # Entry point (autonomy loop)
```

---

## File Descriptions

### 🚀 Entry Point
- **main.py** - Main autonomous navigation loop (SENSE → THINK → ACT)

### ⚙️ Configuration (`config/`)
- **settings.py** - All tunable parameters (thresholds, FPS, dimensions, etc.)
- **airsim_settings.json** - AirSim simulator environment configuration

### 👁️ Perception Module (`modules/perception/`)
Handles visual sensing and obstacle detection:
- **obstacle_detector.py** - Generic obstacle detection using color/shape
- **yolo_detector.py** - Deep learning-based detection with YOLO
- **depth_estimator.py** - Depth/distance estimation from camera images

### 🗺️ Navigation Module (`modules/navigation/`)
Handles motion planning:
- **potential_field.py** - Potential field algorithm for path planning

### 🎮 Control Module (`modules/control/`)
Sends commands to the drone:
- **drone_controller.py** - Converts navigation commands to drone velocity/orientation

### 🖥️ Simulation Module (`modules/simulation/`)
Interfaces with AirSim simulator:
- **airsim_client.py** - AirSim API wrapper and connection management

### 🔧 Utils Module (`modules/utils/`)
Utility functions:
- **visualizer.py** - Debug visualization (displays camera feed, detections, etc.)

### 📚 Documentation (`docs/`)
- **README.md** - Full project documentation

---

## Autonomy Loop Architecture

```
┌─────────────────────────────────────┐
│       MAIN AUTONOMY LOOP (main.py)  │
│                                     │
│  SENSE → THINK → ACT → REPEAT      │
└─────────────────────────────────────┘
         ↓          ↓         ↓
    [Perception] [Navigation] [Control]
         ↓          ↓         ↓
    Vision API  Planning   Motor API
    (airsim_    (potential_ (drone_
     client)     field)      controller)
```

---

## How to Use

### Running the System
```bash
python main.py  # Start autonomous navigation
```

### Modifying Behavior
1. **Change parameters** → Edit `config/settings.py`
2. **Change obstacle detection** → Modify `modules/perception/obstacle_detector.py`
3. **Change navigation strategy** → Modify `modules/navigation/potential_field.py`
4. **Change simulator settings** → Edit `config/airsim_settings.json`

### Adding New Features
1. **New perception algorithm** → Add to `modules/perception/`
2. **New navigation method** → Add to `modules/navigation/`
3. **New visualization** → Add to `modules/utils/`

---

## Dependencies
- See `requirements.txt` in parent directory
- Key packages: airsim, opencv, numpy, scipy

---

## Next Steps
✅ Project structure organized!\
✅ Python packages configured with `__init__.py`\
⏭️ Update imports in your code to use the new structure\
⏭️ Test that `python main.py` still works with new paths
