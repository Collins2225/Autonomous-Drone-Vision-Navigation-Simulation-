"""
main.py
=======
Main autonomous navigation loop — the entry point of the entire system.

MENTOR NOTE — The Autonomy Loop:
  All autonomous robots have the same fundamental loop:
  
      ┌─────────────────────────────────┐
      │         AUTONOMY LOOP           │
      │                                 │
      │  SENSE → THINK → ACT → repeat  │
      │                                 │
      │  (Perception) → (Navigation)   │
      │       → (Control) → loop       │
      └─────────────────────────────────┘
  
  This is the most important architectural concept in robotics.
  No matter how complex the system — from a Roomba to a Mars rover —
  this loop is always at the heart of it.
  
  Our loop:
    1. SENSE:  Capture camera images from AirSim
    2. THINK:  Detect obstacles (Perception) + plan motion (Navigation)
    3. ACT:    Send velocity command to drone (Control)
    4. VIZ:    Update debug display (optional)
    5. TIMING: Sleep to maintain target loop rate (10 Hz)

Usage:
    python main.py              # Live AirSim mode
    python main.py --mock       # Offline mock mode (no AirSim needed)
    python main.py --debug      # Enable visualizer window
    python main.py --mock --debug  # Both
"""

import time
import sys
import argparse
import signal
import numpy as np

# ── Module imports ──
from config.settings import CONTROL_LOOP_HZ, CONTROL_DT, VISUALIZER_ENABLED
from simulation.airsim_client import DroneSimClient
from perception.obstacle_detector import ObstacleDetector
from navigation.potential_field import PotentialFieldNavigator, MotionCommand
from control.drone_controller import DroneController
from utils.visualizer import DroneVisualizer


# ──────────────────────────────────────────────
# GRACEFUL SHUTDOWN
# ──────────────────────────────────────────────

_running = True

def _signal_handler(sig, frame):
    """Handle Ctrl+C gracefully -- don't crash, land safely."""
    global _running
    print("\n[Main] [SIGINT] Ctrl+C received -- initiating safe shutdown...")
    _running = False

signal.signal(signal.SIGINT, _signal_handler)


# ──────────────────────────────────────────────
# ARGUMENT PARSING
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Autonomous Drone Navigation")
    parser.add_argument("--mock",  action="store_true",
                        help="Run in mock mode (no AirSim required)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable real-time visualizer window")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="Max flight duration in seconds (default: 60)")
    return parser.parse_args()


# ──────────────────────────────────────────────
# SYSTEM INITIALIZATION
# ──────────────────────────────────────────────

def initialize_system(mock_mode: bool, debug_mode: bool):
    """
    Instantiate all system components.
    
    MENTOR NOTE — Dependency Injection:
      Notice that DroneController receives the sim_client as a parameter.
      This is dependency injection — it makes the system testable and modular.
      You can swap DroneSimClient with a MockClient without changing
      DroneController at all. This is why we have the mock_mode flag.
    """
    print("="*55)
    print("  [DRONE] Autonomous Drone Vision Navigation System")
    print("="*55)

    print(f"  Mode:  {'MOCK (offline)' if mock_mode else 'LIVE (AirSim)'}")
    print(f"  Debug: {'ON' if debug_mode else 'OFF'}")
    print(f"  Loop:  {CONTROL_LOOP_HZ} Hz\n")

    # Simulation client
    sim = DroneSimClient(mock_mode=mock_mode)

    # Perception module
    detector = ObstacleDetector()

    # Navigation module — goal = straight ahead (image center)
    navigator = PotentialFieldNavigator()
    navigator.set_goal(
        image_x_norm=0.0,   # centered = go straight
        image_y_norm=0.0,
        distance_m=50.0     # arbitrary distance goal
    )

    # Control module
    controller = DroneController(sim)

    # Visualizer (optional)
    visualizer = DroneVisualizer() if debug_mode else None

    return sim, detector, navigator, controller, visualizer


# ──────────────────────────────────────────────
# LOOP TIMING HELPER
# ──────────────────────────────────────────────

class LoopTimer:
    """
    Maintains consistent loop frequency.
    
    MENTOR NOTE — Loop Rate Control:
      In robotics, consistent timing is critical. If your perception takes
      longer than one tick, you fall behind. If you sleep too long, you
      miss obstacles.
      
      Pattern: record start time → do work → sleep the REMAINDER of the tick.
      This way, fast iterations sleep more; slow iterations sleep less.
      If an iteration exceeds the budget, we log a warning (overrun).
    """

    def __init__(self, target_hz: float):
        self.target_dt = 1.0 / target_hz
        self._iter_start: float = 0.0
        self._hz_samples = []

    def start(self):
        self._iter_start = time.time()

    def sleep(self) -> float:
        """Sleep remaining time to hit target rate. Returns actual Hz."""
        elapsed = time.time() - self._iter_start
        sleep_time = self.target_dt - elapsed

        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            print(f"[Timer] ⚠ Loop overrun by {-sleep_time*1000:.1f}ms")

        actual_dt = time.time() - self._iter_start
        hz = 1.0 / actual_dt if actual_dt > 0 else 0.0

        self._hz_samples.append(hz)
        if len(self._hz_samples) > 50:
            self._hz_samples.pop(0)

        return hz

    @property
    def mean_hz(self) -> float:
        if not self._hz_samples:
            return 0.0
        return sum(self._hz_samples) / len(self._hz_samples)


# ──────────────────────────────────────────────
# MAIN ENTRY POINT
# ──────────────────────────────────────────────

def main():
    global _running

    args = parse_args()

    # ── Initialize all modules ──
    sim, detector, navigator, controller, visualizer = initialize_system(
        mock_mode=args.mock,
        debug_mode=args.debug
    )

    # ── Takeoff ──
    sim.takeoff()
    print("\n[Main] [TAKEOFF] Autonomous navigation started. Press Ctrl+C to land.\n")

    # ── Loop timing ──
    timer = LoopTimer(target_hz=CONTROL_LOOP_HZ)
    start_time = time.time()
    iteration  = 0

    # ── MAIN AUTONOMY LOOP ──
    while _running:
        # Check flight duration limit
        if time.time() - start_time > args.duration:
            print(f"[Main] [TIME] Max flight duration ({args.duration}s) reached.")
            break

        timer.start()

        # ────────────────────────────────────────
        # STEP 1: SENSE — Capture camera images
        # ────────────────────────────────────────
        images = sim.get_images()
        rgb    = images["rgb"]
        depth  = images["depth"]

        # ────────────────────────────────────────
        # STEP 2a: PERCEIVE — Detect obstacles
        # ────────────────────────────────────────
        obs_map = detector.process(depth, rgb)

        # ────────────────────────────────────────
        # STEP 2b: THINK — Compute navigation
        # ────────────────────────────────────────
        # Get current speed from AirSim state
        state = sim.get_state()
        vel   = state["velocity"]
        current_speed = float(np.linalg.norm(vel[:2]))  # horizontal speed

        # Compute motion command from APF
        cmd = navigator.compute(obs_map, current_speed)

        # ────────────────────────────────────────
        # STEP 3: ACT — Send velocity command
        # ────────────────────────────────────────
        ok = controller.execute(cmd)

        if not ok:
            print("[Main] Emergency stop triggered — exiting loop.")
            break

        if cmd.goal_reached:
            print("[Main] [GOAL] Goal reached — exiting loop.")
            
            if navigator.goal_image[0] == 0.0 and navigator.goal_image[1] == 0.0:
                navigator.set_goal(0.0, 0.0, 50.0)
            time.sleep(3)    

        # ────────────────────────────────────────
        # STEP 4: VISUALIZE (if debug mode)
        # ────────────────────────────────────────
        actual_hz = timer.sleep()  # Sleep then measure actual Hz

        if visualizer:
            visualizer.render(rgb, depth, obs_map, cmd, actual_hz)

        # ────────────────────────────────────────
        # STEP 5: PERIODIC CONSOLE LOG
        # ────────────────────────────────────────
        iteration += 1
        if iteration % 20 == 0:  # Log every 2 seconds @ 10Hz
            closest_str = (
                f"{obs_map.closest.distance:.1f}m"
                if obs_map.closest else "clear"
            )
            escape_str = " [ESCAPING]" if cmd.is_escaping else ""
            print(
                f"[{iteration:4d}] "
                f"Hz:{actual_hz:4.1f} | "
                f"Vx:{cmd.vx:+.2f} Vy:{cmd.vy:+.2f} Vz:{cmd.vz:+.2f} | "
                f"Closest:{closest_str}"
                f"{escape_str}"
            )

    # ──────────────────────────────────────────────
    # SHUTDOWN
    # ──────────────────────────────────────────────
    print("\n[Main] Initiating shutdown sequence...")

    # Print telemetry summary
    summary = controller.get_telemetry_summary()
    if summary:
        print("\n── Flight Telemetry Summary ──")
        for k, v in summary.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.3f}")
            else:
                print(f"  {k}: {v}")

    print(f"  Mean loop rate: {timer.mean_hz:.1f} Hz")
    print(f"  Total iterations: {iteration}")

    # Close visualizer
    if visualizer:
        visualizer.close()

    # Land and disarm
    sim.land_and_disarm()
    print("\n[Main] [OK] Shutdown complete.")


if __name__ == "__main__":
    main()
