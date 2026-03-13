"""
main.py
========
Main autonomous navigation loop with YOLO + depth perception.

Two-layer perception architecture:
  Layer 1 (every frame):  Depth-based obstacle detection  — fast, metric
  Layer 2 (every N frames): YOLO semantic detection        — smart, class-aware

Decision priority:
  1. YOLO says STOP (person nearby)  → emergency hover
  2. YOLO says SLOW                  → reduce cruise speed
  3. Depth detects obstacle          → APF avoidance
  4. Path clear                      → cruise forward

Usage:
  python main.py                   # live AirSim
  python main.py --mock            # offline mock mode
  python main.py --debug           # with visualizer window
  python main.py --no-yolo         # disable YOLO (depth only)
  python main.py --duration 300    # run for 300 seconds
"""

import time
import sys
import argparse
import signal
import numpy as np

from config.settings import (
    CONTROL_LOOP_HZ, CONTROL_DT,
    YOLO_ENABLED, CRUISE_SPEED
)
from simulation.airsim_client  import DroneSimClient
from perception.obstacle_detector import ObstacleDetector
from navigation.potential_field   import PotentialFieldNavigator, MotionCommand
from control.drone_controller     import DroneController
from utils.visualizer             import DroneVisualizer

# ──────────────────────────────────────────────
# GRACEFUL SHUTDOWN
# ──────────────────────────────────────────────

_running = True

def _signal_handler(sig, frame):
    global _running
    print("\n[Main] Ctrl+C — shutting down safely...")
    _running = False

signal.signal(signal.SIGINT, _signal_handler)

# ──────────────────────────────────────────────
# ARGUMENTS
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Autonomous Drone Navigation")
    parser.add_argument("--mock",     action="store_true", help="Mock mode, no AirSim needed")
    parser.add_argument("--debug",    action="store_true", help="Enable visualizer window")
    parser.add_argument("--no-yolo",  action="store_true", help="Disable YOLO, depth only")
    parser.add_argument("--duration", type=float, default=300.0, help="Max flight duration seconds")
    return parser.parse_args()

# ──────────────────────────────────────────────
# LOOP TIMER
# ──────────────────────────────────────────────

class LoopTimer:
    def __init__(self, target_hz):
        self.target_dt    = 1.0 / target_hz
        self._start       = 0.0
        self._hz_samples  = []

    def start(self):
        self._start = time.time()

    def sleep(self) -> float:
        elapsed    = time.time() - self._start
        sleep_time = self.target_dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            overrun = -sleep_time * 1000
            if overrun > 50:
                print(f"[Timer] ⚠ Loop overrun by {overrun:.1f}ms")

        actual_dt = time.time() - self._start
        hz = 1.0 / actual_dt if actual_dt > 0 else 0.0
        self._hz_samples.append(hz)
        if len(self._hz_samples) > 50:
            self._hz_samples.pop(0)
        return hz

    @property
    def mean_hz(self):
        return sum(self._hz_samples) / len(self._hz_samples) if self._hz_samples else 0.0

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    global _running
    args = parse_args()

    use_yolo = YOLO_ENABLED and not args.no_yolo and not args.mock

    print("=" * 55)
    print("  🚁 Autonomous Drone Vision Navigation System")
    print("=" * 55)
    print(f"  Mode:    {'MOCK' if args.mock else 'LIVE (AirSim)'}")
    print(f"  YOLO:    {'ON' if use_yolo else 'OFF'}")
    print(f"  Debug:   {'ON' if args.debug else 'OFF'}")
    print(f"  Loop:    {CONTROL_LOOP_HZ} Hz")
    print(f"  Duration:{args.duration}s")
    print()

    # ── Initialize modules ──
    sim        = DroneSimClient(mock_mode=args.mock)
    detector   = ObstacleDetector()
    navigator  = PotentialFieldNavigator()
    controller = DroneController(sim)
    visualizer = DroneVisualizer() if args.debug else None

    # ── Load YOLO if enabled ──
    yolo = None
    if use_yolo:
        try:
            from perception.yolo_detector import YOLOObstacleDetector
            yolo = YOLOObstacleDetector()
            print("[Main] YOLO semantic detection: ACTIVE ✓\n")
        except Exception as e:
            print(f"[Main] YOLO failed to load: {e}")
            print("[Main] Continuing with depth-only perception.\n")
            yolo = None

    # ── Set initial goal — fly straight ahead ──
    navigator.set_goal(0.0, 0.0, 50.0)

    # ── Takeoff ──
    sim.takeoff()
    print("\n[Main]  Autonomous navigation started.")
    print("[Main] Press Ctrl+C to land safely.\n")

    timer      = LoopTimer(target_hz=CONTROL_LOOP_HZ)
    start_time = time.time()
    iteration  = 0
    yolo_result = None

    # ══════════════════════════════════════════
    # MAIN AUTONOMY LOOP
    # ══════════════════════════════════════════
    while _running:

        # Check duration limit
        if time.time() - start_time > args.duration:
            print(f"[Main] ⏱ Duration limit reached ({args.duration}s)")
            break

        timer.start()

        # ────────────────────────────────
        # STEP 1: SENSE
        # ────────────────────────────────
        images = sim.get_images()
        rgb    = images["rgb"]
        depth  = images["depth"]

        # ────────────────────────────────
        # STEP 2a: DEPTH PERCEPTION
        # (runs every frame — fast)
        # ────────────────────────────────
        obs_map = detector.process(depth, rgb)

        # ────────────────────────────────
        # STEP 2b: YOLO PERCEPTION
        # (runs every N frames — smart)
        # ────────────────────────────────
        if yolo is not None:
            yolo_result = yolo.process(rgb, depth, iteration)

        # ────────────────────────────────
        # STEP 3: THINK — Navigation
        # ────────────────────────────────
        state         = sim.get_state()
        current_speed = float(np.linalg.norm(state["velocity"][:2]))

        # Check YOLO decision first — overrides APF if critical
        if yolo_result is not None and yolo_result.should_stop:
            # Safety critical object detected — emergency hover
            cmd = MotionCommand(vx=0.0, vy=0.0, vz=0.0)
            det = yolo_result.closest_critical
            if det and iteration % 10 == 0:
                print(f"[YOLO]  STOP — {det.class_name} at {det.distance:.1f}m!")

        elif yolo_result is not None and yolo_result.should_slow:
            # Obstacle detected by YOLO — slow APF navigation
            cmd = navigator.compute(obs_map, current_speed)
            cmd.vx = min(cmd.vx, CRUISE_SPEED * 0.5)   # half speed

        else:
            # Normal APF navigation
            cmd = navigator.compute(obs_map, current_speed)

        # Reset goal if reached — keep flying
        if cmd.goal_reached:
            print("[Main]  Waypoint reached — continuing patrol...")
            navigator.set_goal(0.0, 0.0, 50.0)

        # ────────────────────────────────
        # STEP 4: ACT — Control
        # ────────────────────────────────
        ok = controller.execute(cmd)
        if not ok:
            print("[Main] Emergency stop triggered.")
            break

        # ────────────────────────────────
        # STEP 5: VISUALIZE
        # ────────────────────────────────
        actual_hz = timer.sleep()

        if visualizer:
            # Draw YOLO boxes on RGB if available
            display_rgb = rgb.copy()
            if yolo is not None and yolo_result is not None:
                display_rgb = yolo.draw(display_rgb, yolo_result)
            visualizer.render(display_rgb, depth, obs_map, cmd, actual_hz)

        # ────────────────────────────────
        # STEP 6: TELEMETRY LOG
        # ────────────────────────────────
        iteration += 1
        if iteration % (CONTROL_LOOP_HZ * 2) == 0:
            closest_str = (
                f"{obs_map.closest.distance:.1f}m"
                if obs_map.closest else "clear"
            )

            yolo_str = ""
            if yolo_result and yolo_result.detections:
                top = yolo_result.detections[0]
                yolo_str = f" | YOLO: {top.class_name}({top.distance:.1f}m)"

            escape_str = " [ESCAPING]" if cmd.is_escaping else ""

            print(
                f"[{iteration:4d}] "
                f"Hz:{actual_hz:4.1f} | "
                f"Vx:{cmd.vx:+.2f} "
                f"Vy:{cmd.vy:+.2f} "
                f"Vz:{cmd.vz:+.2f} | "
                f"Depth:{closest_str}"
                f"{yolo_str}"
                f"{escape_str}"
            )

    # ──────────────────────────────────────────────
    # SHUTDOWN
    # ──────────────────────────────────────────────
    print("\n[Main] Shutting down...")

    summary = controller.get_telemetry_summary()
    if summary:
        print("\n── Flight Telemetry Summary ──")
        for k, v in summary.items():
            print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"  Mean loop rate:   {timer.mean_hz:.1f} Hz")
    print(f"  Total iterations: {iteration}")

    if visualizer:
        visualizer.close()

    sim.land_and_disarm()
    print("\n[Main]  Flight complete.")


if __name__ == "__main__":
    main()