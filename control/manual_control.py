"""
manual_control.py
==================
Manually fly the drone using keyboard input via Python.
Uses pynput to detect keypresses and sends velocity commands to AirSim.

Controls:
  W/S       - Forward / Backward
  A/D       - Rotate Left / Right (Yaw)
  Q/E       - Strafe Left / Right
  R/F       - Climb / Descend
  SPACE     - Emergency hover (stop all movement)
  ESC       - Land and exit
"""

import airsim
import time
import threading
import numpy as np

try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    print("Install pynput first: pip install pynput")
    exit()

# ── Flight settings ──
SPEED     = 2.0   # m/s — increase for faster manual flight
YAW_RATE  = 30.0  # degrees/second rotation speed
VERT_SPEED = 1.0  # m/s climb/descend speed

# ── State ──
keys_pressed = set()
running = True

def on_press(key):
    try:
        keys_pressed.add(key.char.lower())
    except AttributeError:
        keys_pressed.add(key)

def on_release(key):
    try:
        keys_pressed.discard(key.char.lower())
    except AttributeError:
        keys_pressed.discard(key)
    if key == keyboard.Key.esc:
        global running
        running = False

def get_velocity():
    """Convert current keypresses into velocity commands."""
    vx, vy, vz, yaw = 0.0, 0.0, 0.0, 0.0

    if 'w' in keys_pressed:   vx += SPEED         # forward
    if 's' in keys_pressed:   vx -= SPEED         # backward
    if 'q' in keys_pressed:   vy -= SPEED         # strafe left
    if 'e' in keys_pressed:   vy += SPEED         # strafe right
    if 'r' in keys_pressed:   vz -= VERT_SPEED    # climb (NED: negative = up)
    if 'f' in keys_pressed:   vz += VERT_SPEED    # descend
    if 'a' in keys_pressed:   yaw -= YAW_RATE     # rotate left
    if 'd' in keys_pressed:   yaw += YAW_RATE     # rotate right

    # SPACE = emergency hover
    if keyboard.Key.space in keys_pressed:
        return 0.0, 0.0, 0.0, 0.0

    return vx, vy, vz, yaw

def main():
    global running

    # Connect to AirSim
    print("Connecting to AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    print("Connected and armed ✓")

    # Takeoff
    print("Taking off...")
    client.takeoffAsync().join()
    client.moveToZAsync(-3.0, velocity=2.0).join()
    print("Airborne! Use keyboard to fly. ESC to land.\n")

    # Print controls reminder
    print("W/S = Forward/Back  |  Q/E = Strafe Left/Right")
    print("A/D = Rotate        |  R/F = Climb/Descend")
    print("SPACE = Hover       |  ESC = Land & Exit\n")

    # Start keyboard listener in background thread
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # Main control loop
    try:
        while running:
            vx, vy, vz, yaw = get_velocity()

            # Send velocity command
            client.moveByVelocityAsync(
                vx, vy, vz,
                duration=0.3,
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw)
            )

            # Print telemetry
            state = client.getMultirotorState()
            pos   = state.kinematics_estimated.position
            vel   = state.kinematics_estimated.linear_velocity
            speed = np.sqrt(vel.x_val**2 + vel.y_val**2)

            print(
                f"\r  Pos: X={pos.x_val:+.1f} Y={pos.y_val:+.1f} Z={pos.z_val:+.1f}m  "
                f"| Speed: {speed:.1f}m/s  "
                f"| Cmd: Vx={vx:+.1f} Vy={vy:+.1f} Vz={vz:+.1f}  "
                f"| Keys: {[str(k) for k in keys_pressed]}   ",
                end=""
            )

            time.sleep(0.05)   # 20Hz manual control loop

    except KeyboardInterrupt:
        pass

    # Land safely
    print("\n\nLanding...")
    listener.stop()
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    print("Landed and disarmed ✓")

if __name__ == "__main__":
    main()