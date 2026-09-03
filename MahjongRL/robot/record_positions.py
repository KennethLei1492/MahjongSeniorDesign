"""Teach named poses by physically moving the LEADER arm.

The advanced kit's leader arm mirrors the follower's kinematics with no
gearing torque, so a person can move it freely. This script reads the
leader's joint positions and saves them as a named pose the follower can
replay - the standard leader/follower teaching workflow.

Usage:
    python -m robot.record_positions
      -> interactive: type a pose name, physically pose the leader arm,
         press Enter to capture. Empty name saves and exits.
    python -m robot.record_positions --limits
      -> continuously prints leader joint ticks so joint limits can be found.
"""
import argparse
import json
import os

from . import config
from .servo_bus import FeetechBus


def read_pose(bus):
    pose = {}
    for joint, sid in config.JOINTS.items():
        tick = bus.read_position(sid)
        if tick is None:
            raise RuntimeError(f"no response from leader servo {sid} ({joint})")
        pose[joint] = tick
    return pose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=config.LEADER_PORT)
    ap.add_argument("--limits", action="store_true")
    args = ap.parse_args()

    bus = FeetechBus(args.port, config.BAUDRATE)
    for sid in config.JOINTS.values():
        bus.torque(sid, False)  # leader must be freely movable

    if args.limits:
        import time
        print("Move the leader arm; Ctrl+C to stop.")
        while True:
            print(read_pose(bus))
            time.sleep(0.5)

    poses = {}
    if os.path.exists(config.POSITIONS_FILE):
        with open(config.POSITIONS_FILE) as f:
            poses = json.load(f)
    print("Required poses: hover_wall grasp_wall hover_discard place_discard "
          "hover_meld place_meld hover_hand_0..13 grasp_hand_0..13")
    while True:
        name = input("pose name (empty = save & quit): ").strip()
        if not name:
            break
        input(f"  pose the leader arm for '{name}', then press Enter...")
        poses[name] = read_pose(bus)
        print(f"  captured {poses[name]}")
    with open(config.POSITIONS_FILE, "w") as f:
        json.dump(poses, f, indent=2)
    print(f"saved {len(poses)} poses to {config.POSITIONS_FILE}")


if __name__ == "__main__":
    main()
