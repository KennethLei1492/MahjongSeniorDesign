"""High-level NexArm control: taught-pose playback and mahjong primitives.

All motion is pose playback between positions taught with the leader arm
(record_positions.py), which sidesteps inverse kinematics entirely and is
exactly the workflow the NexArm advanced kit's leader/follower design is
built for. Primitives used by play_physical.py:

    draw_tile()            wall -> rightmost empty hand slot
    discard_tile(slot)     hand slot -> discard area
    claim_tile()           discard area -> meld area
    lay_meld(slots)        hand slots -> meld area
    sort_gap_close(slot)   shift hand tiles left after a discard
"""
import json
import os
import time

from . import config


class ArmController:
    def __init__(self, backend=None, dry_run=False):
        self.dry_run = dry_run
        self.poses = dict(HOME=config.HOME_POSE)
        if os.path.exists(config.POSITIONS_FILE):
            with open(config.POSITIONS_FILE) as f:
                self.poses.update(json.load(f))
        if dry_run:
            self.bus = None
        elif backend is not None:
            self.bus = backend
        else:
            self.bus = self._auto_backend()

    def _auto_backend(self):
        try:  # preferred: lerobot's own Feetech bus
            from lerobot.common.robot_devices.motors.feetech import (
                FeetechMotorsBus)  # noqa: F401  (API name VERIFY per version)
        except ImportError:
            pass
        from .servo_bus import FeetechBus
        return FeetechBus(config.FOLLOWER_PORT, config.BAUDRATE)

    # ---------- low-level ----------
    def goto(self, pose_name, duration_ms=config.DEFAULT_MOVE_MS):
        pose = self.poses[pose_name]
        if self.dry_run:
            print(f"  [arm] goto {pose_name}")
            return
        targets = {}
        for joint, tick in pose.items():
            lo, hi = config.JOINT_LIMITS[joint]
            targets[config.JOINTS[joint]] = max(lo, min(hi, tick))
        self.bus.move_all(targets, duration_ms)

    def gripper(self, open_, duration_ms=400):
        if self.dry_run:
            print(f"  [arm] gripper {'open' if open_ else 'close'}")
            return
        tick = config.GRIPPER_OPEN_TICK if open_ else config.GRIPPER_CLOSED_TICK
        self.bus.move(config.JOINTS["gripper"], tick, duration_ms)
        time.sleep(duration_ms / 1000.0 + 0.1)

    def home(self):
        self.goto("HOME", 1200)

    def _pick(self, hover, grasp):
        self.goto(hover)
        self.gripper(True)
        self.goto(grasp, 500)
        self.gripper(False)          # close on tile
        self.goto(hover, 500)

    def _place(self, hover, place):
        self.goto(hover)
        self.goto(place, 500)
        self.gripper(True)           # release
        self.goto(hover, 500)

    # ---------- mahjong primitives ----------
    def draw_tile(self, hand_slot):
        """Pick the next wall tile and place it into hand_slot."""
        self._pick("hover_wall", "grasp_wall")
        self._place(f"hover_hand_{hand_slot}", f"grasp_hand_{hand_slot}")
        self.home()

    def discard_tile(self, hand_slot):
        """Pick the tile at hand_slot and place it in the discard area."""
        self._pick(f"hover_hand_{hand_slot}", f"grasp_hand_{hand_slot}")
        self._place("hover_discard", "place_discard")
        self.home()

    def claim_tile(self):
        """Take the last discard from the table into our meld area."""
        self._pick("hover_discard", "place_discard")
        self._place("hover_meld", "place_meld")
        self.home()

    def lay_meld(self, hand_slots):
        """Expose meld tiles from the given hand slots into the meld area."""
        for slot in hand_slots:
            self._pick(f"hover_hand_{slot}", f"grasp_hand_{slot}")
            self._place("hover_meld", "place_meld")
        self.home()

    def sort_gap_close(self, empty_slot, occupied_through):
        """Shift tiles left one slot to close the gap left by a discard."""
        for s in range(empty_slot + 1, occupied_through + 1):
            self._pick(f"hover_hand_{s}", f"grasp_hand_{s}")
            self._place(f"hover_hand_{s-1}", f"grasp_hand_{s-1}")
        self.home()
