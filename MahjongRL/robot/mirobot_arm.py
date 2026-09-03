"""WLKATA Mirobot backend: coordinate-based pick & place for mahjong tiles.

The Mirobot Professional Kit is a 6-axis stepper arm driven by G-code over
USB serial, with a pneumatic suction cup - the ideal end effector for flat
tiles. Unlike the leader-arm/pose-playback NexArm design, the Mirobot is
addressed in Cartesian coordinates, so the table layout is described by a
handful of reference points in robot/mirobot_positions.json:

  {
    "hand_slot_0":  [x, y, z],   // first (leftmost) rack slot, tile-top height
    "hand_step":    [dx, dy],    // offset between adjacent rack slots
    "wall_pick":    [x, y, z],   // where fresh drawn tiles are picked
    "discard_drop": [x, y, z],   // center of our discard area
    "meld_drop":    [x, y, z],   // where claimed melds are laid out
    "safe_z": 60                 // travel height above the table
  }

Record these with robot/mirobot_teach.py (jog with the Bluetooth teach
pendant or keyboard, then save each point).

Preferred driver: pip install wlkata-mirobot-python
Fallback: raw G-code over pyserial (commands marked VERIFY against the
WLKATA G-code manual for your firmware version).
"""
import json
import os
import time

POSITIONS_FILE = "robot/mirobot_positions.json"
DEFAULT_PORT = "COM3"          # VERIFY: check Device Manager
NUM_HAND_SLOTS = 14


class MirobotArm:
    def __init__(self, port=DEFAULT_PORT, dry_run=False):
        self.dry_run = dry_run
        self.pos = {
            "hand_slot_0": [150.0, -130.0, 25.0],
            "hand_step": [0.0, 20.0],
            "wall_pick": [200.0, 0.0, 25.0],
            "discard_drop": [170.0, 40.0, 25.0],
            "meld_drop": [120.0, 120.0, 25.0],
            "safe_z": 60.0,
        }
        if os.path.exists(POSITIONS_FILE):
            with open(POSITIONS_FILE) as f:
                self.pos.update(json.load(f))
        self.bot = None
        if not dry_run:
            self._connect(port)

    def _connect(self, port):
        try:
            from wlkata_mirobot import WlkataMirobot
            self.bot = WlkataMirobot(portname=port)
            self.bot.home()          # stepper arm: must home after power-on
        except ImportError:
            import serial
            self._ser = serial.Serial(port, 115200, timeout=2)
            time.sleep(2)
            self._gcode("$H")        # homing (VERIFY)
            self.bot = "raw"

    # ---------- low-level ----------
    def _gcode(self, line):
        self._ser.write((line + "\r\n").encode())
        time.sleep(0.1)
        return self._ser.read_all()

    def _goto(self, x, y, z, speed=2000):
        if self.dry_run:
            print(f"  [mirobot] goto ({x:.1f}, {y:.1f}, {z:.1f})")
            return
        if self.bot == "raw":
            self._gcode(f"M20 G90 G00 X{x:.1f} Y{y:.1f} Z{z:.1f} F{speed}")
            time.sleep(1.2)          # VERIFY: poll status instead if needed
        else:
            self.bot.set_tool_pose(x, y, z)

    def _pump(self, on):
        if self.dry_run:
            print(f"  [mirobot] suction {'ON' if on else 'OFF'}")
            return
        if self.bot == "raw":
            self._gcode("M3S1000" if on else "M3S0")   # VERIFY pump command
        else:
            if on:
                self.bot.pump_suction()
            else:
                self.bot.pump_off()
        time.sleep(0.4)

    def home(self):
        if self.dry_run:
            print("  [mirobot] home")
            return
        if self.bot == "raw":
            self._gcode("$H")
        else:
            self.bot.home()

    # ---------- geometry ----------
    def _hand_slot(self, slot):
        x0, y0, z0 = self.pos["hand_slot_0"]
        dx, dy = self.pos["hand_step"]
        return x0 + dx * slot, y0 + dy * slot, z0

    def _pick_place(self, src, dst):
        """Suction-pick at src, travel at safe_z, release at dst."""
        sz = self.pos["safe_z"]
        for (x, y, z), grab in ((src, True), (dst, False)):
            self._goto(x, y, sz)
            self._goto(x, y, z, speed=800)
            self._pump(grab)
            self._goto(x, y, sz, speed=800)

    # ---------- mahjong primitives (same API as NexArm ArmController) ----------
    def draw_tile(self, hand_slot):
        self._pick_place(tuple(self.pos["wall_pick"]),
                         self._hand_slot(hand_slot))

    def discard_tile(self, hand_slot):
        self._pick_place(self._hand_slot(hand_slot),
                         tuple(self.pos["discard_drop"]))

    def claim_tile(self):
        self._pick_place(tuple(self.pos["discard_drop"]),
                         tuple(self.pos["meld_drop"]))

    def lay_meld(self, hand_slots):
        for slot in hand_slots:
            self._pick_place(self._hand_slot(slot),
                             tuple(self.pos["meld_drop"]))

    def sort_gap_close(self, empty_slot, occupied_through):
        for s in range(empty_slot + 1, occupied_through + 1):
            self._pick_place(self._hand_slot(s), self._hand_slot(s - 1))

    def gripper(self, open_):   # API compatibility with ArmController
        self._pump(not open_)
