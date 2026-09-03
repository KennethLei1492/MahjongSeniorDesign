"""Hardware configuration for the NexArm follower arm.

IMPORTANT: values marked VERIFY must be checked against the NexArm manual /
LeRobot config that ships with the advanced kit before first run.
"""

# Serial connection (VERIFY: run `python -m serial.tools.list_ports`)
FOLLOWER_PORT = "COM4"        # Windows; "/dev/ttyACM0" on Linux/RPi
LEADER_PORT = "COM5"          # leader arm for teaching positions
BAUDRATE = 1_000_000          # Feetech STS default (VERIFY)

# Servo bus IDs base->gripper (VERIFY against NexArm docs)
JOINTS = {
    "shoulder_pan": 1,
    "shoulder_lift": 2,
    "elbow": 3,
    "wrist_flex": 4,
    "wrist_roll": 5,
    "gripper": 6,
}

# STS3215 position range: 0..4095 ticks over 360 degrees
TICKS_PER_TURN = 4096
CENTER_TICK = 2048

# Software joint limits in ticks (VERIFY: run robot/record_positions.py --limits)
JOINT_LIMITS = {
    "shoulder_pan": (256, 3840),
    "shoulder_lift": (768, 3328),
    "elbow": (768, 3328),
    "wrist_flex": (768, 3328),
    "wrist_roll": (0, 4095),
    "gripper": (1900, 2600),   # closed .. open (VERIFY by jogging slowly)
}

GRIPPER_OPEN_TICK = 2600
GRIPPER_CLOSED_TICK = 2000    # closing force on a mahjong tile (VERIFY)

DEFAULT_MOVE_MS = 800         # default duration of a point-to-point move

# Named poses (ticks per joint, taught with the leader arm and saved by
# robot/record_positions.py into robot/positions.json; these are fallbacks)
HOME_POSE = {"shoulder_pan": 2048, "shoulder_lift": 2048, "elbow": 2048,
             "wrist_flex": 2048, "wrist_roll": 2048, "gripper": 2600}

POSITIONS_FILE = "robot/positions.json"

# Table geometry used by play_physical.py (all taught, not computed):
# required named poses in positions.json:
#   hover_wall, grasp_wall            - where fresh tiles are drawn from
#   hover_hand_0..hover_hand_13       - above each rack slot of our hand
#   grasp_hand_0..grasp_hand_13      - grasp height at each rack slot
#   hover_discard, place_discard      - center of the discard area
#   hover_meld, place_meld            - where claimed melds are laid out
NUM_HAND_SLOTS = 14
