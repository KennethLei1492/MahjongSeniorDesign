"""Record Mirobot table positions into robot/mirobot_positions.json.

Workflow: jog the arm to each reference point (with the Bluetooth teach
pendant, WLKATA Studio, or the keyboard commands below), then save it under
its name. Needed points: hand_slot_0, wall_pick, discard_drop, meld_drop -
plus hand_step (offset between rack slots) and safe_z (travel height),
which are entered as numbers.

Usage:  py -3 -m robot.mirobot_teach [--port COM3]
Commands at the prompt:
  x+5 / y-2 / z+10   jog by mm            p        print current pose
  save <name>        save current x,y,z   step <dx> <dy>   set hand_step
  safez <z>          set safe_z           q        save file and quit
"""
import argparse
import json
import os

from .mirobot_arm import POSITIONS_FILE, DEFAULT_PORT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=DEFAULT_PORT)
    args = ap.parse_args()

    from wlkata_mirobot import WlkataMirobot
    bot = WlkataMirobot(portname=args.port)
    bot.home()

    pos = {}
    if os.path.exists(POSITIONS_FILE):
        with open(POSITIONS_FILE) as f:
            pos = json.load(f)

    def pose():
        bot.get_status()
        c = bot.status.cartesian
        return [round(c.x, 1), round(c.y, 1), round(c.z, 1)]

    print(__doc__)
    while True:
        cur = pose()
        cmd = input(f"{cur} > ").strip().lower().split()
        if not cmd:
            continue
        if cmd[0] == "q":
            break
        elif cmd[0] == "p":
            print(cur)
        elif cmd[0] == "save" and len(cmd) == 2:
            pos[cmd[1]] = pose()
            print(f"  saved {cmd[1]} = {pos[cmd[1]]}")
        elif cmd[0] == "step" and len(cmd) == 3:
            pos["hand_step"] = [float(cmd[1]), float(cmd[2])]
        elif cmd[0] == "safez" and len(cmd) == 2:
            pos["safe_z"] = float(cmd[1])
        elif cmd[0][0] in "xyz" and len(cmd) == 1:
            axis, delta = cmd[0][0], float(cmd[0][1:])
            x, y, z = cur
            if axis == "x":
                x += delta
            elif axis == "y":
                y += delta
            else:
                z += delta
            bot.set_tool_pose(x, y, z)
        else:
            print("  unknown command")
    with open(POSITIONS_FILE, "w") as f:
        json.dump(pos, f, indent=2)
    print(f"saved {len(pos)} entries to {POSITIONS_FILE}")


if __name__ == "__main__":
    main()
