"""Physical play orchestrator: camera -> game-state mirror -> policy -> arm.

Runs on the host computer connected to the NexArm. It maintains a mirror of
the physical game inside the same hk_mahjong data structures used in
training, so the exported TorchScript policy sees exactly the observation
distribution it was trained on.

Loop per decision:
  1. vision.read_hand / read_last_discard update the GameMirror
  2. encoder.encode_observation + legal mask -> policy forward pass
  3. chosen action -> ArmController primitive (discard/draw/claim)
  4. non-arm facts (claims by humans, wall count) are confirmed by the
     operator on the keyboard - full multi-camera table tracking is a
     stretch goal (see docs/PROJECT_DESCRIPTION.md).

Usage:
    python -m robot.play_physical --model deploy/mahjong_policy.ts.pt
    python -m robot.play_physical --dry-run     # no arm, no camera; keyboard I/O
"""
import argparse

import numpy as np
import torch

from hk_mahjong.game import Game, GameConfig
from hk_mahjong.tiles import Tile
from hk_mahjong.actions import Action, ActionType, A_PASS
from rl.encoder import encode_observation, encode_mask
from .arm_controller import ArmController


class GameMirror:
    """Tracks the physical game inside a hk_mahjong Game object.

    The Game's wall is only used for tile counting; actual tile identities
    come from vision/operator input, so we overwrite our seat's hand from
    the camera every turn (drift-proof).
    """

    def __init__(self, our_seat=0, min_faan=0):
        # min_faan must match the value the deployed model was trained with
        self.our_seat = our_seat
        self.game = Game(GameConfig(seed=0, min_faan=min_faan), dealer=0)
        for p in self.game.players:  # clear the simulated deal
            p.hand.clear()

    def set_our_hand(self, kind_ids):
        me = self.game.players[self.our_seat]
        me.hand = sorted(Tile.from_kind_id(k) for k in kind_ids)

    def opponent_discarded(self, seat, kind_id):
        t = Tile.from_kind_id(kind_id)
        self.game.players[seat].discards.append(t)
        self.game.last_discard, self.game.last_discarder = t, seat
        self.game.phase = "claim"
        self.game.claim_queue = [self.our_seat]
        self.game.history.append((seat, t.kind_id))


def choose_action(model, mirror):
    game = mirror.game
    planes, hist = encode_observation(game, mirror.our_seat)
    mask = encode_mask(game)
    if not mask.any():
        return None
    pt = torch.from_numpy(planes).unsqueeze(0)
    ht = torch.from_numpy(hist).unsqueeze(0)
    mt = torch.from_numpy(mask).unsqueeze(0)
    logits, _ = model(pt, ht, mt)
    return int(logits.argmax(-1).item())


def hand_slot_of_kind(mirror, kind_id):
    """Physical rack slot (left-to-right sorted) holding this tile kind."""
    hand = mirror.game.players[mirror.our_seat].hand
    for slot, t in enumerate(hand):
        if t.kind_id == kind_id:
            return slot
    raise ValueError(f"kind {kind_id} not in tracked hand")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deploy/mahjong_policy.ts.pt")
    ap.add_argument("--arm", choices=["mirobot", "nexarm"], default="mirobot")
    ap.add_argument("--port", default=None, help="serial port override")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--auto", action="store_true",
                    help="fully autonomous: camera perceives, model decides, "
                         "arm acts - no keyboard input (see robot/autonomous.py)")
    args = ap.parse_args()

    model = torch.jit.load(args.model)
    model.eval()
    if args.arm == "mirobot":
        from .mirobot_arm import MirobotArm, DEFAULT_PORT
        arm = MirobotArm(port=args.port or DEFAULT_PORT,
                         dry_run=args.dry_run)
    else:
        arm = ArmController(dry_run=args.dry_run)
    arm.home()
    mirror = GameMirror(our_seat=0)

    camera = classifier = None
    if not args.dry_run:
        from .vision import Camera, make_classifier
        camera, classifier = Camera(args.camera), make_classifier()

    if args.auto:
        from .autonomous import autonomous_loop
        autonomous_loop(model, arm, camera, classifier, mirror,
                        choose_action, hand_slot_of_kind)
        return

    print("Physical play started. Commands:")
    print("  h 3,12,25,...   set our hand kinds (or auto from camera)")
    print("  d <seat> <kind> opponent at seat discarded tile kind")
    print("  t               our turn: draw happened, act")
    print("  q               quit")

    while True:
        cmd = input("> ").strip().split()
        if not cmd:
            continue
        if cmd[0] == "q":
            break
        elif cmd[0] == "h":
            if len(cmd) > 1:
                mirror.set_our_hand([int(k) for k in cmd[1].split(",")])
            else:
                from .vision import read_hand
                mirror.set_our_hand(read_hand(camera, classifier))
            print("hand:", mirror.game.players[0].hand)
        elif cmd[0] == "d":
            mirror.opponent_discarded(int(cmd[1]), int(cmd[2]))
            aid = choose_action(model, mirror)
            a = Action.from_id(aid) if aid is not None else None
            if a is None or a.type == ActionType.PASS:
                print("PASS")
            elif a.type == ActionType.WIN:
                print("*** MAHJONG! declare win ***")
            else:
                print(f"claim: {a.type.name}")
                arm.claim_tile()
            mirror.game.phase = "discard"
            mirror.game.turn = 0
        elif cmd[0] == "t":
            mirror.game.phase = "discard"
            mirror.game.turn = 0
            aid = choose_action(model, mirror)
            a = Action.from_id(aid)
            if a.type == ActionType.WIN:
                print("*** MAHJONG! declare win ***")
            elif a.type == ActionType.DISCARD:
                slot = hand_slot_of_kind(mirror, a.tile_kind)
                print(f"discard {Tile.from_kind_id(a.tile_kind)} "
                      f"from slot {slot}")
                arm.discard_tile(slot)
                me = mirror.game.players[0]
                me.discards.append(
                    next(t for t in me.hand if t.kind_id == a.tile_kind))
                me.hand.remove(me.discards[-1])
            else:
                print(f"action: {a.type.name}")
        else:
            print("unknown command")
    arm.home()


if __name__ == "__main__":
    main()
