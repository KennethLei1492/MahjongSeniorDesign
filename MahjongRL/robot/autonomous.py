"""Fully autonomous play loop: camera -> model -> arm, no human input.

The system perceives the table with the overhead camera, tracks the game in
a GameMirror, lets the trained policy decide, and executes decisions with
the Mirobot. Nobody drives anything.

Perception model (regions calibrated in robot/table_calibration.json):
    hand        our face-up rack (re-read every turn - drift-proof)
    discard_p1  right opponent's discard zone   (counterclockwise order)
    discard_p2  opposite opponent's discard zone
    discard_p3  left opponent's discard zone
    discard     our own discard zone

Turn logic (Hong Kong rules, counterclockwise):
  * a NEW tile appearing in an opponent zone = that player discarded
    -> the policy gets a claim decision (win / pung / kong / chow / pass)
  * if we pass and the discarder was our LEFT neighbor (p3), the next turn
    is ours after a debounce window with no further table changes
  * if another human claims (their meld zone changes / a different zone
    discards next), the loop just keeps watching - state stays consistent
    because our hand is re-read from pixels every time we act
  * on our turn the arm draws from the wall, the camera re-reads our rack
    to learn what we drew, the policy discards (or wins)

A human can still press Ctrl+C; 'p' pauses. That's an off switch, not
control.

Usage:
    py -3 -m robot.play_physical --auto --model deploy/mahjong_policy.ts.pt
    py -3 -m robot.play_physical --auto --dry-run    # simulated camera
"""
import time

import numpy as np

from hk_mahjong.actions import Action, ActionType
from hk_mahjong.tiles import Tile

POLL_S = 0.6          # camera polling period
DEBOUNCE_FRAMES = 3   # frames a change must persist before we act on it
TURN_GAP_S = 2.0      # quiet time after p3's discard before we take our turn


class TableWatcher:
    """Polls the camera and turns pixel changes into game events."""

    def __init__(self, camera, classifier):
        self.cam = camera
        self.cls = classifier
        self.counts = {"discard_p1": 0, "discard_p2": 0, "discard_p3": 0}

    def snapshot_hand(self):
        from .vision import read_hand
        return read_hand(self.cam, self.cls)

    def poll_discards(self):
        """Return (zone_name, kind_id) if a new opponent discard appeared."""
        from .vision import detect_tiles
        frame = self.cam.capture()
        for zone in self.counts:
            tiles = detect_tiles(self.cam.roi(frame, zone), self.cls)
            if len(tiles) > self.counts[zone]:
                # debounce: require the same count on consecutive frames
                stable = 0
                kid = None
                while stable < DEBOUNCE_FRAMES:
                    time.sleep(POLL_S / 2)
                    f2 = self.cam.capture()
                    t2 = detect_tiles(self.cam.roi(f2, zone), self.cls)
                    if len(t2) == len(tiles):
                        stable += 1
                        kid = t2[-1][2]
                    else:
                        tiles = t2
                        stable = 0
                self.counts[zone] = len(tiles)
                return zone, kid
            self.counts[zone] = min(self.counts[zone], len(tiles))
        return None, None


ZONE_TO_SEAT = {"discard_p1": 1, "discard_p2": 2, "discard_p3": 3}


def autonomous_loop(model, arm, camera, classifier, mirror, choose_action,
                    hand_slot_of_kind):
    watcher = TableWatcher(camera, classifier)
    print("AUTONOMOUS MODE - watching the table. Ctrl+C to stop.")
    mirror.set_our_hand(watcher.snapshot_hand())
    print("hand:", mirror.game.players[0].hand)
    last_p3_discard_t = None

    while True:
        zone, kid = watcher.poll_discards()

        if zone is not None:                      # an opponent discarded
            seat = ZONE_TO_SEAT[zone]
            tile = Tile.from_kind_id(kid)
            print(f"[see] opponent {seat} discarded {tile}")
            mirror.opponent_discarded(seat, kid)
            aid = choose_action(model, mirror)
            a = Action.from_id(aid) if aid is not None else None
            if a is not None and a.type == ActionType.WIN:
                print("*** MAHJONG - we win on that discard! ***")
                arm.claim_tile()
                return
            if a is not None and a.type in (ActionType.PUNG, ActionType.KONG,
                                            ActionType.CHOW):
                print(f"[act] claiming: {a.type.name}")
                arm.claim_tile()                  # take the tile to meld area
                melds = _meld_slots(mirror, a, kid)
                arm.lay_meld(melds)               # expose our matching tiles
                _apply_claim(mirror, a, kid)
                _our_discard(model, arm, watcher, mirror, choose_action,
                             hand_slot_of_kind)
                last_p3_discard_t = None
            else:
                mirror.game.phase = "discard"     # we passed; keep watching
                last_p3_discard_t = (time.time() if seat == 3 else None)
            continue

        # no event: is it our turn? (left neighbor discarded, table quiet)
        if last_p3_discard_t and time.time() - last_p3_discard_t > TURN_GAP_S:
            last_p3_discard_t = None
            print("[turn] our draw")
            slot = len(mirror.game.players[0].hand)   # next free rack slot
            arm.draw_tile(min(slot, 13))
            time.sleep(0.5)
            mirror.set_our_hand(watcher.snapshot_hand())  # learn what we drew
            _our_discard(model, arm, watcher, mirror, choose_action,
                         hand_slot_of_kind)
        time.sleep(POLL_S)


def _our_discard(model, arm, watcher, mirror, choose_action,
                 hand_slot_of_kind):
    mirror.game.phase = "discard"
    mirror.game.turn = 0
    aid = choose_action(model, mirror)
    a = Action.from_id(aid)
    if a.type == ActionType.WIN:
        print("*** MAHJONG - self-draw win! ***")
        raise SystemExit(0)
    if a.type != ActionType.DISCARD:              # e.g. concealed kong: draw again
        print(f"[act] {a.type.name} (operator: verify, then game continues)")
        return
    slot = hand_slot_of_kind(mirror, a.tile_kind)
    tile = Tile.from_kind_id(a.tile_kind)
    print(f"[act] discarding {tile} from slot {slot}")
    arm.discard_tile(slot)
    hand_len = len(mirror.game.players[0].hand)
    arm.sort_gap_close(slot, hand_len - 2)        # close the rack gap
    me = mirror.game.players[0]
    victim = next(t for t in me.hand if t.kind_id == a.tile_kind)
    me.hand.remove(victim)
    me.discards.append(victim)


def _meld_slots(mirror, action, claimed_kid):
    """Rack slots of our tiles that join the claimed tile in the meld."""
    hand = mirror.game.players[0].hand
    if action.type == ActionType.CHOW:
        base = claimed_kid - action.chow_variant
        needed = [k for k in (base, base + 1, base + 2) if k != claimed_kid]
    else:
        n = 2 if action.type == ActionType.PUNG else 3
        needed = [claimed_kid] * n
    slots, used = [], set()
    for k in needed:
        for i, t in enumerate(hand):
            if t.kind_id == k and i not in used:
                slots.append(i)
                used.add(i)
                break
    return slots


def _apply_claim(mirror, action, claimed_kid):
    from hk_mahjong.game import Meld
    me = mirror.game.players[0]
    if action.type == ActionType.CHOW:
        base = claimed_kid - action.chow_variant
        kinds = [base, base + 1, base + 2]
        name = "chow"
    else:
        n = 3 if action.type == ActionType.PUNG else 4
        kinds = [claimed_kid] * n
        name = "pung" if action.type == ActionType.PUNG else "kong"
    tiles = [Tile.from_kind_id(k) for k in kinds]
    for k in kinds:
        if k != claimed_kid or kinds.count(claimed_kid) > 1:
            victim = next((t for t in me.hand if t.kind_id == k), None)
            if victim:
                me.hand.remove(victim)
    me.melds.append(Meld(name, tiles, claimed_from=1))
