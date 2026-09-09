"""Manual tile hand-off mode: a player feeds tiles into the arm's reach.

The arm's workspace cannot cover the wall or the other players' discards,
so in hand-off mode a person places the relevant tile (the robot's draw,
or a claimed discard) into a fixed HAND-OFF ZONE inside the arm's reach.
Vision watches that zone; once a tile is present and stable, the game
continues exactly as if the arm had fetched it itself. The trained policy
is unaffected - it only ever sees the GameMirror state, never the table.

Setup on top of the normal bring-up:
  1. Choose the zone: physically it is the same point as "wall_pick" in
     robot/mirobot_positions.json - teach it somewhere comfortable for both
     the arm and the human (mirobot_teach.py).
  2. Calibrate a "handoff" ROI over that spot in
     robot/table_calibration.json: {"handoff": [x, y, w, h], ...}
  3. Run:  python -m robot.play_physical --model deploy/challenger_win_400k.ts.pt --handoff

Fairness note: whoever places the robot's drawn tile sees it. Prefer a
neutral dealer, or slide tiles face-down (the suction cup does not care)
and let only the camera see the face after the arm racks it - in that
variant, point the "handoff" ROI at the rack-entry slot instead and rack
before reading.
"""
import time


def wait_for_handoff_tile(camera, classifier, stable_reads=4, poll_s=0.3,
                          timeout_s=120, roi="handoff"):
    """Block until one tile sits stably in the hand-off zone; return kind_id.

    Stability = the same single tile kind seen `stable_reads` polls in a row,
    so a hand mid-placement or a half-covered face is never accepted.
    Returns None on timeout (caller decides how to recover, e.g. keyboard).
    """
    from .vision import detect_tiles
    last_kid, streak = None, 0
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        frame = camera.capture()
        tiles = detect_tiles(camera.roi(frame, roi), classifier)
        if len(tiles) == 1:
            kid = tiles[0][2]
            streak = streak + 1 if kid == last_kid else 1
            last_kid = kid
            if streak >= stable_reads:
                return kid
        else:
            last_kid, streak = None, 0
        time.sleep(poll_s)
    return None


class HandoffSession:
    """Physical-rack bookkeeping for hand-off play.

    Tracks the rack's true left-to-right slot order. The pre-existing
    hand_slot_of_kind() assumes the physical rack mirrors the sorted hand;
    that breaks the moment the arm appends a drawn tile on the right, so in
    hand-off mode every arm move goes through this session instead.
    """

    def __init__(self, mirror, arm):
        self.mirror = mirror
        self.arm = arm
        self.rack = []            # kind_ids in physical slot order

    def set_rack(self, kind_ids):
        """Initial deal: rack read left-to-right (camera or keyboard)."""
        self.rack = list(kind_ids)
        self.mirror.set_our_hand(self.rack)

    def take_draw(self, kind_id):
        """A tile placed in the zone is our draw: rack it rightmost."""
        slot = len(self.rack)
        self.arm.draw_tile(slot)          # wall_pick (= hand-off zone) -> rack
        self.rack.append(kind_id)
        self.mirror.set_our_hand(self.rack)

    def discard(self, kind_id):
        """Discard this kind from its true physical slot, then compact."""
        slot = self.rack.index(kind_id)
        self.arm.discard_tile(slot)
        self.rack.pop(slot)
        if slot < len(self.rack):
            self.arm.sort_gap_close(slot, len(self.rack))
        self.mirror.set_our_hand(self.rack)
        me = self.mirror.game.players[self.mirror.our_seat]
        me.discards.append(next(t for t in me.hand if t.kind_id == kind_id))

    def claim(self):
        """Claimed discard was placed in the zone: move it to the meld area."""
        if hasattr(self.arm, "claim_handoff_tile"):
            self.arm.claim_handoff_tile()
        else:
            # legacy pose-playback arm has no zone pose; nearest primitive
            print("  (arm has no hand-off claim pose - using claim_tile)")
            self.arm.claim_tile()
