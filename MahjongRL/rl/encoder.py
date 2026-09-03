"""Observation encoding: Game -> tensors for the Res2Net+LSTM network.

Spatial input: a (C, 4, 34) float tensor. The 4x34 grid is the natural
"image" of a mahjong state: 34 tile kinds wide, 4 copies tall (row r of a
plane is 1 if at least r+1 copies of that kind are present - a thermometer
encoding, like AlphaJong's availableTiles bookkeeping generalized to planes).

Planes (from the acting player's perspective, opponents in relative seat
order so the network is seat-equivariant):
   0      own concealed hand
   1-3    own melds / flowers indicator / own discards
   4-6    opponent +1: melds, discards, recent-discard
   7-9    opponent +2: melds, discards, recent-discard
  10-12   opponent +3: melds, discards, recent-discard
  13      all visible tiles (for wall counting, cf. AlphaJong visibleTiles)
  14      claimable tile on the table (claim phase only)
  15      scalar broadcast: wall remaining / 84
  16      scalar broadcast: seat wind one-hot over first 4 columns
  17      scalar broadcast: phase (1 = claim phase)

History input for the LSTM: sequence of (seat_onehot(4) + action_onehot(41))
vectors for the last HISTORY_LEN events.
"""
import numpy as np
from hk_mahjong.tiles import NUM_KINDS
from hk_mahjong.actions import NUM_ACTIONS

NUM_PLANES = 18
HISTORY_LEN = 32
EVENT_DIM = 4 + NUM_ACTIONS


def _fill_thermo(plane, counts):
    for k, c in enumerate(counts):
        for r in range(min(c, 4)):
            plane[r, k] = 1.0


def encode_observation(game, seat):
    """Returns (planes: float32 (18,4,34), history: float32 (32,45))."""
    x = np.zeros((NUM_PLANES, 4, NUM_KINDS), dtype=np.float32)
    me = game.players[seat]

    _fill_thermo(x[0], me.counts())

    def meld_counts(p):
        c = [0] * NUM_KINDS
        for m in p.melds:
            for t in m.tiles:
                c[t.kind_id] += 1
        return c

    def discard_counts(p):
        c = [0] * NUM_KINDS
        for t in p.discards:
            c[t.kind_id] += 1
        return c

    _fill_thermo(x[1], meld_counts(me))
    x[2, 0, :len(me.flowers)] = 1.0  # flower count as bar
    _fill_thermo(x[3], discard_counts(me))

    visible = meld_counts(me)
    for k, c in enumerate(discard_counts(me)):
        visible[k] += c

    for rel in (1, 2, 3):
        opp = game.players[(seat + rel) % 4]
        base = 4 + (rel - 1) * 3
        mc, dc = meld_counts(opp), discard_counts(opp)
        _fill_thermo(x[base], mc)
        _fill_thermo(x[base + 1], dc)
        if opp.discards:
            x[base + 2, 0, opp.discards[-1].kind_id] = 1.0
        for k in range(NUM_KINDS):
            visible[k] += mc[k] + dc[k]

    _fill_thermo(x[13], visible)
    if game.phase == "claim" and game.last_discard is not None:
        x[14, 0, game.last_discard.kind_id] = 1.0
    x[15] = game.wall.remaining() / 84.0
    x[16, 0, game.seat_wind(seat) - 1] = 1.0
    x[17] = 1.0 if game.phase == "claim" else 0.0

    # ---- action history for the LSTM ----
    h = np.zeros((HISTORY_LEN, EVENT_DIM), dtype=np.float32)
    events = game.history[-HISTORY_LEN:]
    for i, (s, aid) in enumerate(events):
        h[i, (s - seat) % 4] = 1.0          # relative seat
        h[i, 4 + aid] = 1.0
    return x, h


def encode_mask(game):
    return np.asarray(game.legal_actions(), dtype=bool)
