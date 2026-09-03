"""Hong Kong faan scoring.

Implements the common HK table (mahjong-rules.com): 3-faan minimum by default,
faan values compound, payout doubles per faan with a limit hand cap.
Payment scheme (configurable in GameConfig): discarder pays the full amount,
the other two pay half ("half-spread"); self-draw = everyone pays full.
"""
from .tiles import (Tile, TYPE_HONOR, TYPE_DOT, TYPE_CHAR, TYPE_BAM,
                    DRAGON_WHITE, DRAGON_RED)
from .hand import decompositions, is_thirteen_orphans, CHOW, PUNG, KONG, PAIR

LIMIT_FAAN = 13


def _meld_kinds(claimed_melds):
    """Claimed melds as (kind, base_kind_id); kongs count as pungs for shape."""
    out = []
    for m in claimed_melds:
        kind = CHOW if m.kind == "chow" else PUNG
        out.append((kind, m.tiles[0].kind_id))
    return out


def score_hand(concealed_counts, claimed_melds, win_tile, *,
               self_draw, seat_wind, round_wind, num_flowers,
               is_concealed, robbing_kong=False, last_tile=False):
    """Return (faan, list_of_scoring_elements). faan capped at LIMIT_FAAN."""
    elements = []

    if is_thirteen_orphans(concealed_counts, len(claimed_melds)):
        return LIMIT_FAAN, [("Thirteen Orphans", LIMIT_FAAN)]

    decomps = decompositions(concealed_counts, len(claimed_melds))
    if not decomps:
        return 0, []

    claimed = _meld_kinds(claimed_melds)
    best_faan, best_elems = -1, []

    for d in decomps:
        melds = claimed + [m for m in d if m[0] != PAIR]
        pair_kid = next(k for kind, k in d if kind == PAIR)
        f, e = _score_decomposition(
            melds, pair_kid, win_tile, self_draw=self_draw,
            seat_wind=seat_wind, round_wind=round_wind,
            num_flowers=num_flowers, is_concealed=is_concealed,
            robbing_kong=robbing_kong, last_tile=last_tile,
            has_kong=any(m.kind == "kong" for m in claimed_melds))
        if f > best_faan:
            best_faan, best_elems = f, e

    return min(best_faan, LIMIT_FAAN), best_elems


def _score_decomposition(melds, pair_kid, win_tile, *, self_draw, seat_wind,
                         round_wind, num_flowers, is_concealed, robbing_kong,
                         last_tile, has_kong):
    faan, elems = 0, []
    chows = [k for kind, k in melds if kind == CHOW]
    pungs = [k for kind, k in melds if kind == PUNG]
    all_kids = [k for _, k in melds] + [pair_kid]

    def add(name, f):
        nonlocal faan
        faan += f
        elems.append((name, f))

    # --- 1 faan ---
    if self_draw:
        add("Self Draw", 1)
    for k in pungs:
        if k >= 27:
            idx = k - 27 + 1
            if idx >= DRAGON_WHITE:
                add("Dragon Pung", 1)
            else:
                if idx == seat_wind:
                    add("Seat Wind Pung", 1)
                if idx == round_wind:
                    add("Round Wind Pung", 1)
    if len(chows) == 4 and pair_kid < 27:  # Common Hand (all chows)
        add("Common Hand", 1)
    if num_flowers == 0:
        add("No Flowers", 1)
    else:
        # own-number flower/season each worth 1 in most HK tables; simplified:
        add(f"Flowers x{num_flowers}", num_flowers)
    if robbing_kong:
        add("Robbing the Kong", 1)
    if last_tile:
        add("Last Tile", 1)
    if has_kong and self_draw:
        add("Win on Kong Replacement", 1)

    # --- 3 faan ---
    if len(pungs) == 4:
        add("All Pungs", 3)
    suits = {k // 9 for k in all_kids if k < 27}
    honors = any(k >= 27 for k in all_kids)
    if len(suits) == 1 and honors:
        add("Half Flush", 3)

    # --- 6+ faan ---
    if len(suits) == 1 and not honors:
        add("Full Flush", 6)
    dragon_pungs = sum(1 for k in pungs if k - 27 + 1 in (5, 6, 7))
    if dragon_pungs == 3:
        add("Great Dragons", LIMIT_FAAN)
    elif dragon_pungs == 2 and pair_kid >= 27 and pair_kid - 27 + 1 >= 5:
        add("Small Dragons", 5)
    wind_pungs = sum(1 for k in pungs if 27 <= k <= 30)
    if wind_pungs == 4:
        add("Great Winds", LIMIT_FAAN)
    elif wind_pungs == 3 and 27 <= pair_kid <= 30:
        add("Small Winds", 6)
    if all(k >= 27 for k in all_kids):
        add("All Honors", LIMIT_FAAN)
    if is_concealed and self_draw and len(chows) == 0:
        # all pungs, fully concealed, self-drawn
        if len(pungs) == 4:
            add("Concealed Pungs Self-Draw", 3)

    return faan, elems


def payout(faan, *, self_draw, base=1, cap_faan=LIMIT_FAAN):
    """Base payment value: base * 2^faan, capped. Returns per-loser payments.

    half-spread convention:
      discard win: discarder pays full value, others pay half.
      self-draw:   all three pay full value.
    """
    faan = min(faan, cap_faan)
    value = base * (2 ** faan)
    if self_draw:
        return {"all": value}
    return {"discarder": value, "others": value / 2}
