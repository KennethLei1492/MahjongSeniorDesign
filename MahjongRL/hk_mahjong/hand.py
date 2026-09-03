"""Winning-hand detection and decomposition.

Standard shape: 4 melds (chow/pung/kong, concealed or claimed) + 1 pair.
Special shape: Thirteen Orphans. Decomposition uses backtracking over the
34-kind count vector (same idea as AlphaJong's getTriplesAndPairs/getSequences,
src/utils.js, but exhaustive since HK hands are checked only at win time).
"""
from .tiles import Tile, TYPE_HONOR, NUM_KINDS

# Meld kinds used in decompositions
CHOW, PUNG, KONG, PAIR = "chow", "pung", "kong", "pair"


def _decompose(counts, melds_needed, pair_used, acc, out):
    """Backtrack over counts; acc collects (kind, base_kind_id) melds."""
    if melds_needed == 0 and pair_used:
        if all(c == 0 for c in counts):
            out.append(list(acc))
        return
    # first non-empty kind
    try:
        k = next(i for i, c in enumerate(counts) if c > 0)
    except StopIteration:
        return
    # pair
    if not pair_used and counts[k] >= 2:
        counts[k] -= 2
        acc.append((PAIR, k))
        _decompose(counts, melds_needed, True, acc, out)
        acc.pop()
        counts[k] += 2
    if melds_needed > 0:
        # pung
        if counts[k] >= 3:
            counts[k] -= 3
            acc.append((PUNG, k))
            _decompose(counts, melds_needed - 1, pair_used, acc, out)
            acc.pop()
            counts[k] += 3
        # chow (suited only, not crossing suit boundary)
        if k < 27 and k % 9 <= 6 and counts[k + 1] > 0 and counts[k + 2] > 0:
            for i in (k, k + 1, k + 2):
                counts[i] -= 1
            acc.append((CHOW, k))
            _decompose(counts, melds_needed - 1, pair_used, acc, out)
            acc.pop()
            for i in (k, k + 1, k + 2):
                counts[i] += 1


def decompositions(concealed_counts, num_claimed_melds):
    """All ways to split concealed tiles into (4 - claimed) melds + 1 pair."""
    out = []
    _decompose(list(concealed_counts), 4 - num_claimed_melds, False, [], out)
    return out


THIRTEEN_ORPHAN_KIDS = [0, 8, 9, 17, 18, 26] + [27 + i for i in range(7)]


def is_thirteen_orphans(concealed_counts, num_claimed_melds):
    if num_claimed_melds != 0:
        return False
    if sum(concealed_counts) != 14:
        return False
    for kid in range(NUM_KINDS):
        c = concealed_counts[kid]
        if kid in THIRTEEN_ORPHAN_KIDS:
            if c not in (1, 2):
                return False
        elif c != 0:
            return False
    return sum(concealed_counts[k] for k in THIRTEEN_ORPHAN_KIDS) == 14


def is_winning(concealed_counts, num_claimed_melds):
    """True if concealed tiles + claimed melds form a legal 4-meld + pair hand."""
    if is_thirteen_orphans(concealed_counts, num_claimed_melds):
        return True
    return len(decompositions(concealed_counts, num_claimed_melds)) > 0
