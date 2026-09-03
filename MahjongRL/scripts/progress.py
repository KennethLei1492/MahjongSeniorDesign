"""Show training progress from checkpoints/train_log.csv - no Excel needed.

Usage (from the MahjongRL folder):
    py -3 scripts/progress.py
"""
import csv
import os
import sys

_DIR = sys.argv[1] if len(sys.argv) > 1 else "checkpoints"
LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   _DIR, "train_log.csv")
TARGET = 500_000
BLOCKS = " .:-=+*#%@"


def sparkline(values, width=60):
    """Compress a series into one text line, min..max scaled."""
    if len(values) > width:  # average into buckets
        per = len(values) / width
        values = [sum(values[int(i * per):int((i + 1) * per)])
                  / max(len(values[int(i * per):int((i + 1) * per)]), 1)
                  for i in range(width)]
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    return "".join(BLOCKS[int((v - lo) / span * (len(BLOCKS) - 1))]
                   for v in values)


def main():
    if not os.path.exists(LOG) or os.path.getsize(LOG) == 0:
        print("No training data yet - the log gets its first row a few "
              "minutes after training starts.")
        return
    with open(LOG, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("games")]
    if not rows:
        print("Training is running but hasn't finished its first batch yet.")
        return

    games = int(float(rows[-1]["games"]))
    win = [float(r["win_rate"]) for r in rows]
    spg = [float(r["sec_per_game"]) for r in rows]
    recent = rows[-20:]
    r_win = sum(float(r["win_rate"]) for r in recent) / len(recent)
    r_spg = sum(float(r["sec_per_game"]) for r in recent) / len(recent)
    pct = games / TARGET
    bar = "#" * int(pct * 40)

    print(f"  games trained : {games:,} / {TARGET:,}")
    print(f"  progress      : [{bar:<40}] {pct:.1%}")
    print(f"  win rate      : {r_win:.1%} of recent games end in a win")
    print(f"  speed         : {r_spg:.2f} s/game "
          f"(~{86400 / r_spg:,.0f} games/day at 24/7)")
    remaining = TARGET - games
    if remaining > 0 and r_spg > 0:
        days = remaining * r_spg / 86400
        print(f"  ETA to 500k   : ~{days:.1f} days at current speed")
    print()
    print(f"  win-rate trend  (start -> now, low..high {min(win):.2f}.."
          f"{max(win):.2f})")
    print(f"  [{sparkline(win)}]")
    print()
    print("  If the win-rate trend is drifting upward, the bot is learning.")


if __name__ == "__main__":
    main()
