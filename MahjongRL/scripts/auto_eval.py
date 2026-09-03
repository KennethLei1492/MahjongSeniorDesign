"""Evaluate new checkpoints_big milestones vs the 500k champion, unattended.

Scans checkpoints_big/model_*.pt, and for every milestone that has no line
yet in checkpoints_big/vs_champion_results.txt, runs the standard 200-game
head-to-head (1 seat vs 3 champions) and appends the result. Idempotent -
safe to run on a schedule alongside training.
"""
import datetime
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import torch
torch.set_num_threads(2)

from scripts.head2head import head2head

BIG_DIR = "checkpoints_big"
CHAMPION = os.path.join("checkpoints", "model_500096_champion.pt")
RESULTS = os.path.join(BIG_DIR, "vs_champion_results.txt")
GAMES = 200


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    done = set()
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            done = {m.group(1) for line in f
                    if (m := re.match(r"(\d+) ", line))}
    milestones = sorted(
        (int(m.group(1)), name) for name in os.listdir(BIG_DIR)
        if (m := re.fullmatch(r"model_(\d+)\.pt", name)))
    for games, name in milestones:
        tag = str(games)
        if tag in done:
            continue
        path = os.path.join(BIG_DIR, name)
        r = head2head(path, CHAMPION, games=GAMES, quiet=True)
        stamp = datetime.date.today().isoformat()
        line = (f"{tag} (1 seat) vs 500k-champion table: A won {r['a_wins']}, "
                f"champion won {r['b_wins']}, draws {r['draws']} "
                f"-> {tag}'s share {r['a_share']:.1%} (25% = equal) "
                f"[big model 128ch/6blk/lstm256, {GAMES} games, {stamp}]")
        with open(RESULTS, "a") as f:
            f.write(line + "\n")
        print(line, flush=True)
    print("auto_eval done", flush=True)


if __name__ == "__main__":
    main()
