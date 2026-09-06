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

# Every variant is benchmarked against the SAME baseline (the original 500k
# champion) so curves stay comparable across models.
VARIANT_DIRS = [("checkpoints_big", "big model 128ch/6blk/lstm256 win-reward"),
                ("checkpoints_score", "big model 128ch/6blk/lstm256 score-reward")]
CHAMPION = os.path.join("checkpoints", "model_500096_champion.pt")
GAMES = 200


def eval_dir(ckpt_dir, label):
    results = os.path.join(ckpt_dir, "vs_champion_results.txt")
    done = set()
    if os.path.exists(results):
        with open(results) as f:
            done = {m.group(1) for line in f
                    if (m := re.match(r"(\d+) ", line))}
    milestones = sorted(
        (int(m.group(1)), name) for name in os.listdir(ckpt_dir)
        if (m := re.fullmatch(r"model_(\d+)\.pt", name)))
    for games, name in milestones:
        tag = str(games)
        if tag in done:
            continue
        path = os.path.join(ckpt_dir, name)
        r = head2head(path, CHAMPION, games=GAMES, quiet=True)
        stamp = datetime.date.today().isoformat()
        line = (f"{tag} (1 seat) vs 500k-champion table: A won {r['a_wins']}, "
                f"champion won {r['b_wins']}, draws {r['draws']} "
                f"-> {tag}'s share {r['a_share']:.1%} (25% = equal) "
                f"[{label}, {GAMES} games, {stamp}]")
        with open(results, "a") as f:
            f.write(line + "\n")
        print(line, flush=True)


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for ckpt_dir, label in VARIANT_DIRS:
        if os.path.isdir(ckpt_dir):
            eval_dir(ckpt_dir, label)
    print("auto_eval done", flush=True)


if __name__ == "__main__":
    main()
