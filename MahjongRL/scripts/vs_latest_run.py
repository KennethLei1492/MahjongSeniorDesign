"""Milestone versions vs a table of the 500k champion, 200 games each."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.head2head import head2head

LATEST = "checkpoints/model_500096_champion.pt"
GAMES = 200
lines = []
for a in ["checkpoints/model_25088.pt", "checkpoints/model_100352.pt",
          "checkpoints/model_200192.pt", "checkpoints/model_300032.pt",
          "checkpoints/model_400384.pt"]:
    r = head2head(a, LATEST, games=GAMES, quiet=True)
    tag = a.split("_")[1].split(".")[0]
    line = (f"{tag} (1 seat) vs 500k-champion table: A won {r['a_wins']}, "
            f"champion won {r['b_wins']}, draws {r['draws']} "
            f"-> {tag}'s share {r['a_share']:.1%} (25% = equal)")
    print(line, flush=True)
    lines.append(line)
with open("checkpoints/vs_latest_results.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
print("DONE", flush=True)
