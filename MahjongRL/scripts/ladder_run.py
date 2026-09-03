"""One-off ladder: 100k/200k/300k each vs a table of three 25k bots."""
import glob, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.head2head import head2head

B = "checkpoints/model_25088.pt"     # baseline: 3 seats of the 25k bot
GAMES = 200

def find_300k(timeout_s=7200):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        hits = sorted(glob.glob("checkpoints/model_3*.pt"))
        if hits:
            return hits[0]
        time.sleep(120)
    return None

results = []
for a in ["checkpoints/model_100352.pt", "checkpoints/model_200192.pt"]:
    r = head2head(a, B, games=GAMES, quiet=True)
    line = (f"{a.split('_')[1].split('.')[0]} vs 25k table: "
            f"A won {r['a_wins']}, B won {r['b_wins']}, draws {r['draws']} "
            f"-> A share {r['a_share']:.1%}")
    print(line, flush=True)
    results.append(line)

m300 = find_300k()
if m300:
    r = head2head(m300, B, games=GAMES, quiet=True)
    line = (f"{m300.split('_')[1].split('.')[0]} vs 25k table: "
            f"A won {r['a_wins']}, B won {r['b_wins']}, draws {r['draws']} "
            f"-> A share {r['a_share']:.1%}")
    print(line, flush=True)
    results.append(line)
else:
    results.append("300k milestone never appeared within 2h")

with open("checkpoints/ladder_results.txt", "w") as f:
    f.write("\n".join(results) + "\n")
print("DONE - saved to checkpoints/ladder_results.txt", flush=True)
