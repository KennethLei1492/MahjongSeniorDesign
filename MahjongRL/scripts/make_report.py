"""Generate an HTML training report: learning curves + live evaluations.

Reads checkpoints/train_log.csv, benchmarks the target checkpoint against
random-legal opponents, plays a head-to-head ladder against earlier milestone
checkpoints, and writes a self-contained docs/training_report.html.

Usage:
    py -3 scripts/make_report.py                          # report on latest.pt
    py -3 scripts/make_report.py --eval-games 60 --ladder-games 40
Re-run it any time (e.g. when the 500k milestone lands) - it regenerates
from whatever data exists.
"""
import argparse
import csv
import datetime
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = "docs/training_report.html"

# validated reference palette (dataviz skill): blue series slot, both modes
C = {"light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e",
               "grid": "#e7e6e2", "s1": "#2a78d6"},
     "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7",
              "grid": "#33322f", "s1": "#3987e5"}}


def read_log():
    rows = []
    with open("checkpoints/train_log.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("games"):
                rows.append({"games": int(float(r["games"])),
                             "win": float(r["win_rate"]),
                             "spg": float(r["sec_per_game"])})
    return rows


def smooth(rows, key, window=15, max_points=240):
    ys, xs, out = [r[key] for r in rows], [r["games"] for r in rows], []
    for i in range(len(ys)):
        lo = max(0, i - window + 1)
        out.append(sum(ys[lo:i + 1]) / (i + 1 - lo))
    step = max(1, len(out) // max_points)
    return xs[::step], out[::step]


def line_chart(xs, ys, title, y_fmt, cid, y_max=None):
    """Single-series SVG line chart with grid, endpoint label, hover layer."""
    W, H, L, R, T, B = 640, 240, 52, 20, 18, 30
    y_max = y_max if y_max is not None else max(ys) * 1.12 or 1
    x_max = max(xs) or 1

    def px(x):
        return L + (W - L - R) * x / x_max

    def py(y):
        return T + (H - T - B) * (1 - y / y_max)

    grid, glab = [], []
    for i in range(5):
        gy = y_max * i / 4
        grid.append(f'<line x1="{L}" y1="{py(gy):.1f}" x2="{W - R}" '
                    f'y2="{py(gy):.1f}" stroke="var(--grid)"/>')
        glab.append(f'<text x="{L - 6}" y="{py(gy) + 4:.1f}" text-anchor="end" '
                    f'class="tick">{y_fmt(gy)}</text>')
    for i in range(1, 4):
        gx = x_max * i / 4
        glab.append(f'<text x="{px(gx):.1f}" y="{H - 10}" text-anchor="middle" '
                    f'class="tick">{gx / 1000:.0f}k</text>')
    pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys))
    end_x, end_y = px(xs[-1]), py(ys[-1])
    data = ";".join(f"{x}|{y:.4f}" for x, y in zip(xs, ys))
    return f'''
<div class="chart"><h3>{title}</h3>
<svg id="{cid}" viewBox="0 0 {W} {H}" data-series="{data}" data-ymax="{y_max}"
     data-xmax="{x_max}" data-l="{L}" data-r="{R}" data-t="{T}" data-b="{B}">
  {"".join(grid)}{"".join(glab)}
  <polyline points="{pts}" fill="none" stroke="var(--s1)" stroke-width="2"
            stroke-linejoin="round"/>
  <circle cx="{end_x:.1f}" cy="{end_y:.1f}" r="4" fill="var(--s1)"
          stroke="var(--surface)" stroke-width="2"/>
  <text x="{min(end_x, W - R - 4):.1f}" y="{max(end_y - 10, 14):.1f}"
        text-anchor="end" class="dlabel">{y_fmt(ys[-1])}</text>
  <line class="cross" y1="{T}" y2="{H - B}" stroke="var(--ink2)"
        stroke-dasharray="3,3" visibility="hidden"/>
  <circle class="dot" r="4" fill="var(--s1)" stroke="var(--surface)"
          stroke-width="2" visibility="hidden"/>
</svg><div class="tip" hidden></div></div>'''


def bar_chart(labels, shares, title):
    """A-share vs each milestone; 25% reference line = equal skill."""
    W, per, L, R, T, B = 640, 44, 150, 70, 16, 26
    H = T + B + per * len(labels)
    x_max = max(0.5, max(shares) * 1.15)

    def px(v):
        return L + (W - L - R) * v / x_max

    bars = []
    for i, (lab, v) in enumerate(zip(labels, shares)):
        y = T + per * i + 7
        bars.append(
            f'<text x="{L - 8}" y="{y + 15}" text-anchor="end" class="tick">'
            f'{lab}</text>'
            f'<rect x="{L}" y="{y}" width="{max(px(v) - L, 2):.1f}" height="22"'
            f' rx="4" fill="var(--s1)"><title>{v:.1%} of decided games</title>'
            f'</rect>'
            f'<text x="{px(v) + 6:.1f}" y="{y + 15}" class="dlabel" '
            f'text-anchor="start">{v:.0%}</text>')
    ref = px(0.25)
    return f'''
<div class="chart"><h3>{title}</h3>
<svg viewBox="0 0 {W} {H}">
  <line x1="{ref:.1f}" y1="{T}" x2="{ref:.1f}" y2="{H - B}"
        stroke="var(--ink2)" stroke-dasharray="4,3"/>
  <text x="{ref:.1f}" y="{H - 8}" text-anchor="middle" class="tick">
    25% = equal skill</text>
  {"".join(bars)}
</svg></div>'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/latest.pt")
    ap.add_argument("--eval-games", type=int, default=40)
    ap.add_argument("--ladder-games", type=int, default=32)
    ap.add_argument("--skip-evals", action="store_true",
                    help="charts from the log only (fast)")
    args = ap.parse_args()

    rows = read_log()
    games_now = rows[-1]["games"] if rows else 0
    xs_w, ys_w = smooth(rows, "win")
    xs_s, ys_s = smooth(rows, "spg")
    recent = rows[-20:]
    completion = sum(r["win"] for r in recent) / max(len(recent), 1)
    spg = sum(r["spg"] for r in recent) / max(len(recent), 1)

    # ---------- evaluations ----------
    vs_random_html = ladder_html = ""
    vs_random_line = "not run"
    if not args.skip_evals:
        from scripts.evaluate import play_vs_random
        from scripts.head2head import head2head, load_checkpoint
        import torch  # noqa: F401
        model, ckpt_games = load_checkpoint(args.checkpoint)
        aw = ow = dr = 0
        for g in range(args.eval_games):
            r = play_vs_random(model, "cpu", seed=30_000_000 + g)
            if r["type"] == "draw":
                dr += 1
            elif r["winner"] == 0:
                aw += 1
            else:
                ow += 1
        decided = aw + ow
        share = aw / decided if decided else 0
        vs_random_line = (f"{aw}/{args.eval_games} games won, opponents "
                          f"{ow}, draws {dr} - {share:.0%} of decided games "
                          f"(random seat baseline: 25%)")

        milestones = sorted(glob.glob("checkpoints/model_*.pt"),
                            key=lambda p: int(re.findall(r"\d+", p)[-1]))
        milestones = [m for m in milestones
                      if int(re.findall(r"\d+", m)[-1]) < ckpt_games - 10_000]
        if len(milestones) > 5:  # subsample the ladder to 5 rungs
            idx = [round(i * (len(milestones) - 1) / 4) for i in range(5)]
            milestones = [milestones[i] for i in sorted(set(idx))]
        labs, shares = [], []
        for m in milestones:
            res = head2head(args.checkpoint, m, games=args.ladder_games,
                            quiet=True)
            labs.append(f"vs {int(re.findall(r'[0-9]+', m)[-1]) // 1000}k")
            shares.append(res["a_share"])
        if labs:
            ladder_html = bar_chart(
                labs, shares,
                f"Current bot vs earlier milestones "
                f"(1 seat vs 3, {args.ladder_games} games each)")

    vs_random_html = f'<p class="result">Vs 3 random-legal opponents: ' \
                     f'<strong>{vs_random_line}</strong></p>'

    date = datetime.date.today().isoformat()
    html = f'''<title>MahjongRL Training Report</title>
<style>
:root {{ --surface:{C["light"]["surface"]}; --ink:{C["light"]["ink"]};
  --ink2:{C["light"]["ink2"]}; --grid:{C["light"]["grid"]};
  --s1:{C["light"]["s1"]}; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --surface:{C["dark"]["surface"]}; --ink:{C["dark"]["ink"]};
  --ink2:{C["dark"]["ink2"]}; --grid:{C["dark"]["grid"]};
  --s1:{C["dark"]["s1"]}; }} }}
:root[data-theme="dark"] {{
  --surface:{C["dark"]["surface"]}; --ink:{C["dark"]["ink"]};
  --ink2:{C["dark"]["ink2"]}; --grid:{C["dark"]["grid"]};
  --s1:{C["dark"]["s1"]}; }}
body {{ background:var(--surface); color:var(--ink);
  font:16px/1.55 "Segoe UI", system-ui, sans-serif; margin:0;
  padding:2.5rem 1rem 4rem; }}
.page {{ max-width:720px; margin:0 auto; }}
h1 {{ font-size:1.7rem; margin:0 0 .2rem; }}
h3 {{ font-size:.95rem; font-weight:600; margin:0 0 .4rem; }}
.sub {{ color:var(--ink2); margin:0 0 1.6rem; }}
.tiles {{ display:flex; flex-wrap:wrap; gap:.6rem; margin-bottom:1.8rem; }}
.tile {{ flex:1 1 130px; border:1px solid var(--grid); border-radius:8px;
  padding:.6rem .8rem; }}
.tile b {{ display:block; font-size:1.5rem;
  font-variant-numeric:tabular-nums; }}
.tile small {{ color:var(--ink2); }}
.chart {{ margin:1.6rem 0; }}
svg {{ width:100%; height:auto; display:block; }}
.tick {{ font-size:11px; fill:var(--ink2); }}
.dlabel {{ font-size:12px; font-weight:600; fill:var(--ink); }}
.result {{ border-left:3px solid var(--s1); padding:.5rem .9rem;
  background:color-mix(in srgb, var(--s1) 7%, transparent); }}
.note {{ color:var(--ink2); font-size:.88rem; }}
.tip {{ position:fixed; pointer-events:none; background:var(--ink);
  color:var(--surface); font-size:12px; padding:3px 8px; border-radius:5px; }}
</style>
<div class="page">
<h1>MahjongRL Training Report</h1>
<p class="sub">Res2Net+LSTM self-play (PPO) &middot; Hong Kong rules,
win = 4 melds + pair &middot; generated {date}</p>
<div class="tiles">
<div class="tile"><b>{games_now:,}</b><small>self-play games trained</small></div>
<div class="tile"><b>{completion:.0%}</b><small>of recent games end in a
completed hand</small></div>
<div class="tile"><b>{spg:.2f}s</b><small>per game
(~{86400 / spg if spg else 0:,.0f}/day)</small></div>
</div>
{vs_random_html}
{line_chart(xs_w, ys_w, "Hand-completion rate over training (any seat wins vs drawn wall)", lambda v: f"{v * 100:.0f}%", "c1", 1.0)}
{line_chart(xs_s, ys_s, "Simulation cost per game (lower = faster)", lambda v: f"{v:.1f}s", "c2")}
{ladder_html}
<p class="note">Method: all four seats share one policy during training
(self-play). "Vs random" plays the bot in seat 0 against three random-legal
players; the milestone ladder gives the current bot one seat and an earlier
checkpoint the other three, rotating seats and dealer - 25% of decided games
is the equal-skill baseline in both. Full data: checkpoints/train_log.csv.</p>
</div>
<script>
document.querySelectorAll("svg[data-series]").forEach(svg => {{
  const pts = svg.dataset.series.split(";").map(s => s.split("|").map(Number));
  const [L, R, T, B] = ["l", "r", "t", "b"].map(k => +svg.dataset[k]);
  const ymax = +svg.dataset.ymax, xmax = +svg.dataset.xmax;
  const W = svg.viewBox.baseVal.width, H = svg.viewBox.baseVal.height;
  const cross = svg.querySelector(".cross"), dot = svg.querySelector(".dot");
  const tip = svg.parentElement.querySelector(".tip");
  svg.addEventListener("mousemove", e => {{
    const r = svg.getBoundingClientRect();
    const gx = (e.clientX - r.left) / r.width * W;
    const xv = (gx - L) / (W - L - R) * xmax;
    let best = pts[0];
    for (const p of pts) if (Math.abs(p[0] - xv) < Math.abs(best[0] - xv)) best = p;
    const px = L + (W - L - R) * best[0] / xmax;
    const py = T + (H - T - B) * (1 - best[1] / ymax);
    cross.setAttribute("x1", px); cross.setAttribute("x2", px);
    cross.setAttribute("visibility", "visible");
    dot.setAttribute("cx", px); dot.setAttribute("cy", py);
    dot.setAttribute("visibility", "visible");
    tip.hidden = false;
    tip.textContent = Math.round(best[0] / 1000) + "k games: " +
      (ymax <= 1 ? (best[1] * 100).toFixed(0) + "%" : best[1].toFixed(2) + "s");
    tip.style.left = (e.clientX + 12) + "px";
    tip.style.top = (e.clientY - 10) + "px";
  }});
  svg.addEventListener("mouseleave", () => {{
    cross.setAttribute("visibility", "hidden");
    dot.setAttribute("visibility", "hidden"); tip.hidden = true;
  }});
}});
</script>'''
    os.makedirs("docs", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"report written to {OUT} ({games_now:,} games)")


if __name__ == "__main__":
    main()
