"""Train the mahjong policy by self-play toward a target game count.

Usage:
    python scripts/train.py --max-new-games 5000 --checkpoint-every 500
    python scripts/train.py --games 500000 --workers 8       # parallel
    python scripts/train.py --workers 1                      # old serial mode

Parallelism: --workers N runs N self-play processes feeding one PPO learner,
pipelined (workers play iteration k+1 while the learner updates on k).
Default --workers 0 auto-picks min(8, cores-2). The run is fully resumable:
checkpoints store model + optimizer + games played, so any target can be
reached across many sessions. Progress appends to checkpoints/train_log.csv.
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from rl.model import build_model
from rl.ppo import PPOTrainer
from rl.self_play import collect_batch


def summarize(results):
    wins = [r for r in results if r["type"] == "win"]
    win_rate = len(wins) / max(len(results), 1)
    avg_faan = sum(r["faan"] for r in wins) / max(len(wins), 1)
    return win_rate, avg_faan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=2_000_000,
                    help="total self-play games to reach (contest milestone "
                         "model_500xxx.pt is saved on the way)")
    ap.add_argument("--max-new-games", type=int, default=None,
                    help="stop after this many NEW games this session "
                         "(e.g. a daily slice); overall --games target "
                         "still caps the run")
    ap.add_argument("--games-per-iter", type=int, default=128)
    ap.add_argument("--workers", type=int, default=0,
                    help="parallel self-play processes; 0 = auto "
                         "(min(8, cores-2)), 1 = serial")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--ckpt-dir", default="checkpoints",
                    help="checkpoint/log directory - use a separate dir per "
                         "model variant (e.g. checkpoints_big)")
    ap.add_argument("--resume", default=None,
                    help="checkpoint to resume (default: <ckpt-dir>/latest.pt)")
    ap.add_argument("--checkpoint-every", type=int, default=500,
                    help="save latest.pt every N games")
    ap.add_argument("--milestone-every", type=int, default=25_000,
                    help="keep a permanent model_N.pt copy every N games "
                         "(latest.pt is overwritten; milestones are not)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ppo-epochs", type=int, default=1)
    ap.add_argument("--minibatch", type=int, default=2048)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--lstm-hidden", type=int, default=128)
    ap.add_argument("--reward", choices=["win", "score"], default="win",
                    help="win: +1 for winning, no point scoring (project "
                         "rule). score: faan-based zero-sum payouts.")
    ap.add_argument("--min-faan", type=int, default=0,
                    help="minimum faan to declare a win. 0 = any 4 melds + "
                         "pair wins (project rule); 3 = traditional HK. "
                         "Train and deploy with the SAME value.")
    args = ap.parse_args()

    workers = args.workers
    if workers == 0:
        workers = max(1, min(8, (os.cpu_count() or 4) - 2))

    net_cfg = {"channels": args.channels, "num_blocks": args.blocks,
               "lstm_hidden": args.lstm_hidden}
    model = build_model(args.device, **net_cfg)
    trainer = PPOTrainer(model, device=args.device, lr=args.lr,
                         epochs=args.ppo_epochs,
                         minibatch_size=args.minibatch)

    ckdir = args.ckpt_dir
    if args.resume is None:
        args.resume = os.path.join(ckdir, "latest.pt")

    games_played = 0
    if args.resume and os.path.exists(args.resume):
        try:
            games_played = trainer.load(args.resume)
            print(f"resumed from {args.resume} at {games_played} games")
        except Exception as e:
            print(f"WARNING: could not load {args.resume} ({e}).")
            print("Starting fresh (expected after a model-size change; "
                  "otherwise investigate before losing progress).")

    log_path = os.path.join(ckdir, "train_log.csv")
    os.makedirs(ckdir, exist_ok=True)
    new_log = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
    log = open(log_path, "a", newline="")
    logw = csv.writer(log)
    if new_log:
        logw.writerow(["games", "win_rate", "draw_rate", "avg_faan",
                       "policy_loss", "value_loss", "entropy", "sec_per_game"])

    session_start = games_played
    session_cap = (args.games if args.max_new_games is None
                   else min(args.games, session_start + args.max_new_games))
    gpi = args.games_per_iter
    print(f"training: {games_played} -> {session_cap} games, "
          f"{workers} worker(s), device {args.device}")

    def report(n_new, results, metrics, elapsed):
        nonlocal games_played
        games_played += n_new
        win_rate, avg_faan = summarize(results)
        spg = elapsed / max(n_new, 1)
        print(f"[{games_played}/{args.games}] win%={win_rate:.2f} "
              f"faan={avg_faan:.2f} pi={metrics['policy_loss']:.4f} "
              f"v={metrics['value_loss']:.4f} H={metrics['entropy']:.3f} "
              f"{spg:.2f}s/game")
        logw.writerow([games_played, win_rate, 1 - win_rate, avg_faan,
                       metrics["policy_loss"], metrics["value_loss"],
                       metrics["entropy"], spg])
        log.flush()

    last_save = games_played
    model.eval()

    if workers > 1:
        from rl.parallel_self_play import ParallelCollector
        collector = ParallelCollector(
            workers, net_cfg,
            min_faan=args.min_faan, reward=args.reward)
        try:
            # pipelined: workers play round k+1 while the learner updates on k
            collector.dispatch(model.state_dict(), gpi, games_played)
            while games_played < session_cap:
                t0 = time.time()
                batch, results = collector.gather()
                if games_played + len(results) < session_cap:
                    collector.dispatch(model.state_dict(), gpi,
                                       games_played + len(results))
                model.train()
                metrics = trainer.update(batch)
                model.eval()
                report(len(results), results, metrics, time.time() - t0)
                if games_played - last_save >= args.checkpoint_every:
                    trainer.save(os.path.join(ckdir, "latest.pt"), games_played, net_cfg)
                    if (games_played // args.milestone_every
                            > last_save // args.milestone_every):
                        trainer.save(os.path.join(ckdir, f"model_{games_played}.pt"),
                                     games_played, net_cfg)
                    last_save = games_played
        finally:
            collector.stop()
    else:
        while games_played < session_cap:
            t0 = time.time()
            batch, results = collect_batch(model, gpi, device=args.device,
                                           seed_base=games_played,
                                           min_faan=args.min_faan,
                                           reward=args.reward)
            model.train()
            metrics = trainer.update(batch)
            model.eval()
            report(len(results), results, metrics, time.time() - t0)
            if games_played - last_save >= args.checkpoint_every:
                trainer.save(os.path.join(ckdir, "latest.pt"), games_played, net_cfg)
                if (games_played // args.milestone_every
                        > last_save // args.milestone_every):
                    trainer.save(os.path.join(ckdir, f"model_{games_played}.pt"),
                                 games_played)
                last_save = games_played

    trainer.save(os.path.join(ckdir, "latest.pt"), games_played, net_cfg)
    print(f"session done: +{games_played - session_start} games this session, "
          f"{games_played}/{args.games} total "
          f"({games_played / args.games:.1%} of target)")


if __name__ == "__main__":
    main()
