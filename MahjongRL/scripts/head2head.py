"""Head-to-head evaluation: checkpoint A vs checkpoint B at the same table.

One seat plays model A, the other three play model B, and the A-seat
rotates through all four positions (and the dealer button) so neither
side gets a positional edge. Reports A's win share of decided games -
the honest skill comparison once random opponents are too weak to
measure progress.

Usage:
    py -3 scripts/head2head.py --a checkpoints/latest.pt \
                               --b checkpoints/model_25088.pt --games 80
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from hk_mahjong.game import Game, GameConfig
from rl.encoder import encode_observation, encode_mask
from rl.model import build_model

MAX_STEPS = 700


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt.get("config") or {}
    model = build_model(device, **cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt.get("games_played", 0)


def model_action(model, game, seat):
    mask = encode_mask(game)
    planes, hist = encode_observation(game, seat)
    pt = torch.from_numpy(planes).unsqueeze(0)
    ht = torch.from_numpy(hist).unsqueeze(0)
    mt = torch.from_numpy(mask).unsqueeze(0)
    return int(model.act(pt, ht, mt, greedy=True)[0].item())


def play(model_a, model_b, a_seat, seed, min_faan=0):
    game = Game(GameConfig(seed=seed, min_faan=min_faan), dealer=seed % 4)
    for _ in range(MAX_STEPS):
        if game.phase == "finished":
            break
        mask = np.asarray(game.legal_actions())
        legal = np.flatnonzero(mask)
        if len(legal) == 1:
            game.step(int(legal[0]))
            continue
        seat = game.current_player()
        model = model_a if seat == a_seat else model_b
        game.step(model_action(model, game, seat))
    if game.result is None:
        game._finish_draw()
    return game.result


def head2head(path_a, path_b, games=80, min_faan=0, seed_base=77_000_000,
              quiet=False):
    model_a, games_a = load_checkpoint(path_a)
    model_b, games_b = load_checkpoint(path_b)
    if not quiet:
        print(f"A: {path_a} ({games_a:,} games trained)")
        print(f"B: {path_b} ({games_b:,} games trained)  [3 seats]")
    a_wins = b_wins = draws = 0
    for g in range(games):
        r = play(model_a, model_b, a_seat=g % 4, seed=seed_base + g,
                 min_faan=min_faan)
        if r["type"] == "draw":
            draws += 1
        elif r["winner"] == g % 4:
            a_wins += 1
        else:
            b_wins += 1
    decided = a_wins + b_wins
    # expected A share if equal skill: 25% (A holds 1 of 4 seats)
    a_share = a_wins / decided if decided else 0.0
    if not quiet:
        print(f"\nA won {a_wins}, B seats won {b_wins}, draws {draws} "
              f"(of {games})")
        print(f"A's share of decided games: {a_share:.1%} "
              f"(25% = equal skill, higher = A is stronger)")
    return {"a_wins": a_wins, "b_wins": b_wins, "draws": draws,
            "a_share": a_share, "games_a": games_a, "games_b": games_b}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="challenger checkpoint")
    ap.add_argument("--b", required=True, help="baseline checkpoint (3 seats)")
    ap.add_argument("--games", type=int, default=80)
    ap.add_argument("--min-faan", type=int, default=0)
    args = ap.parse_args()
    head2head(args.a, args.b, args.games, args.min_faan)
