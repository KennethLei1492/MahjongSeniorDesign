"""Evaluate a trained checkpoint: greedy policy vs. random-legal opponents.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/latest.pt --games 200
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from hk_mahjong.game import Game, GameConfig
from rl.encoder import encode_observation, encode_mask
from rl.model import build_model
from rl.self_play import MAX_STEPS


def random_action(mask, rng):
    legal = [i for i, ok in enumerate(mask) if ok]
    return rng.choice(legal)


def play_vs_random(model, device, seed, agent_seat=0, show=False, min_faan=0):
    game = Game(GameConfig(seed=seed, min_faan=min_faan))
    rng = random.Random(seed)
    if show:
        print(f"--- game (seed {seed}): BOT is seat {agent_seat}, "
              f"seats {[s for s in range(4) if s != agent_seat]} play random ---")
        print("bot starting hand:", game.players[agent_seat].hand)
    for _ in range(MAX_STEPS):
        if game.phase == "finished":
            break
        seat = game.current_player()
        mask = encode_mask(game)
        if seat == agent_seat:
            planes, hist = encode_observation(game, seat)
            pt = torch.from_numpy(planes).unsqueeze(0).to(device)
            ht = torch.from_numpy(hist).unsqueeze(0).to(device)
            mt = torch.from_numpy(mask).unsqueeze(0).to(device)
            aid = int(model.act(pt, ht, mt, greedy=True)[0].item())
        else:
            aid = random_action(mask, rng)
        if show:
            from hk_mahjong.actions import Action, ActionType
            from hk_mahjong.tiles import Tile
            a = Action.from_id(aid)
            if a.type == ActionType.DISCARD:
                desc = f"discards {Tile.from_kind_id(a.tile_kind)}"
            else:
                desc = a.type.name.lower()
            who = "BOT " if seat == agent_seat else f"p{seat}  "
            if a.type != ActionType.PASS:  # keep the log readable
                print(f"  {who}{desc}   (wall: {game.wall.remaining()})")
        game.step(aid)
    if game.result is None:
        game._finish_draw()
    if show:
        r = game.result
        if r["type"] == "win":
            winner = "BOT" if r["winner"] == agent_seat else f"p{r['winner']}"
            print(f"--- result: {winner} wins, {r['faan']} faan "
                  f"{[n for n, _ in r['elements']]}, scores {r['scores']} ---\n")
        else:
            print("--- result: wall exhausted, drawn hand ---\n")
    return game.result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/latest.pt")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--show", action="store_true",
                    help="print the first game move by move")
    ap.add_argument("--min-faan", type=int, default=0,
                    help="must match the value the checkpoint was trained with")
    args = ap.parse_args()

    model = build_model(args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    agent_wins, opp_wins, draws, scores = 0, 0, 0, []
    for g in range(args.games):
        r = play_vs_random(model, args.device, seed=10_000_000 + g,
                           show=args.show and g == 0, min_faan=args.min_faan)
        scores.append(r["scores"][0])
        if r["type"] == "draw":
            draws += 1
        elif r["winner"] == 0:
            agent_wins += 1
        else:
            opp_wins += 1
    n = args.games
    print(f"agent wins: {agent_wins}/{n} ({agent_wins/n:.1%})")
    print(f"opponent wins: {opp_wins}/{n}, draws: {draws}/{n}")
    print(f"mean score: {np.mean(scores):+.2f} points/hand")
    print("(random-play baseline for a single seat would win ~25% of decided hands)")


if __name__ == "__main__":
    main()
