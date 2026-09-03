"""Play Hong Kong mahjong against the trained bot at the terminal.

Hot-seat virtual table: the bot takes seat 0, humans take the others
(default 3 humans; with fewer, the remaining seats play random-legal).
Mirrors the physical demo setup - 1 bot vs 3 people - so you can test the
model's behavior before it ever touches the robot arm.

Usage (from the MahjongRL folder):
    py -3 scripts/play_vs_bot.py                     # 3 humans vs the bot
    py -3 scripts/play_vs_bot.py --humans 1          # you + 2 random seats
    py -3 scripts/play_vs_bot.py --checkpoint checkpoints/model_25000.pt

Tile names: 1p-9p dots, 1m-9m characters, 1s-9s bamboo, honors by name.
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from hk_mahjong.game import Game, GameConfig
from hk_mahjong.actions import Action, ActionType, NUM_ACTIONS, A_CHOW_LOWER
from hk_mahjong.tiles import Tile
from rl.encoder import encode_observation, encode_mask
from rl.model import build_model

BOT_SEAT = 0


def bot_action(model, game, seat):
    planes, hist = encode_observation(game, seat)
    mask = encode_mask(game)
    pt = torch.from_numpy(planes).unsqueeze(0)
    ht = torch.from_numpy(hist).unsqueeze(0)
    mt = torch.from_numpy(mask).unsqueeze(0)
    return int(model.act(pt, ht, mt, greedy=True)[0].item())


def show_table(game, seat):
    print()
    for s in range(4):
        who = "BOT " if s == BOT_SEAT else f"P{s}  "
        me = " <- you" if s == seat else ""
        melds = " ".join("[" + " ".join(map(str, m.tiles)) + "]"
                         for m in game.players[s].melds)
        disc = " ".join(map(str, game.players[s].discards[-8:]))
        print(f"  {who} melds: {melds or '-':<28} discards: ...{disc}{me}")
    print(f"  wall: {game.wall.remaining()} tiles left")


def show_hand(game, seat):
    hand = game.players[seat].hand
    idx = "  ".join(f"{i}:{t}" for i, t in enumerate(hand))
    print(f"\n  your hand: {idx}")


def human_discard(game, seat, mask):
    show_table(game, seat)
    show_hand(game, seat)
    hand = game.players[seat].hand
    extra = []
    if mask[38]:
        extra.append("'k' = kong")
    if mask[39]:
        extra.append("'w' = WIN")
    hint = (", " + ", ".join(extra)) if extra else ""
    while True:
        c = input(f"  discard which tile? (0-{len(hand) - 1}{hint}): ").strip().lower()
        if c == "k" and mask[38]:
            return 38
        if c == "w" and mask[39]:
            return 39
        try:
            t = hand[int(c)]
            if mask[t.kind_id]:
                return t.kind_id
        except (ValueError, IndexError):
            pass
        print("  invalid choice")


def human_claim(game, seat, mask):
    t = game.last_discard
    opts = ["Enter = pass"]
    if mask[39]:
        opts.append("'w' = WIN")
    if mask[38]:
        opts.append("'k' = kong")
    if mask[37]:
        opts.append("'p' = pung")
    chows = []
    for v in range(3):
        if mask[A_CHOW_LOWER + v]:
            base = t.kind_id - v
            run = " ".join(str(Tile.from_kind_id(base + i)) for i in range(3))
            opts.append(f"'c{v}' = chow ({run})")
            chows.append(v)
    print(f"\n  P{game.last_discarder if game.last_discarder != BOT_SEAT else 'BOT'}"
          f" discarded {t}.  Options: {', '.join(opts)}")
    show_hand(game, seat)
    while True:
        c = input("  claim? ").strip().lower()
        if c == "":
            return 40
        if c == "w" and mask[39]:
            return 39
        if c == "k" and mask[38]:
            return 38
        if c == "p" and mask[37]:
            return 37
        if c.startswith("c") and len(c) == 2 and c[1].isdigit() \
                and int(c[1]) in chows:
            return A_CHOW_LOWER + int(c[1])
        print("  invalid choice")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/latest.pt")
    ap.add_argument("--humans", type=int, default=3, choices=[1, 2, 3])
    ap.add_argument("--min-faan", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=4)
    args = ap.parse_args()

    model = build_model("cpu", channels=args.channels, num_blocks=args.blocks)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {args.checkpoint} "
          f"({ckpt.get('games_played', '?')} games trained)")

    human_seats = list(range(1, 1 + args.humans))
    random_seats = [s for s in range(1, 4) if s not in human_seats]
    rng = random.Random(args.seed)
    game = Game(GameConfig(seed=args.seed, min_faan=args.min_faan))
    print(f"\nseats: BOT=0, humans={human_seats}, random={random_seats}")
    print("=" * 60)

    while game.phase != "finished":
        mask = np.asarray(game.legal_actions())
        legal = np.flatnonzero(mask)
        seat = game.current_player()
        if len(legal) == 1:            # forced (usually mandatory pass)
            game.step(int(legal[0]))
            continue
        if seat == BOT_SEAT:
            aid = bot_action(model, game, seat)
            a = Action.from_id(aid)
            if a.type == ActionType.DISCARD:
                print(f"\n>> BOT discards {Tile.from_kind_id(a.tile_kind)}")
            elif a.type != ActionType.PASS:
                print(f"\n>> BOT: {a.type.name}!")
        elif seat in human_seats:
            if args.humans > 1:
                input(f"\n--- P{seat}: press Enter when you have the "
                      f"screen to yourself ---")
            if game.phase == "discard":
                aid = human_discard(game, seat, mask)
            else:
                aid = human_claim(game, seat, mask)
            if args.humans > 1:
                print("\n" * 30)       # push previous hand off screen
        else:
            aid = int(rng.choice(legal))
        game.step(aid)

    r = game.result
    print("=" * 60)
    if r["type"] == "draw":
        print("Wall exhausted - drawn hand, nobody wins.")
    else:
        who = "THE BOT" if r["winner"] == BOT_SEAT else f"Player {r['winner']}"
        how = "self-draw" if r["self_draw"] else "on a discard"
        print(f"{who} WINS {how}! ({r['faan']} faan: "
              f"{', '.join(n for n, _ in r['elements']) or 'basic hand'})")
        print("Winning hand:",
              " ".join(map(str, sorted(game.players[r['winner']].hand))),
              "+ melds", " ".join("[" + " ".join(map(str, m.tiles)) + "]"
                                  for m in game.players[r["winner"]].melds))


if __name__ == "__main__":
    main()
