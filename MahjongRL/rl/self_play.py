"""Self-play: 4 copies of the current policy play full HK mahjong hands.

All four seats share one network (parameter sharing), so every game yields
four trajectories - a hand of mahjong produces ~70 decision points per seat,
and 500k games therefore gives ~10^8 training transitions.

Rewards are the zero-sum end-of-hand payouts (normalized by the limit-hand
value), assigned to every transition of that seat (Monte-Carlo return;
gamma=1 within a hand). A small shaping bonus for winning encourages the
early policy to discover mahjong at all.
"""
import numpy as np
import torch

from hk_mahjong.game import Game, GameConfig
from hk_mahjong.scoring import LIMIT_FAAN
from .encoder import encode_observation, encode_mask

MAX_STEPS = 700          # safety cap per hand
SCORE_NORM = 2.0 ** LIMIT_FAAN
WIN_BONUS = 0.05


class Trajectory:
    __slots__ = ("planes", "history", "masks", "actions", "logprobs",
                 "values", "reward")

    def __init__(self):
        self.planes, self.history, self.masks = [], [], []
        self.actions, self.logprobs, self.values = [], [], []
        self.reward = 0.0


def play_game(model, device="cpu", seed=None, greedy=False, min_faan=0,
              reward="win"):
    """Play one hand; returns (list of 4 Trajectories, game.result).

    reward="win":   +1 to the winner, -1/3 to the other three, 0 on a draw.
                    The bot purely races to complete 4 melds + a pair.
    reward="score": zero-sum faan-based payouts (traditional HK scoring);
                    the bot trades speed against hand value.
    """
    game = Game(GameConfig(seed=seed, min_faan=min_faan))
    trajs = [Trajectory() for _ in range(4)]

    for _ in range(MAX_STEPS):
        if game.phase == "finished":
            break
        mask = encode_mask(game)
        legal = np.flatnonzero(mask)
        if len(legal) == 1:
            # forced move (usually a mandatory PASS in the claim phase):
            # no decision to learn - skip the network call and the transition
            game.step(int(legal[0]))
            continue
        seat = game.current_player()
        planes, hist = encode_observation(game, seat)

        pt = torch.from_numpy(planes).unsqueeze(0).to(device)
        ht = torch.from_numpy(hist).unsqueeze(0).to(device)
        mt = torch.from_numpy(mask).unsqueeze(0).to(device)
        action, logprob, value = model.act(pt, ht, mt, greedy=greedy)
        aid = int(action.item())

        t = trajs[seat]
        t.planes.append(planes)
        t.history.append(hist)
        t.masks.append(mask)
        t.actions.append(aid)
        t.logprobs.append(float(logprob.item()))
        t.values.append(float(value.item()))
        game.step(aid)

    if game.result is None:
        game._finish_draw()

    winner = game.result["winner"]
    for seat in range(4):
        if reward == "win":
            if winner < 0:
                r = 0.0
            else:
                r = 1.0 if winner == seat else -1.0 / 3.0
        else:  # "score"
            r = game.result["scores"][seat] / SCORE_NORM
            if winner == seat:
                r += WIN_BONUS
        trajs[seat].reward = r
    return trajs, game.result


def collect_batch(model, num_games, device="cpu", seed_base=0, min_faan=0,
                  reward="win"):
    """Play num_games and flatten transitions into arrays for PPO."""
    P, H, M, A, LP, V, RET = [], [], [], [], [], [], []
    results = []
    for g in range(num_games):
        trajs, result = play_game(model, device, seed=seed_base + g,
                                  min_faan=min_faan, reward=reward)
        results.append(result)
        for t in trajs:
            n = len(t.actions)
            if n == 0:
                continue
            P.extend(t.planes)
            H.extend(t.history)
            M.extend(t.masks)
            A.extend(t.actions)
            LP.extend(t.logprobs)
            V.extend(t.values)
            RET.extend([t.reward] * n)   # MC return, gamma = 1 within hand
    batch = {
        "planes": np.stack(P), "history": np.stack(H), "masks": np.stack(M),
        "actions": np.asarray(A, np.int64),
        "logprobs": np.asarray(LP, np.float32),
        "values": np.asarray(V, np.float32),
        "returns": np.asarray(RET, np.float32),
    }
    return batch, results
