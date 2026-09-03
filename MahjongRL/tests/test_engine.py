"""Engine sanity tests: win detection, scoring, and full random games.

Run:  python -m pytest tests/ -q     (or just: python tests/test_engine.py)
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hk_mahjong.tiles import Tile, counts_from_tiles
from hk_mahjong.hand import is_winning, is_thirteen_orphans
from hk_mahjong.scoring import score_hand, LIMIT_FAAN
from hk_mahjong.game import Game, GameConfig
from hk_mahjong.actions import NUM_ACTIONS


def tiles(*specs):
    """tiles('1p','1p','1p','2m',...) -> Tile list."""
    type_map = {"p": 0, "m": 1, "s": 2, "z": 3}
    return [Tile(type_map[s[-1]], int(s[:-1])) for s in specs]


def test_standard_win():
    hand = tiles("1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
                 "1m", "1m", "1m", "5z", "5z")
    assert is_winning(counts_from_tiles(hand), 0)


def test_not_win():
    hand = tiles("1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
                 "1m", "1m", "2m", "5z", "5z")
    assert not is_winning(counts_from_tiles(hand), 0)


def test_thirteen_orphans():
    hand = tiles("1p", "9p", "1m", "9m", "1s", "9s",
                 "1z", "2z", "3z", "4z", "5z", "6z", "7z", "7z")
    assert is_thirteen_orphans(counts_from_tiles(hand), 0)
    faan, elems = score_hand(counts_from_tiles(hand), [], None,
                             self_draw=True, seat_wind=1, round_wind=1,
                             num_flowers=0, is_concealed=True)
    assert faan == LIMIT_FAAN


def test_full_flush_scoring():
    hand = tiles("1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
                 "1p", "1p", "1p", "5p", "5p")
    faan, elems = score_hand(counts_from_tiles(hand), [], None,
                             self_draw=False, seat_wind=1, round_wind=1,
                             num_flowers=1, is_concealed=True)
    names = [n for n, _ in elems]
    assert "Full Flush" in names and faan >= 6


def test_random_games_complete():
    """Random legal play must always reach a terminal state legally."""
    rng = random.Random(7)
    finished = wins = 0
    for seed in range(30):
        game = Game(GameConfig(seed=seed))
        for _ in range(2000):
            if game.phase == "finished":
                break
            mask = game.legal_actions()
            legal = [i for i in range(NUM_ACTIONS) if mask[i]]
            assert legal, f"no legal actions in phase {game.phase}"
            game.step(rng.choice(legal))
        assert game.phase == "finished"
        finished += 1
        if game.result["type"] == "win":
            wins += 1
            assert game.result["faan"] >= game.cfg.min_faan
            assert abs(sum(game.result["scores"])) < 1e-9  # zero-sum
    assert finished == 30
    print(f"  30 random games completed, {wins} ended in a win")


def test_encoder_shapes():
    import numpy as np
    from rl.encoder import encode_observation, NUM_PLANES, HISTORY_LEN, EVENT_DIM
    game = Game(GameConfig(seed=1))
    planes, hist = encode_observation(game, game.current_player())
    assert planes.shape == (NUM_PLANES, 4, 34)
    assert hist.shape == (HISTORY_LEN, EVENT_DIM)
    assert planes.dtype == np.float32


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            print(name, "...")
            fn()
    print("all tests passed")
