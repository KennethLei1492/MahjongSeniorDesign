"""Hong Kong Mahjong rules engine.

Pure-Python, dependency-free implementation of the Hong Kong (Cantonese)
ruleset per https://www.mahjong-rules.com/hong-kong-mahjong-rules/:
144 tiles (3 suits, winds, dragons, 8 bonus tiles), chow/pung/kong claims,
4 melds + 1 pair winning shape, minimum-faan requirement and faan scoring.

Tile representation mirrors AlphaJong's {index, type} convention so that
logic can be cross-referenced against the AlphaJong-master codebase.
"""
from .tiles import Tile, TYPE_DOT, TYPE_CHAR, TYPE_BAM, TYPE_HONOR, TYPE_BONUS
from .game import Game, GameConfig
from .actions import Action, ActionType
