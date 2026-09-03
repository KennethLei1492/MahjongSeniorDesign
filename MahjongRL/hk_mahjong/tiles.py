"""Tile definitions.

AlphaJong convention: a tile is (index, type).
  type 0 = Dots (pin)      index 1-9
  type 1 = Characters (man) index 1-9
  type 2 = Bamboo (sou)    index 1-9
  type 3 = Honors:         index 1-4 winds (E,S,W,N), 5-7 dragons (White,Green,Red)
  type 4 = Bonus:          index 1-4 flowers, 5-8 seasons (HK addition, not in AlphaJong)

Each of the 34 playable kinds exists 4 times; each bonus tile once -> 144 tiles.
A playable kind also has a flat id in [0, 34): id = type * 9 + (index - 1).
"""
from dataclasses import dataclass

TYPE_DOT, TYPE_CHAR, TYPE_BAM, TYPE_HONOR, TYPE_BONUS = 0, 1, 2, 3, 4
NUM_KINDS = 34  # playable kinds (bonus tiles are never held in hand)

WIND_EAST, WIND_SOUTH, WIND_WEST, WIND_NORTH = 1, 2, 3, 4
DRAGON_WHITE, DRAGON_GREEN, DRAGON_RED = 5, 6, 7

TYPE_NAMES = ["Dots", "Characters", "Bamboo", "Honor", "Bonus"]
HONOR_NAMES = {1: "East", 2: "South", 3: "West", 4: "North",
               5: "White Dragon", 6: "Green Dragon", 7: "Red Dragon"}
BONUS_NAMES = {1: "Plum", 2: "Orchid", 3: "Chrysanthemum", 4: "Bamboo Flower",
               5: "Spring", 6: "Summer", 7: "Autumn", 8: "Winter"}


@dataclass(frozen=True, order=True)
class Tile:
    type: int
    index: int

    @property
    def kind_id(self) -> int:
        """Flat id in [0, 34) for network encoding. Bonus tiles have no kind_id."""
        assert self.type != TYPE_BONUS
        return self.type * 9 + (self.index - 1)

    @classmethod
    def from_kind_id(cls, kid: int) -> "Tile":
        return cls(kid // 9, (kid % 9) + 1)

    def is_honor(self) -> bool:
        return self.type == TYPE_HONOR

    def is_bonus(self) -> bool:
        return self.type == TYPE_BONUS

    def is_terminal(self) -> bool:
        return self.type < TYPE_HONOR and self.index in (1, 9)

    def is_terminal_or_honor(self) -> bool:
        return self.is_honor() or self.is_terminal()

    def __repr__(self) -> str:
        if self.type == TYPE_HONOR:
            return HONOR_NAMES[self.index]
        if self.type == TYPE_BONUS:
            return BONUS_NAMES[self.index]
        return f"{self.index}{'pmns'[self.type]}"


def full_tile_set():
    """All 144 tiles of a Hong Kong set."""
    tiles = []
    for t in (TYPE_DOT, TYPE_CHAR, TYPE_BAM):
        for idx in range(1, 10):
            tiles += [Tile(t, idx)] * 4
    for idx in range(1, 8):
        tiles += [Tile(TYPE_HONOR, idx)] * 4
    for idx in range(1, 9):
        tiles.append(Tile(TYPE_BONUS, idx))
    return tiles


def counts_from_tiles(tiles):
    """34-length count vector from a list of playable tiles."""
    c = [0] * NUM_KINDS
    for t in tiles:
        c[t.kind_id] += 1
    return c
