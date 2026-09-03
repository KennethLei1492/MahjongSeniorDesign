"""Wall building and dealing (counterclockwise, dealer gets 14)."""
import random
from .tiles import full_tile_set


class Wall:
    def __init__(self, rng: random.Random = None):
        self.rng = rng or random.Random()
        self.tiles = full_tile_set()
        self.rng.shuffle(self.tiles)
        self._front = 0           # normal draws from the front
        self._back = len(self.tiles)  # kong/bonus replacement draws from the back

    def remaining(self) -> int:
        return self._back - self._front

    def draw(self):
        if self.remaining() <= 0:
            return None
        t = self.tiles[self._front]
        self._front += 1
        return t

    def draw_replacement(self):
        """Draw from the back of the wall after a kong or bonus tile."""
        if self.remaining() <= 0:
            return None
        self._back -= 1
        return self.tiles[self._back]

    def deal(self, num_players: int = 4, dealer: int = 0):
        """13 tiles each, 14th to the dealer. Returns list of hands."""
        hands = [[] for _ in range(num_players)]
        for _ in range(13):
            for p in range(num_players):
                seat = (dealer + p) % num_players
                hands[seat].append(self.draw())
        hands[dealer].append(self.draw())
        return hands
