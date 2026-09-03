"""Flat action space shared by the RL agent and the game engine.

  0..33   discard tile kind k
  34..36  chow (lower/middle/upper: position of the claimed tile in the run)
  37      pung
  38      kong (exposed, concealed, or added - engine disambiguates)
  39      win (mahjong declaration)
  40      pass (decline a claim)

Total: 41 actions. Legality is provided per-step as a boolean mask.
"""
from dataclasses import dataclass
from enum import IntEnum

NUM_ACTIONS = 41
A_DISCARD_BASE = 0
A_CHOW_LOWER, A_CHOW_MIDDLE, A_CHOW_UPPER = 34, 35, 36
A_PUNG, A_KONG, A_WIN, A_PASS = 37, 38, 39, 40


class ActionType(IntEnum):
    DISCARD = 0
    CHOW = 1
    PUNG = 2
    KONG = 3
    WIN = 4
    PASS = 5


@dataclass
class Action:
    type: ActionType
    tile_kind: int = -1    # for DISCARD: which kind to discard
    chow_variant: int = 0  # 0: claimed tile is lowest, 1: middle, 2: highest

    def to_id(self) -> int:
        if self.type == ActionType.DISCARD:
            return A_DISCARD_BASE + self.tile_kind
        if self.type == ActionType.CHOW:
            return A_CHOW_LOWER + self.chow_variant
        return {ActionType.PUNG: A_PUNG, ActionType.KONG: A_KONG,
                ActionType.WIN: A_WIN, ActionType.PASS: A_PASS}[self.type]

    @classmethod
    def from_id(cls, aid: int) -> "Action":
        if aid < 34:
            return cls(ActionType.DISCARD, tile_kind=aid)
        if aid <= A_CHOW_UPPER:
            return cls(ActionType.CHOW, chow_variant=aid - A_CHOW_LOWER)
        return cls({A_PUNG: ActionType.PUNG, A_KONG: ActionType.KONG,
                    A_WIN: ActionType.WIN, A_PASS: ActionType.PASS}[aid])
