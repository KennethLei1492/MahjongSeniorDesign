"""Hong Kong Mahjong game engine.

Event-driven engine designed for RL self-play: at every decision point it
exposes (current player, state, legal-action mask) and advances with
step(action_id). Claim priority follows HK rules: Win > Pung/Kong > Chow;
chow may only be claimed by the next player counterclockwise.
"""
import random
from dataclasses import dataclass, field
from .tiles import Tile, TYPE_BONUS, counts_from_tiles
from .wall import Wall
from .hand import is_winning
from .scoring import score_hand, payout, LIMIT_FAAN
from .actions import (Action, ActionType, NUM_ACTIONS, A_DISCARD_BASE,
                      A_CHOW_LOWER, A_PUNG, A_KONG, A_WIN, A_PASS)


@dataclass
class Meld:
    kind: str               # "chow" | "pung" | "kong"
    tiles: list             # Tile objects, base tile first for chow
    claimed_from: int = -1  # seat discarded from; -1 = concealed/self kong
    concealed: bool = False


@dataclass
class GameConfig:
    min_faan: int = 3        # minimum faan to declare mahjong (house rule 1-4)
    base_points: float = 1.0
    seed: int = None


@dataclass
class PlayerState:
    hand: list = field(default_factory=list)     # concealed Tile list
    melds: list = field(default_factory=list)    # claimed Melds
    flowers: list = field(default_factory=list)  # bonus tiles set aside
    discards: list = field(default_factory=list)

    def counts(self):
        return counts_from_tiles(self.hand)


class Game:
    """One full hand of HK mahjong. Phases:
    'discard'  - current player holds a 14th tile: discard, self-kong or win
    'claim'    - a discard is on the table; others respond in seat order
    'finished' - hand is over (win, or wall exhausted -> draw)
    """

    def __init__(self, config: GameConfig = None, dealer: int = 0):
        self.cfg = config or GameConfig()
        self.rng = random.Random(self.cfg.seed)
        self.wall = Wall(self.rng)
        self.dealer = dealer
        self.round_wind = 1  # East round (single-hand episodes)
        self.players = [PlayerState() for _ in range(4)]
        self.history = []    # (seat, action_id) pairs, consumed by the LSTM
        self.phase = "discard"
        self.turn = dealer
        self.last_discard = None
        self.last_discarder = -1
        self.claim_queue = []     # seats yet to respond to the discard
        self.pending_claims = {}  # seat -> chosen claim Action
        self.result = None
        self._deal()

    # ---------- setup ----------
    def _deal(self):
        hands = self.wall.deal(dealer=self.dealer)
        for seat, hand in enumerate(hands):
            for t in hand:
                self._take_tile(seat, t)

    def _take_tile(self, seat, tile):
        """Add a tile to the hand; auto-replace bonus tiles from the back wall."""
        while tile is not None and tile.type == TYPE_BONUS:
            self.players[seat].flowers.append(tile)
            tile = self.wall.draw_replacement()
        if tile is not None:
            self.players[seat].hand.append(tile)
            self.players[seat].hand.sort()

    def seat_wind(self, seat):
        return ((seat - self.dealer) % 4) + 1

    # ---------- decision points ----------
    def current_player(self):
        if self.phase == "discard":
            return self.turn
        if self.phase == "claim":
            return self.claim_queue[0]
        return -1

    def legal_actions(self):
        """Boolean mask of length NUM_ACTIONS for the current player."""
        mask = [False] * NUM_ACTIONS
        if self.phase == "finished":
            return mask
        seat = self.current_player()
        p = self.players[seat]

        if self.phase == "discard":
            for k, c in enumerate(p.counts()):
                if c > 0:
                    mask[A_DISCARD_BASE + k] = True
            if self.wall.remaining() > 0 and self._can_self_kong(p):
                mask[A_KONG] = True
            if self._can_win(seat, None):
                mask[A_WIN] = True
        else:  # claim phase
            mask[A_PASS] = True
            t = self.last_discard
            c = p.counts()
            if self._can_win(seat, t):
                mask[A_WIN] = True
            if c[t.kind_id] >= 2:
                mask[A_PUNG] = True
            if c[t.kind_id] >= 3 and self.wall.remaining() > 0:
                mask[A_KONG] = True
            if seat == (self.last_discarder + 1) % 4 and t.kind_id < 27:
                for variant in range(3):  # claimed tile = low/mid/high of run
                    base = t.kind_id - variant
                    needed = [base + i for i in range(3) if base + i != t.kind_id]
                    if (base >= 0 and (base + 2) < 27
                            and base // 9 == (base + 2) // 9
                            and all(c[k] > 0 for k in needed)):
                        mask[A_CHOW_LOWER + variant] = True
        return mask

    def _can_self_kong(self, p):
        c = p.counts()
        if any(x == 4 for x in c):
            return True
        return any(m.kind == "pung" and c[m.tiles[0].kind_id] >= 1
                   for m in p.melds)

    def _can_win(self, seat, extra_tile):
        p = self.players[seat]
        counts = p.counts()
        if extra_tile is not None:
            counts[extra_tile.kind_id] += 1
        if not is_winning(counts, len(p.melds)):
            return False
        faan, _ = self._score(seat, extra_tile)
        return faan >= self.cfg.min_faan

    def _score(self, seat, extra_tile):
        p = self.players[seat]
        counts = p.counts()
        if extra_tile is not None:
            counts[extra_tile.kind_id] += 1
        return score_hand(
            counts, p.melds, extra_tile,
            self_draw=extra_tile is None,
            seat_wind=self.seat_wind(seat), round_wind=self.round_wind,
            num_flowers=len(p.flowers),
            is_concealed=all(m.concealed for m in p.melds),
            last_tile=self.wall.remaining() == 0)

    # ---------- stepping ----------
    def step(self, action_id: int):
        """Advance the game with the current player's action."""
        assert self.legal_actions()[action_id], f"illegal action {action_id}"
        seat = self.current_player()
        self.history.append((seat, action_id))
        a = Action.from_id(action_id)
        if self.phase == "discard":
            self._step_discard_phase(seat, a)
        else:
            self._step_claim_phase(seat, a)
        return self.phase

    def _step_discard_phase(self, seat, a):
        p = self.players[seat]
        if a.type == ActionType.WIN:
            self._finish_win(seat, extra_tile=None)
            return
        if a.type == ActionType.KONG:
            self._do_self_kong(seat)
            return
        tile = next(t for t in p.hand if t.kind_id == a.tile_kind)
        p.hand.remove(tile)
        p.discards.append(tile)
        self.last_discard, self.last_discarder = tile, seat
        self.claim_queue = [(seat + i) % 4 for i in (1, 2, 3)]
        self.pending_claims = {}
        self.phase = "claim"

    def _step_claim_phase(self, seat, a):
        self.pending_claims[seat] = a
        self.claim_queue.pop(0)
        if self.claim_queue:
            return
        # everyone answered: resolve by priority Win > Pung/Kong > Chow.
        prio = {ActionType.WIN: 3, ActionType.KONG: 2, ActionType.PUNG: 2,
                ActionType.CHOW: 1, ActionType.PASS: 0}
        best_seat, best_a, best_p = -1, None, 0
        for i in (1, 2, 3):  # seat order breaks win ties (closest to discarder)
            s = (self.last_discarder + i) % 4
            act = self.pending_claims.get(s)
            if act is not None and prio[act.type] > best_p:
                best_seat, best_a, best_p = s, act, prio[act.type]
        if best_p == 0:
            self._next_turn()
            return
        t = self.last_discard
        if best_a.type == ActionType.WIN:
            self._finish_win(best_seat, extra_tile=t)
            return
        self.players[self.last_discarder].discards.pop()  # tile is taken
        p = self.players[best_seat]
        if best_a.type == ActionType.PUNG:
            self._remove_kind(p, t.kind_id, 2)
            p.melds.append(Meld("pung", [t] * 3, self.last_discarder))
        elif best_a.type == ActionType.KONG:
            self._remove_kind(p, t.kind_id, 3)
            p.melds.append(Meld("kong", [t] * 4, self.last_discarder))
            self._take_tile(best_seat, self.wall.draw_replacement())
        else:  # chow
            base = t.kind_id - best_a.chow_variant
            run = [Tile.from_kind_id(base + i) for i in range(3)]
            for rt in run:
                if rt.kind_id != t.kind_id:
                    self._remove_kind(p, rt.kind_id, 1)
            p.melds.append(Meld("chow", run, self.last_discarder))
        self.turn = best_seat
        self.phase = "discard"  # claimer must now discard

    def _remove_kind(self, p, kid, n):
        for _ in range(n):
            tile = next(t for t in p.hand if t.kind_id == kid)
            p.hand.remove(tile)

    def _do_self_kong(self, seat):
        p = self.players[seat]
        c = p.counts()
        kid = next((k for k, x in enumerate(c) if x == 4), None)
        if kid is not None:  # concealed kong
            self._remove_kind(p, kid, 4)
            p.melds.append(Meld("kong", [Tile.from_kind_id(kid)] * 4,
                                -1, concealed=True))
        else:  # added kong on an existing pung
            m = next(m for m in p.melds
                     if m.kind == "pung" and c[m.tiles[0].kind_id] >= 1)
            self._remove_kind(p, m.tiles[0].kind_id, 1)
            m.kind = "kong"
            m.tiles.append(m.tiles[0])
        self._take_tile(seat, self.wall.draw_replacement())
        if self.wall.remaining() <= 0 and len(p.hand) % 3 != 2:
            self._finish_draw()

    def _next_turn(self):
        self.turn = (self.last_discarder + 1) % 4
        tile = self.wall.draw()
        if tile is None:
            self._finish_draw()
            return
        self._take_tile(self.turn, tile)
        if len(self.players[self.turn].hand) % 3 != 2:  # bonus replacements ran dry
            self._finish_draw()
            return
        self.phase = "discard"

    # ---------- ending ----------
    def _finish_win(self, seat, extra_tile):
        faan, elements = self._score(seat, extra_tile)
        faan = min(faan, LIMIT_FAAN)
        self_draw = extra_tile is None
        pay = payout(faan, self_draw=self_draw, base=self.cfg.base_points)
        scores = [0.0] * 4
        if self_draw:
            for s in range(4):
                if s != seat:
                    scores[s] -= pay["all"]
                    scores[seat] += pay["all"]
        else:
            for s in range(4):
                if s == seat:
                    continue
                amt = pay["discarder"] if s == self.last_discarder else pay["others"]
                scores[s] -= amt
                scores[seat] += amt
        self.result = {"type": "win", "winner": seat, "faan": faan,
                       "elements": elements, "self_draw": self_draw,
                       "scores": scores}
        self.phase = "finished"

    def _finish_draw(self):
        self.result = {"type": "draw", "winner": -1, "faan": 0,
                       "elements": [], "self_draw": False,
                       "scores": [0.0] * 4}
        self.phase = "finished"
