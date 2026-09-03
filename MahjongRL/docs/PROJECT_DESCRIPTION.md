# MahjongRL: A Self-Taught Hong Kong Mahjong Robot

**Senior Design Project Description** · ~2.5 pages

## 1. Objective

Build an autonomous system that plays Hong Kong–style mahjong at a physical
table in place of a human player. The system has three parts: (1) a faithful
Hong Kong mahjong rules engine that doubles as a reinforcement-learning
environment, (2) a deep neural network — a **Res2Net** convolutional backbone
fused with an **LSTM** — trained entirely by **self-play reinforcement
learning** over roughly **500,000 games**, and (3) a deployment layer that
runs the trained policy on a host computer driving a **Hiwonder NexArm**
robotic arm (advanced kit with leader arm) which physically draws, discards,
and claims tiles. The open-source AlphaJong bot (a heuristic JavaScript
Mahjong Soul bot) served as the design reference: its tile representation,
visible-tile bookkeeping, and defense signals informed our state encoding,
but where AlphaJong hand-codes its evaluation, we *learn* it.

## 2. Rules Engine — `hk_mahjong/` (task: game logic & RL environment)

The engine implements the ruleset at mahjong-rules.com/hong-kong-mahjong-rules:
144 tiles (three suits 1–9, four winds, three dragons, eight flower/season
bonus tiles), counterclockwise turns, and the claims **chow** (next player
only), **pung**, and **kong**. A winning hand is 4 melds + 1 pair — or
Thirteen Orphans — and must reach the **3-faan minimum** to declare mahjong.

- `tiles.py` — tiles are `(index, type)` pairs following AlphaJong's
  convention, with a flat *kind id* in [0, 34) for the network. Bonus tiles
  are type 4 and never held in hand.
- `wall.py` — shuffles all 144 tiles; normal draws come from the front,
  kong/flower replacement draws from the back; deals 13 tiles each + 14th to
  the dealer.
- `hand.py` — win detection by backtracking decomposition of the 34-kind
  count vector into melds + pair (exhaustive, since HK hands are only checked
  at win time), plus the Thirteen Orphans special case.
- `scoring.py` — the faan table (Self Draw, wind/dragon pungs, Common Hand,
  flowers, All Pungs 3, Half Flush 3, Full Flush 6, and limit hands at 13
  faan) and the payout rule: value = 2^faan; on a discard win the discarder
  pays full and the others half, on self-draw all three pay full — zero-sum
  by construction, which is exactly what the RL reward needs.
- `game.py` + `actions.py` — a turn/claim state machine exposed as an RL
  environment. Every decision point yields (current player, observation,
  **legal-action mask**) over a flat 41-action space (34 discards, 3 chow
  variants, pung, kong, win, pass), and claim conflicts resolve by HK
  priority: Win > Pung/Kong > Chow.

The engine is dependency-free and fast (~thousands of games/minute/core),
which is what makes half a million training games tractable.

## 3. Learning System — `rl/` (task: deep learning & training)

**State encoding** (`encoder.py`). Each observation is an 18-plane 4×34
"image": thermometer-encoded counts of our hand, each player's melds and
discards (in relative seat order, making the network seat-equivariant), all
visible tiles for wall-counting (AlphaJong's `availableTiles` idea as a
learned input), the claimable tile, wall depletion, seat wind, and phase.
Alongside it, the last 32 game events (who, which action) form a sequence.

**Network** (`model.py`). The spatial planes pass through a stem convolution
and six **Res2Net blocks**: inside each residual bottleneck the channels are
split into 4 hierarchical sub-groups where each group's 3×3 convolution
receives the previous group's output — multi-scale receptive fields in a
single block (Gao et al., 2021). On a mahjong grid this simultaneously
captures adjacent-tile chow shapes and whole-suit flush structure. The event
sequence feeds a 256-unit **LSTM**, giving the policy memory of turn order
and opponents' discard tempo — the signals AlphaJong's `ai_defense.js` reads
with hand-written heuristics (tenpai detection, suji, danger). The pooled
Res2Net features and final LSTM state are concatenated into two heads: a
**policy** over the 41 actions (illegal actions masked to −∞) and a **value**
estimate of the hand's final normalized score.

**Training** (`self_play.py`, `ppo.py`, `scripts/train.py`). Four copies of
the *same* network play full hands against each other ("4 mahjong bots
against each other"); every game yields four trajectories. Rewards are the
zero-sum end-of-hand payouts normalized by the limit-hand value, plus a
small win bonus so the early random policy discovers that winning exists at
all. Updates use **PPO** (clipped surrogate, entropy regularization, action
masking, advantage normalization). Training is checkpointed with optimizer
state and a games-played counter, so the **500,000-game target is resumable
across sessions** — the intended workflow is to let Claude Code (or a lab
machine) repeatedly run `scripts/train.py`, which logs win rate, average
faan, and losses to `checkpoints/train_log.csv` after every 128-game
iteration. At ~70 decisions per seat per hand, 500k games ≈ 1.4×10⁸
training transitions. `scripts/evaluate.py` benchmarks any checkpoint
against random-legal opponents; `rl/export.py` freezes the result to
TorchScript and ONNX for deployment.

## 4. Robot Deployment — `robot/` (task: hardware integration)

The NexArm is a LeRobot-compatible arm built on Feetech STS-series serial
bus servos with a K230 vision module; the advanced kit's **leader arm** is a
force-free twin used for teaching by demonstration. We exploit exactly that:

- `record_positions.py` — torque-off the leader arm, physically pose it over
  the wall / each of the 14 hand-rack slots / the discard and meld areas,
  and capture each named pose to `positions.json`. **No inverse kinematics
  needed** — all motion is taught-pose playback.
- `servo_bus.py` / `arm_controller.py` — a minimal implementation of the
  Feetech half-duplex packet protocol over USB serial (with the `lerobot`
  package as the preferred backend when installed), wrapped in mahjong
  motion primitives: `draw_tile()`, `discard_tile(slot)`, `claim_tile()`,
  `lay_meld()`, and a gap-closing hand-sort. Ports, servo IDs, and joint
  limits live in `config.py` and are verified against the shipped manual
  during bring-up.
- `vision.py` — the overhead camera (K230 or any USB cam) finds tile-shaped
  contours inside calibrated regions (hand rack, discard pool, wall) and
  classifies faces by template matching against reference crops, with a CNN
  classifier as the upgrade path (trainable on K230 via nncase).
- `play_physical.py` — the live loop. A **GameMirror** tracks the physical
  game inside the same `hk_mahjong` structures used in training, so the
  exported policy sees exactly its training distribution: camera reads our
  hand (drift-proof — re-read every turn), opponents' discards enter the
  mirror, the policy picks an action, and the arm executes the matching
  primitive. Human claims and corrections are confirmed by an operator at
  the keyboard in the first milestone; full multi-camera table tracking is
  the stretch goal.

## 5. Task Map & Verification

| Task | Where | Status |
|---|---|---|
| HK rules, scoring, payouts | `hk_mahjong/` | Done, tested (`tests/test_engine.py`: win shapes, Thirteen Orphans, Full Flush faan, 300+ random games terminate legally, payouts zero-sum) |
| RL environment + action masks | `hk_mahjong/game.py`, `actions.py` | Done, tested |
| State encoding | `rl/encoder.py` | Done, shape-tested |
| Res2Net + LSTM network | `rl/model.py` | Done |
| PPO self-play to 500k games | `rl/self_play.py`, `rl/ppo.py`, `scripts/train.py` | Code done; training compute is the next milestone |
| Model export for the arm host | `rl/export.py` | Done |
| Arm control + leader-arm teaching | `robot/servo_bus.py`, `arm_controller.py`, `record_positions.py` | Code done; needs hardware bring-up |
| Tile vision | `robot/vision.py` | Pipeline done; needs per-table calibration + templates |
| Physical play loop | `robot/play_physical.py` | Done (`--dry-run` works today) |

Everything trains offline in our own simulator — unlike the AlphaJong
reference, no online game service is involved at any point.
