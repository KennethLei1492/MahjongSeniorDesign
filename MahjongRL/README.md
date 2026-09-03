# MahjongRL — Hong Kong Mahjong RL Bot for the Hiwonder NexArm

Senior design project: train a deep-RL agent (Res2Net + LSTM, PPO self-play)
to play **Hong Kong–style mahjong**, then deploy it on a **Hiwonder NexArm**
(advanced kit + leader arm) so a robot arm physically draws, discards, and
claims tiles in place of a human player.

- Ruleset: [Hong Kong rules](https://www.mahjong-rules.com/hong-kong-mahjong-rules/) —
  144 tiles with flowers, chow/pung/kong, 4 melds + pair, 3-faan minimum, faan scoring.
- Reference codebase: `../AlphaJong-master` (heuristic Mahjong Soul bot).
  Tile representation and several encoding ideas mirror its conventions.
- Full architecture write-up: [docs/PROJECT_DESCRIPTION.md](docs/PROJECT_DESCRIPTION.md)

## Layout

```
hk_mahjong/   Pure-Python HK rules engine (no dependencies)
  tiles.py      144-tile set, (index,type) representation, 34 kind ids
  wall.py       shuffle, deal 13+14, front/back wall draws
  hand.py       win detection (4 melds + pair, Thirteen Orphans)
  scoring.py    faan table, limit hands, payout math
  game.py       turn/claim state machine exposed as an RL environment
  actions.py    flat 41-action space + legality masks
rl/           Deep learning stack (PyTorch)
  encoder.py    game state -> (18,4,34) planes + event history sequence
  model.py      Res2Net blocks + LSTM + policy/value heads
  self_play.py  4 shared-policy seats play full hands, collect trajectories
  ppo.py        PPO with action masking, checkpoint save/resume
  export.py     TorchScript + ONNX export for the robot host
robot/        Robot deployment (WLKATA Mirobot primary, NexArm legacy)
  mirobot_arm.py      Mirobot G-code/suction-cup backend (current hardware)
  mirobot_teach.py    jog & record table coordinates for the Mirobot
  config.py           NexArm ports/servo IDs (legacy)
  servo_bus.py        raw Feetech STS bus protocol (NexArm legacy)
  arm_controller.py   NexArm taught-pose playback (legacy)
  record_positions.py NexArm leader-arm teaching (legacy)
  vision.py           tile detection/classification from overhead camera
  play_physical.py    live loop: camera -> policy -> arm (--arm mirobot)
scripts/      train.py (500k-game target, resumable), evaluate.py,
              play_vs_bot.py (virtual table: 3 humans vs the bot),
              progress.py (training progress at a glance)
tests/        engine + encoder sanity tests
```

## Quick start

```bash
pip install -r requirements.txt
python tests/test_engine.py          # verify the rules engine
python scripts/train.py --games 500000 --device cuda   # long-running, resumable
python scripts/evaluate.py --checkpoint checkpoints/latest.pt
python -m rl.export --checkpoint checkpoints/latest.pt --out deploy
python -m robot.play_physical --dry-run   # policy loop without hardware
```

## Training to 500k games

`scripts/train.py` checkpoints every 2,000 games and resumes from
`checkpoints/latest.pt`, so the half-million-game target can be split across
any number of sessions (e.g. run it in a loop under Claude Code, on a lab
GPU box, or overnight). Progress metrics land in `checkpoints/train_log.csv`.

## Hardware bring-up order

1. Fill in ports/IDs in `robot/config.py` (marked `VERIFY`).
2. `python -m robot.record_positions` — teach all named poses with the leader arm.
3. Calibrate camera ROIs into `robot/table_calibration.json` and capture tile
   template crops into `robot/assets/templates/`.
4. `python -m robot.play_physical --model deploy/mahjong_policy.ts.pt`

## Safety note

Unlike AlphaJong, nothing here connects to Mahjong Soul or any online
service — training is 100% self-play in our own engine, and deployment is a
physical table. Keep the arm's joint limits conservative until poses are
verified at low speed.
