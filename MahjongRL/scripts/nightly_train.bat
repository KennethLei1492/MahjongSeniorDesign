@echo off
rem Nightly MahjongRL training slice - run by Windows Task Scheduler.
rem Resumes from checkpoints/latest.pt automatically; logs to checkpoints/.
cd /d "C:\Users\kenne\OneDrive\Documents\MahjongSeniorDesign\MahjongRL"
py -3 scripts\train.py --max-new-games 5000 --checkpoint-every 500 >> checkpoints\nightly_run.log 2>&1
