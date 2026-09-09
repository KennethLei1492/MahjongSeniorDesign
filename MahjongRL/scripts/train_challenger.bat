@echo off
rem The "MahjongRL Training 24-7" scheduled task points here.
rem ALL TRAINING RUNS COMPLETE as of 2026-09-08:
rem   - win-reward big model: finished at 400k; champion = model_400384.pt
rem     (31.8% vs old champion), deployed as deploy\challenger_win_400k.ts.pt
rem   - score-reward big model: stopped at 200k (negative result - faded to
rem     ~22-25% after a strong 28.7% debut; noisier reward, no better ceiling)
rem To start a new run, replace this file's body with the new trainer command
rem (see train_score.bat for the pattern).
echo [%date% %time%] no active training run - nothing to do
exit /b 0
