@echo off
rem The "MahjongRL Training 24-7" scheduled task points here. The win-reward
rem big-model run finished 2026-09-06 (best checkpoint model_400384.pt,
rem 31.8% vs champion, exported to deploy\challenger_win_400k.ts.pt).
rem Current active run: the SCORE-REWARD model - delegate to its script.
call "C:\Users\kenne\OneDrive\Documents\MahjongSeniorDesign\MahjongRL\scripts\train_score.bat"
