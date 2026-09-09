@echo off
rem LEAGUE FINE-TUNE: continue the win-reward champion (model_400384, 31.8%%
rem vs the original 500k champion) with league play - half the games seat a
rem frozen pool opponent on 3 seats vs 1 learner seat, so strategy drift
rem that pure self-play cannot see gets punished during training.
rem Pool: the original small 500k champion + the big model's own 250k and
rem 400k snapshots. Lives in checkpoints_league\ (fine-tune of the win model,
rem so game counts continue from 400384).
rem Do NOT run at the same time as another trainer.
cd /d "C:\Users\kenne\OneDrive\Documents\MahjongSeniorDesign\MahjongRL"
if not exist checkpoints_league mkdir checkpoints_league
:loop
echo [%date% %time%] starting league fine-tune >> checkpoints_league\forever_run.log
py -3 -u scripts\train.py --ckpt-dir checkpoints_league --resume checkpoints_league\latest.pt --channels 128 --blocks 6 --lstm-hidden 256 --workers 6 --opponent-pool "checkpoints\model_500096_champion.pt,checkpoints_big\model_250368.pt,checkpoints_big\model_400384_champion.pt" --league-prob 0.5 >> checkpoints_league\forever_run.log 2>&1
if %errorlevel%==0 goto done
echo [%date% %time%] crashed with error %errorlevel%, restarting in 30s >> checkpoints_league\forever_run.log
timeout /t 30 /nobreak >nul
goto loop
:done
echo [%date% %time%] target reached >> checkpoints_league\forever_run.log
