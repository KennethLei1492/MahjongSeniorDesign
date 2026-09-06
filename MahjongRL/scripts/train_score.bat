@echo off
rem SCORE-REWARD challenger: same big Res2Net (128ch x 6 blocks x lstm256)
rem but trained on faan-based zero-sum payouts (--reward score) instead of
rem win-only +1/0. Lives entirely in checkpoints_score\ so it is a separate
rem model from the win-reward one in checkpoints_big\ - either can be
rem exported and loaded onto the arm independently.
rem Do NOT run at the same time as another trainer (they fight over CPU).
cd /d "C:\Users\kenne\OneDrive\Documents\MahjongSeniorDesign\MahjongRL"
if not exist checkpoints_score mkdir checkpoints_score
:loop
echo [%date% %time%] starting score-reward training >> checkpoints_score\forever_run.log
py -3 -u scripts\train.py --ckpt-dir checkpoints_score --channels 128 --blocks 6 --lstm-hidden 256 --reward score >> checkpoints_score\forever_run.log 2>&1
if %errorlevel%==0 goto done
echo [%date% %time%] crashed with error %errorlevel%, restarting in 30s >> checkpoints_score\forever_run.log
timeout /t 30 /nobreak >nul
goto loop
:done
echo [%date% %time%] target reached >> checkpoints_score\forever_run.log
