@echo off
rem CHALLENGER training: larger Res2Net (128ch x 6 blocks) in its own
rem checkpoint directory. Start this AFTER the main run reaches 500k -
rem do NOT run both trainers at the same time (they would fight over CPU).
cd /d "C:\Users\kenne\OneDrive\Documents\MahjongSeniorDesign\MahjongRL"
:loop
echo [%date% %time%] starting challenger training >> checkpoints_big\forever_run.log
py -3 -u scripts\train.py --ckpt-dir checkpoints_big --channels 128 --blocks 6 --lstm-hidden 256 >> checkpoints_big\forever_run.log 2>&1
if %errorlevel%==0 goto done
echo [%date% %time%] crashed with error %errorlevel%, restarting in 30s >> checkpoints_big\forever_run.log
timeout /t 30 /nobreak >nul
goto loop
:done
echo [%date% %time%] target reached >> checkpoints_big\forever_run.log
