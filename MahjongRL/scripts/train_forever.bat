@echo off
rem 24/7 MahjongRL training loop.
rem Runs until the 500k-game target is reached; if training crashes for any
rem reason it restarts from the last checkpoint after 30 seconds.
rem Progress: checkpoints\train_log.csv   Output: checkpoints\forever_run.log
cd /d "C:\Users\kenne\OneDrive\Documents\MahjongSeniorDesign\MahjongRL"
:loop
echo [%date% %time%] starting training >> checkpoints\forever_run.log
py -3 -u scripts\train.py >> checkpoints\forever_run.log 2>&1
if %errorlevel%==0 goto done
echo [%date% %time%] training exited with error %errorlevel%, restarting in 30s >> checkpoints\forever_run.log
timeout /t 30 /nobreak >nul
goto loop
:done
echo [%date% %time%] 500k target reached, stopping >> checkpoints\forever_run.log
