@echo off
rem FAAN-SCORED fine-tune: warm-starts from the frozen race champion and
rem retrains under traditional HK rules (3-faan minimum, point payouts).
rem Transfer-learning experiment - expect metrics to crash then recover.
rem Do NOT run at the same time as any other trainer.
cd /d "C:\Users\kenne\OneDrive\Documents\MahjongSeniorDesign\MahjongRL"
if not exist checkpoints_faan\latest.pt (
  echo [%date% %time%] warm-starting from race champion >> checkpoints_faan\forever_run.log
  copy /y checkpoints\latest.pt checkpoints_faan\latest.pt >nul
)
:loop
echo [%date% %time%] starting faan fine-tune >> checkpoints_faan\forever_run.log
py -3 -u scripts\train.py --ckpt-dir checkpoints_faan --min-faan 3 --reward score >> checkpoints_faan\forever_run.log 2>&1
if %errorlevel%==0 goto done
echo [%date% %time%] crashed with error %errorlevel%, restarting in 30s >> checkpoints_faan\forever_run.log
timeout /t 30 /nobreak >nul
goto loop
:done
echo [%date% %time%] target reached >> checkpoints_faan\forever_run.log
