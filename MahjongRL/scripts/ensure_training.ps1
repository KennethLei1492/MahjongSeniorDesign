# Watchdog: restart challenger training if the learner has died.
# The learner occasionally segfaults (0xc0000005 in VCRUNTIME140.dll) and the
# crash takes the whole console tree with it, so the .bat retry loop never
# fires. This script runs every 15 min from Task Scheduler: if no train.py
# process is alive it kills orphaned multiprocessing workers and restarts
# training via the "MahjongRL Training 24-7" task.
$log = 'C:\Users\kenne\OneDrive\Documents\MahjongSeniorDesign\MahjongRL\checkpoints_big\watchdog.log'
$runlog = 'C:\Users\kenne\OneDrive\Documents\MahjongSeniorDesign\MahjongRL\checkpoints_score\forever_run.log'
# -OperationTimeoutSec: WMI goes sluggish when the CPU is thermally
# throttled; without a timeout the whole watchdog run can wedge here.
$procs = Get-CimInstance Win32_Process -Filter "Name='python.exe' or Name='py.exe'" -ErrorAction SilentlyContinue -OperationTimeoutSec 60
$learner = $procs | Where-Object { $_.CommandLine -match 'train\.py' }

# Task Scheduler starts tasks at BelowNormal priority; when other apps are
# active this starves the learner, its result queue balloons (multi-GB), the
# system pages, and training collapses. Keep every training process at Normal.
foreach ($p in $procs) {
    try { (Get-Process -Id $p.ProcessId -ErrorAction Stop).PriorityClass = 'Normal' } catch {}
}

# Restart ONLY for the pathology a restart actually cures: the learner's
# result-queue ballooning into multi-GB memory (paging death spiral, seen
# 2026-09-03). Slow s/game alone is usually thermal throttling - restarting
# for that just burns progress in a kill-loop (learned 2026-09-07).
if ($learner) {
    $memGB = [math]::Round((Get-Process -Id $learner.ProcessId -ErrorAction SilentlyContinue).WorkingSet64 / 1GB, 2)
    # 6GB, not lower: the score-reward learner runs at ~4.5GB in normal
    # healthy operation; the Sept-3 death spiral was 7GB+ and climbing.
    if ($memGB -gt 6) {
        foreach ($p in $procs) {
            try { Stop-Process -Id $p.ProcessId -Force -Confirm:$false -ErrorAction Stop } catch {}
        }
        "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] learner memory balloon (${memGB}GB) - killed training for restart" | Add-Content $log
        $learner = $null
        Start-Sleep 5
        $procs = @()
    }
}

if (-not $learner) {
    $orphans = $procs | Where-Object { $_.CommandLine -match 'multiprocessing' }
    foreach ($o in $orphans) {
        try { Stop-Process -Id $o.ProcessId -Force -Confirm:$false -ErrorAction Stop } catch {}
    }
    Start-ScheduledTask -TaskName 'MahjongRL Training 24-7'
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] learner dead - killed $(@($orphans).Count) orphan worker(s), restarted training" | Add-Content $log
}
