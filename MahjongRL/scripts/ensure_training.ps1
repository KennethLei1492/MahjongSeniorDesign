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

# The learner leaks memory over ~12h runs; once the system starts paging,
# games slow from ~0.6s to 3s+. A restart (resumes from latest.pt, loses
# <=500 games) restores full speed - kill the tree and let the .bat loop
# or the dead-learner branch below bring it back.
if ($learner) {
    # Only treat sustained slowness as degradation: the last two iteration
    # lines both slow, and at least 3 iterations since the last restart
    # (the first iteration after a restart is always warmup-slow).
    $tail = Get-Content $runlog -Tail 30 -ErrorAction SilentlyContinue
    $lastStart = ($tail | Select-String 'starting challenger training' | Select-Object -Last 1).LineNumber
    if (-not $lastStart) { $lastStart = 0 }
    $iters = @($tail | Select-Object -Skip $lastStart | Where-Object { $_ -match 's/game' })
    $speeds = @($iters | ForEach-Object { if ($_ -match '([\d.]+)s/game') { [double]$Matches[1] } })
    if ($speeds.Count -ge 3 -and $speeds[-1] -gt 2.5 -and $speeds[-2] -gt 2.5) {
        $Matches = @{ 1 = $speeds[-1] }
        foreach ($p in $procs) {
            try { Stop-Process -Id $p.ProcessId -Force -Confirm:$false -ErrorAction Stop } catch {}
        }
        "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] slowdown ($($Matches[1])s/game, likely memory leak) - killed training for restart" | Add-Content $log
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
