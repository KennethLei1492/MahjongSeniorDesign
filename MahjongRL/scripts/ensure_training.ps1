# Watchdog: restart challenger training if the learner has died.
# The learner occasionally segfaults (0xc0000005 in VCRUNTIME140.dll) and the
# crash takes the whole console tree with it, so the .bat retry loop never
# fires. This script runs every 15 min from Task Scheduler: if no train.py
# process is alive it kills orphaned multiprocessing workers and restarts
# training via the "MahjongRL Training 24-7" task.
$log = 'C:\Users\kenne\OneDrive\Documents\MahjongSeniorDesign\MahjongRL\checkpoints_big\watchdog.log'
$procs = Get-CimInstance Win32_Process -Filter "Name='python.exe' or Name='py.exe'" -ErrorAction SilentlyContinue
$learner = $procs | Where-Object { $_.CommandLine -match 'train\.py' }
if (-not $learner) {
    $orphans = $procs | Where-Object { $_.CommandLine -match 'multiprocessing' }
    foreach ($o in $orphans) {
        try { Stop-Process -Id $o.ProcessId -Force -Confirm:$false -ErrorAction Stop } catch {}
    }
    Start-ScheduledTask -TaskName 'MahjongRL Training 24-7'
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] learner dead - killed $(@($orphans).Count) orphan worker(s), restarted training" | Add-Content $log
}
