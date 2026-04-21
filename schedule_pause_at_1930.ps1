param(
    [datetime]$TriggerAt = (Get-Date -Hour 19 -Minute 30 -Second 0 -Millisecond 0),
    # Freeze list:
    #   14708 dowhy_ci worker, 15412 run_all wrapper, 38008 rerun+migrate watcher,
    #   41392 streamlit wrapper, 30352 streamlit server (user requested full freeze).
    [int[]]$ProcessIds  = @(14708, 15412, 38008, 41392, 30352)
)

# --- P/Invoke NtSuspendProcess ------------------------------------------------
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class Proc2 {
    [DllImport("ntdll.dll", SetLastError = true)]
    public static extern int NtSuspendProcess(IntPtr h);
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern IntPtr OpenProcess(uint access, bool inherit, int pid);
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool CloseHandle(IntPtr h);
    public const uint ACCESS = 0x0800 | 0x0400;
}
"@ -ErrorAction SilentlyContinue

$now = Get-Date
if ($TriggerAt -lt $now) {
    # In case the job was queued after the target, fire immediately.
    Write-Host ("[{0}] Trigger time is in the past; pausing immediately." -f $now.ToString("HH:mm:ss"))
} else {
    $sleep = [int]([math]::Round(($TriggerAt - $now).TotalSeconds))
    Write-Host ("[{0}] Scheduler armed. Will pause PIDs {1} at {2} (in {3}s)." -f `
        $now.ToString("HH:mm:ss"), ($ProcessIds -join ','), $TriggerAt.ToString("HH:mm:ss"), $sleep)
    # Sleep in 30s chunks so an observer can see the file is alive.
    while ((Get-Date) -lt $TriggerAt) {
        $remaining = [int][math]::Ceiling(($TriggerAt - (Get-Date)).TotalSeconds)
        if ($remaining -le 0) { break }
        $chunk = [math]::Min($remaining, 30)
        Start-Sleep -Seconds $chunk
    }
}

Write-Host ("`n[{0}] ===== PAUSE TRIGGER FIRED =====" -f (Get-Date).ToString("HH:mm:ss"))

$anyFailed = $false
foreach ($pid_ in $ProcessIds) {
    $p = Get-Process -Id $pid_ -ErrorAction SilentlyContinue
    if (-not $p) {
        Write-Host ("  [SKIP] PID {0} not running (already exited)." -f $pid_)
        continue
    }
    $h = [Proc2]::OpenProcess([Proc2]::ACCESS, $false, $pid_)
    if ($h -eq [IntPtr]::Zero) {
        Write-Host ("  [FAIL] Could not open PID {0} (access denied?)." -f $pid_)
        $anyFailed = $true
        continue
    }
    $rc = [Proc2]::NtSuspendProcess($h)
    [Proc2]::CloseHandle($h) | Out-Null
    if ($rc -eq 0) {
        $ram = [math]::Round($p.WorkingSet64/1MB,0)
        Write-Host ("  [PAUSED] PID {0} ({1}) -- threads={2}, RAM={3} MB" -f `
            $pid_, $p.ProcessName, $p.Threads.Count, $ram)
    } else {
        Write-Host ("  [FAIL] NtSuspendProcess returned {0:X} for PID {1}" -f $rc, $pid_)
        $anyFailed = $true
    }
}

Write-Host ""
if ($anyFailed) {
    Write-Host "One or more PIDs failed to suspend. Check above." -ForegroundColor Red
} else {
    Write-Host "Pipeline AND Streamlit are both frozen." -ForegroundColor Yellow
    Write-Host "Browser connections to http://localhost:8502 will hang until resume." -ForegroundColor Yellow
    Write-Host "Do NOT reboot or hibernate while paused."
    Write-Host "Resume everything with:  .\resume_pipeline.ps1"
}
