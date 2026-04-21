param(
    # Default = everything the 7:30 PM pause scheduler froze:
    #   14708 dowhy_ci worker, 15412 run_all wrapper, 38008 rerun+migrate watcher,
    #   41392 streamlit wrapper, 30352 streamlit server.
    [int[]]$ProcessIds = @(14708, 15412, 38008, 41392, 30352)
)

# Dual-strategy resume.
#
# A plain NtResumeProcess call sometimes leaves threads stuck suspended on
# Windows — typically when a thread's individual suspend count was > 1, or
# when a thread was parked in a wait state at suspend time and its kernel
# bookkeeping drifts. To make resume reliable every time, this script:
#   1. Calls NtResumeProcess (process-level) up to 3x; extra calls past a
#      suspend count of 0 return a non-zero status and are ignored.
#   2. Iterates every thread and calls kernel32!ResumeThread until each
#      thread's suspend count is 0 (hard cap 16 iterations per thread).
#
# Both operations are safe no-ops against a thread/process that is already
# running; this is why the approach is idempotent.

Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class Resumer {
    [DllImport("ntdll.dll", SetLastError = true)]
    public static extern int NtResumeProcess(IntPtr hProcess);
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern IntPtr OpenProcess(uint access, bool inherit, int pid);
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern IntPtr OpenThread(uint access, bool inherit, uint tid);
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint ResumeThread(IntPtr hThread);
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool CloseHandle(IntPtr h);
    public const uint PROCESS_ACCESS = 0x0800 | 0x0400; // SUSPEND_RESUME | QUERY_INFORMATION
    public const uint THREAD_ACCESS  = 0x0002;          // THREAD_SUSPEND_RESUME
}
"@ -ErrorAction SilentlyContinue

$ok = $true
foreach ($pid_ in $ProcessIds) {
    $p = Get-Process -Id $pid_ -ErrorAction SilentlyContinue
    if (-not $p) {
        Write-Host ("[SKIP] PID {0} not running." -f $pid_)
        continue
    }

    # Phase 1: process-level resume (up to 3 calls for inflated suspend counts).
    $h = [Resumer]::OpenProcess([Resumer]::PROCESS_ACCESS, $false, $pid_)
    if ($h -ne [IntPtr]::Zero) {
        for ($i = 0; $i -lt 3; $i++) {
            $rc = [Resumer]::NtResumeProcess($h)
            if ($rc -ne 0) { break }  # non-zero => count already 0, stop.
        }
        [Resumer]::CloseHandle($h) | Out-Null
    } else {
        Write-Host ("[WARN] Could not open process handle for PID {0}" -f $pid_)
    }

    # Phase 2: per-thread ResumeThread loop.
    $threads = $p.Threads
    foreach ($t in $threads) {
        $th = [Resumer]::OpenThread([Resumer]::THREAD_ACCESS, $false, [uint32]$t.Id)
        if ($th -eq [IntPtr]::Zero) { continue }
        for ($k = 0; $k -lt 16; $k++) {
            $prev = [Resumer]::ResumeThread($th)
            if ($prev -eq 0 -or $prev -eq [uint32]0xFFFFFFFF) { break }
        }
        [Resumer]::CloseHandle($th) | Out-Null
    }

    Write-Host ("[RESUMED] PID {0} ({1}) -- threads touched: {2}" -f `
        $pid_, $p.ProcessName, $threads.Count)
}

Write-Host ""
Write-Host "Pipeline and Streamlit resumed. Verify the worker is truly running with:" -ForegroundColor Green
Write-Host '  $p=Get-Process -Id 14708; $a=$p.TotalProcessorTime.TotalSeconds; sleep 5;' -ForegroundColor DarkGray
Write-Host '  $b=(Get-Process -Id 14708).TotalProcessorTime.TotalSeconds; "CPU delta: {0:F2}s" -f ($b-$a)' -ForegroundColor DarkGray
