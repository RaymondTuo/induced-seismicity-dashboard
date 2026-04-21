param(
    [int[]]$ProcessIds = @(14708, 15412)
)

# --- P/Invoke NtSuspendProcess from ntdll.dll ---------------------------------
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class ProcControl {
    [DllImport("ntdll.dll", SetLastError = true)]
    public static extern int NtSuspendProcess(IntPtr processHandle);
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern IntPtr OpenProcess(uint access, bool inherit, int pid);
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool CloseHandle(IntPtr handle);
    // PROCESS_ALL_ACCESS minimal subset needed for Nt*Process: SUSPEND_RESUME | QUERY_INFORMATION
    public const uint ACCESS = 0x0800 | 0x0400;
}
"@ -ErrorAction SilentlyContinue

$ok = $true
foreach ($target in $ProcessIds) {
    $proc = Get-Process -Id $target -ErrorAction SilentlyContinue
    if (-not $proc) {
        Write-Host "[WARN] PID $target not running; skipping."
        continue
    }
    $h = [ProcControl]::OpenProcess([ProcControl]::ACCESS, $false, $target)
    if ($h -eq [IntPtr]::Zero) {
        Write-Host "[ERROR] Could not open PID $target (access denied?)."
        $ok = $false
        continue
    }
    $rc = [ProcControl]::NtSuspendProcess($h)
    [ProcControl]::CloseHandle($h) | Out-Null
    if ($rc -eq 0) {
        Write-Host "[PAUSED] PID $target ($($proc.ProcessName)) — threads=$($proc.Threads.Count), RAM=$([math]::Round($proc.WorkingSet64/1MB,0)) MB"
    } else {
        Write-Host "[ERROR] NtSuspendProcess returned 0x{0:X} for PID $target" -f $rc
        $ok = $false
    }
}

if ($ok) {
    Write-Host ""
    Write-Host "Pipeline frozen in place. RAM is still held. Do NOT reboot or hibernate." -ForegroundColor Yellow
    Write-Host "Run .\resume_pipeline.ps1 to continue."
} else {
    Write-Host ""
    Write-Host "One or more PIDs failed to suspend. Check errors above." -ForegroundColor Red
    exit 1
}
