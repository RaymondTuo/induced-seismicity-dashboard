param(
    [int]$ValidationPid = 36872,
    [string]$ValidationLog = "safe_parallel_first_run.log",
    [int[]]$StreamlitPids = @(44072, 4648),
    [int]$StreamlitPort = 8502,
    [string]$LogDir = "$PSScriptRoot"
)

# Orchestrator: wait for safe_parallel_first_run.py to finish, verify it
# passed, then run the full pipeline and restart Streamlit.
# Runs unattended; every stage is wrapped in try/catch.

$ErrorActionPreference = "Continue"
$stamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$log = Join-Path $LogDir "full_pipeline_chain_$stamp.log"
$marker = Join-Path $LogDir "FULL_PIPELINE_DONE.txt"
$failMarker = Join-Path $LogDir "FULL_PIPELINE_FAILED.txt"

function Write-Log($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Write-Host $line
    Add-Content -Path $log -Value $line -ErrorAction SilentlyContinue
}

function Invoke-Stage {
    param([string]$Name, [scriptblock]$Action)
    Write-Log ""
    Write-Log "=========================================================="
    Write-Log "STAGE: $Name"
    Write-Log "=========================================================="
    $t0 = Get-Date
    $ok = $true
    try { & $Action } catch {
        $ok = $false
        Write-Log ("STAGE FAILED ({0}): {1}" -f $Name, $_.Exception.Message)
    }
    $elapsed = [int]((Get-Date) - $t0).TotalSeconds
    $tag = if ($ok) { 'OK' } else { 'FAIL' }
    Write-Log ("STAGE {0}: {1} ({2}s)" -f $tag, $Name, $elapsed)
    return $ok
}

Write-Log "Orchestrator armed. Waiting for safe validation PID $ValidationPid to finish..."

try {
    Wait-Process -Id $ValidationPid -ErrorAction Stop
    Write-Log "Safe validation process $ValidationPid has exited."
} catch {
    Write-Log "PID $ValidationPid not found (already exited). Proceeding immediately."
}

Start-Sleep -Seconds 5

# -------------------------------------------------------------------------
# Gate 1: Did the safe validation actually pass?
# -------------------------------------------------------------------------
$validationPassed = $false
$logPath = Join-Path $PSScriptRoot $ValidationLog
if (Test-Path $logPath) {
    $content = Get-Content -Raw -Path $logPath
    $hasPass = $content -match "Parallel path works end-to-end"
    $hasFail = $content -match "Validation FAILED"
    if ($hasPass -and -not $hasFail) {
        $validationPassed = $true
        Write-Log "Validation log shows PASS."
    } elseif ($hasFail) {
        Write-Log "Validation log shows FAIL. Aborting full pipeline."
    } else {
        Write-Log "Validation log inconclusive. Aborting full pipeline as a precaution."
    }
} else {
    Write-Log "Validation log not found at $logPath. Aborting as a precaution."
}

if (-not $validationPassed) {
    Set-Content -Path $failMarker -Encoding UTF8 -Value @(
        "FULL PIPELINE SKIPPED",
        "=====================",
        "At:              $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')",
        "Reason:          safe validation did not pass",
        "Validation log:  $logPath",
        "Chain log:       $log"
    )
    Write-Log "Wrote failure marker: $failMarker"
    exit 1
}

$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Log "FATAL: .venv python not found at $py. Aborting."
    exit 1
}

# UTF-8 for the Python subprocesses — same emoji/encoding concern as the
# simple-dowhy watcher.
$env:PYTHONIOENCODING = "utf-8"

# -------------------------------------------------------------------------
# STAGE 1: full pipeline (run_all.py). Logs inline to chain log.
# -------------------------------------------------------------------------
$results = [ordered]@{}

$results["run_all"] = Invoke-Stage "Full pipeline: run_all.py" {
    $s = Join-Path $PSScriptRoot "run_all.py"
    if (-not (Test-Path $s)) { throw "run_all.py not found" }
    $jobsDisplay = if ($env:DOWHY_CI_JOBS) { $env:DOWHY_CI_JOBS } else { '(default: min(8, cpu-1))' }
    Write-Log "  DOWHY_CI_JOBS = $jobsDisplay"
    & $py $s 2>&1 | Tee-Object -FilePath $log -Append | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "run_all.py exit code $LASTEXITCODE" }
}

# -------------------------------------------------------------------------
# STAGE 2: Streamlit restart regardless of pipeline pass/fail, so whatever
# the pipeline did produce gets picked up and cache_data is invalidated.
# -------------------------------------------------------------------------
$results["streamlit_restart"] = Invoke-Stage "Streamlit restart (port $StreamlitPort)" {
    foreach ($sp in $StreamlitPids) {
        $p = Get-Process -Id $sp -ErrorAction SilentlyContinue
        if ($p) {
            Write-Log ("  stopping streamlit PID {0}" -f $sp)
            Stop-Process -Id $sp -Force -ErrorAction SilentlyContinue
        } else {
            Write-Log ("  streamlit PID {0} already gone" -f $sp)
        }
    }
    Start-Sleep -Seconds 3

    try {
        $listeners = Get-NetTCPConnection -LocalPort $StreamlitPort -State Listen -ErrorAction SilentlyContinue
        foreach ($l in $listeners) {
            Write-Log ("  port {0} still bound by PID {1}; stopping it" -f $StreamlitPort, $l.OwningProcess)
            Stop-Process -Id $l.OwningProcess -Force -ErrorAction SilentlyContinue
        }
    } catch { }
    Start-Sleep -Seconds 2

    $stdout = Join-Path $LogDir "streamlit_stdout.log"
    $stderr = Join-Path $LogDir "streamlit_stderr.log"
    $sargs = @("-m", "streamlit", "run", "dashboard_app.py",
               "--server.headless", "true",
               "--server.port", "$StreamlitPort")
    $proc = Start-Process -FilePath $py -ArgumentList $sargs `
        -WorkingDirectory $PSScriptRoot `
        -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    Write-Log ("  spawned streamlit launcher PID {0}" -f $proc.Id)

    $ok = $false
    for ($i = 0; $i -lt 45; $i++) {
        Start-Sleep -Seconds 1
        try {
            $r = Invoke-WebRequest -Uri "http://localhost:$StreamlitPort/_stcore/health" -TimeoutSec 3 -UseBasicParsing
            if ($r.StatusCode -eq 200) { Write-Log ("  Streamlit /health 200 after {0}s" -f ($i + 1)); $ok = $true; break }
        } catch { }
    }
    if (-not $ok) { throw "Streamlit did not come back within 45 s" }
}

# -------------------------------------------------------------------------
# STAGE 3: Write completion marker.
# -------------------------------------------------------------------------
$null = Invoke-Stage "Write marker: $(Split-Path $marker -Leaf)" {
    $csvs = Get-ChildItem -Path $PSScriptRoot -Filter '*.csv' -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -like 'dowhy_mediation_analysis*' -or
            $_.Name -like 'dowhy_event_level_mediation*' -or
            $_.Name -eq 'causal_analysis_results_by_radius.csv' -or
            $_.Name -eq 'event_level_causal_analysis_by_radius.csv'
        } |
        Where-Object { $_.LastWriteTime -gt (Get-Date).AddHours(-8) } |
        Sort-Object LastWriteTime -Descending |
        ForEach-Object { "  {0,-60}  {1}  ({2:N1} KB)" -f $_.Name, $_.LastWriteTime, ($_.Length / 1KB) }

    $lines = @()
    $lines += "FULL PIPELINE CHAIN COMPLETE"
    $lines += "============================"
    $lines += "Chain finished at:  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    $lines += "Chain log:          $log"
    $lines += "Validation log:     $logPath"
    $lines += ""
    $lines += "Stage results:"
    foreach ($k in $results.Keys) {
        $v = if ($results[$k]) { "OK  " } else { "FAIL" }
        $lines += "  [{0}]  {1}" -f $v, $k
    }
    $lines += ""
    $lines += "Fresh CSVs (last 8 h):"
    if ($csvs) { $lines += $csvs } else { $lines += "  (none)" }
    $lines += ""
    $lines += "Dashboard:          http://localhost:$StreamlitPort"
    $lines += "Streamlit cache was cleared by the restart above."
    $lines | Set-Content -Path $marker -Encoding UTF8
    Write-Log ("  wrote: {0}" -f $marker)
}

Write-Log ""
Write-Log "Chain complete at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')."
