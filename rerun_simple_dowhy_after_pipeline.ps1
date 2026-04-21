param(
    [int]$PipelinePid = 15412,
    [int[]]$StreamlitPids = @(41392, 30352),
    [int]$StreamlitPort = 8502,
    [string]$LogDir = "$PSScriptRoot"
)

# Watcher runs UNATTENDED after the main pipeline finishes.
# Every stage is wrapped in try/catch so a failure in one stage does not block
# the others. All output is teed into a timestamped log so the user can audit
# what happened while they were away.

$ErrorActionPreference = "Continue"
$stamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$log   = Join-Path $LogDir "simple_dowhy_rerun_$stamp.log"
$summary = Join-Path $LogDir "PIPELINE_DONE_SUMMARY.txt"

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
    try {
        & $Action
    } catch {
        $ok = $false
        Write-Log ("STAGE FAILED ({0}): {1}" -f $Name, $_.Exception.Message)
    }
    $elapsed = [int]((Get-Date) - $t0).TotalSeconds
    Write-Log ("STAGE {0}: {1} ({2}s)" -f ($(if ($ok) { 'OK' } else { 'FAIL' })), $Name, $elapsed)
    return $ok
}

Write-Log "Watcher armed. Waiting for run_all.py (PID $PipelinePid) to finish..."

try {
    Wait-Process -Id $PipelinePid -ErrorAction Stop
    Write-Log "Main pipeline process $PipelinePid has exited."
} catch {
    Write-Log "PID $PipelinePid not found (already exited). Proceeding immediately."
}

Start-Sleep -Seconds 10  # Flush guard so step 8 CSV writes complete.

$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Log "FATAL: .venv python not found at $py. Aborting."
    exit 1
}

# UTF-8 for child Python processes: the simple_dowhy scripts print emoji
# (U+23F3, U+274C, U+1F4CB). With captured stdout Windows defaults to cp1252
# which raises UnicodeEncodeError on the very first banner line. Setting this
# env var before spawning makes stdout/stderr utf-8 compatible.
$env:PYTHONIOENCODING = "utf-8"

$results = [ordered]@{}

# -------------------------------------------------------------------------
# STAGE 1: Rerun simple DoWhy scripts (steps 5 & 6 with 90d lookback fix).
# -------------------------------------------------------------------------
$results["simple_5"] = Invoke-Stage "Step 5 rerun: dowhy_simple_all.py" {
    $s = Join-Path $PSScriptRoot "dowhy_simple_all.py"
    if (-not (Test-Path $s)) { throw "script missing: $s" }
    & $py $s 2>&1 | Tee-Object -FilePath $log -Append | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "exit code $LASTEXITCODE" }
}

$results["simple_6"] = Invoke-Stage "Step 6 rerun: dowhy_simple_all_aggregate.py" {
    $s = Join-Path $PSScriptRoot "dowhy_simple_all_aggregate.py"
    if (-not (Test-Path $s)) { throw "script missing: $s" }
    & $py $s 2>&1 | Tee-Object -FilePath $log -Append | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "exit code $LASTEXITCODE" }
}

# -------------------------------------------------------------------------
# STAGE 2: Apply parallelism migration. Dry-run first; only commit on clean
# dry-run. Smoke-test with ast.parse on the patched files.
# -------------------------------------------------------------------------
$results["migrate"] = Invoke-Stage "Parallelism migration: dowhy_ci{,_aggregated}.py" {
    $m = Join-Path $PSScriptRoot "migrate_to_parallel.py"
    if (-not (Test-Path $m)) { throw "migrate_to_parallel.py missing" }
    & $py $m --dry-run 2>&1 | Tee-Object -FilePath $log -Append | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "dry-run FAILED; serial scripts left untouched" }
    & $py $m 2>&1 | Tee-Object -FilePath $log -Append | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "commit FAILED; serial scripts untouched" }
    $smoke1 = & $py -c "import ast, pathlib; ast.parse(pathlib.Path('dowhy_ci.py').read_text(encoding='utf-8')); print('dowhy_ci.py AST OK')" 2>&1
    Write-Log ("  smoke dowhy_ci.py: " + ($smoke1 -join ' '))
    $smoke2 = & $py -c "import ast, pathlib; ast.parse(pathlib.Path('dowhy_ci_aggregated.py').read_text(encoding='utf-8')); print('dowhy_ci_aggregated.py AST OK')" 2>&1
    Write-Log ("  smoke dowhy_ci_aggregated.py: " + ($smoke2 -join ' '))
}

# -------------------------------------------------------------------------
# STAGE 3: Restart Streamlit so its @st.cache_data picks up the new CSVs.
# The dashboard caches load_dashboard_data() without mtime invalidation, so
# an in-place CSV refresh alone is NOT enough — we must kill + relaunch.
# -------------------------------------------------------------------------
$results["streamlit_restart"] = Invoke-Stage "Streamlit restart (port $StreamlitPort)" {
    foreach ($sp in $StreamlitPids) {
        $p = Get-Process -Id $sp -ErrorAction SilentlyContinue
        if ($p) {
            Write-Log ("  stopping streamlit PID {0} ({1})" -f $sp, $p.ProcessName)
            Stop-Process -Id $sp -Force -ErrorAction SilentlyContinue
        } else {
            Write-Log ("  streamlit PID {0} already gone" -f $sp)
        }
    }
    Start-Sleep -Seconds 3

    # Defensive port-kill: find any process still bound to $StreamlitPort.
    try {
        $listeners = Get-NetTCPConnection -LocalPort $StreamlitPort -State Listen -ErrorAction SilentlyContinue
        foreach ($l in $listeners) {
            Write-Log ("  port {0} still bound by PID {1}; stopping it" -f $StreamlitPort, $l.OwningProcess)
            Stop-Process -Id $l.OwningProcess -Force -ErrorAction SilentlyContinue
        }
    } catch { }
    Start-Sleep -Seconds 2

    # Launch Streamlit detached so it survives this watcher exiting.
    # -WindowStyle Hidden hides the console; output is redirected to files.
    $stdout = Join-Path $LogDir "streamlit_stdout.log"
    $stderr = Join-Path $LogDir "streamlit_stderr.log"
    $streamlitArgs = @("-m", "streamlit", "run", "dashboard_app.py",
                       "--server.headless", "true",
                       "--server.port", "$StreamlitPort")
    $proc = Start-Process -FilePath $py -ArgumentList $streamlitArgs `
        -WorkingDirectory $PSScriptRoot `
        -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    Write-Log ("  spawned streamlit launcher PID {0}" -f $proc.Id)

    # Poll /health for up to 45 s.
    $ok = $false
    for ($i = 0; $i -lt 45; $i++) {
        Start-Sleep -Seconds 1
        try {
            $r = Invoke-WebRequest -Uri "http://localhost:$StreamlitPort/_stcore/health" -TimeoutSec 3 -UseBasicParsing
            if ($r.StatusCode -eq 200) {
                Write-Log ("  Streamlit /health 200 after {0}s" -f ($i + 1))
                $ok = $true; break
            }
        } catch { }
    }
    if (-not $ok) { throw "Streamlit did not come back on port $StreamlitPort within 45 s" }
}

# -------------------------------------------------------------------------
# STAGE 4: Write a human-readable summary file.
# -------------------------------------------------------------------------
$null = Invoke-Stage "Write summary file: $(Split-Path $summary -Leaf)" {
    $csvTable = Get-ChildItem -Path $PSScriptRoot -Filter '*.csv' -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -like 'dowhy_mediation_analysis*' -or
            $_.Name -like 'dowhy_event_level_mediation*' -or
            $_.Name -eq 'causal_analysis_results_by_radius.csv' -or
            $_.Name -eq 'event_level_causal_analysis_by_radius.csv'
        } |
        Where-Object { $_.LastWriteTime -gt (Get-Date).AddHours(-24) } |
        Sort-Object LastWriteTime -Descending |
        ForEach-Object { "  {0,-55}  {1}  ({2:N1} KB)" -f $_.Name, $_.LastWriteTime, ($_.Length / 1KB) }

    $lines = @()
    $lines += "PIPELINE COMPLETE SUMMARY"
    $lines += "========================="
    $lines += "Watcher finished at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    $lines += "Full log:            $log"
    $lines += ""
    $lines += "Stage results:"
    foreach ($k in $results.Keys) {
        $v = if ($results[$k]) { "OK  " } else { "FAIL" }
        $lines += "  [{0}]  {1}" -f $v, $k
    }
    $lines += ""
    $lines += "Fresh CSVs (last 24 h):"
    if ($csvTable) { $lines += $csvTable } else { $lines += "  (none)" }
    $lines += ""
    $lines += "Dashboard:           http://localhost:$StreamlitPort"
    $lines += "Rollback parallelism: mv dowhy_ci.py.serial_backup dowhy_ci.py (and _aggregated)"
    $lines += ""
    $lines | Set-Content -Path $summary -Encoding UTF8
    Write-Log ("  wrote: {0}" -f $summary)
}

Write-Log ""
Write-Log "All post-pipeline tasks finished at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')."
Write-Log "Summary: $summary"
