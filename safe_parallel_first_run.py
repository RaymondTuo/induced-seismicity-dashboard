"""
Safe parallel-path validation for dowhy_ci.py + dowhy_ci_aggregated.py.

Context:
    The parallelism migration (applied 2026-04-21 by migrate_to_parallel.py)
    restructured both scripts to fan radii out across joblib workers. The
    migration was validated syntactically (AST parse + smoke test) but the
    actual parallel run has never been exercised end-to-end.

What this wrapper does:
    1. Monkey-patches each module's find_csv_files() to return only a small
       subset of radii (default: 1, 5, 10 km).
    2. Runs _run_main() on each module in-process so we exercise the real
       joblib.Parallel code path with 3 workers.
    3. Measures wall-clock time per module.
    4. Reports whether each call wrote a fresh timestamped output CSV.

Safety:
    The scripts write timestamped output files (dowhy_mediation_analysis_
    90d_<TIMESTAMP>.csv and dowhy_event_level_mediation_<TIMESTAMP>.csv),
    NOT the canonical dowhy_mediation_analysis.csv that the dashboard reads.
    So this wrapper cannot corrupt the dashboard's current data.

Run:
    .venv\\Scripts\\python.exe safe_parallel_first_run.py
    # or override subset:
    .venv\\Scripts\\python.exe safe_parallel_first_run.py 1 3 5
"""
import importlib
import os
import re
import sys
import time
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Configure a small, fast subset. Radii chosen to cover the size spectrum
# without hitting the 20 km monster (which is the critical path and would
# make this validation itself a 2-hour affair).
# ---------------------------------------------------------------------------
if len(sys.argv) > 1:
    TEST_RADII = {int(a) for a in sys.argv[1:]}
else:
    TEST_RADII = {1, 5, 10}

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("DOWHY_CI_JOBS", str(min(3, len(TEST_RADII))))

HERE = Path(__file__).resolve().parent
os.chdir(HERE)


def _radius_from_name(path: str) -> int | None:
    """Extract the integer radius from 'event_well_links_with_faults_90d_12km.csv'."""
    m = re.search(r"_(\d+)km\.csv$", path)
    return int(m.group(1)) if m else None


def _snapshot_outputs() -> set[str]:
    """Files that look like dowhy output CSVs, so we can tell what got written."""
    patterns = ("dowhy_mediation_analysis*", "dowhy_event_level_mediation*")
    out: set[str] = set()
    for pat in patterns:
        out.update(str(p.name) for p in HERE.glob(pat))
    return out


def run_module(mod_name: str) -> tuple[int, float, list[str]]:
    """Import the module, patch find_csv_files, invoke _run_main().

    Returns (return_code, elapsed_seconds, list_of_new_output_files).
    """
    print("\n" + "=" * 78)
    print(f"VALIDATION: {mod_name}   (radii = {sorted(TEST_RADII)})")
    print("=" * 78)

    mod = importlib.import_module(mod_name)
    original = mod.find_csv_files

    def subset_csv_files():
        all_files = original()
        kept = [f for f in all_files
                if _radius_from_name(f) in TEST_RADII
                and "event_well_links_with_faults" in f
                and "_90d_" in f]
        kept.sort(key=lambda p: _radius_from_name(p) or 0)
        print(f"  [safe-run] subset: {len(kept)}/{len(all_files)} files: "
              f"{[Path(f).name for f in kept]}")
        return kept

    mod.find_csv_files = subset_csv_files

    before = _snapshot_outputs()
    t0 = time.time()
    rc = 0
    try:
        mod._run_main()
    except SystemExit as e:
        rc = int(e.code) if e.code is not None else 0
    except Exception:
        traceback.print_exc()
        rc = 1
    elapsed = time.time() - t0

    new_files = sorted(_snapshot_outputs() - before)
    status = "OK" if rc == 0 else f"FAIL (rc={rc})"
    print(f"\n  [safe-run] {mod_name}: {status}   elapsed {elapsed:.1f}s")
    if new_files:
        for name in new_files:
            size_kb = (HERE / name).stat().st_size / 1024
            print(f"  [safe-run]   new output: {name} ({size_kb:.1f} KB)")
    else:
        print("  [safe-run]   WARNING: no new output CSV detected")
    return rc, elapsed, new_files


def main() -> int:
    print("SAFE PARALLEL VALIDATION")
    print(f"  cwd:                 {HERE}")
    print(f"  radii subset:        {sorted(TEST_RADII)}")
    print(f"  DOWHY_CI_JOBS:       {os.environ['DOWHY_CI_JOBS']}")
    print(f"  PYTHONIOENCODING:    {os.environ['PYTHONIOENCODING']}")

    results: list[tuple[str, int, float, list[str]]] = []
    for name in ("dowhy_ci", "dowhy_ci_aggregated"):
        rc, elapsed, new_files = run_module(name)
        results.append((name, rc, elapsed, new_files))

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    all_ok = True
    for name, rc, elapsed, new_files in results:
        tag = "OK  " if rc == 0 else "FAIL"
        print(f"  [{tag}]  {name:24s}  {elapsed:7.1f}s   "
              f"{len(new_files)} output file(s)")
        if rc != 0:
            all_ok = False

    print()
    if all_ok:
        print("✅  Parallel path works end-to-end on the subset. "
              "Safe to run the full pipeline.")
        return 0
    else:
        print("❌  Validation FAILED. See output above. Do NOT run the full "
              "pipeline until the failure is diagnosed.")
        print("    Rollback: mv dowhy_ci.py.serial_backup dowhy_ci.py "
              "(and dowhy_ci_aggregated.py similarly)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
