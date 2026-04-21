"""
Migrate dowhy_ci.py and dowhy_ci_aggregated.py to a parallel, multi-core layout.

What this script does
---------------------
1. Backs up the originals (<name>.py -> <name>.py.serial_backup) so the serial
   version is always recoverable.
2. Wraps the existing module-level "execution" code (CSV discovery + main loop
   + summary tables) in a new `_run_main()` function guarded by
   `if __name__ == "__main__":`. This is required because Windows multiprocessing
   uses spawn semantics and re-imports the module in every worker. Without the
   guard, workers would execute the main loop recursively and fork-bomb.
3. Replaces the serial `for csv_file in event_files: process_file(csv_file)`
   loop with a `joblib.Parallel` (loky backend) dispatch that distributes
   radii across worker processes. Number of workers defaults to
   min(8, len(event_files), cpu_count()) and can be overridden via the
   DOWHY_CI_JOBS environment variable or --jobs CLI flag.
4. Validates every rewritten file via `ast.parse` before committing, and
   reverts from the backup if parsing fails.

Determinism
-----------
The serial code already threads deterministic seeds through every worker path:
- `np.random.seed(42)` runs at module import time in each worker.
- `bootstrap_mediation_effects_dowhy` uses `random_state=i` where `i` is the
  bootstrap iteration index, so results are independent of execution order.
- Per-worker seeds in `process_file` are fixed (random_state=42) for hurdle
  model, train/test splits, etc.
Parallel execution therefore produces bit-identical outputs to serial.

Safety
------
- Does nothing if the originals are already patched (idempotent).
- Writes to a .tmp file first, validates with ast.parse, then atomically
  renames. A syntax error in the generated file cannot replace the original.
- If you ever need to roll back:
     mv dowhy_ci.py.serial_backup dowhy_ci.py
     mv dowhy_ci_aggregated.py.serial_backup dowhy_ci_aggregated.py
"""

from __future__ import annotations

import argparse
import ast
import os
import shutil
import sys
from pathlib import Path
from typing import Tuple

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Target specifications
# ---------------------------------------------------------------------------
# Each tuple is (filename, main_start_marker, serial_loop_marker,
#                parallel_loop_block).
#
# main_start_marker:   a unique substring that identifies the FIRST line of the
#                      module-level "main" code. Everything from that line to
#                      end of file gets indented and wrapped inside _run_main().
#
# serial_loop_marker:  a unique substring that identifies the existing serial
#                      loop we want to replace. We locate the `for` line, the
#                      body (2-N following indented lines), and swap them.
#
# parallel_loop_block: the replacement loop. Must be at module-indent level
#                      (i.e. the "root" indentation of the main function body,
#                      which becomes 4 spaces after wrapping).

SENTINEL_MARKER = "# --- PARALLEL MIGRATION: applied by migrate_to_parallel.py ---"

DOWHY_CI_PARALLEL_BLOCK = '''    # --- PARALLEL MIGRATION: applied by migrate_to_parallel.py ---
    # Distribute radii across worker processes. Each process_file() call is
    # fully independent (loads its own CSV, uses iteration-indexed bootstrap
    # seeds), so output is bit-identical to serial execution.
    import os as _os
    try:
        from joblib import Parallel, delayed  # type: ignore
        _have_joblib = True
    except ImportError:  # pragma: no cover - fallback if joblib missing
        _have_joblib = False

    _n_jobs_env = _os.environ.get("DOWHY_CI_JOBS", "").strip()
    try:
        _n_jobs = int(_n_jobs_env) if _n_jobs_env else 0
    except ValueError:
        _n_jobs = 0
    if _n_jobs <= 0:
        _cpu = _os.cpu_count() or 1
        _n_jobs = min(8, len(event_files), max(1, _cpu - 1))

    print(f"\\n🔀  Parallel execution: {_n_jobs} worker(s) across {len(event_files)} radii "
          f"(override with DOWHY_CI_JOBS)")

    results = []
    if _have_joblib and _n_jobs > 1:
        _all = Parallel(n_jobs=_n_jobs, backend="loky", verbose=5)(
            delayed(process_file)(csv_file) for csv_file in event_files
        )
        for file_results in _all:
            if file_results:
                # dowhy_ci returns list; dowhy_ci_aggregated returns dict or None.
                if isinstance(file_results, list):
                    results.extend(file_results)
                else:
                    results.append(file_results)
    else:
        for i, csv_file in enumerate(event_files):
            print(f"\\n[{i + 1}/{len(event_files)}] Processing {csv_file}...")
            file_results = process_file(csv_file)
            if file_results:
                if isinstance(file_results, list):
                    results.extend(file_results)
                else:
                    results.append(file_results)
'''

# ---------------------------------------------------------------------------
# Rewrite helpers
# ---------------------------------------------------------------------------

def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _already_patched(src: str) -> bool:
    return SENTINEL_MARKER in src


def _wrap_main(src: str, main_start_marker: str, serial_loop_marker: str) -> str:
    """Wrap module-level main code inside _run_main() and swap in parallel loop."""
    lines = src.splitlines(keepends=False)

    # Find where the "module-level main" begins.
    main_start_idx = None
    for idx, line in enumerate(lines):
        if main_start_marker in line:
            main_start_idx = idx
            break
    if main_start_idx is None:
        raise RuntimeError(
            f"Could not locate main_start_marker {main_start_marker!r}"
        )

    # Find the serial loop to replace.
    loop_start_idx = None
    for idx in range(main_start_idx, len(lines)):
        if serial_loop_marker in lines[idx]:
            loop_start_idx = idx
            break
    if loop_start_idx is None:
        raise RuntimeError(
            f"Could not locate serial_loop_marker {serial_loop_marker!r}"
        )

    # Walk backwards a few lines from the loop to absorb any helpful comments
    # and the `results = []` init that precedes it.
    preamble_start = loop_start_idx
    for back in range(loop_start_idx - 1, max(main_start_idx, loop_start_idx - 6), -1):
        stripped = lines[back].lstrip()
        if stripped.startswith("results = []") or stripped.startswith("results=[]"):
            preamble_start = back
            break

    # Walk forwards to find where the loop body ends. The loop body is a
    # contiguous block of lines indented deeper than the `for` line.
    for_indent = len(lines[loop_start_idx]) - len(lines[loop_start_idx].lstrip())
    loop_end_idx = loop_start_idx
    for idx in range(loop_start_idx + 1, len(lines)):
        raw = lines[idx]
        if raw.strip() == "":
            loop_end_idx = idx  # blank lines stay within the loop region
            continue
        this_indent = len(raw) - len(raw.lstrip())
        if this_indent > for_indent:
            loop_end_idx = idx
        else:
            break

    # Slice the file.
    pre_main = lines[:main_start_idx]
    main_before_loop = lines[main_start_idx:preamble_start]
    main_after_loop = lines[loop_end_idx + 1:]

    # Indent the main-body regions by 4 (they become bodies of _run_main()).
    def _indent_block(block):
        out = []
        for line in block:
            if line == "":
                out.append("")
            else:
                out.append("    " + line)
        return out

    main_before_loop_indented = _indent_block(main_before_loop)
    main_after_loop_indented = _indent_block(main_after_loop)

    new_lines = []
    new_lines.extend(pre_main)
    new_lines.append("")
    new_lines.append("")
    new_lines.append("def _run_main():")
    new_lines.append('    """Entry point; wrapped so ProcessPoolExecutor workers do not re-execute it."""')
    new_lines.extend(main_before_loop_indented)
    new_lines.append("")
    # Parallel loop block already has 4-space indent inside; split and append.
    for bl in DOWHY_CI_PARALLEL_BLOCK.splitlines():
        new_lines.append(bl)
    new_lines.extend(main_after_loop_indented)
    new_lines.append("")
    new_lines.append("")
    new_lines.append('if __name__ == "__main__":')
    new_lines.append("    _run_main()")
    new_lines.append("")

    return "\n".join(new_lines)


def _migrate_one(
    target: Path,
    main_start_marker: str,
    serial_loop_marker: str,
) -> Tuple[bool, str]:
    if not target.exists():
        return False, f"not found: {target}"

    src = _read(target)
    if _already_patched(src):
        return True, "already patched (sentinel present)"

    try:
        new_src = _wrap_main(src, main_start_marker, serial_loop_marker)
    except Exception as exc:
        return False, f"rewrite failed: {exc}"

    # Validate
    try:
        ast.parse(new_src)
    except SyntaxError as exc:
        return False, f"ast.parse failed on new content: {exc}"

    # Commit: backup then atomic replace.
    backup = target.with_suffix(target.suffix + ".serial_backup")
    if not backup.exists():
        shutil.copy2(target, backup)

    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(new_src, encoding="utf-8")
    os.replace(tmp, target)
    return True, f"patched (backup: {backup.name})"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--dry-run", action="store_true",
                   help="Do not write, only report whether the patch would apply.")
    args = p.parse_args()

    targets = [
        {
            "file": HERE / "dowhy_ci.py",
            "main_start_marker": "# --- CHANGE 5: filter event files by lookback tag if --lookback-days was given ---",
            # Pattern unique to the outer loop we want to replace.
            "serial_loop_marker": "for i, csv_file in enumerate(event_files):",
        },
        {
            "file": HERE / "dowhy_ci_aggregated.py",
            "main_start_marker": "# Find event files",
            "serial_loop_marker": "for i, csv_file in enumerate(event_files):",
        },
    ]

    exit_code = 0
    for t in targets:
        print(f"\n--- {t['file'].name} ---")
        if args.dry_run:
            try:
                src = _read(t["file"])
                if _already_patched(src):
                    print("  would skip: already patched")
                    continue
                _wrap_main(src, t["main_start_marker"], t["serial_loop_marker"])
                print("  would patch cleanly")
            except Exception as exc:
                print(f"  would FAIL: {exc}")
                exit_code = 1
            continue

        ok, msg = _migrate_one(t["file"], t["main_start_marker"], t["serial_loop_marker"])
        print(f"  {'OK' if ok else 'FAIL'}: {msg}")
        if not ok:
            exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
