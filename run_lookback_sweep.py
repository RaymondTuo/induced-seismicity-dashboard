#!/usr/bin/env python3
"""
run_lookback_sweep.py  –  Change 5: Lookback-window sensitivity sweep

Runs the full pipeline for each lookback window in LOOKBACK_WINDOWS,
then prints a cross-window comparison table of key causal metrics.

Pipeline per window:
  1. filter_active_wells_before_events.py   --lookback-days N
  2. filter_merge_events_and_nonevents.py   --lookback-days N
  3. add_geoscience_to_event_well_links_with_injection.py  --lookback-days N
  4. dowhy_ci.py                            --lookback-days N

Usage:
  python run_lookback_sweep.py
  python run_lookback_sweep.py --windows 7 14 30   # subset
  python run_lookback_sweep.py --skip-pipeline      # compare existing CSVs only
"""

import argparse
import glob
import os
import subprocess
import sys
from datetime import datetime

import pandas as pd

LOOKBACK_WINDOWS = [7, 14, 30, 60, 90]

PIPELINE_STEPS = [
    "filter_active_wells_before_events.py",
    "filter_merge_events_and_nonevents.py",
    "add_geoscience_to_event_well_links_with_injection.py",
    "dowhy_ci.py",
]


def run_step(script: str, lookback: int, no_log: bool = False) -> bool:
    cmd = [sys.executable, script, "--lookback-days", str(lookback)]
    if script == "dowhy_ci.py" and no_log:
        cmd.append("--no-log-transform")
    print(f"\n  >> {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  ❌ {script} failed for {lookback}d (exit {result.returncode})")
        return False
    return True


def find_result_csv(lookback: int, no_log: bool = False) -> str | None:
    tag = f"{lookback}d"
    log_part = "_nolog" if no_log else ""
    pattern = f"dowhy_mediation_analysis_{tag}{log_part}_*.csv"
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def build_comparison(windows: list[int], no_log: bool = False) -> pd.DataFrame:
    rows = []
    for d in windows:
        path = find_result_csv(d, no_log)
        if path is None:
            print(f"  ⚠️  No result CSV found for {d}d — skipping from comparison.")
            continue
        df = pd.read_csv(path)
        # Use GBM results if available, else OLS
        for model in ("gbm_nonparam", "ols_baseline"):
            mdf = df[df["model_name"] == model]
            if not mdf.empty:
                break
        rows.append({
            "lookback_days": d,
            "model": model,
            "n_radii": len(mdf),
            "avg_total_effect": mdf["total_effect"].mean(),
            "avg_pct_mediated": mdf["prop_mediated"].mean(),
            "avg_pred_r2": mdf["predictive_r2"].mean(),
            "avg_mae": mdf["mae"].mean(),
            "avg_skill_score": mdf["skill_score"].mean(),
            "avg_spearman_rho": mdf["spearman_rho"].mean(),
            "bootstrap_success": mdf["bootstrap_success_rate"].mean(),
            "source_file": os.path.basename(path),
        })
    return pd.DataFrame(rows)


def print_comparison(cdf: pd.DataFrame) -> None:
    if cdf.empty:
        print("\n❌ No results to compare.")
        return

    print(f"\n{'=' * 110}")
    print("LOOKBACK WINDOW SENSITIVITY COMPARISON  (Change 5)")
    print(f"{'=' * 110}")
    print(f"{'Days':>6}  {'Model':<14}  {'Radii':>5}  {'AvgEffect':>10}  "
          f"{'%Med':>6}  {'PredR²':>7}  {'MAE':>6}  {'Skill':>6}  {'SpRho':>6}  {'Boot%':>6}")
    print("-" * 110)
    for _, r in cdf.iterrows():
        print(f"{int(r.lookback_days):>6}d  {r.model:<14}  {int(r.n_radii):>5}  "
              f"{r.avg_total_effect:>+10.3e}  "
              f"{r.avg_pct_mediated:>6.1f}  "
              f"{r.avg_pred_r2:>7.3f}  "
              f"{r.avg_mae:>6.3f}  "
              f"{r.avg_skill_score:>6.3f}  "
              f"{r.avg_spearman_rho:>6.3f}  "
              f"{r.bootstrap_success * 100:>5.1f}%")
    print("=" * 110)

    # Highlight best-fit window
    if len(cdf) > 1:
        best_ss = cdf.loc[cdf["avg_skill_score"].idxmax()]
        best_rho = cdf.loc[cdf["avg_spearman_rho"].idxmax()]
        print(f"\n• Best Skill Score:   {int(best_ss.lookback_days)}d  (SS={best_ss.avg_skill_score:.3f})")
        print(f"• Best Spearman rho:  {int(best_rho.lookback_days)}d  (ρ={best_rho.avg_spearman_rho:.3f})")
        print("\n  Note: best-fit window should be cross-checked against physical plausibility")
        print("        (see Q3 in Feb18 reference document) before treating as the preferred model.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", nargs="+", type=int, default=LOOKBACK_WINDOWS,
                        help="Lookback windows to run (default: 7 14 30 60 90)")
    parser.add_argument("--skip-pipeline", action="store_true",
                        help="Skip pipeline steps and only build comparison from existing CSVs")
    parser.add_argument("--no-log-transform", action="store_true",
                        help="Pass --no-log-transform to dowhy_ci.py to identify optimal "
                             "window in raw space before re-applying Change 3 log transform")
    args = parser.parse_args()
    no_log = args.no_log_transform

    windows = sorted(args.windows)
    log_label = "  [NO log-transform]" if no_log else "  [log-transform ON]"
    print(f"\n{'=' * 70}")
    print(f"LOOKBACK SWEEP  –  windows: {windows} days{log_label}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 70}")

    if not args.skip_pipeline:
        for d in windows:
            print(f"\n{'─' * 50}")
            print(f"Running pipeline for {d}-day lookback window …")
            print(f"{'─' * 50}")
            for step in PIPELINE_STEPS:
                if not run_step(step, d, no_log=no_log):
                    print(f"  Stopping sweep at {d}d due to pipeline failure.")
                    break

    cdf = build_comparison(windows, no_log=no_log)
    print_comparison(cdf)

    log_tag = "_nolog" if no_log else ""
    out = f"lookback_sweep_comparison{log_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    cdf.to_csv(out, index=False)
    print(f"\n💾 Comparison saved to: {out}")


if __name__ == "__main__":
    main()
