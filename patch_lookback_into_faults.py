#!/usr/bin/env python3
"""
patch_lookback_into_faults.py  –  Change 5 helper

For each lookback window and radius, takes the existing
event_well_links_with_faults_<R>km.csv (which already has G1/G2)
and replaces the Vol Prev N and Avg Press Prev N columns with values
from event_well_links_with_injection_<tag>_<R>km.csv produced by
filter_active_wells_before_events.py.

This avoids re-running the expensive fault-proximity step (step 3)
for every new lookback window since G1/G2 are purely spatial.

Usage:
    python patch_lookback_into_faults.py --windows 7 14 60 90
"""

import argparse
import os
import re
import pandas as pd

RADII = list(range(1, 21))

VOL_FRAG   = "vol prev"
PRESS_FRAG = "avg press"


def find_col(frags, columns):
    for col in columns:
        if all(f in col.lower() for f in frags):
            return col
    return None


def patch_one(tag: str, radius: int) -> bool:
    injection_file = f"event_well_links_with_injection_{tag}_{radius}km.csv"
    faults_src     = f"event_well_links_with_faults_{radius}km.csv"
    faults_out     = f"event_well_links_with_faults_{tag}_{radius}km.csv"

    if not os.path.exists(injection_file):
        print(f"  ⚠️  Missing {injection_file} — skipping radius {radius}km")
        return False
    if not os.path.exists(faults_src):
        print(f"  ⚠️  Missing {faults_src} — skipping radius {radius}km")
        return False

    inj = pd.read_csv(injection_file, low_memory=False)
    flt = pd.read_csv(faults_src, low_memory=False)

    # Find W and P columns in the injection file
    vol_col   = find_col(VOL_FRAG.split(), inj.columns)
    press_col = find_col(PRESS_FRAG.split(), inj.columns)

    if not vol_col or not press_col:
        print(f"  ❌ Could not find Vol/Press columns in {injection_file}")
        return False

    # Find matching columns in the faults file to replace
    flt_vol_col   = find_col(VOL_FRAG.split(), flt.columns)
    flt_press_col = find_col(PRESS_FRAG.split(), flt.columns)

    if not flt_vol_col or not flt_press_col:
        print(f"  ❌ Could not find Vol/Press columns in {faults_src}")
        return False

    # Align on EventID + API Number and patch the columns
    key_cols = ["EventID", "API Number"]
    inj_sub = inj[key_cols + [vol_col, press_col]].copy()

    merged = flt.merge(inj_sub, on=key_cols, how="left", suffixes=("_old", "_new"))

    # Replace old columns with new values where available
    new_vol_col   = vol_col   + "_new" if vol_col   + "_new" in merged.columns else vol_col
    new_press_col = press_col + "_new" if press_col + "_new" in merged.columns else press_col

    merged[flt_vol_col]   = merged[new_vol_col].combine_first(merged[flt_vol_col + "_old"] if flt_vol_col + "_old" in merged.columns else merged[flt_vol_col])
    merged[flt_press_col] = merged[new_press_col].combine_first(merged[flt_press_col + "_old"] if flt_press_col + "_old" in merged.columns else merged[flt_press_col])

    # Drop the temporary merge columns
    drop_cols = [c for c in merged.columns if c.endswith("_old") or c.endswith("_new")]
    merged = merged.drop(columns=drop_cols, errors="ignore")

    merged.to_csv(faults_out, index=False)
    print(f"  ✅ {faults_out}  ({len(merged):,} rows)")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", nargs="+", type=int, default=[7, 14, 60, 90],
                        help="Lookback windows to patch (default: 7 14 60 90)")
    args = parser.parse_args()

    for days in sorted(args.windows):
        tag = f"{days}d"
        print(f"\n── Patching {days}d window ──────────────────────")
        ok = 0
        for r in RADII:
            if patch_one(tag, r):
                ok += 1
        print(f"   {ok}/{len(RADII)} radii patched for {days}d")


if __name__ == "__main__":
    main()
