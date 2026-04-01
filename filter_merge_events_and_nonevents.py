#!/usr/bin/env python3
"""
Batch version
─────────────
For each radius R in {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0} km:

1. Read innocent_wells_<R>km.csv
2. Replace the `EventID` prefix  texnet → faknet
3. Save the patched file as           innocent_wells_with_fakeids_<R>km.csv
4. Concatenate that table with
       event_well_links_with_injection_<R>km.csv
   and write                           combined_event_well_links_<R>km.csv

Usage:  python batch_concat_innocent_and_links.py
"""

import argparse
from pathlib import Path
import pandas as pd

# ── file templates ─────────────────────────────────────────────────────────
RADII = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]  # km


def patch_event_ids(src: Path, dst: Path) -> pd.DataFrame:
    """Replace EventID prefix (texnet → faknet) and save to *dst*."""
    df = pd.read_csv(src, dtype=str)
    if "EventID" not in df.columns:
        raise KeyError(f"'EventID' column not found in {src.name}")

    df["EventID"] = df["EventID"].str.replace(r"^texnet", "faknet", regex=True)
    df.to_csv(dst, index=False)
    print(f"[OK] {dst.name:40}  ({len(df):,} rows)")
    return df


def concatenate(event_path: Path, fake_df: pd.DataFrame, out_path: Path) -> None:
    """Concatenate *fake_df* with event_well_links file and save."""
    event_df = pd.read_csv(event_path, dtype=str)
    combined = pd.concat([event_df, fake_df], ignore_index=True, sort=False)
    combined.to_csv(out_path, index=False)
    print(f"[OK] {out_path.name:40}  ({len(combined):,} rows)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback-days", type=int, default=90,
                        help="Lookback window in days (must match filter_active_wells run)")
    args = parser.parse_args()
    tag = f"{args.lookback_days}d"

    innocent_fmt   = f"innocent_wells_{tag}_{{R}}km.csv"
    patched_fmt    = f"innocent_wells_with_fakeids_{tag}_{{R}}km.csv"
    event_link_fmt = f"event_well_links_with_injection_{tag}_{{R}}km.csv"
    output_fmt     = f"combined_event_well_links_{tag}_{{R}}km.csv"

    for R in RADII:
        # Resolve paths for this radius
        innocent    = Path(innocent_fmt.format(R=R))
        patched     = Path(patched_fmt.format(R=R))
        event_link  = Path(event_link_fmt.format(R=R))
        output_file = Path(output_fmt.format(R=R))

        print(f"\n--- Processing {R} km files -------------------------------------")
        fake_df = patch_event_ids(innocent, patched)
        concatenate(event_link, fake_df, output_file)


if __name__ == "__main__":
    main()