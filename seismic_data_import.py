#!/usr/bin/env python3
# seismic_data_import_and_plot.py
# ──────────────────────────────────────────────────────────────────────
# 1) Filters TexNet catalog to Midland Basin (ML ≥ 1.0)   → CSV
# 2) Plots a publication-quality map of those events      → PNG
# ---------------------------------------------------------------------

# ── Standard-library imports ──────────────────────────────────────────
import os, sys, datetime as dt
# ── Third-party imports ───────────────────────────────────────────────
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# This script is run inside a non-UTF8 Windows console in some environments.
# Keep console output ASCII-only to avoid UnicodeEncodeError.
def _to_ascii(s: str) -> str:
    return (
        str(s)
        .replace("❌", "[ERROR]")
        .replace("✅", "[OK]")
        .replace("⚠️", "[WARN]")
        .replace("🖼️", "[MAP]")
        .replace("→", "->")
        .replace("•", "-")
        .replace("‒", "-")
    )

# ── File paths / constants ────────────────────────────────────────────
FILENAME = "texnet_events.csv"
OUTFILE  = "texnet_events_filtered.csv"
PLOT_PNG = "midland_basin_events_map.png"

KEEP_COLS = [
    "EventID",
    "Origin Date",
    "Local Magnitude",
    "Latitude (WGS84)",
    "Longitude (WGS84)",
    "Depth of Hypocenter (Km.  Rel to MSL)",
    "Depth of Hypocenter (Km. Rel to Ground Surface)",
]

MIN_LAT, MAX_LAT = 30.6, 33.4
MIN_LON, MAX_LON = -103.2, -100.2
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # 1 ▸ Load raw CSV
    if not os.path.isfile(FILENAME):
        sys.exit(_to_ascii(f"[ERROR] '{FILENAME}' not found in {os.getcwd()}"))

    try:
        df = pd.read_csv(FILENAME, low_memory=False)
    except Exception as exc:
        sys.exit(_to_ascii(f"[ERROR] Could not read '{FILENAME}': {exc}"))

    # 2 ▸ Verify required columns
    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        sys.exit(_to_ascii(f"[ERROR] Missing column(s): {missing}"))

    # 3 ▸ Subset / cleanse
    df = df[KEEP_COLS].copy()
    lat, lon = "Latitude (WGS84)", "Longitude (WGS84)"

    df[lat]            = pd.to_numeric(df[lat],            errors="coerce")
    df[lon]            = pd.to_numeric(df[lon],            errors="coerce")
    df["Local Magnitude"] = pd.to_numeric(df["Local Magnitude"], errors="coerce")
    df["Origin Date"]     = pd.to_datetime(df["Origin Date"],    errors="coerce")

    df = df.dropna(subset=[lat, lon, "Local Magnitude"])
    df = df[(df[lat] != 0) & (df[lon] != 0)]

    # 4 ▸ Spatial filter
    df = df[
        df[lat].between(MIN_LAT, MAX_LAT) &
        df[lon].between(MIN_LON, MAX_LON)
    ]

    # 5 ▸ Magnitude completeness
    df_f = df[df["Local Magnitude"] >= 1.0]

    # 6 ▸ Save CSV
    df_f.to_csv(OUTFILE, index=False)
    print(_to_ascii(f"[OK] Saved filtered catalog -> {OUTFILE} ({len(df_f):,} rows)"))

    # 7 ▸ World-class Matplotlib map
    if df_f.empty:
        print("[WARN] No events to plot.")
        return

    # ── styling --------------------------------------------------------
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150,
                            facecolor="#0e1117")
    fig.patch.set_facecolor("#0e1117")

    # colour by magnitude, size by magnitude² for pop
    mags   = df_f["Local Magnitude"]
    norm   = Normalize(mags.min(), mags.max())
    cmap   = plt.cm.turbo
    sizes  = ((mags - mags.min()) + 1.0)**2.3 * 6  # tweak to taste

    sc = ax.scatter(
        df_f[lon], df_f[lat],
        c=mags, cmap=cmap, norm=norm,
        s=sizes, alpha=0.85, edgecolor="black", linewidth=0.3
    )

    # bounding box
    ax.set_xlim(MIN_LON, MAX_LON)
    ax.set_ylim(MIN_LAT, MAX_LAT)
    ax.set_aspect("equal", adjustable="box")

    # labels & grid
    ax.set_xlabel("Longitude (°E)", labelpad=8, color="white")
    ax.set_ylabel("Latitude (°N)",  labelpad=8, color="white")
    ax.grid(ls="--", lw=0.4, color="#333940", alpha=0.6)

    # title & subtitle
    t0, t1 = df_f["Origin Date"].min(), df_f["Origin Date"].max()
    year_span = f"{t0.year}‒{t1.year}" if t0.year != t1.year else str(t0.year)
    ax.set_title(
        f"Midland Basin Earthquakes (ML ≥ 1.0) • {year_span}",
        fontweight="bold", fontsize=17, pad=14
    )
    ax.set_title(f"{len(df_f):,} events", fontsize=12, loc="right", color="#bbbbbb")

    # colour-bar
    cbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=ax, pad=0.01, aspect=35, shrink=0.82
    )
    cbar.set_label("Local Magnitude", labelpad=6)

    # save
    plt.tight_layout()
    fig.savefig(PLOT_PNG, bbox_inches="tight")
    plt.close(fig)
    print(_to_ascii(f"[MAP] Event map saved -> {PLOT_PNG}"))

if __name__ == "__main__":
    main()