#!/usr/bin/env python3
# aggregate_event_level_causal_multi_radius.py
"""
Multi-Radius Event-Level Causal Analysis
───────────────────────────────────────────────────────────────
This script processes multiple event-well linkage files at different radius thresholds
(2km, 5km, 10km, 15km, 20km) to examine how the causal effects change with distance.

For each radius file:

1. **Group by EventID** and aggregate:
     • Σ Volume Injected (BBLs)      → `t_W`
     • Median Injection Pressure     → `t_P`
     • Min Nearest Fault Dist (km)   → `t_G1`
     • Σ Fault Segments ≤R km        → `t_G2`
     • Count of wells in radius      → `well_count`
     • (Magnitude is identical per event, so we just take the first)

2. Run a DoWhy back-door linear-regression causal estimate of
   ΔMagnitude per extra injected BBL.

3. Perform mediation analysis to decompose the total effect into
   direct (W → S) and indirect (W → P → S) pathways.

4. Show a predictive sanity-check and optional DAG PNG.

All results are compiled into a comparative summary table to evaluate sensitivity
to radius selection.

Edit the `COL_FRAGS` dict if your headers differ. All heavy steps have
timed banners so the console never looks frozen.
"""
# ─── 0 ▸ IMPORTS & QUIET MODE ──────────────────────────────────
import re, sys, time, logging, warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

from dowhy import CausalModel
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("dowhy").setLevel(logging.ERROR)

# List of input CSV files (one per radius).
# Upstream pipeline switched from a 30-day to a 90-day injection lookback; keep
# this list aligned with `event_well_links_with_faults_90d_*km.csv` on disk.
LOOKBACK_TAG = "90d"
CSV_FILES = [
    f"event_well_links_with_faults_{LOOKBACK_TAG}_{r}km.csv"
    for r in range(1, 21)
]

# Output summary file
RESULTS_CSV = "event_level_causal_analysis_by_radius.csv"

# Change 6: well-level P_missing → aggregate as fraction; zero-fill P before agg
P_MISSING_COL = "P_missing"

# ─── 1 ▸ COLUMN-NAME FRAGMENTS  (tweak as needed) ──────────────
COL_FRAGS = {
    "E": ["event", "id"],  # EventID
    "W": ["volume", "bbl"],  # injected volume
    "P": ["pressure", "psig"],  # average injection pressure
    "S": ["local", "mag"],  # earthquake magnitude
    "G1": ["nearest", "fault", "dist"],
    "G2": ["fault", "segment"],  # fault segment count
}


# ─── helper – timed banner ─────────────────────────────────────
@contextmanager
def step(title: str):
    print(f"\n⏳  {title} …", end="", flush=True)
    t0 = time.time()
    yield
    print(f" done in {time.time() - t0:.1f}s")


def safe(col: str) -> str:
    return re.sub(r"\W", "_", col)


def find_column(frags, columns):
    for c in columns:
        if all(f.lower() in c.lower() for f in frags):
            return c
    return None


def extract_radius(filename: str) -> int:
    """Extract the radius value (in km) from the filename."""
    match = re.search(r'(\d+)km', filename)
    if match:
        return int(match.group(1))
    return 0  # Default fallback


# ─── PROCESS ONE FILE ──────────────────────────────────────────
def process_file(csv_file: str) -> dict:
    """Process one radius file and return results as a dictionary."""
    radius = extract_radius(csv_file)
    print(f"\n{'=' * 80}\nPROCESSING: {csv_file} (radius: {radius} km)\n{'=' * 80}")

    # ─── 2 ▸ LOAD CSV & AUTO-MAP ───────────────────────────────────
    with step(f"Loading CSV ({csv_file})"):
        if not Path(csv_file).exists():
            print(f"❌ File not found: {csv_file} - skipping")
            return None
        df_raw = pd.read_csv(csv_file, low_memory=False)

    print("\n🔎  Auto-detecting required columns:")
    found = {}
    for key, frags in COL_FRAGS.items():
        col = find_column(frags, df_raw.columns)
        print(f"   {key:>2}  ←  {col if col else '❌ NOT FOUND'}")
        if col:
            found[key] = col

    missing = [k for k in COL_FRAGS if k not in found]
    if missing:
        print(f"\n❌  Missing column(s): {', '.join(missing)} – skipping file.")
        return None

    # ─── 3 ▸ NA SUMMARY & EVENT-LEVEL AGGREGATION ──────────────────
    print("\n🕵️  Missing-value summary (raw rows):")
    na = df_raw[found.values()].isna().sum().sort_values(ascending=False)
    for c, n in na.items():
        print(f"   {c:<35} : {n:,} NA ({n / len(df_raw):.1%})")

    # a) rename to safe identifiers
    rename_map = {found[k]: safe(k) for k in found}
    df = df_raw[list(found.values())].rename(columns=rename_map)

    # Change 6: indicator then zero-fill P at well level before aggregating
    p_col = safe("P")
    df[P_MISSING_COL] = df[p_col].isna().astype(np.float64)
    df[p_col] = df[p_col].fillna(0.0)

    # b) aggregate to one row per EventID
    aggregated = (
        df
        .groupby(safe("E"))
        .agg({
            safe("W"): "sum",
            safe("P"): "median",
            P_MISSING_COL: "mean",  # fraction of wells in radius with missing P (pre-fill)
            safe("G1"): "min",
            safe("G2"): "sum",
            safe("S"): "first",  # magnitude identical within group
        })
        .rename_axis(None)
        .reset_index()
    )
    aggregated["well_count"] = df.groupby(safe("E")).size().values

    # c) NA handling
    essential = [safe("W"), safe("S")]
    rows_before = len(aggregated)
    aggregated = aggregated.dropna(subset=essential)
    rows_after = len(aggregated)
    rows_dropped = rows_before - rows_after
    print(f"Dropped {rows_dropped:,} events missing treatment or outcome ({rows_dropped / rows_before:.1%} if any)")

    covariates = [c for c in [safe("P"), P_MISSING_COL, safe("G1"), safe("G2"), "well_count"]
                  if c in aggregated.columns]
    covariates_median = [c for c in covariates if c not in (safe("P"), P_MISSING_COL)]
    if covariates_median:
        aggregated[covariates_median] = aggregated[covariates_median].fillna(
            aggregated[covariates_median].median(numeric_only=True)
        )

    if aggregated.empty:
        print("\n❌  No events retain both treatment and outcome — skipping file.")
        return None

    W, P, S, G1, G2 = map(safe, ["W", "P", "S", "G1", "G2"])
    PM = P_MISSING_COL
    print("✅  Events after aggregation & cleaning:", len(aggregated))
    print(f"   Mean fraction wells with missing pressure (within radius): {aggregated[PM].mean():.1%}")

    # ─── 4 ▸ DAG (networkx) ────────────────────────────────────────
    G = nx.DiGraph([
        (G1, W), (G1, P), (G1, S), (G1, PM),
        (G2, W), (G2, P), (G2, S), (G2, PM),
        ("well_count", W), ("well_count", PM),
        (W, P), (W, S), (W, PM),
        (P, S), (PM, S),
    ])

    # ─── 5 ▸ DoWhy causal estimate ─────────────────────────────────
    with step("Building DoWhy model"):
        dag_dot = nx.nx_pydot.to_pydot(G).to_string()
        model = CausalModel(aggregated, treatment=W, outcome=S, graph=dag_dot)
    estimand = model.identify_effect()

    backdoor_vars = sorted({v for lst in estimand.backdoor_variables.values() for v in lst})
    print("📝  Back-door controls:", ", ".join(backdoor_vars) or "None")

    with step("Estimating total causal effect"):
        est = model.estimate_effect(
            estimand,
            method_name="backdoor.linear_regression",
            control_value=0, treatment_value=1,
        )

    X_sm = sm.add_constant(aggregated[[W] + backdoor_vars])
    ols_result = sm.OLS(aggregated[S], X_sm).fit()
    pval = ols_result.pvalues[W]

    print(f"\n📈  TOTAL causal effect (ΔMagnitude per extra BBL within {radius} km)")
    print(f"    DoWhy point estimate : {est.value:+.4g}")
    print(f"    Statsmodels p-value  : {pval:.3g}")

    # ─── 5.5 ▸ MEDIATION ANALYSIS (W → P → S) ───────────────────────
    print("\n🔄  Running mediation analysis...")

    # Step 1: Effect of W on P (path a)
    model_a = sm.OLS(aggregated[P], sm.add_constant(aggregated[[W] + backdoor_vars])).fit()
    a_coef = model_a.params[W]
    a_pval = model_a.pvalues[W]

    # Step 2: Effect of P on S controlling for W (path b)
    model_b = sm.OLS(aggregated[S], sm.add_constant(aggregated[[P, W] + backdoor_vars])).fit()
    b_coef = model_b.params[P]
    b_pval = model_b.pvalues[P]

    # Step 3: Total effect (path c)
    model_c = sm.OLS(aggregated[S], sm.add_constant(aggregated[[W] + backdoor_vars])).fit()
    c_coef = model_c.params[W]
    c_pval = model_c.pvalues[W]

    # Step 4: Direct effect (path c')
    c_prime_coef = model_b.params[W]
    c_prime_pval = model_b.pvalues[W]

    # Calculate indirect effect (a*b)
    indirect_effect = a_coef * b_coef
    proportion_mediated = (indirect_effect / c_coef) * 100 if c_coef != 0 else 0

    # Print results
    print(f"Path a (W → P)  : {a_coef:+.4g} (p={a_pval:.3g})")
    print(f"Path b (P → S)  : {b_coef:+.4g} (p={b_pval:.3g})")
    print(f"Total effect    : {c_coef:+.4g} (p={c_pval:.3g})")
    print(f"Direct effect   : {c_prime_coef:+.4g} (p={c_prime_pval:.3g})")
    print(f"Indirect effect : {indirect_effect:+.4g}")
    print(f"Proportion mediated: {proportion_mediated:.1f}%")

    # ─── 6 ▸ Refutations ───────────────────────────────────────────
    with step("Running refutation checks"):
        placebo = model.refute_estimate(
            estimand, est, method_name="placebo_treatment_refuter",
            placebo_type="permute")
        subset = model.refute_estimate(
            estimand, est, method_name="data_subset_refuter",
            subset_fraction=0.8)

    print("\n🔍  Refutation checks (effect should stay similar)")
    print(f"    Placebo : {placebo.new_effect:+.4g}")
    print(f"    Subset  : {subset.new_effect:+.4g}")

    # ─── 7 ▸ Predictive sanity-check ───────────────────────────────
    with step("Training predictive sanity-check model"):
        X = aggregated[[W, P, G1, G2, "well_count", PM]].copy()
        X[W] = np.log1p(X[W])
        y = aggregated[S]
        x_tr, x_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.20, random_state=42, shuffle=True)
        pred = LinearRegression().fit(x_tr, y_tr).predict(x_te)

    rmse = np.sqrt(mean_squared_error(y_te, pred))
    r2 = r2_score(y_te, pred)
    print("\n⚙️   Prediction metrics (non-causal sanity check)")
    print(f"    RMSE : {rmse:.4f}")
    print(f"    R²   : {r2:.4f}")

    # ─── 8 ▸ Optional DAG PNG ──────────────────────────────────────
    dag_filename = f"dag_event_level_{radius}km.png"
    try:
        from graphviz import Source
        import networkx.drawing.nx_pydot as nxpd
        Source(nxpd.to_pydot(G)).render(f"dag_event_level_{radius}km", format="png", quiet=True)
        print(f"\n🖼️   DAG image saved as {dag_filename}")
    except Exception:
        print("\n🖼️   graphviz or pydot not installed → DAG image skipped")

    # ─── 9 ▸ SUMMARY OF FINDINGS ──────────────────────────────────
    print(f"\n📋  SUMMARY OF MEDIATION ANALYSIS (RADIUS: {radius} KM)")
    print("─────────────────────────────────────────────────────")
    print(f"Total effect (W → S)            : {c_coef:+.4g}")
    print(f"Direct effect (W → S)           : {c_prime_coef:+.4g}")
    print(f"Indirect effect (W → P → S)     : {indirect_effect:+.4g}")
    print(f"Percentage mediated by pressure : {proportion_mediated:.1f}%")
    print("─────────────────────────────────────────────────────")

    # Optional: Probability of Exceedance analysis
    # Note: Only generating for the largest radius file to avoid too many output files
    poe_generated = False
    if radius == 20:  # Only for the largest radius
        try:
            import matplotlib.pyplot as plt
            from scipy.stats import norm

            # --- user-tweakable knobs ---------------------------------------
            MAG_THRESH = [2.0, 3.0, 4.0, 5.0]  # M ≥ thresholds
            VOL_BINS = np.logspace(4, 6, 7)  # 1e4 … 1e6 BBL
            MIN_EVENTS = 10  # skip very thin bins

            # ----------------------------------------------------------------

            # helper: maximum-likelihood b-value (Aki 1965)
            def b_value(mags, mmin):
                if len(mags) < MIN_EVENTS:
                    return np.nan
                return np.log10(np.e) / (mags.mean() - (mmin - 0.05))  # 0.1-mag bins

            results = []  # will collect rows: dist, Mth, Vmid, Pex

            # ── loop over injection-volume bins ────────────────────────────
            for i in range(len(VOL_BINS) - 1):
                v_lo, v_hi = VOL_BINS[i], VOL_BINS[i + 1]
                slc = aggregated[(aggregated[W] >= v_lo) & (aggregated[W] < v_hi)]
                if len(slc) < MIN_EVENTS:
                    continue

                mags = slc[S].values
                v_mid = np.sqrt(v_lo * v_hi)  # log-mid-point for plotting

                # empirical μ, σ for the "normal" assumption
                mu, sigma = mags.mean(), mags.std(ddof=1)

                # Gutenberg–Richter parameters
                mmin = mags.min()
                b = b_value(mags, mmin)

                for mthr in MAG_THRESH:
                    # – Normal tail –
                    p_norm = 1.0 - norm.cdf(mthr, mu, sigma)

                    # – G-R tail –
                    p_gr = 10 ** (-b * (mthr - mmin)) if not np.isnan(b) else np.nan

                    results.append(("Normal", mthr, v_mid, p_norm))
                    results.append(("Gutenberg-Richter", mthr, v_mid, p_gr))

            poe = pd.DataFrame(results,
                               columns=["model", "M_threshold", "V_mid", "P_exceed"])
            out_csv = f"earthquake_probability_curves_{radius}km.csv"
            poe.to_csv(out_csv, index=False)

            # ─── plot: two panels -- Normal vs Gutenberg-Richter —──────────
            fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
            palette = ["C0", "C1", "C2", "C3"]

            for ax, model in zip(axes, ["Normal", "Gutenberg-Richter"]):
                sub = poe[poe["model"] == model]
                for k, mthr in enumerate(MAG_THRESH):
                    g = sub[sub["M_threshold"] == mthr]
                    ax.plot(g["V_mid"], g["P_exceed"], "-o",
                            label=f"M ≥ {mthr:.1f}", color=palette[k])
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.grid(True, which="both", ls="--", lw=0.4, alpha=0.6)
                ax.set_xlabel("Injection Volume (BBLs)")
                if ax is axes[0]:
                    ax.set_ylabel("Exceedance Probability")
                ax.set_title(f"Probability of Exceeding Magnitude Thresholds\n({model}, {radius}km radius)")

            axes[0].legend(title="", loc="upper right", ncol=1)
            plt.tight_layout()
            out_png = f"earthquake_probability_curves_{radius}km.png"
            plt.savefig(out_png, dpi=180)
            plt.close()

            print(f"\n📈  PoE curves written → {out_png}")
            print(f"📑  Data table written → {out_csv}")
            poe_generated = True
        except ImportError:
            print("\n📈  matplotlib / scipy not installed → PoE curves skipped")

    # Return results as a dictionary for later saving to CSV
    return {
        "radius": radius,
        "n_events": len(aggregated),
        "avg_well_count": aggregated["well_count"].mean(),
        "total_effect": c_coef,
        "total_effect_pvalue": c_pval,
        "direct_effect": c_prime_coef,
        "direct_effect_pvalue": c_prime_pval,
        "indirect_effect": indirect_effect,
        "proportion_mediated": proportion_mediated,
        "path_a_coef": a_coef,
        "path_a_pvalue": a_pval,
        "path_b_coef": b_coef,
        "path_b_pvalue": b_pval,
        "rmse": rmse,
        "r_squared": r2,
        "refutation_placebo": placebo.new_effect,
        "refutation_subset": subset.new_effect,
    }


# ─── MAIN EXECUTION ─────────────────────────────────────────────
if __name__ == "__main__":
    start_time = time.time()
    print(f"\n{'=' * 80}\nBATCH EVENT-LEVEL CAUSAL ANALYSIS ACROSS RADII\n{'=' * 80}")

    # Process each file and collect results
    results = []
    for i, csv_file in enumerate(CSV_FILES):
        print(f"\nProcessing file {i + 1} of {len(CSV_FILES)}: {csv_file}")
        try:
            result = process_file(csv_file)
            if result:
                results.append(result)
        except Exception as e:
            print(f"❌ Error processing {csv_file}: {str(e)}")

    # If we have any results, save them to CSV
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("radius")
        results_df.to_csv(RESULTS_CSV, index=False)
        print(f"\n✅ Results saved to {RESULTS_CSV}")

        # Print comparative summary
        print(f"\n{'=' * 80}\nCOMPARATIVE SUMMARY ACROSS RADII\n{'=' * 80}")
        print("\nEffect Estimates by Radius (Event Level):")
        summary_df = results_df[["radius", "n_events", "avg_well_count",
                                 "total_effect", "direct_effect",
                                 "indirect_effect", "proportion_mediated"]]
        print(summary_df.to_string(index=False,
                                   float_format=lambda x: f"{x:+.5f}" if isinstance(x, (int, float)) and not isinstance(
                                       x, bool) else str(x)))

        # Find radius with strongest effect
        strongest_idx = results_df["total_effect"].abs().idxmax()
        strongest = results_df.loc[strongest_idx]

        # Find radius with most mediation
        most_mediated_idx = results_df["proportion_mediated"].idxmax()
        most_mediated = results_df.loc[most_mediated_idx]

        print(
            f"\nStrongest total effect at event level: {strongest['radius']}km radius: {strongest['total_effect']:.5f}")
        print(
            f"Most mediation by pressure at event level: {most_mediated['radius']}km radius: {most_mediated['proportion_mediated']:.1f}%")

        # Calculate total runtime
        total_runtime = time.time() - start_time
        minutes, seconds = divmod(total_runtime, 60)
        print(f"\nTotal analysis runtime: {int(minutes)}m {int(seconds)}s")

        # Find amplification factor (comparing to 20km results)
        if 20 in results_df["radius"].values:
            df_20km = results_df[results_df["radius"] == 20].iloc[0]
            print("\nAmplification factors compared to 20km radius:")
            for i, row in results_df.iterrows():
                if row["radius"] != 20:
                    amp_factor = row["total_effect"] / df_20km["total_effect"] if df_20km[
                                                                                      "total_effect"] != 0 else float(
                        'nan')
                    print(f"  {row['radius']}km: {amp_factor:.3f}x")
    else:
        print("\n❌ No valid results obtained from any file.")

    print(f"\n{'=' * 80}\nAnalysis complete!\n{'=' * 80}")