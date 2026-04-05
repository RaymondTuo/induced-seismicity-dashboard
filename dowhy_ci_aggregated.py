import numpy as np
import pandas as pd
import warnings
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from dowhy import CausalModel
import dowhy.causal_estimators.linear_regression_estimator
import networkx as nx
import os
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("Current working directory:", os.getcwd())

# Navigate to the parent directory if we're in .git-rewrite
if os.getcwd().endswith('.git-rewrite'):
    parent_dir = Path(os.getcwd()).parent
    os.chdir(parent_dir)
    print(f"Changed to parent directory: {os.getcwd()}")

# Also try going up one more level if needed
if not any(f.endswith('.csv') for f in os.listdir('.') if os.path.isfile(f)):
    if Path('..').exists():
        os.chdir('..')
        print(f"Changed to: {os.getcwd()}")

print("\nLooking for data files...")

# Change 6: zero-fill P at well level; aggregate P_missing as fraction per event
P_MISSING_COL = "P_missing"


def find_csv_files():
    csv_files = []

    # Look in current directory
    for file in os.listdir('.'):
        if file.endswith('.csv') and os.path.isfile(file):
            csv_files.append(file)

    # Look in common subdirectories
    for subdir in ['data', 'Data', 'csv', 'CSV']:
        if os.path.exists(subdir) and os.path.isdir(subdir):
            try:
                for file in os.listdir(subdir):
                    if file.endswith('.csv'):
                        csv_files.append(os.path.join(subdir, file))
            except PermissionError:
                continue

    return csv_files


def extract_radius(filename):
    """Extract radius from filename"""
    match = re.search(r'(\d+)km', filename)
    return int(match.group(1)) if match else 0


def find_column(frags, columns):
    """Return the first column that contains all substrings in frags."""
    for c in columns:
        if all(f.lower() in c.lower() for f in frags):
            return c
    return None


def safe(col: str) -> str:
    """Convert a column name to a strict Python identifier."""
    return re.sub(r"\W", "_", col)


def calculate_vif(X):
    """Calculate VIF for each variable in design matrix (skip ill-conditioned columns)."""
    vif_data = pd.DataFrame()
    vif_data["variable"] = X.columns
    vifs = []
    for i in range(X.shape[1]):
        try:
            vifs.append(variance_inflation_factor(np.asarray(X.values, dtype=float), i))
        except Exception:
            vifs.append(np.nan)
    vif_data["VIF"] = vifs
    return vif_data


def calculate_mediation_effects_dowhy(data):
    """Calculate mediation effects using DoWhy framework with Baron & Kenny approach"""

    data = data.copy().reset_index(drop=True)

    pm = P_MISSING_COL
    G = nx.DiGraph([
        ('G1', 'W'), ('G1', 'P'), ('G1', 'S'), ('G1', pm),
        ('G2', 'W'), ('G2', 'P'), ('G2', 'S'), ('G2', pm),
        ('well_count', 'W'), ('well_count', 'P'), ('well_count', 'S'), ('well_count', pm),
        ('W', 'P'), ('W', 'S'), ('W', pm),
        ('P', 'S'),
        (pm, 'S'),
    ])

    # Step 1: Path a (W → P) using DoWhy
    try:
        dag_dot = nx.nx_pydot.to_pydot(G).to_string()
        model_a_dowhy = CausalModel(data, treatment='W', outcome='P', graph=dag_dot)
        estimand_a = model_a_dowhy.identify_effect()
        estimate_a = model_a_dowhy.estimate_effect(
            estimand_a,
            method_name="backdoor.linear_regression",
            control_value=0, treatment_value=1,
        )
        a_coef_dowhy = estimate_a.value

        # Get p-value from statsmodels for path a
        X_a = sm.add_constant(data[['W', 'G1', 'G2', 'well_count', P_MISSING_COL]])
        model_a_sm = sm.OLS(data['P'], X_a).fit()
        a_pval = model_a_sm.pvalues['W']

    except Exception as e:
        print(f"Warning: DoWhy path a estimation failed: {e}")
        # Fallback to statsmodels
        model_a_sm = sm.OLS(data['P'], sm.add_constant(data[['W', 'G1', 'G2', 'well_count', P_MISSING_COL]])).fit()
        a_coef_dowhy = model_a_sm.params['W']
        a_pval = model_a_sm.pvalues['W']

    # Step 2: Path b (P → S | W) using DoWhy
    try:
        dag_dot = nx.nx_pydot.to_pydot(G).to_string()
        model_b_dowhy = CausalModel(data, treatment='P', outcome='S', graph=dag_dot)
        estimand_b = model_b_dowhy.identify_effect()
        estimate_b = model_b_dowhy.estimate_effect(
            estimand_b,
            method_name="backdoor.linear_regression",
            control_value=0, treatment_value=1,
        )
        b_coef_dowhy = estimate_b.value

        # Get p-value from statsmodels for path b
        X_b = sm.add_constant(data[['P', 'W', 'G1', 'G2', 'well_count', P_MISSING_COL]])
        model_b_sm = sm.OLS(data['S'], X_b).fit()
        b_pval = model_b_sm.pvalues['P']

    except Exception as e:
        print(f"Warning: DoWhy path b estimation failed: {e}")
        # Fallback to statsmodels
        model_b_sm = sm.OLS(data['S'], sm.add_constant(data[['P', 'W', 'G1', 'G2', 'well_count', P_MISSING_COL]])).fit()
        b_coef_dowhy = model_b_sm.params['P']
        b_pval = model_b_sm.pvalues['P']

    # Step 3: Total effect (W → S) using DoWhy
    try:
        dag_dot = nx.nx_pydot.to_pydot(G).to_string()
        model_c_dowhy = CausalModel(data, treatment='W', outcome='S', graph=dag_dot)
        estimand_c = model_c_dowhy.identify_effect()
        estimate_c = model_c_dowhy.estimate_effect(
            estimand_c,
            method_name="backdoor.linear_regression",
            control_value=0, treatment_value=1,
        )
        c_coef_dowhy = estimate_c.value

        # Get p-value and R² from statsmodels for total effect
        X_c = sm.add_constant(data[['W', 'G1', 'G2', 'well_count', P_MISSING_COL]])
        model_c_sm = sm.OLS(data['S'], X_c).fit()
        c_pval = model_c_sm.pvalues['W']
        model_c_r2 = model_c_sm.rsquared

    except Exception as e:
        print(f"Warning: DoWhy total effect estimation failed: {e}")
        # Fallback to statsmodels
        model_c_sm = sm.OLS(data['S'], sm.add_constant(data[['W', 'G1', 'G2', 'well_count', P_MISSING_COL]])).fit()
        c_coef_dowhy = model_c_sm.params['W']
        c_pval = model_c_sm.pvalues['W']
        model_c_r2 = model_c_sm.rsquared

    # Step 4: Direct effect (W → S | P) - use statsmodels for consistency
    model_direct = sm.OLS(data['S'], sm.add_constant(data[['P', 'W', 'G1', 'G2', 'well_count', P_MISSING_COL]])).fit()
    c_prime_coef = model_direct.params['W']
    c_prime_pval = model_direct.pvalues['W']

    return {
        'total_effect': c_coef_dowhy,
        'total_pval': c_pval,
        'direct_effect': c_prime_coef,
        'direct_pval': c_prime_pval,
        'indirect_effect': a_coef_dowhy * b_coef_dowhy,
        'path_a_coef': a_coef_dowhy,
        'path_a_pval': a_pval,
        'path_b_coef': b_coef_dowhy,
        'path_b_pval': b_pval,
        'model_c_r2': model_c_r2,
        'dowhy_estimates': {
            'path_a': estimate_a if 'estimate_a' in locals() else None,
            'path_b': estimate_b if 'estimate_b' in locals() else None,
            'total': estimate_c if 'estimate_c' in locals() else None
        }
    }


def bootstrap_mediation_effects_dowhy(data, n_bootstrap=100):
    """Calculate bootstrap confidence intervals for DoWhy mediation effects"""
    total_effects = []
    direct_effects = []
    indirect_effects = []

    for i in range(n_bootstrap):
        boot_data = resample(data, random_state=i, replace=True).reset_index(drop=True)

        try:
            boot_effects = calculate_mediation_effects_dowhy(boot_data)
            total_effects.append(boot_effects['total_effect'])
            direct_effects.append(boot_effects['direct_effect'])
            indirect_effects.append(boot_effects['indirect_effect'])
        except Exception as e:
            continue

    if len(total_effects) < n_bootstrap * 0.8:
        print(f"Warning: Only {len(total_effects)}/{n_bootstrap} bootstrap iterations succeeded")

    total_ci = np.percentile(total_effects, [2.5, 97.5]) if total_effects else [np.nan, np.nan]
    direct_ci = np.percentile(direct_effects, [2.5, 97.5]) if direct_effects else [np.nan, np.nan]
    indirect_ci = np.percentile(indirect_effects, [2.5, 97.5]) if indirect_effects else [np.nan, np.nan]

    return {
        'total_ci': total_ci,
        'direct_ci': direct_ci,
        'indirect_ci': indirect_ci,
        'bootstrap_success_rate': len(total_effects) / n_bootstrap
    }


def run_dowhy_refutations(data):
    """Run DoWhy refutation tests on the aggregated event-level data"""
    try:
        pm = P_MISSING_COL
        G = nx.DiGraph([
            ('G1', 'W'), ('G1', 'P'), ('G1', 'S'), ('G1', pm),
            ('G2', 'W'), ('G2', 'P'), ('G2', 'S'), ('G2', pm),
            ('well_count', 'W'), ('well_count', 'P'), ('well_count', 'S'), ('well_count', pm),
            ('W', 'P'), ('P', 'S'), ('W', 'S'), ('W', pm),
            (pm, 'S'),
        ])

        # Total effect model for refutations
        dag_dot = nx.nx_pydot.to_pydot(G).to_string()
        model = CausalModel(data, treatment='W', outcome='S', graph=dag_dot)
        estimand = model.identify_effect()
        estimate = model.estimate_effect(
            estimand,
            method_name="backdoor.linear_regression",
            control_value=0, treatment_value=1,
        )

        # Placebo treatment refutation
        placebo_refute = model.refute_estimate(
            estimand, estimate,
            method_name="placebo_treatment_refuter",
            placebo_type="permute"
        )

        # Data subset refutation
        subset_refute = model.refute_estimate(
            estimand, estimate,
            method_name="data_subset_refuter",
            subset_fraction=0.8
        )

        # Add common cause refutation (add random confounder)
        try:
            random_refute = model.refute_estimate(
                estimand, estimate,
                method_name="add_unobserved_common_cause",
                confounders_effect_on_treatment="binary_flip",
                confounders_effect_on_outcome="linear",
                effect_strength_on_treatment=0.01,
                effect_strength_on_outcome=0.02
            )
            random_effect = random_refute.new_effect
        except:
            random_effect = np.nan

        return {
            'placebo_effect': placebo_refute.new_effect,
            'subset_effect': subset_refute.new_effect,
            'random_confounder_effect': random_effect,
            'original_effect': estimate.value
        }

    except Exception as e:
        print(f"Warning: DoWhy refutations failed: {e}")
        return {
            'placebo_effect': np.nan,
            'subset_effect': np.nan,
            'random_confounder_effect': np.nan,
            'original_effect': np.nan
        }


def calculate_predictive_r2(data):
    """Calculate predictive R² with log transformation for event-level data"""
    try:
        X = data[['W', 'P', 'G1', 'G2', 'well_count', P_MISSING_COL]].copy()
        X['W'] = np.log1p(X['W'].clip(lower=0))  # Log transform volumes (non-negative)
        X['well_count'] = np.log1p(X['well_count'].clip(lower=0))  # Log transform well count
        y = data['S']

        # Train-test split
        x_tr, x_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.20, random_state=42, shuffle=True
        )

        # Fit and predict
        pred = LinearRegression().fit(x_tr, y_tr).predict(x_te)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_te, pred))
        r2 = r2_score(y_te, pred)

        return {'predictive_r2': r2, 'rmse': rmse}

    except Exception as e:
        print(f"Warning: Predictive R² calculation failed: {e}")
        return {'predictive_r2': np.nan, 'rmse': np.nan}


# =============================================================================
# CHANGE 7: Hurdle model — two-stage framework separating occurrence from magnitude
# =============================================================================

def fit_hurdle_model(data):
    """CHANGE 7: Two-stage hurdle model for event-level aggregated data.

    Identical logic to the well-level version in dowhy_ci.py. At event level,
    S is magnitude aggregated per event, so S > 0 identifies events with a
    detected earthquake. Non-event control rows have S = 0.

    Stage 1: GBM binary classifier predicts P(S > 0) using W, P, G1, G2,
             P_missing, well_count.
    Stage 2: GBM regressor predicts magnitude conditional on S > 0.

    Formation depth (shallow vs. Ellenburger) is a known missing confounder
    for Stage 1 — not available in RRC data without external well records.
    """
    _nan_result = {
        'hurdle_occurrence_prob': np.nan,
        'hurdle_conditional_magnitude': np.nan,
        'hurdle_stage1_auc': np.nan,
        'hurdle_stage2_mae': np.nan,
    }

    try:
        feature_cols = [c for c in ['W', 'P', 'G1', 'G2', 'well_count', P_MISSING_COL]
                        if c in data.columns]
        X = data[feature_cols].copy()
        S_binary = (data['S'] > 0).astype(int)
        S_magnitude = data['S'].copy()

        if S_binary.nunique() < 2:
            print("  ⚠️  Hurdle Stage 1: only one class in S_binary — skipping.")
            return _nan_result

        from sklearn.model_selection import train_test_split
        x_tr, x_te, yb_tr, yb_te, ym_tr, ym_te = train_test_split(
            X, S_binary, S_magnitude, test_size=0.20, random_state=42, shuffle=True
        )

        # Stage 1 — binary occurrence classifier
        clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
        clf.fit(x_tr, yb_tr)
        occ_prob = clf.predict_proba(x_te)[:, 1]
        stage1_auc = float(roc_auc_score(yb_te, occ_prob)) if yb_te.nunique() > 1 else np.nan

        # Stage 2 — conditional magnitude regression (events only)
        event_idx_tr = yb_tr[yb_tr == 1].index
        event_idx_te = yb_te[yb_te == 1].index

        if len(event_idx_tr) < 10:
            print(f"  ⚠️  Hurdle Stage 2: only {len(event_idx_tr)} event rows in train — skipping Stage 2.")
            return {
                'hurdle_occurrence_prob': float(occ_prob.mean()),
                'hurdle_conditional_magnitude': np.nan,
                'hurdle_stage1_auc': stage1_auc,
                'hurdle_stage2_mae': np.nan,
            }

        reg = GradientBoostingRegressor(n_estimators=100, random_state=42)
        reg.fit(x_tr.loc[event_idx_tr], ym_tr.loc[event_idx_tr])

        if len(event_idx_te) > 0:
            mag_pred = reg.predict(x_te.loc[event_idx_te])
            stage2_mae = float(mean_absolute_error(ym_te.loc[event_idx_te], mag_pred))
            cond_mag = float(mag_pred.mean())
        else:
            stage2_mae = np.nan
            cond_mag = np.nan

        cond_str = f"{cond_mag:.3f}" if cond_mag is not None and not np.isnan(cond_mag) else "n/a"
        mae_str = f"{stage2_mae:.3f}" if stage2_mae is not None and not np.isnan(stage2_mae) else "n/a"
        print(f"  ✅ Hurdle model: P(occur)={float(occ_prob.mean()):.3f}, "
              f"AUC={stage1_auc:.3f}, E[mag|occur]={cond_str}, Stage2 MAE={mae_str}")

        return {
            'hurdle_occurrence_prob': float(occ_prob.mean()),
            'hurdle_conditional_magnitude': cond_mag,
            'hurdle_stage1_auc': stage1_auc,
            'hurdle_stage2_mae': stage2_mae,
        }

    except Exception as e:
        print(f"  Warning: Hurdle model failed: {e}")
        return _nan_result

# =============================================================================
# END CHANGE 7
# =============================================================================


def process_file(csv_file):
    """Process a single file and return results"""
    print(f"\n{'=' * 60}")
    print(f"PROCESSING: {csv_file}")
    print(f"{'=' * 60}")

    radius = extract_radius(csv_file)

    try:
        # Load the data
        df_raw = pd.read_csv(csv_file, low_memory=False)
        print(f"✅ Loaded {len(df_raw):,} rows")

        # Auto-detect columns (same as original aggregated version)
        COL_FRAGS = {
            "E": ["event", "id"],  # EventID
            "W": ["volume", "bbl"],  # Treatment – injection volume
            "P": ["pressure", "psig"],  # Mediator – average injection pressure
            "S": ["local", "mag"],  # Outcome – earthquake magnitude
            "G1": ["nearest", "fault", "dist"],  # Confounder 1
            "G2": ["fault", "segment"],  # Confounder 2
        }

        found = {}
        for k, frags in COL_FRAGS.items():
            col = find_column(frags, df_raw.columns)
            if col:
                found[k] = col

        # Check if all required columns were found
        missing = [k for k in COL_FRAGS.keys() if k not in found]
        if missing:
            print(f"❌ Missing columns: {missing}")
            return None

        # Rename columns to safe identifiers
        rename_map = {found[k]: safe(k) for k in found}
        df = df_raw[list(found.values())].rename(columns=rename_map)

        p_col = safe("P")
        df[P_MISSING_COL] = df[p_col].isna().astype(np.float64)
        df[p_col] = df[p_col].fillna(0.0)

        print(f"Raw data before aggregation: {len(df):,} rows")

        # ═══ AGGREGATE BY EVENT ID ═══
        print("🔄 Aggregating wells by EventID...")

        # Group by EventID and aggregate
        aggregated = (
            df
            .groupby(safe("E"))
            .agg({
                safe("W"): "sum",  # Sum of injection volumes
                safe("P"): "median",  # Median injection pressure
                P_MISSING_COL: "mean",  # Fraction of wells with missing pressure
                safe("G1"): "min",  # Min nearest fault distance
                safe("G2"): "sum",  # Sum of fault segments
                safe("S"): "first",  # Magnitude (same within event)
            })
            .rename_axis(None)
            .reset_index()
        )

        # Add well count per event
        aggregated["well_count"] = df.groupby(safe("E")).size().values

        print(f"✅ Aggregated to {len(aggregated):,} events")

        # Clean the aggregated data
        essential = [safe("W"), safe("S")]
        rows_before = len(aggregated)
        aggregated = aggregated.dropna(subset=essential)
        rows_after = len(aggregated)

        if rows_after == 0:
            print("❌ No events retain both treatment and outcome")
            return None

        covariates = [c for c in [safe("P"), P_MISSING_COL, safe("G1"), safe("G2"), "well_count"]
                      if c in aggregated.columns]
        covariates_median = [c for c in covariates if c not in (safe("P"), P_MISSING_COL)]
        if covariates_median:
            aggregated[covariates_median] = aggregated[covariates_median].fillna(
                aggregated[covariates_median].median(numeric_only=True)
            )

        # Rename columns to standard names
        data = aggregated.rename(columns={
            safe("W"): "W",
            safe("P"): "P",
            safe("S"): "S",
            safe("G1"): "G1",
            safe("G2"): "G2",
        })

        print(f"✅ Clean aggregated data: {len(data):,} events (dropped {rows_before - rows_after:,})")
        print(f"   Mean fraction wells with missing pressure: {100.0 * data[P_MISSING_COL].mean():.2f}%")
        print(f"   Average wells per event: {data['well_count'].mean():.1f}")

        # Calculate DoWhy mediation effects
        print("🔄 Running DoWhy mediation analysis...")
        effects = calculate_mediation_effects_dowhy(data)

        # Bootstrap confidence intervals with DoWhy
        print("🔄 Running bootstrap confidence intervals...")
        ci_results = bootstrap_mediation_effects_dowhy(data, n_bootstrap=50)

        # Run DoWhy refutations
        print("🔄 Running DoWhy refutation tests...")
        refutations = run_dowhy_refutations(data)

        # Calculate predictive R²
        print("🔄 Calculating predictive metrics...")
        pred_metrics = calculate_predictive_r2(data)

        # --- CHANGE 7: Hurdle model ---
        print("🔄 Fitting hurdle model (Change 7)...")
        hurdle = fit_hurdle_model(data)
        # --- END CHANGE 7 ---

        # Calculate VIFs (including well_count)
        X_a = sm.add_constant(data[['W', 'G1', 'G2', 'well_count', P_MISSING_COL]])
        vif_a = calculate_vif(X_a)
        vif_a_avg = vif_a['VIF'].mean()

        X_b = sm.add_constant(data[['W', 'P', 'G1', 'G2', 'well_count', P_MISSING_COL]])
        vif_b = calculate_vif(X_b)
        vif_b_avg = vif_b['VIF'].mean()

        # Calculate proportion mediated
        prop_mediated = (effects['indirect_effect'] / effects['total_effect']) * 100 if effects[
                                                                                            'total_effect'] != 0 else 0

        print(f"✅ DoWhy event-level analysis complete")
        print(f"   Total effect: {effects['total_effect']:+.3e} (p={effects['total_pval']:.3e})")
        print(f"   Direct effect: {effects['direct_effect']:+.3e} (p={effects['direct_pval']:.3e})")
        print(f"   Indirect effect: {effects['indirect_effect']:+.3e}")
        print(f"   Proportion mediated: {prop_mediated:.1f}%")
        print(f"   Bootstrap success rate: {ci_results['bootstrap_success_rate']:.1%}")
        print(f"   Predictive R²: {pred_metrics['predictive_r2']:.3f}")

        return {
            'radius': radius,
            'filename': csv_file,
            'n_events': len(data),
            'avg_well_count': data['well_count'].mean(),
            'total_effect': effects['total_effect'],
            'total_pval': effects['total_pval'],
            'total_ci_low': ci_results['total_ci'][0],
            'total_ci_high': ci_results['total_ci'][1],
            'direct_effect': effects['direct_effect'],
            'direct_pval': effects['direct_pval'],
            'direct_ci_low': ci_results['direct_ci'][0],
            'direct_ci_high': ci_results['direct_ci'][1],
            'indirect_effect': effects['indirect_effect'],
            'indirect_ci_low': ci_results['indirect_ci'][0],
            'indirect_ci_high': ci_results['indirect_ci'][1],
            'prop_mediated': prop_mediated,
            'path_a_coef': effects['path_a_coef'],
            'path_a_pval': effects['path_a_pval'],
            'path_b_coef': effects['path_b_coef'],
            'path_b_pval': effects['path_b_pval'],
            'causal_r2': effects['model_c_r2'],
            'predictive_r2': pred_metrics['predictive_r2'],
            'rmse': pred_metrics['rmse'],
            'vif_a_avg': vif_a_avg,
            'vif_b_avg': vif_b_avg,
            'bootstrap_success_rate': ci_results['bootstrap_success_rate'],
            'placebo_effect': refutations['placebo_effect'],
            'subset_effect': refutations['subset_effect'],
            'random_confounder_effect': refutations['random_confounder_effect'],
            'dowhy_original_effect': refutations['original_effect'],
            'pressure_missing_pct': 100.0 * float(data[P_MISSING_COL].mean()),
            # CHANGE 7: hurdle model outputs
            'hurdle_occurrence_prob': hurdle['hurdle_occurrence_prob'],
            'hurdle_conditional_magnitude': hurdle['hurdle_conditional_magnitude'],
            'hurdle_stage1_auc': hurdle['hurdle_stage1_auc'],
            'hurdle_stage2_mae': hurdle['hurdle_stage2_mae'],
        }

    except Exception as e:
        print(f"❌ Error processing {csv_file}: {str(e)}")
        return None


# Find event files
csv_files = find_csv_files()
event_files = [f for f in csv_files if 'event_well_links_with_faults' in f]

if not event_files:
    print("❌ No event files found!")
    exit(1)

# Sort event files by radius
event_files.sort(key=lambda x: extract_radius(x))

print(f"\n🎯 Found {len(event_files)} event files to process:")
for i, file in enumerate(event_files):
    radius = extract_radius(file)
    size = os.path.getsize(file) / (1024 * 1024)
    print(f"  {i + 1:2d}. {radius:2d}km - {file} ({size:.1f} MB)")

# Process all files
print(f"\n{'=' * 80}")
print("BATCH DOWHY EVENT-LEVEL MEDIATION ANALYSIS WITH BOOTSTRAP CI")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'=' * 80}")

results = []
for i, csv_file in enumerate(event_files):
    print(f"\n[{i + 1}/{len(event_files)}] Processing {csv_file}...")

    result = process_file(csv_file)
    if result:
        results.append(result)

# Create summary table
if results:
    results_df = pd.DataFrame(results).sort_values('radius')

    print(f"\n{'=' * 150}")
    print("📊 COMPREHENSIVE DOWHY EVENT-LEVEL MEDIATION ANALYSIS SUMMARY")
    print(f"{'=' * 150}")

    # Main results table
    print("\n🎯 CAUSAL EFFECTS BY RADIUS (DOWHY EVENT-LEVEL AGGREGATED):")
    print("─" * 150)
    print(
        f"{'Radius':>6} {'Events':>7} {'Avg Wells':>9} {'Total Effect':>13} {'95% CI':>20} {'Direct Effect':>14} {'95% CI':>20} {'Indirect':>11} {'% Med':>6} {'Causal R²':>9} {'Pred R²':>8}")
    print("─" * 150)

    for _, row in results_df.iterrows():
        print(f"{row['radius']:>4}km {row['n_events']:>7,} "
              f"{row['avg_well_count']:>8.1f} "
              f"{row['total_effect']:>+10.3e} "
              f"[{row['total_ci_low']:>+7.2e},{row['total_ci_high']:>+7.2e}] "
              f"{row['direct_effect']:>+11.3e} "
              f"[{row['direct_ci_low']:>+7.2e},{row['direct_ci_high']:>+7.2e}] "
              f"{row['indirect_effect']:>+8.2e} "
              f"{row['prop_mediated']:>5.1f}% "
              f"{row['causal_r2']:>8.3f} "
              f"{row['predictive_r2']:>7.3f}")

    print("─" * 150)

    # Statistical significance and quality metrics table
    print("\n📈 STATISTICAL SIGNIFICANCE & QUALITY METRICS:")
    print("─" * 120)
    print(
        f"{'Radius':>6} {'Total p-val':>12} {'Direct p-val':>13} {'Path a p-val':>13} {'Path b p-val':>13} {'Bootstrap':>10} {'VIF':>6}")
    print("─" * 120)

    for _, row in results_df.iterrows():
        total_sig = "***" if row['total_pval'] < 0.001 else "**" if row['total_pval'] < 0.01 else "*" if row[
                                                                                                             'total_pval'] < 0.05 else ""
        direct_sig = "***" if row['direct_pval'] < 0.001 else "**" if row['direct_pval'] < 0.01 else "*" if row[
                                                                                                                'direct_pval'] < 0.05 else ""
        path_a_sig = "***" if row['path_a_pval'] < 0.001 else "**" if row['path_a_pval'] < 0.01 else "*" if row[
                                                                                                                'path_a_pval'] < 0.05 else ""
        path_b_sig = "***" if row['path_b_pval'] < 0.001 else "**" if row['path_b_pval'] < 0.01 else "*" if row[
                                                                                                                'path_b_pval'] < 0.05 else ""

        print(f"{row['radius']:>4}km "
              f"{row['total_pval']:>8.2e}{total_sig:<3} "
              f"{row['direct_pval']:>8.2e}{direct_sig:<3} "
              f"{row['path_a_pval']:>8.2e}{path_a_sig:<3} "
              f"{row['path_b_pval']:>8.2e}{path_b_sig:<3} "
              f"{row['bootstrap_success_rate']:>8.1%} "
              f"{row['vif_b_avg']:>5.2f}")

    print("─" * 120)
    print("Significance: *** p<0.001, ** p<0.01, * p<0.05")

    # DoWhy refutation results
    print("\n🔍 DOWHY REFUTATION TESTS:")
    print("─" * 100)
    print(f"{'Radius':>6} {'Original':>12} {'Placebo':>12} {'Subset':>12} {'Random Conf':>12} {'Status':>15}")
    print("─" * 100)

    for _, row in results_df.iterrows():
        placebo_ratio = abs(row['placebo_effect'] / row['total_effect']) if row['total_effect'] != 0 else np.inf
        subset_ratio = abs((row['subset_effect'] - row['total_effect']) / row['total_effect']) if row[
                                                                                                      'total_effect'] != 0 else np.inf

        # Status assessment
        status = "✅ PASS" if placebo_ratio < 0.1 and subset_ratio < 0.2 else "⚠️ CAUTION" if placebo_ratio < 0.3 else "❌ FAIL"

        print(f"{row['radius']:>4}km "
              f"{row['total_effect']:>+9.2e} "
              f"{row['placebo_effect']:>+9.2e} "
              f"{row['subset_effect']:>+9.2e} "
              f"{row['random_confounder_effect']:>+9.2e} "
              f"{status:>15}")

    print("─" * 100)

    # Key insights
    print("\n🔍 KEY INSIGHTS:")
    print("─" * 70)

    strongest_total = results_df.loc[results_df['total_effect'].abs().idxmax()]
    highest_mediation = results_df.loc[results_df['prop_mediated'].idxmax()]
    best_causal_r2 = results_df.loc[results_df['causal_r2'].idxmax()]
    best_pred_r2 = results_df.loc[results_df['predictive_r2'].idxmax()]

    print(f"• Strongest total effect:    {strongest_total['radius']}km ({strongest_total['total_effect']:+.3e})")
    print(f"• Highest mediation:         {highest_mediation['radius']}km ({highest_mediation['prop_mediated']:.1f}%)")
    print(f"• Best causal model fit:     {best_causal_r2['radius']}km (R²={best_causal_r2['causal_r2']:.3f})")
    print(f"• Best predictive fit:       {best_pred_r2['radius']}km (R²={best_pred_r2['predictive_r2']:.3f})")
    print(f"• Event sample size range:   {results_df['n_events'].min():,} - {results_df['n_events'].max():,} events")
    print(
        f"• Wells per event range:     {results_df['avg_well_count'].min():.1f} - {results_df['avg_well_count'].max():.1f}")

    # Mediation patterns
    near_field = results_df[results_df['radius'] <= 5]
    far_field = results_df[results_df['radius'] >= 10]

    if len(near_field) > 0 and len(far_field) > 0:
        print(f"• Near-field mediation:      {near_field['prop_mediated'].mean():.1f}% (≤5km)")
        print(f"• Far-field mediation:       {far_field['prop_mediated'].mean():.1f}% (≥10km)")

    # Quality metrics
    avg_bootstrap_success = results_df['bootstrap_success_rate'].mean()
    avg_vif = results_df['vif_b_avg'].mean()

    print(f"• Average bootstrap success: {avg_bootstrap_success:.1%}")
    print(f"• Average VIF:               {avg_vif:.2f}")

    # Refutation quality assessment
    passed_refutations = sum([
        abs(row['placebo_effect'] / row['total_effect']) < 0.1 and
        abs((row['subset_effect'] - row['total_effect']) / row['total_effect']) < 0.2
        for _, row in results_df.iterrows()
        if row['total_effect'] != 0 and not np.isnan(row['placebo_effect'])
    ])

    print(
        f"• Refutation tests passed:   {passed_refutations}/{len(results_df)} ({passed_refutations / len(results_df):.1%})")

    print("─" * 70)

    # Event-level vs Well-level comparison insights
    print("\n📊 EVENT-LEVEL AGGREGATION BENEFITS:")
    print("─" * 50)
    print("• Reduced noise from duplicate earthquake magnitudes")
    print("• Accounts for multiple wells affecting single events")
    print("• Better signal-to-noise ratio for causal inference")
    print("• Well count included as additional confounder")
    print("• More interpretable units of analysis (events vs well-pairs)")
    print("─" * 50)

    # Save results to CSV
    output_file = f"dowhy_event_level_mediation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")

    # Save detailed summary with methodology notes
    summary_file = f"dowhy_event_methodology_notes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(summary_file, 'w') as f:
        f.write("DoWhy Event-Level Mediation Analysis - Methodology Notes\n")
        f.write("=" * 60 + "\n\n")
        f.write("DATA AGGREGATION:\n")
        f.write("- Raw data aggregated by EventID (earthquake events)\n")
        f.write("- Injection volumes: summed across wells per event\n")
        f.write("- Injection pressure: median across wells per event\n")
        f.write("- Fault distance: minimum across wells per event\n")
        f.write("- Fault segments: summed across wells per event\n")
        f.write("- Well count: number of wells per event (added as confounder)\n\n")

        f.write("CAUSAL MODEL:\n")
        f.write("- Treatment (W): Total injection volume per event\n")
        f.write("- Mediator (P): Median injection pressure per event\n")
        f.write("- Outcome (S): Earthquake magnitude\n")
        f.write("- Confounders: G1 (fault distance), G2 (fault segments), well_count\n\n")

        f.write("DOWHY FRAMEWORK:\n")
        f.write("- Explicit causal DAG with backdoor adjustment\n")
        f.write("- Bootstrap confidence intervals (50 iterations)\n")
        f.write("- Refutation tests: placebo, subset, random confounder\n")
        f.write("- Dual R² metrics: causal (linear) and predictive (log-transformed)\n\n")

        f.write("QUALITY METRICS:\n")
        f.write(f"- Average bootstrap success rate: {avg_bootstrap_success:.1%}\n")
        f.write(f"- Average VIF (multicollinearity): {avg_vif:.2f}\n")
        f.write(
            f"- Refutation tests passed: {passed_refutations}/{len(results_df)} ({passed_refutations / len(results_df):.1%})\n")

    print(f"📝 Methodology notes saved to: {summary_file}")

else:
    print("\n❌ No valid results obtained from any files.")

print(f"\n{'=' * 80}")
print(f"DoWhy Event-Level Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'=' * 80}")